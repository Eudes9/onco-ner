# training/train_optimized.py
"""
Entraînement optimisé de DrBERT avec :
1. Sliding window : couvre l'intégralité des documents (stride=256, window=512)
2. Class weights (sqrt) : compense le déséquilibre differenciation (1:24)
3. Hyperparamètres affinés depuis l'analyse d'erreurs Sprint 5

Corrections appliquées :
- Bug BIO sliding window : entity_started global sur tout le document
- Class weights : racine carrée + O à 1.0 (évite hallucinations)
- Overlap détecté via offsets caractères (pas token_idx)
- processing_class supprimé du Trainer
"""

import os
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
import evaluate

from onco_ner.utils.logging import get_logger
from onco_ner.utils.seed import set_seed

logger = get_logger(__name__)

LABELS = [
    "O",
    "B-morphologie", "I-morphologie",
    "B-topographie", "I-topographie",
    "B-differenciation", "I-differenciation",
    "B-expression_CIM", "I-expression_CIM",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


def load_splits(splits_dir: Path) -> tuple:
    """Charge les splits Parquet produits par preprocessing.py."""
    train_df = pl.read_parquet(splits_dir / "train.parquet")
    val_df = pl.read_parquet(splits_dir / "val.parquet")
    return train_df, val_df


def compute_class_weights(train_df: pl.DataFrame) -> torch.Tensor:
    """
    Calcule les class weights depuis la distribution des entités.

    Formule : weight = sqrt(max_count / count)
    La racine carrée atténue le déséquilibre sans pénaliser
    agressivement la classe majoritaire O.

    La classe O garde un poids de 1.0 — la sous-pondérer
    provoquerait des hallucinations d'entités dans des contextes
    sains (négations, anatomie normale, historique patient).
    """
    label_counts = {
        row["label"]: row["count"]
        for row in train_df["label"].value_counts().to_dicts()
    }

    max_count = max(label_counts.values())
    weights = torch.ones(len(LABELS))

    for label_idx, label in enumerate(LABELS):
        if label == "O":
            weights[label_idx] = 1.0
            continue

        entity_type = label.split("-")[1]
        count = label_counts.get(entity_type, 1)
        # Lissage par racine carrée : atténue sans écraser
        weights[label_idx] = (max_count / count) ** 0.5

    logger.info("Class weights (sqrt) :")
    for label, w in zip(LABELS, weights.tolist()):
        logger.info(f"  {label:30s} : {w:.3f}")

    return weights


def build_sliding_window_dataset(
    df: pl.DataFrame,
    tokenizer,
    txt_dir: Path,
    max_length: int = 512,
    stride: int = 256,
) -> Dataset:
    """
    Construit un dataset avec sliding window pour couvrir
    l'intégralité des documents longs (98.5% > 512 tokens).

    Corrections appliquées :
    1. entity_started global sur TOUT le document (pas par chunk)
       Une entité qui chevauche deux chunks reçoit B- dans le premier
       chunk et I- dans le suivant — jamais deux B- pour la même entité.

    2. Zone de chevauchement détectée via offsets caractères
       (pas via token_idx) pour éviter le décalage des tokens spéciaux.
       Un token est dans la zone de chevauchement si son offset_start
       est < offset_start du chunk précédent + stride.

    3. Les tokens de chevauchement reçoivent -100 (ignorés par la loss)
       mais entity_started est quand même mis à jour pour préserver
       la cohérence BIO entre chunks.
    """
    examples = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    doc_names = df["doc_name"].unique().to_list()
    total_chunks = 0

    for doc_name in doc_names:
        txt_name = doc_name.replace(".ann", ".txt")
        txt_path = txt_dir / txt_name

        if not txt_path.exists():
            logger.warning(f"Fichier manquant : {txt_path}, ignoré")
            continue

        text = txt_path.read_text(encoding="utf-8")

        doc_df = df.filter(pl.col("doc_name") == doc_name)
        span_to_label = {
            (row["span_start"], row["span_end"]): row["label"]
            for row in doc_df.to_dicts()
        }

        # Tokenizer avec sliding window
        encoding = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # entity_started global sur TOUT le document
        # Réinitialisé une seule fois par document, jamais par chunk
        entity_started = {span: False for span in span_to_label}

        # Détecter l'offset caractère de début de chaque chunk
        # via le premier token non spécial de chaque chunk
        chunk_start_offsets = []
        for offset_mapping in encoding["offset_mapping"]:
            first_real_offset = next(
                (
                    off[0]
                    for off in offset_mapping
                    if not (off[0] == 0 and off[1] == 0)
                ),
                0,
            )
            chunk_start_offsets.append(first_real_offset)

        for chunk_idx, (input_ids, attention_mask, offset_mapping) in enumerate(
            zip(
                encoding["input_ids"],
                encoding["attention_mask"],
                encoding["offset_mapping"],
            )
        ):
            labels = []

            # Offset de fin de la zone de chevauchement pour ce chunk
            # = offset de début du chunk précédent + stride (en caractères)
            if chunk_idx > 0:
                prev_chunk_start = chunk_start_offsets[chunk_idx - 1]
                overlap_end_char = prev_chunk_start + stride
            else:
                overlap_end_char = 0

            for offset_start, offset_end in offset_mapping:
                # Tokens spéciaux ([CLS], [SEP], [PAD]) -> offset (0, 0)
                if offset_start == 0 and offset_end == 0:
                    labels.append(-100)
                    continue

                # Zone de chevauchement via offset caractère
                in_overlap = (
                    chunk_idx > 0 and offset_start < overlap_end_char
                )

                token_label = "O"
                for (span_start, span_end), label in span_to_label.items():
                    if offset_start >= span_start and offset_end <= span_end:
                        if not entity_started[(span_start, span_end)]:
                            token_label = f"B-{label}"
                            entity_started[(span_start, span_end)] = True
                        else:
                            token_label = f"I-{label}"
                        break

                # Zone de chevauchement : -100 dans la loss
                # mais entity_started a été mis à jour ci-dessus
                if in_overlap:
                    labels.append(-100)
                else:
                    labels.append(LABEL2ID[token_label])

            examples["input_ids"].append(input_ids)
            examples["attention_mask"].append(attention_mask)
            examples["labels"].append(labels)
            total_chunks += 1

    logger.info(
        f"{len(doc_names)} documents -> {total_chunks} chunks "
        f"(window={max_length}, stride={stride})"
    )
    return Dataset.from_dict(examples)


def compute_metrics(eval_pred, metric):
    """Calcule precision/recall/F1 avec seqeval."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [ID2LABEL[l] for l in label if l != -100]
        for label in labels
    ]
    true_predictions = [
        [ID2LABEL[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(
        predictions=true_predictions,
        references=true_labels,
    )
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


class WeightedLossTrainer(Trainer):
    """
    Trainer personnalisé avec class weights dans la CrossEntropyLoss.
    Compense le déséquilibre differenciation (82 ex test) vs autres classes.
    Utilise une racine carrée pour un lissage progressif.
    """

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device),
            ignore_index=-100,
        )
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train_optimized(cfg: DictConfig) -> None:
    logger.info(f"Config :\n{OmegaConf.to_yaml(cfg)}")

    set_seed(cfg.training.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device : {device}")

    # Chemins
    splits_dir = Path(cfg.paths.splits_dir)
    txt_dir = Path(cfg.paths.data_dir) / "ann_txt_files"
    output_dir = (
        Path(cfg.models.output_dir).parent / f"{cfg.models.name}_optimized"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chargement des splits
    logger.info("Chargement des splits...")
    train_df, val_df = load_splits(splits_dir)

    # Tokenizer
    logger.info(f"Chargement du tokenizer : {cfg.models.pretrained_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.models.pretrained_model_name,
        use_fast=True,
    )

    # Datasets avec sliding window
    logger.info("Construction des datasets avec sliding window...")
    stride = cfg.training.max_length // 2

    train_dataset = build_sliding_window_dataset(
        train_df, tokenizer, txt_dir,
        max_length=cfg.training.max_length,
        stride=stride,
    )
    val_dataset = build_sliding_window_dataset(
        val_df, tokenizer, txt_dir,
        max_length=cfg.training.max_length,
        stride=stride,
    )

    logger.info(f"Train : {len(train_dataset)} chunks")
    logger.info(f"Val   : {len(val_dataset)} chunks")

    # Modèle
    logger.info(f"Chargement du modèle : {cfg.models.pretrained_model_name}")
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.models.pretrained_model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Class weights
    class_weights = compute_class_weights(train_df)

    # Métrique
    metric = evaluate.load("seqeval")

    # W&B
    os.environ["WANDB_PROJECT"] = cfg.wandb.project
    os.environ["WANDB_RUN_NAME"] = f"{cfg.models.name}_optimized"

    # Arguments d'entraînement
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.models.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="wandb",
        seed=cfg.training.seed,
        fp16=torch.cuda.is_available(),
    )

    # Trainer avec class weights
    # processing_class supprimé : data_collator gère déjà le padding
    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=lambda p: compute_metrics(p, metric),
    )

    # Entraînement
    logger.info(f"Début entraînement optimisé : {cfg.models.name}")
    trainer.train()

    # Sauvegarde
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Modèle optimisé sauvegardé dans {output_dir}")


if __name__ == "__main__":
    train_optimized()