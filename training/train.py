# training/train.py

import os
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import evaluate
import polars as pl
import numpy as np

from onco_ner.utils.logging import get_logger
from onco_ner.utils.seed import set_seed

logger = get_logger(__name__)

# Labels ICD-O pour la tâche NER (schéma BIO)
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
    test_df = pl.read_parquet(splits_dir / "test.parquet")
    return train_df, val_df, test_df


def build_token_classification_dataset(
    df: pl.DataFrame,
    tokenizer,
    max_length: int,
    txt_dir: Path,
) -> Dataset:
    """
    Construit un dataset HuggingFace pour token classification.

    Stratégie d'alignement robuste :
    - On tokenize le texte complet avec return_offsets_mapping=True
    - Pour chaque token, on vérifie si son offset chevauche une entité annotée
    - Le PREMIER subtoken d'une entité reçoit B-label, les suivants I-label
    - Un flag `entity_started` suit l'état courant pour éviter le piège
      des espaces absorbés par SentencePiece (CamemBERT, XLM-RoBERTa)
    - Les tokens spéciaux reçoivent -100 (ignorés par la loss)

    Pour les spans discontinus : on annote uniquement le premier sous-span.
    Limitation documentée dans le rapport (section Limitations).
    """
    examples = {"input_ids": [], "attention_mask": [], "labels": []}

    for doc_name in df["doc_name"].unique().to_list():
        txt_name = doc_name.replace(".ann", ".txt")
        txt_path = txt_dir / txt_name

        if not txt_path.exists():
            logger.warning(f"Fichier txt manquant : {txt_path}, ignoré")
            continue

        text = txt_path.read_text(encoding="utf-8")

        doc_df = df.filter(pl.col("doc_name") == doc_name)
        entities = doc_df.select(["span_start", "span_end", "label"]).to_dicts()

        span_to_label = {
            (e["span_start"], e["span_end"]): e["label"]
            for e in entities
        }

        encoding = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            padding="max_length",
        )

        labels = []
        # Flag : pour chaque entité, on suit si on a déjà vu son premier subtoken
        # Évite le piège des espaces absorbés par SentencePiece
        entity_started = {span: False for span in span_to_label}

        for offset_start, offset_end in encoding["offset_mapping"]:
            # Tokens spéciaux ([CLS], [SEP], [PAD]) -> offset (0, 0) -> ignorés
            if offset_start == 0 and offset_end == 0:
                labels.append(-100)
                continue

            token_label = "O"

            for (span_start, span_end), label in span_to_label.items():
                # Le token chevauche cette entité
                if offset_start >= span_start and offset_end <= span_end:
                    if not entity_started[(span_start, span_end)]:
                        # Premier subtoken de l'entité -> B-label
                        token_label = f"B-{label}"
                        entity_started[(span_start, span_end)] = True
                    else:
                        # Subtokens suivants -> I-label
                        token_label = f"I-{label}"
                    break

            labels.append(LABEL2ID[token_label])

        examples["input_ids"].append(encoding["input_ids"])
        examples["attention_mask"].append(encoding["attention_mask"])
        examples["labels"].append(labels)

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
        references=true_labels
    )
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    logger.info(f"Config :\n{OmegaConf.to_yaml(cfg)}")

    set_seed(cfg.training.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device : {device}")

    # Chemins
    splits_dir = Path(cfg.paths.splits_dir)
    txt_dir = Path(cfg.paths.data_dir) / "ann_txt_files"
    output_dir = Path(cfg.model.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chargement des splits
    logger.info("Chargement des splits...")
    train_df, val_df, test_df = load_splits(splits_dir)

    # Tokenizer
    logger.info(f"Chargement du tokenizer : {cfg.model.pretrained_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model_name)

    # Datasets HuggingFace
    logger.info("Construction des datasets token classification...")
    train_dataset = build_token_classification_dataset(
        train_df, tokenizer, cfg.training.max_length, txt_dir
    )
    val_dataset = build_token_classification_dataset(
        val_df, tokenizer, cfg.training.max_length, txt_dir
    )

    # Modèle
    logger.info(f"Chargement du modèle : {cfg.model.pretrained_model_name}")
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model.pretrained_model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # W&B via variables d'environnement
    # Le Trainer gère l'init et le finish en interne
    os.environ["WANDB_PROJECT"] = cfg.wandb.project
    os.environ["WANDB_RUN_NAME"] = cfg.model.name

    # Métrique
    metric = evaluate.load("seqeval")

    # Arguments d'entraînement
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.model.gradient_accumulation_steps,
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

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=lambda p: compute_metrics(p, metric),
    )

    # Entraînement
    logger.info(f"Début entraînement : {cfg.model.name}")
    trainer.train()

    # Sauvegarde
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Modèle sauvegardé dans {output_dir}")


if __name__ == "__main__":
    train()