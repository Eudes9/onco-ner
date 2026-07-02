# evaluation/error_analysis.py
"""
Analyse d'erreurs approfondie sur le set de test.

Types d'erreurs distingués :
- TP       : span exact + label correct
- FN_PURE  : entité gold complètement manquée (aucun chevauchement)
- FP_PURE  : entité prédite sans correspondance gold
- BOUNDARY : bon label mais frontières décalées (chevauchement partiel)

Évaluation stricte : les boundary errors pénalisent precision et recall
(elles comptent comme FP + FN dans le calcul du F1).
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForTokenClassification, AutoTokenizer

from onco_ner.exceptions import ModelNotLoadedError
from onco_ner.utils.logging import get_logger

logger = get_logger(__name__)

LABELS = [
    "O",
    "B-morphologie", "I-morphologie",
    "B-topographie", "I-topographie",
    "B-differenciation", "I-differenciation",
    "B-expression_CIM", "I-expression_CIM",
]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ENTITY_LABELS = ["morphologie", "topographie", "differenciation", "expression_CIM"]


def load_model_and_tokenizer(model_path: str):
    """Charge le modèle et le tokenizer depuis HuggingFace Hub ou un chemin local."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info(f"Modèle chargé depuis {model_path} sur {device}")
        return model, tokenizer, device
    except Exception as e:
        raise ModelNotLoadedError(
            f"Impossible de charger le modèle {model_path}"
        ) from e


def predict_document(
    text: str,
    tokenizer,
    model,
    max_length: int = 512,
    device=None,
) -> list[dict]:
    """
    Prédit les entités dans un texte complet.
    Le device est détecté automatiquement depuis le modèle si non fourni.
    """
    if device is None:
        device = next(model.parameters()).device

    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
        padding="max_length",
    )

    offset_mapping = encoding["offset_mapping"][0].tolist()
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().tolist()

    results = []
    for offset, pred in zip(offset_mapping, predictions):
        if offset[0] == 0 and offset[1] == 0:
            continue
        results.append({
            "offset_start": offset[0],
            "offset_end": offset[1],
            "pred_label": ID2LABEL[pred],
        })

    return results


def extract_entities_from_predictions(predictions: list[dict]) -> list[dict]:
    """
    Extrait les entités (span, label) depuis les prédictions BIO.

    Gestion des subtokens orphelins I- sans B- précédent :
    un I- sans B- est traité comme un nouveau B- (début implicite).
    """
    entities = []
    current_entity = None

    for token in predictions:
        label = token["pred_label"]

        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "start": token["offset_start"],
                "end": token["offset_end"],
                "label": label[2:],
            }

        elif label.startswith("I-"):
            entity_type = label[2:]
            if current_entity and current_entity["label"] == entity_type:
                current_entity["end"] = token["offset_end"]
            else:
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "start": token["offset_start"],
                    "end": token["offset_end"],
                    "label": entity_type,
                }

        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    return entities


def _greedy_match(
    gold_entities: dict,
    pred_set: dict,
    text: str,
) -> tuple[list, list, list]:
    """
    Matching greedy entre entités gold et prédites.
    Chaque entité gold et prédite ne peut être matchée qu'une seule fois.
    Évite le double comptage des erreurs de frontières.

    Trie les candidats par overlap décroissant pour favoriser
    les meilleurs matches en premier.

    Retourne : (matched_pairs, unmatched_gold, unmatched_pred)
    """
    candidates = []
    for gold_span, gold_label in gold_entities.items():
        gold_start, gold_end = gold_span
        for pred_span, pred_label in pred_set.items():
            pred_start, pred_end = pred_span
            overlap = min(gold_end, pred_end) - max(gold_start, pred_start)
            if overlap > 0 and gold_label == pred_label:
                candidates.append((overlap, gold_span, pred_span, gold_label))

    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_pairs = []
    used_gold = set()
    used_pred = set()

    for overlap, gold_span, pred_span, label in candidates:
        if gold_span in used_gold or pred_span in used_pred:
            continue
        matched_pairs.append((gold_span, pred_span, label))
        used_gold.add(gold_span)
        used_pred.add(pred_span)

    unmatched_gold = [
        (span, label) for span, label in gold_entities.items()
        if span not in used_gold
    ]
    unmatched_pred = [
        (span, label) for span, label in pred_set.items()
        if span not in used_pred
    ]

    return matched_pairs, unmatched_gold, unmatched_pred


def compute_per_label_metrics(
    test_df: pl.DataFrame,
    tokenizer,
    model,
    ann_dir: Path,
    max_length: int = 512,
    device=None,
) -> tuple[dict, list[dict]]:
    """
    Calcule precision/recall/F1 par type d'entité sur le test set.

    Évaluation stricte : les boundary errors comptent comme FP + FN
    dans le calcul du F1 (span exact requis pour un TP).

    Utilise un matching greedy pour éviter le double comptage.
    """
    if device is None:
        device = next(model.parameters()).device

    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    boundary_errors = defaultdict(int)
    errors = []

    doc_names = test_df["doc_name"].unique().to_list()
    logger.info(f"Analyse de {len(doc_names)} documents...")

    for doc_name in doc_names:
        txt_name = doc_name.replace(".ann", ".txt")
        txt_path = ann_dir / txt_name

        if not txt_path.exists():
            logger.warning(f"Fichier manquant : {txt_path}, ignoré")
            continue

        text = txt_path.read_text(encoding="utf-8")

        doc_df = test_df.filter(pl.col("doc_name") == doc_name)
        gold_entities = {
            (row["span_start"], row["span_end"]): row["label"]
            for row in doc_df.to_dicts()
        }

        predictions = predict_document(text, tokenizer, model, max_length, device)
        pred_entities = extract_entities_from_predictions(predictions)
        pred_set = {
            (e["start"], e["end"]): e["label"]
            for e in pred_entities
        }

        # Matching greedy : évite le double comptage
        matched_pairs, unmatched_gold, unmatched_pred = _greedy_match(
            gold_entities, pred_set, text
        )

        # TP ou BOUNDARY
        for gold_span, pred_span, label in matched_pairs:
            if gold_span == pred_span:
                true_positives[label] += 1
            else:
                boundary_errors[label] += 1
                gold_start, gold_end = gold_span
                pred_start, pred_end = pred_span
                errors.append({
                    "doc": doc_name,
                    "type": "BOUNDARY",
                    "gold_label": label,
                    "pred_label": label,
                    "gold_span": gold_span,
                    "pred_span": pred_span,
                    "gold_text": text[gold_start:gold_end],
                    "pred_text": text[pred_start:pred_end],
                })

        # FN purs
        for gold_span, gold_label in unmatched_gold:
            false_negatives[gold_label] += 1
            gold_start, gold_end = gold_span
            errors.append({
                "doc": doc_name,
                "type": "FN_PURE",
                "gold_label": gold_label,
                "pred_label": "O",
                "gold_span": gold_span,
                "pred_span": None,
                "gold_text": text[gold_start:gold_end],
                "pred_text": None,
            })

        # FP purs
        for pred_span, pred_label in unmatched_pred:
            false_positives[pred_label] += 1
            pred_start, pred_end = pred_span
            errors.append({
                "doc": doc_name,
                "type": "FP_PURE",
                "gold_label": "O",
                "pred_label": pred_label,
                "gold_span": None,
                "pred_span": pred_span,
                "gold_text": None,
                "pred_text": text[pred_start:pred_end],
            })

    # Calcul des métriques avec boundary errors dans les dénominateurs
    metrics = {}
    all_labels = set(
        list(true_positives.keys()) +
        list(false_negatives.keys()) +
        list(false_positives.keys())
    )

    for label in all_labels:
        tp = true_positives[label]
        fp_pure = false_positives[label]
        fn_pure = false_negatives[label]
        boundary = boundary_errors[label]

        # Évaluation stricte : boundary errors pénalisent precision ET recall
        fp_total = fp_pure + boundary
        fn_total = fn_pure + boundary

        precision = tp / (tp + fp_total) if (tp + fp_total) > 0 else 0
        recall = tp / (tp + fn_total) if (tp + fn_total) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0
        )

        metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp_pure": fp_pure,
            "fn_pure": fn_pure,
            "boundary_errors": boundary,
            "fp_total": fp_total,
            "fn_total": fn_total,
        }

    logger.info(f"Analyse terminée — {len(errors)} erreurs identifiées")
    return metrics, errors


def analyze_errors_by_length(
    test_df: pl.DataFrame,
    errors: list[dict],
    ann_dir: Path,
) -> pd.DataFrame:
    """Analyse le taux d'erreur selon la longueur des documents."""
    doc_lengths = {}
    for doc_name in test_df["doc_name"].unique().to_list():
        txt_path = ann_dir / doc_name.replace(".ann", ".txt")
        if txt_path.exists():
            doc_lengths[doc_name] = len(txt_path.read_text(encoding="utf-8"))

    # Sécurisation : DataFrame vide si aucune erreur
    if not errors:
        logger.warning("Aucune erreur détectée — DataFrame vide retourné")
        return pd.DataFrame(
            columns=["doc_name", "length", "error_rate", "category"]
        )

    errors_df = pd.DataFrame(errors)

    if "type" not in errors_df.columns:
        logger.warning("Colonne 'type' manquante dans errors_df")
        return pd.DataFrame(
            columns=["doc_name", "length", "error_rate", "category"]
        )

    results = []
    for doc_name, length in doc_lengths.items():
        doc_errors = errors_df[errors_df["doc"] == doc_name]
        doc_gold = test_df.filter(pl.col("doc_name") == doc_name).shape[0]

        if doc_gold == 0:
            continue

        error_rate = len(
            doc_errors[doc_errors["type"].isin(["FN_PURE", "FP_PURE"])]
        ) / doc_gold

        results.append({
            "doc_name": doc_name,
            "length": length,
            "error_rate": error_rate,
            "category": (
                "court (<3000)" if length < 3000
                else "moyen (3000-8000)" if length < 8000
                else "long (>8000)"
            ),
        })

    return pd.DataFrame(results)


def plot_confusion_matrix(
    errors: list[dict],
    output_path: Path,
) -> None:
    """Génère et sauvegarde la confusion matrix."""
    gold_labels_cm = []
    pred_labels_cm = []

    for error in errors:
        if error["type"] == "FN_PURE":
            gold_labels_cm.append(error["gold_label"])
            pred_labels_cm.append("O")
        elif error["type"] == "FP_PURE":
            gold_labels_cm.append("O")
            pred_labels_cm.append(error["pred_label"])
        elif error["type"] == "BOUNDARY":
            gold_labels_cm.append(error["gold_label"])
            pred_labels_cm.append(error["pred_label"])

    if not gold_labels_cm:
        logger.warning("Aucune erreur à afficher dans la confusion matrix")
        return

    label_names = [
        "morphologie", "topographie",
        "differenciation", "expression_CIM", "O"
    ]
    cm = confusion_matrix(gold_labels_cm, pred_labels_cm, labels=label_names)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=label_names,
        yticklabels=label_names,
        cmap="Blues",
    )
    plt.title(
        "Confusion Matrix — DrBERT\n(FN purs + FP purs + Boundary errors)",
        fontsize=13,
    )
    plt.ylabel("Gold label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix sauvegardée dans {output_path}")


def save_results(
    metrics: dict,
    errors: list[dict],
    length_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Sauvegarde tous les résultats d'analyse."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(metrics).T.sort_values("f1", ascending=False)
    metrics_df.to_csv(output_dir / "per_label_metrics.csv")

    if errors:
        errors_df = pd.DataFrame(errors)
        errors_df.to_csv(output_dir / "error_analysis.csv", index=False)

    if not length_df.empty:
        length_df.to_csv(output_dir / "length_analysis.csv", index=False)

    logger.info(f"Résultats sauvegardés dans {output_dir}")


if __name__ == "__main__":
    MODEL_PATH = "Eudes9/onco-ner-drbert"
    TEST_SPLIT = Path("data/splits/test.parquet")
    ANN_DIR = Path("data/ann_txt_files")
    OUTPUT_DIR = Path("benchmark/results")

    # Chargement
    model, tokenizer, device = load_model_and_tokenizer(MODEL_PATH)
    test_df = pl.read_parquet(TEST_SPLIT)
    logger.info(
        f"Test set : {test_df.shape[0]} entités, "
        f"{test_df['doc_name'].n_unique()} documents"
    )

    # Analyse
    metrics, errors = compute_per_label_metrics(
        test_df, tokenizer, model, ANN_DIR, device=device
    )

    # Métriques par label
    metrics_df = pd.DataFrame(metrics).T.sort_values("f1", ascending=False)
    print("\n=== Métriques par type d'entité (évaluation stricte) ===")
    print(
        metrics_df[[
            "precision", "recall", "f1",
            "tp", "fp_pure", "fn_pure",
            "boundary_errors", "fp_total", "fn_total"
        ]].to_string()
    )

    # Résumé des erreurs
    errors_df = pd.DataFrame(errors) if errors else pd.DataFrame()
    if not errors_df.empty:
        print(f"\n=== Résumé des erreurs ===")
        print(errors_df["type"].value_counts())

        print("\n=== FN purs par label ===")
        print(errors_df[errors_df["type"] == "FN_PURE"]["gold_label"].value_counts())

        print("\n=== FP purs par label ===")
        print(errors_df[errors_df["type"] == "FP_PURE"]["pred_label"].value_counts())

        print("\n=== Boundary errors par label ===")
        print(errors_df[errors_df["type"] == "BOUNDARY"]["gold_label"].value_counts())

        print("\n=== Exemples de boundary errors ===")
        boundary_examples = errors_df[errors_df["type"] == "BOUNDARY"].head(5)
        for _, row in boundary_examples.iterrows():
            print(f"\nLabel : {row['gold_label']}")
            print(f"  Gold  {row['gold_span']} : '{row['gold_text']}'")
            print(f"  Prédit {row['pred_span']} : '{row['pred_text']}'")

    # Analyse par longueur
    length_df = analyze_errors_by_length(test_df, errors, ANN_DIR)
    if not length_df.empty:
        print("\n=== Taux d'erreur moyen par longueur de document ===")
        print(
            length_df.groupby("category")["error_rate"]
            .mean()
            .sort_values(ascending=False)
        )

    # Confusion matrix
    plot_confusion_matrix(errors, OUTPUT_DIR / "confusion_matrix.png")

    # Sauvegarde
    save_results(metrics, errors, length_df, OUTPUT_DIR)

    print(f"\n=== Analyse terminée — {len(errors)} erreurs au total ===")