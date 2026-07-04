# evaluation/benchmark.py
"""
Benchmark comparatif des modèles onco-ner sur le test set.

Deux niveaux d'évaluation :
1. Seqeval (token level) : métrique standard HuggingFace
2. Stricte par spans : via error_analysis.py (precision/recall/F1 exact)

Modèles évalués :
- Baseline : DrBERT, XLM-RoBERTa-base, ModernCamemBERT-base
- Optimisés : DrBERT-optimized, XLM-RoBERTa-optimized, ModernCamemBERT-optimized

Usage :
    python evaluation/benchmark.py
"""

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
import evaluate
import importlib.util

from onco_ner.utils.logging import get_logger
from onco_ner.utils.seed import set_seed

logger = get_logger(__name__)

# Configuration
LABELS = [
    "O",
    "B-morphologie", "I-morphologie",
    "B-topographie", "I-topographie",
    "B-differenciation", "I-differenciation",
    "B-expression_CIM", "I-expression_CIM",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

MODELS = {
    "DrBERT-baseline": "Eudes9/onco-ner-drbert",
    "ModernCamemBERT-baseline": "Eudes9/onco-ner-moderncamembert",
    "XLM-RoBERTa-baseline": "Eudes9/onco-ner-xlm-roberta",
    "DrBERT-optimized": "Eudes9/onco-ner-drbert-optimized",
    "ModernCamemBERT-optimized": "Eudes9/onco-ner-moderncamembert-optimized",
    "XLM-RoBERTa-optimized": "Eudes9/onco-ner-xlm-roberta-optimized",
}

# Modèles optimisés uniquement pour l'évaluation stricte (coûteuse)
MODELS_STRICT_EVAL = {
    "DrBERT-optimized": "Eudes9/onco-ner-drbert-optimized",
    "XLM-RoBERTa-optimized": "Eudes9/onco-ner-xlm-roberta-optimized",
    "ModernCamemBERT-optimized": "Eudes9/onco-ner-moderncamembert-optimized",
}


def _import_module(module_name: str, file_path: str):
    """Importe un module depuis un chemin de fichier."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_seqeval(
    model_id: str,
    test_df: pl.DataFrame,
    ann_dir: Path,
    max_length: int = 512,
    stride: int = 256,
    batch_size: int = 8,
    device: torch.device = None,
) -> dict:
    """
    Évalue un modèle avec seqeval (token level) sur le test set.
    Utilise le sliding window pour couvrir les documents longs.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Évaluation seqeval : {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    model = model.to(device)
    model.eval()

    # Importer build_sliding_window_dataset
    train_mod = _import_module(
        "train_optimized",
        str(Path(__file__).parent.parent / "training" / "train_optimized.py"),
    )
    test_dataset = train_mod.build_sliding_window_dataset(
        test_df, tokenizer, ann_dir,
        max_length=max_length,
        stride=stride,
    )
    logger.info(f"Test dataset : {len(test_dataset)} chunks")

    metric = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
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

    training_args = TrainingArguments(
        output_dir="/tmp/eval",
        per_device_eval_batch_size=batch_size,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    results = trainer.evaluate()
    logger.info(
        f"{model_id} — "
        f"P={results['eval_precision']:.4f} "
        f"R={results['eval_recall']:.4f} "
        f"F1={results['eval_f1']:.4f}"
    )
    return {
        "precision": results["eval_precision"],
        "recall": results["eval_recall"],
        "f1": results["eval_f1"],
        "accuracy": results["eval_accuracy"],
        "n_chunks": len(test_dataset),
    }


def evaluate_strict(
    model_id: str,
    test_df: pl.DataFrame,
    ann_dir: Path,
    max_length: int = 512,
    device: torch.device = None,
) -> tuple[dict, list]:
    """
    Évalue un modèle avec l'évaluation stricte par spans.
    Utilise error_analysis.py pour le calcul détaillé.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Évaluation stricte : {model_id}")

    error_mod = _import_module(
        "error_analysis",
        str(Path(__file__).parent / "error_analysis.py"),
    )

    model, tokenizer, device = error_mod.load_model_and_tokenizer(model_id)

    metrics, errors = error_mod.compute_per_label_metrics(
        test_df, tokenizer, model, ann_dir,
        max_length=max_length,
        device=device,
    )

    return metrics, errors


def run_full_benchmark(
    test_df: pl.DataFrame,
    ann_dir: Path,
    output_dir: Path,
    max_length: int = 512,
    stride: int = 256,
    batch_size: int = 8,
) -> pd.DataFrame:
    """
    Lance le benchmark complet sur tous les modèles.

    Étape 1 : Évaluation seqeval sur tous les modèles
    Étape 2 : Évaluation stricte sur les modèles optimisés uniquement
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device : {device}")

    # Étape 1 : Seqeval sur tous les modèles
    seqeval_results = {}
    for name, model_id in MODELS.items():
        try:
            results = evaluate_seqeval(
                model_id, test_df, ann_dir,
                max_length=max_length,
                stride=stride,
                batch_size=batch_size,
                device=device,
            )
            seqeval_results[name] = results
        except Exception as e:
            logger.error(f"Erreur pour {name} : {e}")
            seqeval_results[name] = {
                "precision": None, "recall": None,
                "f1": None, "accuracy": None, "n_chunks": None
            }

    # Sauvegarder résultats seqeval
    seqeval_df = pd.DataFrame(seqeval_results).T
    seqeval_df.index.name = "model"
    seqeval_df = seqeval_df.sort_values("f1", ascending=False)
    seqeval_df.to_csv(output_dir / "seqeval_results.csv")

    logger.info("\n=== Benchmark Seqeval ===")
    logger.info(seqeval_df[["precision", "recall", "f1"]].to_string())

    # Étape 2 : Évaluation stricte sur modèles optimisés
    strict_results = {}
    for name, model_id in MODELS_STRICT_EVAL.items():
        try:
            metrics, errors = evaluate_strict(
                model_id, test_df, ann_dir,
                max_length=max_length,
                device=device,
            )
            strict_results[name] = metrics

            # Sauvegarder erreurs détaillées
            errors_df = pd.DataFrame(errors) if errors else pd.DataFrame()
            errors_df.to_csv(
                output_dir / f"errors_{name.lower().replace('-', '_')}.csv",
                index=False,
            )

            # Confusion matrix
            error_mod = _import_module(
                "error_analysis",
                str(Path(__file__).parent / "error_analysis.py"),
            )
            error_mod.plot_confusion_matrix(
                errors,
                output_dir / f"confusion_matrix_{name.lower().replace('-', '_')}.png",
            )

        except Exception as e:
            logger.error(f"Erreur évaluation stricte {name} : {e}")

    # Sauvegarder résultats stricts par label
    for name, metrics in strict_results.items():
        metrics_df = pd.DataFrame(metrics).T.sort_values("f1", ascending=False)
        metrics_df.to_csv(
            output_dir / f"strict_per_label_{name.lower().replace('-', '_')}.csv"
        )

    # Tableau comparatif final
    comparison_rows = []
    for name in MODELS:
        row = {"model": name, "optimized": "optimized" in name.lower()}
        if name in seqeval_results:
            row["seqeval_precision"] = seqeval_results[name]["precision"]
            row["seqeval_recall"] = seqeval_results[name]["recall"]
            row["seqeval_f1"] = seqeval_results[name]["f1"]
            row["n_chunks"] = seqeval_results[name]["n_chunks"]
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        "seqeval_f1", ascending=False
    )
    comparison_df.to_csv(output_dir / "comparison_final.csv", index=False)

    logger.info("\n=== Tableau Comparatif Final ===")
    logger.info(
        comparison_df[["model", "seqeval_precision", "seqeval_recall", "seqeval_f1"]]
        .to_string(index=False)
    )

    return comparison_df


if __name__ == "__main__":
    set_seed(42)

    TEST_SPLIT = Path("data/splits/test.parquet")
    ANN_DIR = Path("data/ann_txt_files")
    OUTPUT_DIR = Path("benchmark/results")

    test_df = pl.read_parquet(TEST_SPLIT)
    logger.info(
        f"Test set : {test_df.shape[0]} entités, "
        f"{test_df['doc_name'].n_unique()} documents"
    )

    comparison_df = run_full_benchmark(
        test_df=test_df,
        ann_dir=ANN_DIR,
        output_dir=OUTPUT_DIR,
        max_length=512,
        stride=256,
        batch_size=8,
    )

    print("\n=== Benchmark terminé ===")
    print(f"Résultats sauvegardés dans {OUTPUT_DIR}")
    print(comparison_df[["model", "seqeval_f1"]].to_string(index=False))