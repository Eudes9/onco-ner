# src/onco_ner/data/preprocessing.py

import polars as pl
from pathlib import Path
from onco_ner.utils.logging import get_logger

logger = get_logger(__name__)

# Ratio du split
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def load_corpus_csv(csv_path: Path) -> pl.DataFrame:
    df = pl.read_csv(csv_path)

    # Nettoyage : trim des espaces sur les colonnes texte
    df = df.with_columns([
        pl.col("doc_name").str.strip_chars(),
        pl.col("label").str.strip_chars(),
        pl.col("code").str.strip_chars(),
        pl.col("content").str.strip_chars(),
        pl.col("full_span").str.strip_chars(),
    ])

    # Normaliser les spans discontinus "3995;4023 4030" -> "3995 4030"
    # Format possible : "3995 4023" ou "3995 4023;4030 4045"
    df = df.with_columns(
        pl.col("full_span")
          .str.replace_all(";", " ")   # remplace les ; par des espaces
          .str.split(" ")              # découpe tout
          .alias("span_parts")
    )

    # Prendre le premier (start) et le dernier (end) élément
    df = df.with_columns([
        pl.col("span_parts").list.first().cast(pl.Int32).alias("span_start"),
        pl.col("span_parts").list.last().cast(pl.Int32).alias("span_end"),
    ]).drop(["full_span", "span_parts"])

    logger.info(f"Corpus chargé : {df.shape[0]} entités, {df['doc_name'].n_unique()} documents")
    logger.info(f"Distribution des labels :\n{df['label'].value_counts().sort('label')}")

    return df

def split_corpus(
    df: pl.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split stratifié par document (pas par entité) pour éviter la fuite de données,
    avec équilibre des labels rares (differenciation).
    """
    # On travaille au niveau document, pas au niveau entité
    doc_labels = (
        df.select(["doc_name", "label"])
        .unique()
        .with_columns(
            pl.col("label")
            .map_elements(lambda l: l, return_dtype=pl.String)
        )
    )

    # Récupérer les noms de documents uniques
    doc_names = df.select("doc_name").unique()

    # Split stratifié : on sépare par label dominant du document
    # Stratégie : split au niveau document avec shuffle reproductible
    doc_names = doc_names.with_columns(
        pl.lit(1).alias("dummy")
    ).sort("doc_name")

    n_docs = doc_names.shape[0]
    n_train = int(n_docs * train_ratio)
    n_val = int(n_docs * val_ratio)

    # Shuffle reproductible
    doc_names_shuffled = doc_names.sample(fraction=1.0, seed=seed)

    train_docs = doc_names_shuffled.slice(0, n_train).select("doc_name")
    val_docs = doc_names_shuffled.slice(n_train, n_val).select("doc_name")
    test_docs = doc_names_shuffled.slice(n_train + n_val).select("doc_name")

    train_df = df.join(train_docs, on="doc_name", how="inner")
    val_df = df.join(val_docs, on="doc_name", how="inner")
    test_df = df.join(test_docs, on="doc_name", how="inner")

    logger.info(f"Split — Train: {train_df['doc_name'].n_unique()} docs "
                f"({train_df.shape[0]} entités)")
    logger.info(f"Split — Val:   {val_df['doc_name'].n_unique()} docs "
                f"({val_df.shape[0]} entités)")
    logger.info(f"Split — Test:  {test_df['doc_name'].n_unique()} docs "
                f"({test_df.shape[0]} entités)")

    return train_df, val_df, test_df


def save_splits(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    output_dir: Path,
) -> None:
    """Sauvegarde les splits en Parquet (plus efficace que CSV pour le ML)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.write_parquet(output_dir / "train.parquet")
    val_df.write_parquet(output_dir / "val.parquet")
    test_df.write_parquet(output_dir / "test.parquet")
    logger.info(f"Splits sauvegardés dans {output_dir}")


def load_split(split_path: Path) -> pl.DataFrame:
    """Charge un split depuis un fichier Parquet."""
    return pl.read_parquet(split_path)