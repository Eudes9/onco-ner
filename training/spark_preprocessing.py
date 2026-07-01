# training/spark_preprocessing.py
"""
Démo Spark : preprocessing du corpus FRACCO à grande échelle.
Sur 1301 documents, Polars est plus rapide.
Ce script montre la maîtrise de l'API Spark pour des volumes
de l'ordre de l'EDS AP-HP (20M+ patients).

Optimisations appliquées :
- Hash déterministe (xxhash64) pour le split sans shuffle réseau
- .cache() après nettoyage pour éviter de relire le CSV à chaque action
- split("[; ]+") pour parser les spans en une seule opération
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pathlib import Path


def build_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("onco-ner-fracco-preprocessing")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def load_corpus_spark(csv_path: str, spark: SparkSession):
    """Charge le CSV FRACCO avec Spark."""
    return spark.read.csv(csv_path, header=True, inferSchema=False)


def clean_and_parse_spans(df):
    """
    Nettoyage et parsing des spans en une seule passe.
    Gère les spans discontinus (format "3995;4023 4030").
    Le pattern [; ]+ découpe sur espace ou point-virgule en une seule opération.
    """
    df = df.withColumn("doc_name", F.trim(F.col("doc_name")))
    df = df.withColumn("label", F.trim(F.col("label")))
    df = df.withColumn("content", F.trim(F.col("content")))
    df = df.withColumn("code", F.trim(F.col("code")))

    # Parser spans en une seule opération (pas de colonne intermédiaire)
    df = df.withColumn(
        "span_parts",
        F.split(F.col("full_span"), r"[; ]+")
    )
    df = df.withColumn(
        "span_start",
        F.element_at(F.col("span_parts"), 1).cast(IntegerType())
    )
    df = df.withColumn(
        "span_end",
        F.element_at(F.col("span_parts"), -1).cast(IntegerType())
    )

    return df.drop("full_span", "span_parts")


def compute_statistics(df):
    """
    Stats descriptives sur le corpus.
    À appeler uniquement après .cache() pour éviter de relire le CSV.
    """
    print("\n=== Distribution des labels ===")
    df.groupBy("label").count().orderBy("count", ascending=False).show()

    print("=== Nombre de documents uniques ===")
    df.select(F.countDistinct("doc_name").alias("n_documents")).show()

    print("=== Longueur moyenne du contenu par label ===")
    df.withColumn("content_length", F.length(F.col("content"))) \
      .groupBy("label") \
      .agg(F.avg("content_length").alias("avg_length")) \
      .orderBy("label") \
      .show()


def split_corpus_spark(df, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Split déterministe par hash sur doc_name.
    Zéro shuffle réseau : chaque document est assigné à un split
    via xxhash64, sans jointure ni échange de données entre partitions.
    Reproductible : même seed = même split, toujours.
    """
    train_max = int(train_ratio * 100)
    val_max = train_max + int(val_ratio * 100)

    df_hashed = df.withColumn(
        "doc_hash",
        F.abs(F.xxhash64(F.concat(F.col("doc_name"), F.lit(str(seed))))) % 100
    )

    train_df = df_hashed.filter(F.col("doc_hash") < train_max).drop("doc_hash")
    val_df = df_hashed.filter(
        (F.col("doc_hash") >= train_max) & (F.col("doc_hash") < val_max)
    ).drop("doc_hash")
    test_df = df_hashed.filter(F.col("doc_hash") >= val_max).drop("doc_hash")

    return train_df, val_df, test_df


if __name__ == "__main__":
    CSV_PATH = "data/DetectOnco_Final.csv"

    spark = build_spark_session()
    print(f"Spark version : {spark.version}")

    # Chargement + nettoyage
    df = load_corpus_spark(CSV_PATH, spark)
    df = clean_and_parse_spans(df)

    # Matérialisation explicite du cache
    # Sans cette ligne, la première action paie le coût complet du calcul
    df = df.cache()
    _ = df.count()  # force Spark à remplir le cache maintenant

    # Toutes les actions suivantes lisent depuis le cache
    compute_statistics(df)

    train_df, val_df, test_df = split_corpus_spark(df, seed=42)

    print(f"\nTrain : {train_df.count()} entités")
    print(f"Val   : {val_df.count()} entités")
    print(f"Test  : {test_df.count()} entités")

    spark.stop()
    print("\nDémo Spark terminée.")