# tests/test_spark_preprocessing.py

import pytest

# Marqueur : ces tests sont ignorés si Java n'est pas disponible
# Lance avec : python -m pytest -m spark
# Exclut avec : python -m pytest -m "not spark"
pytest.importorskip("pyspark", reason="PySpark non disponible")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from training.spark_preprocessing import (
    clean_and_parse_spans,
    split_corpus_spark,
    compute_statistics,
)


@pytest.fixture(scope="module")
def spark():
    """SparkSession locale pour les tests — partagée entre tous les tests du module."""
    session = (
        SparkSession.builder
        .master("local[1]")
        .appName("onco-ner-test")
        .config("spark.driver.memory", "1g")
        .config("spark.ui.enabled", "false")  # désactive l'UI web pendant les tests
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture
def sample_df(spark):
    """DataFrame Spark minimal simulant le format FRACCO."""
    data = [
        ("cc_onco1.ann", "T1", "morphologie", "8010/3", "carcinome", "0 9"),
        ("cc_onco1.ann", "T2", "topographie", "C50.9", "sein gauche", "10 20"),
        ("cc_onco2.ann", "T1", "morphologie", "8041/3", "carcinome microcytaire", "0 9;20 31"),
        ("cc_onco2.ann", "T2", "differenciation", "G2", "bien différencié", "32 47"),
        ("cc_onco3.ann", "T1", "topographie", "C20.9", "rectum", "0 6"),
        ("cc_onco3.ann", "T2", "expression_CIM", "8010/3", "carcinome du sein", "7 23"),
    ]
    return spark.createDataFrame(
        data,
        ["doc_name", "id", "label", "code", "content", "full_span"]
    )


# --- Tests clean_and_parse_spans ---

@pytest.mark.spark
def test_spark_parse_columns(sample_df):
    """Les colonnes span_start et span_end doivent exister après parsing."""
    df = clean_and_parse_spans(sample_df)
    assert "span_start" in df.columns
    assert "span_end" in df.columns
    assert "full_span" not in df.columns


@pytest.mark.spark
def test_spark_parse_simple_span(sample_df):
    """Span simple '0 9' -> start=0, end=9."""
    df = clean_and_parse_spans(sample_df)
    row = df.filter(
        (F.col("doc_name") == "cc_onco1.ann") & (F.col("id") == "T1")
    ).collect()[0]
    assert row["span_start"] == 0
    assert row["span_end"] == 9


@pytest.mark.spark
def test_spark_parse_discontinuous_span(sample_df):
    """Span discontinu '0 9;20 31' -> start=0, end=31."""
    df = clean_and_parse_spans(sample_df)
    row = df.filter(
        (F.col("doc_name") == "cc_onco2.ann") & (F.col("id") == "T1")
    ).collect()[0]
    assert row["span_start"] == 0
    assert row["span_end"] == 31


@pytest.mark.spark
def test_spark_row_count_preserved(sample_df):
    """Le nettoyage ne doit pas supprimer de lignes."""
    df = clean_and_parse_spans(sample_df)
    assert df.count() == 6


# --- Tests split_corpus_spark ---

@pytest.mark.spark
def test_spark_split_no_leakage(sample_df):
    """Un document ne doit pas apparaître dans deux splits."""
    df = clean_and_parse_spans(sample_df)
    train_df, val_df, test_df = split_corpus_spark(df, seed=42)

    train_docs = {r["doc_name"] for r in train_df.select("doc_name").distinct().collect()}
    val_docs = {r["doc_name"] for r in val_df.select("doc_name").distinct().collect()}
    test_docs = {r["doc_name"] for r in test_df.select("doc_name").distinct().collect()}

    assert train_docs.isdisjoint(val_docs), "Fuite train/val"
    assert train_docs.isdisjoint(test_docs), "Fuite train/test"
    assert val_docs.isdisjoint(test_docs), "Fuite val/test"


@pytest.mark.spark
def test_spark_split_total_count(sample_df):
    """Train + val + test doit couvrir toutes les entités."""
    df = clean_and_parse_spans(sample_df)
    train_df, val_df, test_df = split_corpus_spark(df, seed=42)
    total = train_df.count() + val_df.count() + test_df.count()
    assert total == df.count()


@pytest.mark.spark
def test_spark_split_reproducible(sample_df):
    """Même seed = mêmes splits."""
    df = clean_and_parse_spans(sample_df)
    train1, _, _ = split_corpus_spark(df, seed=42)
    train2, _, _ = split_corpus_spark(df, seed=42)

    docs1 = sorted([r["doc_name"] for r in train1.select("doc_name").distinct().collect()])
    docs2 = sorted([r["doc_name"] for r in train2.select("doc_name").distinct().collect()])
    assert docs1 == docs2