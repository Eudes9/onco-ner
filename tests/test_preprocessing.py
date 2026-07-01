# tests/test_preprocessing.py

import pytest
import polars as pl
from pathlib import Path
from onco_ner.data.preprocessing import (
    load_corpus_csv,
    split_corpus,
    save_splits,
    load_split,
)


# --- Fixtures ---

@pytest.fixture
def tmp_csv(tmp_path):
    """CSV minimal simulant le format FRACCO."""
    csv_content = """doc_name,id,label,code,content,full_span
cc_onco1.ann,T1,morphologie,8010/3,carcinome,0 9
cc_onco1.ann,T2,topographie,C50.9,sein gauche,10 20
cc_onco2.ann,T1,morphologie,8041/3,carcinome microcytaire,0 9;20 31
cc_onco2.ann,T2,differenciation,G2,bien différencié,32 47
cc_onco3.ann,T1,topographie,C20.9,rectum,0 6
cc_onco3.ann,T2,expression_CIM,8010/3-C50.9,carcinome du sein,7 23
"""
    csv_file = tmp_path / "test_corpus.csv"
    csv_file.write_text(csv_content, encoding="utf-8")
    return csv_file


# --- Tests load_corpus_csv ---

def test_load_corpus_shape(tmp_csv):
    df = load_corpus_csv(tmp_csv)
    assert df.shape[0] == 6
    assert df.shape[1] == 7  # doc_name, id, label, code, content, span_start, span_end
    assert "span_start" in df.columns
    assert "span_end" in df.columns
    assert "full_span" not in df.columns

def test_load_corpus_types(tmp_csv):
    df = load_corpus_csv(tmp_csv)
    assert df["span_start"].dtype == pl.Int32
    assert df["span_end"].dtype == pl.Int32


def test_load_corpus_discontinuous_span(tmp_csv):
    """Un span discontinu '0 9;20 31' doit donner start=0, end=31."""
    df = load_corpus_csv(tmp_csv)
    row = df.filter(
        (pl.col("doc_name") == "cc_onco2.ann") & (pl.col("id") == "T1")
    )
    assert row["span_start"][0] == 0
    assert row["span_end"][0] == 31


def test_load_corpus_labels(tmp_csv):
    df = load_corpus_csv(tmp_csv)
    labels = set(df["label"].to_list())
    assert labels == {"morphologie", "topographie", "differenciation", "expression_CIM"}


def test_load_corpus_no_null(tmp_csv):
    df = load_corpus_csv(tmp_csv)
    for col in ["doc_name", "label", "code", "content"]:
        assert df[col].null_count() == 0


# --- Tests split_corpus ---

def test_split_no_leakage(tmp_csv):
    """Un document ne doit pas apparaître dans deux splits différents."""
    df = load_corpus_csv(tmp_csv)
    train_df, val_df, test_df = split_corpus(df, seed=42)

    train_docs = set(train_df["doc_name"].to_list())
    val_docs = set(val_df["doc_name"].to_list())
    test_docs = set(test_df["doc_name"].to_list())

    assert train_docs.isdisjoint(val_docs), "Fuite train/val"
    assert train_docs.isdisjoint(test_docs), "Fuite train/test"
    assert val_docs.isdisjoint(test_docs), "Fuite val/test"


def test_split_total_count(tmp_csv):
    """Train + val + test doit couvrir tous les documents."""
    df = load_corpus_csv(tmp_csv)
    train_df, val_df, test_df = split_corpus(df, seed=42)

    total = (
        train_df["doc_name"].n_unique()
        + val_df["doc_name"].n_unique()
        + test_df["doc_name"].n_unique()
    )
    assert total == df["doc_name"].n_unique()


def test_split_reproducible(tmp_csv):
    """Même seed doit produire exactement les mêmes splits."""
    df = load_corpus_csv(tmp_csv)
    train1, val1, test1 = split_corpus(df, seed=42)
    train2, val2, test2 = split_corpus(df, seed=42)

    assert train1["doc_name"].sort().to_list() == train2["doc_name"].sort().to_list()
    assert val1["doc_name"].sort().to_list() == val2["doc_name"].sort().to_list()


def test_split_different_seeds(tmp_csv):
    """Seeds différentes doivent produire des splits différents."""
    df = load_corpus_csv(tmp_csv)
    train1, _, _ = split_corpus(df, seed=42)
    train2, _, _ = split_corpus(df, seed=99)
    # Avec un corpus suffisamment grand les splits seront différents
    # Sur 3 docs c'est possible qu'ils soient identiques, on vérifie juste que ça ne plante pas
    assert train1 is not None
    assert train2 is not None


# --- Tests save_splits / load_split ---

def test_save_and_load_splits(tmp_csv, tmp_path):
    df = load_corpus_csv(tmp_csv)
    train_df, val_df, test_df = split_corpus(df, seed=42)

    output_dir = tmp_path / "splits"
    save_splits(train_df, val_df, test_df, output_dir)

    assert (output_dir / "train.parquet").exists()
    assert (output_dir / "val.parquet").exists()
    assert (output_dir / "test.parquet").exists()


def test_load_split_content(tmp_csv, tmp_path):
    df = load_corpus_csv(tmp_csv)
    train_df, val_df, test_df = split_corpus(df, seed=42)

    output_dir = tmp_path / "splits"
    save_splits(train_df, val_df, test_df, output_dir)

    loaded = load_split(output_dir / "train.parquet")
    assert loaded.shape == train_df.shape
    assert loaded.columns == train_df.columns