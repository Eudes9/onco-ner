# tests/test_parser.py

import pytest
from pathlib import Path
from pydantic import ValidationError
from onco_ner.data.brat_parser import parse_ann_file, parse_document, parse_corpus
from onco_ner.schemas import Entity, Span
from onco_ner.exceptions import InvalidAnnotationError


# --- Fixtures ---

@pytest.fixture
def tmp_ann_simple(tmp_path):
    """Paire .txt/.ann avec un seul span continu."""
    txt = tmp_path / "doc1.txt"
    ann = tmp_path / "doc1.ann"
    txt.write_text("carcinome du poumon gauche", encoding="utf-8")
    ann.write_text(
        "T1\tmorphologie 0 9\tcarcinome\n"
        "#1\tICD-O T1\t8010/3\n",
        encoding="utf-8"
    )
    return txt, ann


@pytest.fixture
def tmp_ann_discontinuous(tmp_path):
    """Paire .txt/.ann avec un span discontinu."""
    txt = tmp_path / "doc2.txt"
    ann = tmp_path / "doc2.ann"
    txt.write_text("carcinome microcytaire du poumon", encoding="utf-8")
    ann.write_text(
        "T1\tmorphologie 0 9;20 31\tcarcinome microcytaire\n"
        "#1\tICD-O T1\t8041/3\n",
        encoding="utf-8"
    )
    return txt, ann


@pytest.fixture
def tmp_ann_malformed(tmp_path):
    """Fichier .ann mal formé."""
    ann = tmp_path / "doc3.ann"
    ann.write_text("T1\tmal formé\n", encoding="utf-8")
    return ann


@pytest.fixture
def tmp_ann_no_icdo(tmp_path):
    """Entité sans code ICD-O associé."""
    txt = tmp_path / "doc4.txt"
    ann = tmp_path / "doc4.ann"
    txt.write_text("adénopathie cervicale", encoding="utf-8")
    ann.write_text("T1\ttopographie 0 11\tadénopathie\n", encoding="utf-8")
    return txt, ann


@pytest.fixture
def tmp_ann_unknown_ref(tmp_path):
    """Ligne # qui référence un ID d'entité inexistant."""
    ann = tmp_path / "doc5.ann"
    ann.write_text(
        "T1\tmorphologie 0 9\tcarcinome\n"
        "#1\tICD-O T99\t8010/3\n",  # T99 n'existe pas
        encoding="utf-8"
    )
    return ann


# --- Tests parse_ann_file ---

def test_parse_simple_entity(tmp_ann_simple):
    _, ann = tmp_ann_simple
    entities = parse_ann_file(ann)
    assert len(entities) == 1
    e = entities[0]
    assert e.id == "T1"
    assert e.label == "morphologie"
    assert e.text == "carcinome"
    assert e.icdo_code == "8010/3"
    assert not e.is_discontinuous


def test_parse_discontinuous_span(tmp_ann_discontinuous):
    _, ann = tmp_ann_discontinuous
    entities = parse_ann_file(ann)
    assert len(entities) == 1
    e = entities[0]
    assert e.is_discontinuous
    assert len(e.spans) == 2
    assert e.spans[0] == Span(start=0, end=9)
    assert e.spans[1] == Span(start=20, end=31)
    assert e.icdo_code == "8041/3"


def test_entity_without_icdo(tmp_ann_no_icdo):
    _, ann = tmp_ann_no_icdo
    entities = parse_ann_file(ann)
    assert len(entities) == 1
    assert entities[0].icdo_code is None


def test_malformed_ann_raises(tmp_ann_malformed):
    with pytest.raises(InvalidAnnotationError):
        parse_ann_file(tmp_ann_malformed)


def test_unknown_entity_reference_raises(tmp_ann_unknown_ref):
    """Une note # qui pointe vers un ID inexistant doit lever InvalidAnnotationError."""
    with pytest.raises(InvalidAnnotationError):
        parse_ann_file(tmp_ann_unknown_ref)


def test_entity_is_immutable(tmp_ann_simple):
    """Pydantic v2 lève ValidationError sur toute tentative de modification."""
    _, ann = tmp_ann_simple
    entities = parse_ann_file(ann)
    with pytest.raises(ValidationError):
        entities[0].icdo_code = "9999/9"


def test_entity_is_hashable(tmp_ann_simple):
    _, ann = tmp_ann_simple
    entities = parse_ann_file(ann)
    entity_set = set(entities)
    assert len(entity_set) == 1


def test_span_positions_positive(tmp_ann_simple):
    _, ann = tmp_ann_simple
    entities = parse_ann_file(ann)
    for span in entities[0].spans:
        assert span.start >= 0
        assert span.end >= 0


# --- Tests parse_document ---

def test_parse_document_structure(tmp_ann_simple):
    txt, ann = tmp_ann_simple
    doc = parse_document(txt, ann)
    assert "doc_name" in doc
    assert "text" in doc
    assert "entities" in doc
    assert doc["doc_name"] == "doc1"
    assert isinstance(doc["entities"], list)


def test_parse_document_text_content(tmp_ann_simple):
    txt, ann = tmp_ann_simple
    doc = parse_document(txt, ann)
    assert doc["text"] == "carcinome du poumon gauche"


# --- Tests parse_corpus ---

def test_parse_corpus_count(tmp_path):
    for i in range(3):
        (tmp_path / f"doc{i}.txt").write_text("texte médical", encoding="utf-8")
        (tmp_path / f"doc{i}.ann").write_text(
            "T1\tmorphologie 0 5\ttexte\n#1\tICD-O T1\t8000/0\n",
            encoding="utf-8"
        )
    docs = parse_corpus(tmp_path)
    assert len(docs) == 3


def test_parse_corpus_missing_ann(tmp_path):
    """Un .txt sans .ann doit être ignoré sans planter."""
    (tmp_path / "doc_orphan.txt").write_text("texte sans ann", encoding="utf-8")
    docs = parse_corpus(tmp_path)
    assert len(docs) == 0