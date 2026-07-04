# tests/test_normalizer.py

import pytest
import polars as pl
from pathlib import Path
from onco_ner.models.normalizer import ICDONormalizer


# --- Fixtures ---

@pytest.fixture
def tmp_csv(tmp_path):
    """
    CSV minimal simulant DetectOnco_Final.csv.
    Utilisé pour tous les tests — pas de dépendance au vrai CSV FRACCO.
    """
    csv_content = """doc_name,id,label,code,content,full_span
cc_onco1.ann,T1,morphologie,8500/3,carcinome canalaire infiltrant,0 30
cc_onco1.ann,T2,topographie,C50.9,sein gauche,31 41
cc_onco2.ann,T1,morphologie,8041/3,carcinome microcytaire,0 21
cc_onco2.ann,T2,differenciation,G2,bien différencié,22 37
cc_onco3.ann,T1,topographie,C20.9,rectum,0 6
cc_onco3.ann,T2,expression_CIM,8500/3,carcinome du sein,7 24
"""
    csv_file = tmp_path / "test_corpus.csv"
    csv_file.write_text(csv_content, encoding="utf-8")
    return csv_file


@pytest.fixture
def normalizer(tmp_csv):
    """
    ICDONormalizer chargé depuis le CSV de test minimal.
    Indépendant du vrai CSV FRACCO pour la portabilité des tests.
    """
    return ICDONormalizer.from_csv(tmp_csv)


# --- Tests from_csv ---

def test_from_csv_loads(tmp_csv):
    normalizer = ICDONormalizer.from_csv(tmp_csv)
    assert normalizer is not None


def test_from_csv_missing_file():
    with pytest.raises(FileNotFoundError):
        ICDONormalizer.from_csv("inexistant.csv")


# --- Tests normalize (exact match) ---

def test_exact_match(normalizer):
    code = normalizer.normalize("carcinome canalaire infiltrant", "morphologie")
    assert code == "8500/3"


def test_exact_match_case_insensitive(normalizer):
    code = normalizer.normalize("Carcinome Canalaire Infiltrant", "morphologie")
    assert code == "8500/3"


def test_exact_match_with_spaces(normalizer):
    code = normalizer.normalize("  sein gauche  ", "topographie")
    assert code == "C50.9"


def test_wrong_label_returns_none(normalizer):
    """Même texte mais mauvais label -> None."""
    code = normalizer.normalize(
        "carcinome canalaire infiltrant", "topographie", fuzzy=False
    )
    assert code is None


def test_unknown_text_returns_none(normalizer):
    code = normalizer.normalize("texte inconnu xyz", "morphologie", fuzzy=False)
    assert code is None


# --- Tests nettoyage labels BIO ---

def test_bio_prefix_B_cleaned(normalizer):
    """
    Le modèle NER produit "B-morphologie" — le normalizer doit accepter ça.
    """
    code = normalizer.normalize(
        "carcinome canalaire infiltrant", "B-morphologie"
    )
    assert code == "8500/3"


def test_bio_prefix_I_cleaned(normalizer):
    """Le préfixe I- doit aussi être nettoyé."""
    code = normalizer.normalize(
        "carcinome canalaire infiltrant", "I-morphologie"
    )
    assert code == "8500/3"


def test_bio_prefix_consistent_with_clean(normalizer):
    """B-morphologie et morphologie donnent le même résultat."""
    code_bio = normalizer.normalize(
        "carcinome canalaire infiltrant", "B-morphologie"
    )
    code_clean = normalizer.normalize(
        "carcinome canalaire infiltrant", "morphologie"
    )
    assert code_bio == code_clean == "8500/3"


# --- Tests fuzzy match ---

def test_fuzzy_match_one_missing_letter(normalizer):
    """
    Faute de frappe médicale : lettre manquante.
    "carcinom" au lieu de "carcinome" -> similarité ~0.96
    """
    code = normalizer.normalize(
        "carcinom canalaire infiltrant",
        "morphologie",
        fuzzy=True,
        fuzzy_threshold=0.8,
    )
    assert code == "8500/3"


def test_fuzzy_match_truncated_word(normalizer):
    """
    Abréviation médicale : mot tronqué.
    "carcinome canalair" (sans 'e' final) -> similarité élevée
    """
    code = normalizer.normalize(
        "carcinome canalair infiltrant",
        "morphologie",
        fuzzy=True,
        fuzzy_threshold=0.8,
    )
    assert code == "8500/3"


def test_fuzzy_match_with_punctuation(normalizer):
    """
    Ponctuation parasite : "carcinom. canalaire infiltrant"
    """
    code = normalizer.normalize(
        "carcinom. canalaire infiltrant",
        "morphologie",
        fuzzy=True,
        fuzzy_threshold=0.8,
    )
    assert code == "8500/3"


def test_fuzzy_match_disabled(normalizer):
    """Avec fuzzy=False, pas de match approximatif."""
    code = normalizer.normalize(
        "carcinom canalaire infiltrant",
        "morphologie",
        fuzzy=False,
    )
    assert code is None


def test_fuzzy_threshold_too_high(normalizer):
    """Seuil trop élevé -> pas de match."""
    code = normalizer.normalize(
        "carcinome canalaire",
        "morphologie",
        fuzzy=True,
        fuzzy_threshold=0.99,
    )
    assert code is None


# --- Tests normalize_entities ---

def test_normalize_entities(normalizer):
    entities = [
        {
            "text": "carcinome canalaire infiltrant",
            "label": "morphologie",
            "start": 0,
            "end": 30,
        },
        {
            "text": "sein gauche",
            "label": "topographie",
            "start": 31,
            "end": 41,
        },
    ]
    result = normalizer.normalize_entities(entities)
    assert len(result) == 2
    assert result[0]["icdo_code"] == "8500/3"
    assert result[1]["icdo_code"] == "C50.9"


def test_normalize_entities_with_bio_labels(normalizer):
    """normalize_entities accepte les labels BIO directement."""
    entities = [
        {
            "text": "carcinome canalaire infiltrant",
            "label": "B-morphologie",
            "start": 0,
            "end": 30,
        },
    ]
    result = normalizer.normalize_entities(entities)
    assert result[0]["icdo_code"] == "8500/3"


def test_normalize_entities_unknown(normalizer):
    """Entité inconnue -> icdo_code=None."""
    entities = [
        {"text": "texte inconnu", "label": "morphologie", "start": 0, "end": 13},
    ]
    result = normalizer.normalize_entities(entities, fuzzy=False)
    assert result[0]["icdo_code"] is None


def test_normalize_entities_preserves_fields(normalizer):
    """Les champs originaux de l'entité sont préservés."""
    entities = [
        {
            "text": "carcinome canalaire infiltrant",
            "label": "morphologie",
            "start": 0,
            "end": 30,
            "confidence": 0.95,
        },
    ]
    result = normalizer.normalize_entities(entities)
    assert result[0]["confidence"] == 0.95
    assert result[0]["start"] == 0
    assert result[0]["icdo_code"] == "8500/3"


# --- Tests get_stats ---

def test_get_stats_returns_all_labels(normalizer):
    stats = normalizer.get_stats()
    assert "morphologie" in stats
    assert "topographie" in stats
    assert "differenciation" in stats
    assert "expression_CIM" in stats

def test_get_stats_counts(normalizer):
    stats = normalizer.get_stats()
    assert stats["morphologie"]["n_unique_codes"] > 0
    assert stats["morphologie"]["n_unique_expressions"] > 0


# --- Tests _similarity ---

def test_similarity_identical():
    assert ICDONormalizer._similarity("abc", "abc") == 1.0


def test_similarity_empty():
    assert ICDONormalizer._similarity("", "abc") == 0.0


def test_similarity_different():
    score = ICDONormalizer._similarity("abc", "xyz")
    assert 0.0 <= score < 1.0


def test_similarity_symmetric():
    """La similarité doit être symétrique."""
    s1, s2 = "carcinome", "carcinom"
    assert ICDONormalizer._similarity(s1, s2) == ICDONormalizer._similarity(s2, s1)