# tests/test_pipeline.py

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from onco_ner.pipeline import Pipeline
from onco_ner.models.ner_model import NERModel
from onco_ner.models.normalizer import ICDONormalizer
from onco_ner.exceptions import ModelNotLoadedError


# --- Fixtures ---

@pytest.fixture
def tmp_csv(tmp_path):
    """CSV minimal pour le normalizer."""
    csv_content = """doc_name,id,label,code,content,full_span
cc_onco1.ann,T1,morphologie,8500/3,carcinome canalaire infiltrant,0 30
cc_onco1.ann,T2,topographie,C50.9,sein gauche,31 41
"""
    csv_file = tmp_path / "test_corpus.csv"
    csv_file.write_text(csv_content, encoding="utf-8")
    return csv_file


@pytest.fixture
def mock_entities():
    """Entités simulées retournées par le modèle NER."""
    return [
        {
            "text": "carcinome canalaire infiltrant",
            "label": "morphologie",
            "start": 13,
            "end": 42,
            "score": 0.94,
        },
        {
            "text": "sein gauche",
            "label": "topographie",
            "start": 46,
            "end": 57,
            "score": 0.91,
        },
    ]


@pytest.fixture
def mock_ner_model(mock_entities):
    """NERModel mocké qui retourne des entités prédéfinies."""
    model = MagicMock(spec=NERModel)
    model.predict.return_value = mock_entities
    return model


@pytest.fixture
def pipeline_without_normalizer(mock_ner_model):
    """Pipeline sans normalizer."""
    return Pipeline(ner_model=mock_ner_model, normalizer=None)


@pytest.fixture
def pipeline_with_normalizer(mock_ner_model, tmp_csv):
    """Pipeline avec normalizer chargé depuis le CSV de test."""
    normalizer = ICDONormalizer.from_csv(tmp_csv)
    mock_ner_model.normalizer = normalizer
    return Pipeline(ner_model=mock_ner_model, normalizer=normalizer)


# --- Tests predict ---

def test_predict_returns_correct_structure(pipeline_without_normalizer):
    """Le résultat de predict() a les bonnes clés."""
    result = pipeline_without_normalizer.predict("texte clinique")
    assert "text" in result
    assert "entities" in result
    assert "n_entities" in result


def test_predict_text_preserved(pipeline_without_normalizer):
    """Le texte d'entrée est préservé dans le résultat."""
    text = "Patient avec carcinome canalaire infiltrant"
    result = pipeline_without_normalizer.predict(text)
    assert result["text"] == text


def test_predict_n_entities_correct(pipeline_without_normalizer, mock_entities):
    """n_entities correspond au nombre d'entités détectées."""
    result = pipeline_without_normalizer.predict("texte clinique")
    assert result["n_entities"] == len(mock_entities)


def test_predict_empty_text(pipeline_without_normalizer):
    """Texte vide retourne un résultat vide sans planter."""
    result = pipeline_without_normalizer.predict("")
    assert result["entities"] == []
    assert result["n_entities"] == 0


def test_predict_whitespace_only(pipeline_without_normalizer):
    """Texte avec espaces seulement retourne un résultat vide."""
    result = pipeline_without_normalizer.predict("   ")
    assert result["entities"] == []
    assert result["n_entities"] == 0


def test_predict_transmits_fuzzy_params(pipeline_without_normalizer):
    """Les paramètres fuzzy sont transmis au modèle NER."""
    pipeline_without_normalizer.predict(
        "texte", fuzzy=False, fuzzy_threshold=0.9
    )
    pipeline_without_normalizer.ner_model.predict.assert_called_once_with(
        "texte", fuzzy=False, fuzzy_threshold=0.9
    )


# --- Tests predict_batch ---

def test_predict_batch_returns_list(pipeline_without_normalizer):
    """predict_batch retourne une liste."""
    texts = ["texte 1", "texte 2", "texte 3"]
    results = pipeline_without_normalizer.predict_batch(texts)
    assert isinstance(results, list)
    assert len(results) == 3


def test_predict_batch_each_result_has_structure(pipeline_without_normalizer):
    """Chaque résultat de predict_batch a la bonne structure."""
    texts = ["texte 1", "texte 2"]
    results = pipeline_without_normalizer.predict_batch(texts)
    for result in results:
        assert "text" in result
        assert "entities" in result
        assert "n_entities" in result


def test_predict_batch_empty_list(pipeline_without_normalizer):
    """predict_batch sur liste vide retourne liste vide."""
    results = pipeline_without_normalizer.predict_batch([])
    assert results == []


def test_predict_batch_transmits_fuzzy_params(pipeline_without_normalizer):
    """Les paramètres fuzzy sont transmis pour chaque texte."""
    pipeline_without_normalizer.predict_batch(
        ["texte 1"], fuzzy=False, fuzzy_threshold=0.95
    )
    pipeline_without_normalizer.ner_model.predict.assert_called_with(
        "texte 1", fuzzy=False, fuzzy_threshold=0.95
    )


# --- Tests predict_file ---

def test_predict_file_returns_source_file(pipeline_without_normalizer, tmp_path):
    """predict_file ajoute le champ source_file."""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("texte clinique", encoding="utf-8")
    result = pipeline_without_normalizer.predict_file(txt_file)
    assert "source_file" in result
    assert "test.txt" in result["source_file"]


def test_predict_file_missing_file(pipeline_without_normalizer):
    """predict_file lève FileNotFoundError si fichier absent."""
    with pytest.raises(FileNotFoundError):
        pipeline_without_normalizer.predict_file("inexistant.txt")


def test_predict_file_reads_content(pipeline_without_normalizer, tmp_path):
    """predict_file lit correctement le contenu du fichier."""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("carcinome canalaire", encoding="utf-8")
    pipeline_without_normalizer.predict_file(txt_file)
    # Vérifie que predict a été appelé avec le contenu du fichier
    call_args = pipeline_without_normalizer.ner_model.predict.call_args
    assert "carcinome canalaire" in call_args[0][0]


# --- Tests from_pretrained (avec mock HuggingFace) ---

@patch("onco_ner.pipeline.NERModel.from_pretrained")
def test_from_pretrained_without_csv(mock_from_pretrained, tmp_path):
    """from_pretrained sans csv_path ne charge pas de normalizer."""
    mock_from_pretrained.return_value = MagicMock(spec=NERModel)
    pipeline = Pipeline.from_pretrained("fake/model")
    assert pipeline.normalizer is None


@patch("onco_ner.pipeline.NERModel.from_pretrained")
def test_from_pretrained_with_csv(mock_from_pretrained, tmp_csv):
    """from_pretrained avec csv_path charge un normalizer."""
    mock_from_pretrained.return_value = MagicMock(spec=NERModel)
    pipeline = Pipeline.from_pretrained("fake/model", csv_path=tmp_csv)
    assert pipeline.normalizer is not None
    assert isinstance(pipeline.normalizer, ICDONormalizer)


@patch("onco_ner.pipeline.NERModel.from_pretrained")
def test_from_pretrained_missing_csv(mock_from_pretrained, tmp_path):
    """from_pretrained avec csv_path inexistant lève FileNotFoundError."""
    mock_from_pretrained.return_value = MagicMock(spec=NERModel)
    with pytest.raises(FileNotFoundError):
        Pipeline.from_pretrained(
            "fake/model",
            csv_path=tmp_path / "inexistant.csv"
        )