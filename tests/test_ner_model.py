# tests/test_ner_model.py

import pytest
import torch
from unittest.mock import MagicMock, patch
from pathlib import Path
from onco_ner.models.ner_model import NERModel
from onco_ner.models.normalizer import ICDONormalizer
from onco_ner.exceptions import ModelNotLoadedError


# --- Fixtures ---

@pytest.fixture
def tmp_csv(tmp_path):
    csv_content = """doc_name,id,label,code,content,full_span
cc_onco1.ann,T1,morphologie,8500/3,carcinome canalaire infiltrant,0 30
cc_onco1.ann,T2,topographie,C50.9,sein gauche,31 41
"""
    csv_file = tmp_path / "test_corpus.csv"
    csv_file.write_text(csv_content, encoding="utf-8")
    return csv_file


@pytest.fixture
def mock_tokenizer():
    """
    Tokenizer mocké qui retourne un encoding minimal compatible
    avec la sliding window de NERModel.predict().
    """
    tokenizer = MagicMock()

    # Simuler l'encoding retourné par le tokenizer avec sliding window
    encoding_mock = MagicMock()
    encoding_mock.__len__ = lambda self: 1

    # Un seul chunk avec 4 tokens : [CLS], "carcinome", "canalaire", [SEP]
    encoding_mock.__getitem__ = lambda self, key: {
        "input_ids": [torch.zeros(4, dtype=torch.long)],
        "attention_mask": [torch.ones(4, dtype=torch.long)],
        "offset_mapping": [
            [(0, 0), (0, 9), (10, 19), (0, 0)]  # CLS, token1, token2, SEP
        ],
    }[key]

    tokenizer.return_value = encoding_mock
    return tokenizer


@pytest.fixture
def mock_hf_model():
    """
    Modèle HuggingFace mocké.
    Retourne des logits qui prédisent B-morphologie sur le token 1.
    """
    model = MagicMock()

    # Shape : (1, 4, 9) — 4 tokens, 9 labels
    # Token 0 (CLS) : label 0 (O) avec score max
    # Token 1 : label 1 (B-morphologie) avec score max
    # Token 2 : label 2 (I-morphologie) avec score max
    # Token 3 (SEP) : label 0 (O)
    logits = torch.zeros(1, 4, 9)
    logits[0, 1, 1] = 10.0  # B-morphologie
    logits[0, 2, 2] = 10.0  # I-morphologie

    model.return_value.logits = logits
    model.to.return_value = model
    model.eval.return_value = model
    return model


@pytest.fixture
def ner_model_with_mocks(mock_tokenizer, mock_hf_model):
    """NERModel instancié manuellement avec des mocks."""
    return NERModel(
        model=mock_hf_model,
        tokenizer=mock_tokenizer,
        device=torch.device("cpu"),
        max_length=4,
        stride=2,
        normalizer=None,
    )


# --- Tests _extract_entities ---

def test_extract_entities_simple():
    """Extraction d'une entité simple B-I."""
    model = MagicMock(spec=NERModel)
    model._extract_entities = NERModel._extract_entities.__get__(model)

    sorted_tokens = [
        (0, (9, "B-morphologie", 0.95)),
        (10, (19, "I-morphologie", 0.93)),
        (20, (25, "O", 0.99)),
    ]
    text = "carcinome canalaire sain"
    entities = model._extract_entities(text, sorted_tokens)

    assert len(entities) == 1
    assert entities[0]["label"] == "morphologie"
    assert entities[0]["start"] == 0
    assert entities[0]["end"] == 19


def test_extract_entities_empty_tokens():
    """Aucun token -> aucune entité."""
    model = MagicMock(spec=NERModel)
    model._extract_entities = NERModel._extract_entities.__get__(model)
    entities = model._extract_entities("texte", [])
    assert entities == []


def test_extract_entities_only_O():
    """Que des labels O -> aucune entité."""
    model = MagicMock(spec=NERModel)
    model._extract_entities = NERModel._extract_entities.__get__(model)
    sorted_tokens = [
        (0, (5, "O", 0.99)),
        (6, (11, "O", 0.98)),
    ]
    entities = model._extract_entities("texte sain", sorted_tokens)
    assert entities == []


def test_extract_entities_orphan_I():
    """I- sans B- précédent est traité comme B-."""
    model = MagicMock(spec=NERModel)
    model._extract_entities = NERModel._extract_entities.__get__(model)
    sorted_tokens = [
        (0, (9, "I-morphologie", 0.85)),
    ]
    entities = model._extract_entities("carcinome", sorted_tokens)
    assert len(entities) == 1
    assert entities[0]["label"] == "morphologie"


def test_extract_entities_score_averaged():
    """Le score est la moyenne des scores des subtokens."""
    model = MagicMock(spec=NERModel)
    model._extract_entities = NERModel._extract_entities.__get__(model)
    sorted_tokens = [
        (0, (3, "B-morphologie", 0.9)),
        (4, (7, "I-morphologie", 0.8)),
    ]
    entities = model._extract_entities("car cin", sorted_tokens)
    assert entities[0]["score"] == round((0.9 + 0.8) / 2, 4)


def test_extract_entities_multiple():
    """Deux entités distinctes sont bien séparées."""
    model = MagicMock(spec=NERModel)
    model._extract_entities = NERModel._extract_entities.__get__(model)
    sorted_tokens = [
        (0, (9, "B-morphologie", 0.95)),
        (10, (15, "O", 0.99)),
        (16, (27, "B-topographie", 0.90)),
    ]
    entities = model._extract_entities(
        "carcinome     sein gauche", sorted_tokens
    )
    assert len(entities) == 2
    assert entities[0]["label"] == "morphologie"
    assert entities[1]["label"] == "topographie"


def test_extract_entities_last_entity_not_lost():
    """La dernière entité est bien retournée même sans O final."""
    model = MagicMock(spec=NERModel)
    model._extract_entities = NERModel._extract_entities.__get__(model)
    sorted_tokens = [
        (0, (9, "B-morphologie", 0.95)),
        (10, (19, "I-morphologie", 0.93)),
        # Pas de O à la fin
    ]
    entities = model._extract_entities("carcinome canalaire", sorted_tokens)
    assert len(entities) == 1
    assert entities[0]["end"] == 19


# --- Tests predict ---

def test_predict_empty_text(ner_model_with_mocks):
    """Texte vide retourne liste vide sans planter."""
    result = ner_model_with_mocks.predict("")
    assert result == []


def test_predict_whitespace_only(ner_model_with_mocks):
    """Texte avec espaces seulement retourne liste vide."""
    result = ner_model_with_mocks.predict("   ")
    assert result == []


def test_predict_nominal_flow(ner_model_with_mocks, mock_tokenizer, mock_hf_model):
    """
    Vérifie le flux nominal de predict() avec des mocks complets.
    Le tokenizer et le modèle doivent être appelés,
    et _extract_entities doit recevoir les bonnes prédictions.
    """
    with patch.object(
        ner_model_with_mocks,
        "_extract_entities",
        return_value=[{"text": "carcinome canalaire", "label": "morphologie",
                       "start": 0, "end": 19, "score": 0.94}]
    ) as mock_extract:
        result = ner_model_with_mocks.predict("carcinome canalaire")

        assert mock_tokenizer.called
        assert mock_hf_model.called
        assert mock_extract.called
        assert len(result) == 1
        assert result[0]["label"] == "morphologie"


def test_predict_returns_list(ner_model_with_mocks):
    """predict retourne toujours une liste."""
    result = ner_model_with_mocks.predict("texte quelconque")
    assert isinstance(result, list)


def test_predict_with_normalizer(ner_model_with_mocks, tmp_csv):
    """predict enrichit les entités avec icdo_code si normalizer configuré."""
    normalizer = ICDONormalizer.from_csv(tmp_csv)
    ner_model_with_mocks.normalizer = normalizer

    with patch.object(
        ner_model_with_mocks,
        "_extract_entities",
        return_value=[{
            "text": "carcinome canalaire infiltrant",
            "label": "morphologie",
            "start": 0,
            "end": 30,
            "score": 0.94,
        }]
    ):
        result = ner_model_with_mocks.predict("carcinome canalaire infiltrant")
        assert "icdo_code" in result[0]
        assert result[0]["icdo_code"] == "8500/3"


# --- Tests from_pretrained ---

@patch("onco_ner.models.ner_model.AutoTokenizer.from_pretrained")
@patch("onco_ner.models.ner_model.AutoModelForTokenClassification.from_pretrained")
def test_from_pretrained_returns_ner_model(mock_model, mock_tokenizer):
    """from_pretrained retourne bien une instance NERModel."""
    mock_model.return_value = MagicMock()
    mock_model.return_value.to.return_value = mock_model.return_value
    mock_model.return_value.eval.return_value = mock_model.return_value
    mock_tokenizer.return_value = MagicMock()

    model = NERModel.from_pretrained("fake/model")
    assert isinstance(model, NERModel)


@patch("onco_ner.models.ner_model.AutoTokenizer.from_pretrained")
@patch("onco_ner.models.ner_model.AutoModelForTokenClassification.from_pretrained")
def test_from_pretrained_with_normalizer(mock_model, mock_tokenizer, tmp_csv):
    """from_pretrained avec normalizer l'attache au modèle."""
    mock_model.return_value = MagicMock()
    mock_model.return_value.to.return_value = mock_model.return_value
    mock_model.return_value.eval.return_value = mock_model.return_value
    mock_tokenizer.return_value = MagicMock()

    normalizer = ICDONormalizer.from_csv(tmp_csv)
    model = NERModel.from_pretrained("fake/model", normalizer=normalizer)
    assert model.normalizer is not None
    assert isinstance(model.normalizer, ICDONormalizer)


@patch("onco_ner.models.ner_model.AutoTokenizer.from_pretrained")
@patch("onco_ner.models.ner_model.AutoModelForTokenClassification.from_pretrained")
def test_from_pretrained_default_no_normalizer(mock_model, mock_tokenizer):
    """from_pretrained sans normalizer -> normalizer=None."""
    mock_model.return_value = MagicMock()
    mock_model.return_value.to.return_value = mock_model.return_value
    mock_model.return_value.eval.return_value = mock_model.return_value
    mock_tokenizer.return_value = MagicMock()

    model = NERModel.from_pretrained("fake/model")
    assert model.normalizer is None


@patch("onco_ner.models.ner_model.AutoTokenizer.from_pretrained")
@patch("onco_ner.models.ner_model.AutoModelForTokenClassification.from_pretrained")
def test_from_pretrained_raises_on_error(mock_model, mock_tokenizer):
    """from_pretrained lève ModelNotLoadedError si chargement échoue."""
    mock_model.side_effect = Exception("Model not found")
    with pytest.raises(ModelNotLoadedError):
        NERModel.from_pretrained("fake/nonexistent-model")


@patch("onco_ner.models.ner_model.AutoTokenizer.from_pretrained")
@patch("onco_ner.models.ner_model.AutoModelForTokenClassification.from_pretrained")
def test_from_pretrained_device_cpu(mock_model, mock_tokenizer):
    """from_pretrained avec device='cpu' charge sur CPU."""
    mock_model.return_value = MagicMock()
    mock_model.return_value.to.return_value = mock_model.return_value
    mock_model.return_value.eval.return_value = mock_model.return_value
    mock_tokenizer.return_value = MagicMock()

    model = NERModel.from_pretrained("fake/model", device="cpu")
    assert model.device == torch.device("cpu")