# src/onco_ner/models/ner_model.py

"""
Wrapper NER autour de XLM-RoBERTa optimisé.

Gère :
- Chargement du modèle depuis HuggingFace Hub ou chemin local
- Inférence avec sliding window (documents > 512 tokens)
- Extraction des entités depuis les prédictions BIO
- Score de confiance par entité (softmax sur les logits)
- Normalisation ICD-O optionnelle via ICDONormalizer
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
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


class NERModel:
    """
    Wrapper autour du modèle de token classification XLM-RoBERTa optimisé.

    Usage sans normalisation :
        model = NERModel.from_pretrained("Eudes9/onco-ner-xlm-roberta-optimized")
        entities = model.predict("Patient avec carcinome canalaire infiltrant")

    Usage avec normalisation ICD-O intégrée :
        from onco_ner.models.normalizer import ICDONormalizer
        normalizer = ICDONormalizer.from_csv("data/DetectOnco_Final.csv")
        model = NERModel.from_pretrained(
            "Eudes9/onco-ner-xlm-roberta-optimized",
            normalizer=normalizer
        )
        entities = model.predict("Patient avec carcinome canalaire infiltrant")
        # -> entités avec champ 'icdo_code' inclus
    """

    def __init__(
        self,
        model: AutoModelForTokenClassification,
        tokenizer: AutoTokenizer,
        device: torch.device,
        max_length: int = 512,
        stride: int = 256,
        normalizer=None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.stride = stride
        self.normalizer = normalizer

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        max_length: int = 512,
        stride: int = 256,
        device: str | None = None,
        normalizer=None,
    ) -> "NERModel":
        """
        Charge le modèle depuis HuggingFace Hub ou un chemin local.

        Args:
            model_path  : identifiant HuggingFace ou chemin local
            max_length  : taille de la fenêtre de tokenization
            stride      : chevauchement entre les fenêtres
            device      : "cuda", "cpu" ou None (auto-détection)
            normalizer  : ICDONormalizer optionnel pour la normalisation ICD-O
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), use_fast=True
            )
            model = AutoModelForTokenClassification.from_pretrained(
                str(model_path)
            )
            if device is None:
                _device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                _device = torch.device(device)

            model = model.to(_device)
            model.eval()

            logger.info(f"Modèle chargé depuis '{model_path}' sur {_device}")
            return cls(model, tokenizer, _device, max_length, stride, normalizer)

        except Exception as e:
            raise ModelNotLoadedError(
                f"Impossible de charger le modèle depuis '{model_path}'"
            ) from e

    def predict(
        self,
        text: str,
        fuzzy: bool = True,
        fuzzy_threshold: float = 0.8,
    ) -> list[dict]:
        """
        Prédit les entités oncologiques dans un texte clinique.

        Utilise le sliding window pour couvrir les documents longs.
        Si un normalizer est configuré, enrichit les entités avec
        les codes ICD-O correspondants.

        Args:
            text            : texte clinique en français
            fuzzy           : activer le matching approximatif
            fuzzy_threshold : seuil de similarité pour le fuzzy match

        Returns:
            Liste de dicts avec clés :
            - text      : texte de l'entité
            - label     : type d'entité (sans préfixe BIO)
            - start     : position de début dans le texte
            - end       : position de fin dans le texte
            - score     : score de confiance moyen (0-1)
            - icdo_code : code ICD-O (si normalizer configuré, sinon absent)
        """
        if not text.strip():
            return []

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )

        all_token_preds = {}

        n_chunks = len(encoding["input_ids"])
        for chunk_idx in range(n_chunks):
            input_ids = encoding["input_ids"][chunk_idx].unsqueeze(0).to(self.device)
            attention_mask = encoding["attention_mask"][chunk_idx].unsqueeze(0).to(self.device)

            # Gestion des deux cas : tenseur PyTorch (production) ou liste (tests)
            offset_mapping_raw = encoding["offset_mapping"][chunk_idx]
            offset_mapping = (
                offset_mapping_raw.tolist()
                if hasattr(offset_mapping_raw, "tolist")
                else offset_mapping_raw
            )

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            probs = F.softmax(outputs.logits[0], dim=-1)
            predictions = torch.argmax(probs, dim=-1).cpu().tolist()
            scores = probs.max(dim=-1).values.cpu().tolist()

            if chunk_idx > 0:
                overlap_end_char = next(
                    (off[0] for off in offset_mapping
                     if not (off[0] == 0 and off[1] == 0)),
                    0,
                )
            else:
                overlap_end_char = 0

            for (offset_start, offset_end), pred, score in zip(
                offset_mapping, predictions, scores
            ):
                if offset_start == 0 and offset_end == 0:
                    continue
                if chunk_idx > 0 and offset_start < overlap_end_char:
                    continue
                if offset_start not in all_token_preds:
                    all_token_preds[offset_start] = (
                        offset_end, ID2LABEL[pred], score
                    )

        sorted_tokens = sorted(all_token_preds.items())
        entities = self._extract_entities(text, sorted_tokens)

        # Normalisation ICD-O optionnelle avec transmission des paramètres fuzzy
        if self.normalizer is not None:
            entities = self.normalizer.normalize_entities(
                entities,
                fuzzy=fuzzy,
                fuzzy_threshold=fuzzy_threshold,
            )

        logger.info(f"Texte analysé : {len(entities)} entités détectées")
        return entities

    def _extract_entities(
        self,
        text: str,
        sorted_tokens: list[tuple],
    ) -> list[dict]:
        """
        Reconstruit les entités depuis les prédictions BIO triées par position.
        Gère les subtokens orphelins I- sans B- précédent.
        Calcule le score de confiance moyen par entité.
        """
        entities = []
        current_entity = None
        current_scores = []

        for offset_start, (offset_end, label, score) in sorted_tokens:
            if label.startswith("B-"):
                if current_entity is not None:
                    current_entity["score"] = round(
                        sum(current_scores) / len(current_scores), 4
                    )
                    entities.append(current_entity)

                entity_type = label[2:]
                current_entity = {
                    "text": text[offset_start:offset_end],
                    "label": entity_type,
                    "start": offset_start,
                    "end": offset_end,
                    "score": 0.0,
                }
                current_scores = [score]

            elif label.startswith("I-"):
                entity_type = label[2:]
                if (current_entity is not None and
                        current_entity["label"] == entity_type):
                    current_entity["end"] = offset_end
                    current_entity["text"] = text[
                        current_entity["start"]:offset_end
                    ]
                    current_scores.append(score)
                else:
                    if current_entity is not None:
                        current_entity["score"] = round(
                            sum(current_scores) / len(current_scores), 4
                        )
                        entities.append(current_entity)

                    current_entity = {
                        "text": text[offset_start:offset_end],
                        "label": entity_type,
                        "start": offset_start,
                        "end": offset_end,
                        "score": 0.0,
                    }
                    current_scores = [score]

            else:
                if current_entity is not None:
                    current_entity["score"] = round(
                        sum(current_scores) / len(current_scores), 4
                    )
                    entities.append(current_entity)
                    current_entity = None
                    current_scores = []

        if current_entity is not None:
            current_entity["score"] = round(
                sum(current_scores) / len(current_scores), 4
            )
            entities.append(current_entity)

        return entities