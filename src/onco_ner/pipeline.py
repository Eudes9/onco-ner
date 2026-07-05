# src/onco_ner/pipeline.py
"""
Pipeline haut niveau pour l'extraction et la normalisation
d'entités oncologiques depuis des textes cliniques en français.

Usage minimal :
    from onco_ner import Pipeline

    pipeline = Pipeline.from_pretrained(
        model_path="Eudes9/onco-ner-xlm-roberta-optimized",
        csv_path="data/DetectOnco_Final.csv",
    )
    result = pipeline.predict("Patient avec carcinome canalaire infiltrant du sein gauche")

Output :
    {
        "text": "Patient avec carcinome canalaire infiltrant du sein gauche",
        "entities": [
            {
                "text": "carcinome canalaire infiltrant",
                "label": "morphologie",
                "start": 13,
                "end": 42,
                "score": 0.94,
                "icdo_code": "8500/3"
            },
            {
                "text": "sein gauche",
                "label": "topographie",
                "start": 46,
                "end": 57,
                "score": 0.91,
                "icdo_code": "C50.9"
            }
        ],
        "n_entities": 2,
    }
"""

from pathlib import Path
from typing import Optional

from onco_ner.models.ner_model import NERModel
from onco_ner.models.normalizer import ICDONormalizer
from onco_ner.exceptions import ModelNotLoadedError
from onco_ner.utils.logging import get_logger

logger = get_logger(__name__)


class Pipeline:
    """
    Pipeline oncologie : texte clinique -> entités structurées + codes ICD-O.

    Orchestre NERModel + ICDONormalizer en une interface simple.

    Usage :
        pipeline = Pipeline.from_pretrained(
            model_path="Eudes9/onco-ner-xlm-roberta-optimized",
            csv_path="data/DetectOnco_Final.csv",
        )
        result = pipeline.predict("texte clinique...")
        results = pipeline.predict_batch(["texte 1", "texte 2", ...])
    """

    def __init__(
        self,
        ner_model: NERModel,
        normalizer: Optional[ICDONormalizer] = None,
    ) -> None:
        self.ner_model = ner_model
        self.normalizer = normalizer

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        csv_path: Optional[str | Path] = None,
        max_length: int = 512,
        stride: int = 256,
        device: Optional[str] = None,
        fuzzy: bool = True,
        fuzzy_threshold: float = 0.8,
    ) -> "Pipeline":
        """
        Crée un Pipeline depuis un modèle HuggingFace et un CSV FRACCO.

        Args:
            model_path      : identifiant HuggingFace ou chemin local
            csv_path        : chemin vers DetectOnco_Final.csv
                              (None = pas de normalisation ICD-O)
            max_length      : taille de la fenêtre de tokenization
            stride          : chevauchement entre les fenêtres
            device          : "cuda", "cpu" ou None (auto-détection)
            fuzzy           : activer le matching approximatif
            fuzzy_threshold : seuil de similarité pour le fuzzy match
        """
        # Charger le normalizer si csv_path fourni
        normalizer = None
        if csv_path is not None:
            normalizer = ICDONormalizer.from_csv(csv_path)
            logger.info(f"Normalizer chargé depuis '{csv_path}'")

        # Charger le modèle NER avec le normalizer intégré
        ner_model = NERModel.from_pretrained(
            model_path=model_path,
            max_length=max_length,
            stride=stride,
            device=device,
            normalizer=normalizer,
        )

        logger.info(
            f"Pipeline prêt — modèle : '{model_path}' | "
            f"normalisation ICD-O : {'activée' if normalizer else 'désactivée'}"
        )
        return cls(ner_model, normalizer)

    def predict(
        self,
        text: str,
        fuzzy: bool = True,
        fuzzy_threshold: float = 0.8,
    ) -> dict:
        """
        Extrait et normalise les entités oncologiques d'un texte clinique.

        Args:
            text            : texte clinique en français
            fuzzy           : activer le matching approximatif pour la normalisation
            fuzzy_threshold : seuil de similarité pour le fuzzy match

        Returns:
            Dict avec clés :
            - text       : texte d'entrée
            - entities   : liste d'entités structurées
            - n_entities : nombre d'entités détectées
        """
        if not text.strip():
            return {"text": text, "entities": [], "n_entities": 0}

        # Transmission correcte des paramètres fuzzy au modèle NER
        entities = self.ner_model.predict(
            text,
            fuzzy=fuzzy,
            fuzzy_threshold=fuzzy_threshold,
        )

        logger.info(
            f"Pipeline.predict : {len(entities)} entités détectées "
            f"sur {len(text)} caractères"
        )

        return {
            "text": text,
            "entities": entities,
            "n_entities": len(entities),
        }

    def predict_batch(
        self,
        texts: list[str],
        fuzzy: bool = True,
        fuzzy_threshold: float = 0.8,
    ) -> list[dict]:
        """
        Prédit sur une liste de textes cliniques.

        Args:
            texts           : liste de textes cliniques
            fuzzy           : activer le matching approximatif
            fuzzy_threshold : seuil de similarité

        Returns:
            Liste de résultats (même format que predict())
        """
        results = []
        for idx, text in enumerate(texts):
            logger.info(f"Traitement document {idx + 1}/{len(texts)}")
            results.append(
                self.predict(
                    text,
                    fuzzy=fuzzy,
                    fuzzy_threshold=fuzzy_threshold,
                )
            )
        return results

    def predict_file(
        self,
        txt_path: str | Path,
        fuzzy: bool = True,
        fuzzy_threshold: float = 0.8,
    ) -> dict:
        """
        Prédit sur un fichier texte (.txt).

        Args:
            txt_path        : chemin vers le fichier texte
            fuzzy           : activer le matching approximatif
            fuzzy_threshold : seuil de similarité

        Returns:
            Résultat au même format que predict() avec champ 'source_file'
        """
        txt_path = Path(txt_path)
        if not txt_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {txt_path}")

        text = txt_path.read_text(encoding="utf-8")
        logger.info(f"Fichier chargé : {txt_path.name} ({len(text)} caractères)")

        result = self.predict(
            text,
            fuzzy=fuzzy,
            fuzzy_threshold=fuzzy_threshold,
        )
        result["source_file"] = str(txt_path)
        return result