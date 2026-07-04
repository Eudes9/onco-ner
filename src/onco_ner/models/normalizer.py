# src/onco_ner/models/normalizer.py
"""
ICDONormalizer : normalisation des entités oncologiques vers les codes ICD-O.

Prend le texte d'une entité détectée (ex: "carcinome canalaire infiltrant")
et retourne le code ICD-O correspondant (ex: "8500/3") en cherchant
dans le corpus d'annotations FRACCO (DetectOnco_Final.csv).

Stratégie de matching :
1. Exact match : recherche exacte du texte dans le corpus
2. Fuzzy match : si pas de match exact, recherche par similarité
   via rapidfuzz (compilé en C, ~100x plus rapide que Python pur)
3. Si aucun match : retourne None avec un warning

Dépendances :
    pip install rapidfuzz
"""

from pathlib import Path

import polars as pl
from rapidfuzz.distance import Levenshtein

from onco_ner.exceptions import UnknownICDOCodeError
from onco_ner.utils.logging import get_logger

logger = get_logger(__name__)


class ICDONormalizer:
    """
    Normalise les entités oncologiques détectées vers les codes ICD-O.

    Usage :
        normalizer = ICDONormalizer.from_csv("data/DetectOnco_Final.csv")
        code = normalizer.normalize("carcinome canalaire infiltrant", "morphologie")
        # -> "8500/3"

        # Avec labels BIO (sortie directe du modèle NER)
        code = normalizer.normalize("carcinome canalaire infiltrant", "B-morphologie")
        # -> "8500/3" (préfixe BIO nettoyé automatiquement)
    """

    def __init__(self, lookup_df: pl.DataFrame) -> None:
        self._lookup = lookup_df
        self._exact_index = self._build_exact_index()
        logger.info(
            f"ICDONormalizer initialisé : "
            f"{len(self._exact_index)} entrées uniques"
        )

    @classmethod
    def from_csv(cls, csv_path: Path | str) -> "ICDONormalizer":
        """Crée un ICDONormalizer depuis le CSV FRACCO."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV introuvable : {csv_path}")

        df = pl.read_csv(csv_path).with_columns([
            pl.col("content").str.strip_chars().str.to_lowercase(),
            pl.col("code").str.strip_chars(),
            pl.col("label").str.strip_chars(),
        ])

        df = df.select(["label", "content", "code"]).unique()
        logger.info(f"CSV chargé : {df.shape[0]} annotations depuis {csv_path}")

        return cls(df)

    def _build_exact_index(self) -> dict[tuple[str, str], str]:
        """Construit un index de lookup exact."""
        index = {}
        for row in self._lookup.to_dicts():
            key = (row["content"].lower().strip(), row["label"])
            if key not in index:
                index[key] = row["code"]
        return index

    @staticmethod
    def _clean_label(label: str) -> str:
        """
        Nettoie un label BIO en label pur.
        Exemple : "B-morphologie" -> "morphologie"
                  "I-topographie" -> "topographie"
                  "morphologie"   -> "morphologie" (inchangé)
        """
        return label.replace("B-", "").replace("I-", "").lower().strip()

    @staticmethod
    def _similarity(s1: str, s2: str) -> float:
        """
        Calcule la similarité entre deux chaînes via rapidfuzz.
        Compilé en C — ~100x plus rapide que Python pur.
        Score entre 0.0 (aucune similarité) et 1.0 (identiques).
        """
        return Levenshtein.normalized_similarity(s1, s2)

    def normalize(
        self,
        text: str,
        label: str,
        fuzzy: bool = True,
        fuzzy_threshold: float = 0.8,
    ) -> str | None:
        """
        Normalise une entité vers son code ICD-O.

        Args:
            text : texte de l'entité (ex: "carcinome canalaire infiltrant")
            label : type d'entité, avec ou sans préfixe BIO
                    (ex: "morphologie" ou "B-morphologie")
            fuzzy : activer le matching approximatif si pas de match exact
            fuzzy_threshold : seuil de similarité minimum (0-1)

        Returns:
            Code ICD-O (ex: "8500/3") ou None si aucun match trouvé
        """
        text_normalized = text.lower().strip()
        # Nettoyage automatique des préfixes BIO
        clean_label = self._clean_label(label)

        # 1. Exact match
        code = self._exact_index.get((text_normalized, clean_label))
        if code is not None:
            logger.debug(f"Match exact : '{text}' -> {code}")
            return code

        # 2. Fuzzy match si activé
        if fuzzy:
            code = self._fuzzy_match(
                text_normalized, clean_label, fuzzy_threshold
            )
            if code is not None:
                return code

        logger.warning(
            f"Aucun code ICD-O trouvé pour : '{text}' (label={clean_label})"
        )
        return None

    def _fuzzy_match(
        self,
        text: str,
        label: str,
        threshold: float,
    ) -> str | None:
        """
        Recherche par similarité via rapidfuzz.
        Filtre d'abord par label pour réduire l'espace de recherche.
        """
        label_entries = [
            (content, code)
            for (content, lbl), code in self._exact_index.items()
            if lbl == label
        ]

        if not label_entries:
            return None

        best_score = 0.0
        best_code = None
        best_match = None

        for content, code in label_entries:
            score = self._similarity(text, content)
            if score > best_score:
                best_score = score
                best_code = code
                best_match = content

        if best_score >= threshold:
            logger.debug(
                f"Fuzzy match : '{text}' -> '{best_match}' "
                f"(score={best_score:.3f}) -> {best_code}"
            )
            return best_code

        return None

    def normalize_entities(
        self,
        entities: list[dict],
        fuzzy: bool = True,
        fuzzy_threshold: float = 0.8,
    ) -> list[dict]:
        """
        Normalise une liste d'entités détectées par le modèle NER.
        Gère automatiquement les labels BIO (B-morphologie, I-morphologie).

        Args:
            entities : liste de dicts avec clés 'text' et 'label'
            fuzzy : activer le matching approximatif
            fuzzy_threshold : seuil de similarité minimum

        Returns:
            Liste de dicts enrichis avec 'icdo_code'
        """
        normalized = []
        for entity in entities:
            code = self.normalize(
                entity["text"],
                entity["label"],
                fuzzy=fuzzy,
                fuzzy_threshold=fuzzy_threshold,
            )
            normalized.append({**entity, "icdo_code": code})
        return normalized

    def get_stats(self) -> dict:
        """Retourne des statistiques sur le référentiel ICD-O chargé."""
        stats = {}
        for label in self._lookup["label"].unique().to_list():
            label_df = self._lookup.filter(pl.col("label") == label)
            stats[label] = {
                "n_unique_codes": label_df["code"].n_unique(),
                "n_unique_expressions": label_df["content"].n_unique(),
            }
        return stats