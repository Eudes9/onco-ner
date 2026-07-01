class OncoNERError(Exception):
    """Exception de base pour la librairie onco_ner."""


class InvalidAnnotationError(OncoNERError):
    """Levée quand un fichier .ann est mal formé ou incohérent."""


class ModelNotLoadedError(OncoNERError):
    """Levée quand on tente une prédiction sans modèle chargé."""


class UnknownICDOCodeError(OncoNERError):
    """Levée quand un code ICD-O ne peut pas être résolu."""