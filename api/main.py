# api/main.py
"""
API REST onco-ner — Extraction d'entités oncologiques.

Endpoints :
    GET  /              : health check
    GET  /info          : informations sur le modèle chargé
    POST /predict       : prédiction sur un texte clinique
    POST /predict/batch : prédiction sur une liste de textes

Usage :
    uvicorn api.main:app --host 0.0.0.0 --port 8000
    docker compose up
"""

import os
import torch
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from onco_ner import Pipeline
from onco_ner.utils.logging import get_logger

logger = get_logger(__name__)

# --- Schémas de requête / réponse ---

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Texte clinique en français",
        examples=["Patient présentant un carcinome canalaire infiltrant du sein gauche"],
    )
    fuzzy: bool = Field(
        default=True,
        description="Activer le matching approximatif pour la normalisation ICD-O",
    )
    fuzzy_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Seuil de similarité pour le fuzzy match (0-1)",
    )


class PredictBatchRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        description="Liste de textes cliniques en français",
    )
    fuzzy: bool = Field(default=True)
    fuzzy_threshold: float = Field(default=0.8, ge=0.0, le=1.0)


class EntityResponse(BaseModel):
    text: str
    label: str
    start: int
    end: int
    score: float
    icdo_code: str | None = None


class PredictResponse(BaseModel):
    text: str
    entities: list[EntityResponse]
    n_entities: int


class HealthResponse(BaseModel):
    status: str
    model: str
    normalizer: bool


class InfoResponse(BaseModel):
    model_path: str
    max_length: int
    stride: int
    normalizer_loaded: bool
    device: str


# --- État global de l'application ---

app_state: dict = {}


# --- Lifespan : chargement du modèle au démarrage ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le Pipeline au démarrage, libère les ressources à l'arrêt."""
    model_path = os.getenv(
        "MODEL_PATH",
        "Eudes9/onco-ner-xlm-roberta-optimized"
    )
    csv_path = os.getenv("CSV_PATH", None) or None
    max_length = int(os.getenv("MAX_LENGTH", "512"))
    stride = int(os.getenv("STRIDE", "256"))
    device = os.getenv("DEVICE", None) or None

    # Limiter le parallélisme intra-op PyTorch en production
    # Évite la contention CPU quand plusieurs requêtes arrivent en parallèle
    torch.set_num_threads(1)
    logger.info("PyTorch num_threads fixé à 1 (mode production)")

    logger.info(f"Chargement du pipeline : {model_path}")

    try:
        pipeline = Pipeline.from_pretrained(
            model_path=model_path,
            csv_path=csv_path,
            max_length=max_length,
            stride=stride,
            device=device,
        )
        app_state["pipeline"] = pipeline
        app_state["model_path"] = model_path
        logger.info("Pipeline chargé avec succès")
    except Exception as e:
        logger.error(f"Erreur chargement pipeline : {e}")
        raise

    yield

    # Nettoyage à l'arrêt
    app_state.clear()
    logger.info("Pipeline libéré")


# --- Application FastAPI ---

app = FastAPI(
    title="onco-ner API",
    description=(
        "API d'extraction et normalisation d'entités oncologiques "
        "depuis des textes cliniques en français. "
        "Modèle : XLM-RoBERTa optimisé sur le corpus FRACCO."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# --- Endpoints ---

@app.get("/", response_model=HealthResponse)
def health_check():
    """Health check — vérifie que le pipeline est chargé et opérationnel."""
    if "pipeline" not in app_state:
        raise HTTPException(status_code=503, detail="Pipeline non chargé")
    pipeline = app_state["pipeline"]
    return HealthResponse(
        status="ok",
        model=app_state.get("model_path", "unknown"),
        normalizer=pipeline.normalizer is not None,
    )


@app.get("/info", response_model=InfoResponse)
def model_info():
    """Informations détaillées sur le modèle chargé."""
    if "pipeline" not in app_state:
        raise HTTPException(status_code=503, detail="Pipeline non chargé")
    pipeline = app_state["pipeline"]
    ner = pipeline.ner_model
    return InfoResponse(
        model_path=app_state.get("model_path", "unknown"),
        max_length=ner.max_length,
        stride=ner.stride,
        normalizer_loaded=pipeline.normalizer is not None,
        device=str(ner.device),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Extrait les entités oncologiques d'un texte clinique.

    Retourne les entités avec leur type, position, score de confiance
    et code ICD-O (si le normalizer est chargé).
    """
    if "pipeline" not in app_state:
        raise HTTPException(status_code=503, detail="Pipeline non chargé")

    if not request.text.strip():
        raise HTTPException(
            status_code=422,
            detail="Le texte ne peut pas être vide"
        )

    try:
        pipeline = app_state["pipeline"]
        result = pipeline.predict(
            text=request.text,
            fuzzy=request.fuzzy,
            fuzzy_threshold=request.fuzzy_threshold,
        )
        return PredictResponse(
            text=result["text"],
            entities=[EntityResponse(**e) for e in result["entities"]],
            n_entities=result["n_entities"],
        )
    except Exception as e:
        logger.error(f"Erreur prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(request: PredictBatchRequest):
    """
    Extrait les entités oncologiques d'une liste de textes cliniques.

    Traite chaque texte séquentiellement et retourne une liste de résultats.
    """
    if "pipeline" not in app_state:
        raise HTTPException(status_code=503, detail="Pipeline non chargé")

    if not request.texts:
        raise HTTPException(
            status_code=422,
            detail="La liste de textes ne peut pas être vide"
        )

    try:
        pipeline = app_state["pipeline"]
        results = pipeline.predict_batch(
            texts=request.texts,
            fuzzy=request.fuzzy,
            fuzzy_threshold=request.fuzzy_threshold,
        )
        return [
            PredictResponse(
                text=r["text"],
                entities=[EntityResponse(**e) for e in r["entities"]],
                n_entities=r["n_entities"],
            )
            for r in results
        ]
    except Exception as e:
        logger.error(f"Erreur prédiction batch : {e}")
        raise HTTPException(status_code=500, detail=str(e))