# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Étape 1 : copier UNIQUEMENT les fichiers de config
# -> le cache Docker est préservé tant que pyproject.toml ne change pas
COPY pyproject.toml .

# Étape 2 : installer les dépendances lourdes
# -> cette couche est mise en cache et ne se réinstalle pas
#    à chaque modification de code source
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    "transformers==4.49.0" \
    torch \
    rapidfuzz \
    "polars==1.9.0" \
    pydantic

# Étape 3 : copier le code source APRÈS les dépendances
# -> une modification de src/ ou api/ n'invalide que cette couche
COPY src/ src/
COPY api/ api/

# Installer le package onco_ner en mode normal (pas editable en prod)
RUN pip install --no-cache-dir .

# Variables d'environnement
ENV MODEL_PATH=Eudes9/onco-ner-xlm-roberta-optimized
ENV CSV_PATH=
ENV MAX_LENGTH=512
ENV STRIDE=256
ENV DEVICE=cpu

# Limiter le parallélisme CPU au niveau système
# Complète torch.set_num_threads(1) défini dans le lifespan
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]