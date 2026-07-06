\# onco-ner



> Extraction et normalisation automatique d'entités oncologiques depuis des textes cliniques en français, avec normalisation vers les codes ICD-O.



\[!\[CI](https://github.com/Eudes9/onco-ner/actions/workflows/ci.yml/badge.svg)](https://github.com/Eudes9/onco-ner/actions/workflows/ci.yml)

\[!\[Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)



\---



\## Table des matières



\- \[Architecture](#architecture)

\- \[Installation](#installation)

\- \[Training](#training)

\- \[Inference](#inference)

\- \[API](#api)

\- \[Docker](#docker)

\- \[Benchmark](#benchmark)

\- \[Résultats](#résultats)

\- \[Dataset](#dataset)

\- \[Limitations](#limitations)

\- \[Citation](#citation)



\---



\## Architecture



onco-ner/

├── src/onco\_ner/          # Bibliothèque Python installable

│   ├── data/              # Parser brat + preprocessing Polars

│   ├── models/            # NERModel + ICDONormalizer

│   ├── pipeline.py        # Interface haut niveau

│   └── schemas.py         # Structures de données Pydantic

├── training/              # Scripts d'entraînement (Hydra + W\&B)

├── evaluation/            # Benchmark + analyse d'erreurs

├── api/                   # API REST FastAPI

├── streamlit/             # Interface utilisateur

└── tests/                 # 81 tests unitaires



Le pipeline complet :



Texte clinique (français)

↓

XLM-RoBERTa optimisé (NER, sliding window 512 tokens, stride 256)

↓

ICDONormalizer (fuzzy match via rapidfuzz)

↓

JSON structuré { entités + codes ICD-O }





\---



\## Installation



```bash

git clone https://github.com/Eudes9/onco-ner.git

cd onco-ner

pip install -e .

pip install transformers==4.49.0 torch rapidfuzz polars==1.9.0

```



\---



\## Training



L'entraînement utilise \[Hydra](https://hydra.cc/) pour la configuration et \[W\&B](https://wandb.ai/) pour le tracking.



\### Baseline (512 tokens, sans sliding window)



```bash

python training/train.py models=drbert training.batch\_size=8

python training/train.py models=xlm\_roberta\_base training.batch\_size=8

python training/train.py models=moderncamembert\_base training.batch\_size=8

```



\### Optimisé (sliding window + class weights)



```bash

python training/train\_optimized.py models=xlm\_roberta\_base \\

&#x20;   training.batch\_size=8 \\

&#x20;   paths.data\_dir=/path/to/data

```



\### Paramètres clés



| Paramètre | Valeur |

|---|---|

| `training.learning\_rate` | `2e-5` |

| `training.num\_epochs` | `5` |

| `training.batch\_size` | `8` (T4 GPU) |

| `training.max\_length` | `512` |

| Stride (sliding window) | `256` |



\---



\## Inference



\### En Python



```python

from onco\_ner import Pipeline



\# Sans normalisation ICD-O

pipeline = Pipeline.from\_pretrained(

&#x20;   "Eudes9/onco-ner-xlm-roberta-optimized"

)



\# Avec normalisation ICD-O

pipeline = Pipeline.from\_pretrained(

&#x20;   model\_path="Eudes9/onco-ner-xlm-roberta-optimized",

&#x20;   csv\_path="data/DetectOnco\_Final.csv",

)



result = pipeline.predict(

&#x20;   "Patient présentant un carcinome canalaire infiltrant du sein gauche"

)

print(result)

\# {

\#   "text": "Patient présentant un carcinome canalaire infiltrant du sein gauche",

\#   "entities": \[

\#     {

\#       "text": "carcinome canalaire infiltrant",

\#       "label": "morphologie",

\#       "start": 22,

\#       "end": 52,

\#       "score": 0.9953,

\#       "icdo\_code": "8500/3"

\#     },

\#     {

\#       "text": "sein gauche",

\#       "label": "topographie",

\#       "start": 56,

\#       "end": 67,

\#       "score": 0.9432,

\#       "icdo\_code": "C50.9"

\#     }

\#   ],

\#   "n\_entities": 2

\# }

```



\### Paramètres disponibles



```python

result = pipeline.predict(

&#x20;   text="...",

&#x20;   fuzzy=True,              # matching approximatif ICD-O

&#x20;   fuzzy\_threshold=0.8,     # seuil de similarité (0-1)

)



\# Prédiction batch

results = pipeline.predict\_batch(\["texte 1", "texte 2", ...])



\# Prédiction depuis un fichier

result = pipeline.predict\_file("chemin/vers/fichier.txt")

```



\---



\## API



Lance l'API REST :



```bash

\# Sans normalisation ICD-O

uvicorn api.main:app --host 0.0.0.0 --port 8000



\# Avec normalisation ICD-O

CSV\_PATH=data/DetectOnco\_Final.csv uvicorn api.main:app --host 0.0.0.0 --port 8000

```



\### Endpoints



| Méthode | Endpoint | Description |

|---|---|---|

| `GET` | `/` | Health check |

| `GET` | `/info` | Informations sur le modèle |

| `POST` | `/predict` | Prédiction sur un texte |

| `POST` | `/predict/batch` | Prédiction sur une liste de textes |



\### Exemple



```bash

curl -X POST http://localhost:8000/predict \\

&#x20; -H "Content-Type: application/json" \\

&#x20; -d '{"text": "carcinome canalaire infiltrant du sein gauche"}'

```



\### Interface Streamlit



```bash

streamlit run streamlit/app.py

```



!\[Interface Streamlit](docs/streamlit\_screenshot.png)



\---



\## Docker



```bash

\# Build et lancement

docker compose up



\# Avec normalisation ICD-O

CSV\_PATH=/app/data/DetectOnco\_Final.csv docker compose up

```



L'API est accessible sur `http://localhost:8000`.

La documentation Swagger est disponible sur `http://localhost:8000/docs`.



\---



\## Benchmark



Trois modèles évalués en deux configurations : \*\*baseline\*\* (troncature à 512 tokens) et \*\*optimisé\*\* (sliding window + class weights).



\### Résultats sur le Val Set (seqeval)



| Modèle | Precision | Recall | F1 |

|---|---|---|---|

| ModernCamemBERT-base (baseline) | 0.482 | 0.616 | 0.541 |

| XLM-RoBERTa-base (baseline) | 0.580 | 0.756 | 0.656 |

| DrBERT (baseline) | 0.651 | 0.773 | 0.707 |

| ModernCamemBERT-base (optimisé) | 0.672 | 0.773 | 0.719 |

| DrBERT (optimisé) | 0.731 | 0.813 | 0.770 |

| \*\*XLM-RoBERTa-base (optimisé)\*\* | \*\*0.731\*\* | \*\*0.841\*\* | \*\*0.782\*\* |



\### Résultats sur le Test Set (seqeval) — Modèles optimisés uniquement



| Modèle | Precision | Recall | F1 |

|---|---|---|---|

| ModernCamemBERT-base (optimisé) | 0.645 | 0.754 | 0.696 |

| DrBERT (optimisé) | 0.700 | 0.801 | 0.747 |

| \*\*XLM-RoBERTa-base (optimisé)\*\* | \*\*0.713\*\* | \*\*0.816\*\* | \*\*0.761\*\* |



\### Analyse d'erreurs — XLM-RoBERTa optimisé (évaluation stricte par spans)



| Label | Precision | Recall | F1 | Boundary errors |

|---|---|---|---|---|

| differenciation | 0.581 | 0.439 | 0.500 | 8 |

| morphologie | 0.775 | 0.324 | 0.457 | 65 |

| topographie | 0.760 | 0.274 | 0.403 | 78 |

| expression\_CIM | 0.039 | 0.012 | 0.019 | 334 |



> \*\*Note\*\* : L'évaluation stricte par spans exige une correspondance exacte des positions de début/fin — elle est plus sévère que seqeval qui opère au niveau token.



\---



\## Résultats



\### Points clés



\*\*1. Impact du sliding window (+12 à +18 points de F1)\*\*

98.5% des documents FRACCO dépassent 512 tokens. Sans sliding window, la majorité des entités dans la seconde moitié des documents sont invisibles pour le modèle. L'optimisation apporte un gain de +12 à +18 points de F1 selon le modèle.



\*\*2. XLM-RoBERTa surpasse DrBERT malgré la spécialisation médicale\*\*

Contre-intuitivement, XLM-RoBERTa-base (modèle multilingue généraliste) surpasse DrBERT (modèle médical français spécialisé) une fois le problème de troncature résolu. Cela suggère que la capacité de généralisation du pré-entraînement multilingue à large échelle compense l'avantage du domaine médical sur ce corpus.



\*\*3. Amélioration de `differenciation` grâce aux class weights\*\*

La classe la plus rare (1053 exemples, ratio 1:24 avec `morphologie`) obtient F1=0.000 en baseline et F1=0.500 après optimisation avec class weights (racine carrée).



\*\*4. Les boundary errors dominent les erreurs résiduelles\*\*

Sur les 5005 erreurs totales, 485 (9.7%) sont des boundary errors — le modèle détecte la bonne entité mais avec des frontières légèrement décalées. C'est un problème d'alignement tokenizer plutôt que de détection.



\---



\## Dataset



Ce projet utilise le corpus \*\*FRACCO\*\* (French Annotated Corpus for Cancer Oncology).



\- \*\*Source\*\* : \[Zenodo DOI: 10.5281/zenodo.17284817](https://zenodo.org/record/17284817)

\- \*\*Taille\*\* : 1 301 cas cliniques synthétiques en français

\- \*\*Annotations\*\* : 71 126 entités annotées selon ICD-O-3.1

\- \*\*Labels\*\* : `morphologie`, `topographie`, `differenciation`, `expression\_CIM`

\- \*\*Format\*\* : brat (`.txt` + `.ann`) + CSV consolidé

\- \*\*Split\*\* : 80% train / 10% val / 10% test (au niveau document)



> Les textes FRACCO sont entièrement \*\*synthétiques\*\* — aucune donnée patient réelle.



\---



\## Limitations



1\. \*\*Boundary errors sur `expression\_CIM`\*\* : 334 erreurs de frontières sur cette classe composite — les expressions longues ont des délimitations floues qui complexifient l'alignement BIO.



2\. \*\*Spans discontinus non gérés en entraînement\*\* : 43.9% des annotations FRACCO contiennent des spans discontinus. Pour l'entraînement NER, seul le premier sous-span est annoté. Une approche avec schéma de tagging spécialisé (BIOUL, spans) améliorerait ce point.



3\. \*\*Normalisation ICD-O limitée pour `expression\_CIM`\*\* : les expressions composites longues dépassent souvent le seuil de similarité fuzzy — le taux de normalisation sur cette classe est inférieur aux autres.



4\. \*\*Fenêtre contextuelle\*\* : malgré le sliding window à 512 tokens, les entités dans les zones de chevauchement reçoivent `-100` dans la loss — une approche avec fenêtre plus large (ModernCamemBERT supporte 8192 tokens) pourrait améliorer le recall.



5\. \*\*Perspectives\*\* : fine-tuning des hyperparamètres (learning rate, stride), augmentation de données sur `differenciation`, évaluation sur des données AP-HP réelles.



\---



\## Citation



Si vous utilisez ce projet, merci de citer :



```bibtex

@misc{onco-ner-2026,

&#x20; author       = {Eudes Gbada},

&#x20; title        = {onco-ner: Extraction et normalisation d'entités oncologiques},

&#x20; year         = {2026},

&#x20; publisher    = {GitHub},

&#x20; url          = {https://github.com/Eudes9/onco-ner}

}

```



Corpus FRACCO :



```bibtex

@dataset{fracco-2025,

&#x20; title        = {FRACCO: French Annotated Corpus for Cancer Oncology},

&#x20; year         = {2025},

&#x20; publisher    = {Zenodo},

&#x20; doi          = {10.5281/zenodo.17284817},

&#x20; url          = {https://zenodo.org/record/17284817}

}

```



\---



\## Tests



```bash

\# Tous les tests (hors Spark)

python -m pytest tests/ -v -m "not spark"



\# Avec coverage

python -m pytest tests/ --cov=src/onco\_ner --cov-report=term-missing -m "not spark"



\# Tests Spark (nécessite Java)

python -m pytest tests/test\_spark\_preprocessing.py -v -m spark

```



\---



\*Développé dans le cadre d'un projet de NLP médical — 2026\*

