# onco-ner : Extraction et Normalisation d'Entités Oncologiques en Français

[![CI](https://github.com/Eudes9/onco-ner/actions/workflows/ci.yml/badge.svg)](https://github.com/Eudes9/onco-ner/actions)
[![Codecov](https://codecov.io/gh/Eudes9/onco-ner/graph/badge.svg)](https://codecov.io/gh/Eudes9/onco-ner)
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0+-009688.svg)](https://fastapi.tiangolo.com)

`onco-ner` est un pipeline complet de Traitement du Langage Naturel (NLP) dédié à l'extraction d'entités cliniques oncologiques et à leur normalisation automatique vers les codes de la **Classification Internationale des Maladies pour l'Oncologie (CIM-O / ICD-O)**.

Basé sur un modèle **XLM-RoBERTa** finement entraîné sur le corpus médical français **FRACCO**, le projet intègre une gestion par fenêtre glissante (*Sliding Window*) pour traiter les rapports médicaux longs et un algorithme de *Fuzzy Matching* ultra-rapide pour la codification automatique.

---

## 🚀 Fonctionnalités Clés

* **NER Spécialisé (Token Classification) :** Extraction des classes `Morphologie` (types de tumeurs), `Topographie` (organes/localisations) et `Différenciation` (grade histologique).
* **Fenêtre Glissante Appliquée (Sliding Window) :** Découpage et reconstruction intelligente des textes cliniques longs dépassant la limite native de 512 tokens des Transformers.
* **Normalisation ICD-O :** Alignement sémantique approximatif (via `rapidfuzz`) pour mapper les entités extraites vers la nomenclature officielle.
* **Architecture Prête pour la Production :** API REST asynchrone propulsée par **FastAPI** et conteneurisée avec **Docker** (optimisée pour inférence CPU).
* **Industrialisation (CI/CD) :** Pipeline de tests unitaires automatiques sur GitHub Actions avec couverture de code complète.

---

## 📦 Architecture du Projet

```text
onco_ner/
├── .github/workflows/   # Pipeline d'Intégration Continue (GitHub Actions)
├── api/
│   └── main.py          # Serveur FastAPI (Points d'entrée REST & Lifespan)
├── data/
│   └── DetectOnco_Final.csv  # Référentiel de codes ICD-O pour le Normalizer
├── src/onco_ner/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ner_model.py   # Logique d'inférence PyTorch & Sliding Window
│   │   └── normalizer.py  # Moteur de Fuzzy Matching ICD-O
│   ├── utils/
│   │   └── logging.py     # Gestionnaire de logs centralisé
│   ├── exceptions.py      # Exceptions métiers customisées
│   └── pipeline.py        # Interface unifiée (High-Level API)
├── tests/                 # Suite de tests unitaires (33 tests au vert)
├── Dockerfile             # Image Docker optimisée (< 1 Go)
├── docker-compose.yml     # Orchestration des services et variables d'env
└── pyproject.toml         # Configuration du package Python et dépendances
