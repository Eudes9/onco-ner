# Extraction et Normalisation Automatique d'Entités Oncologiques depuis des Textes Cliniques en Français

**Auteur** : Eudes Gbada  
**Date** : Juillet 2026  
**Dépôt** : [github.com/Eudes9/onco-ner](https://github.com/Eudes9/onco-ner)

---

## Résumé

Ce rapport présente **onco-ner**, un système de reconnaissance d'entités nommées (NER) et de normalisation automatique d'entités oncologiques depuis des textes cliniques en français. Le système extrait quatre types d'entités — morphologie tumorale, topographie, différenciation histologique et expression CIM — et les normalise vers les codes ICD-O-3.1. 

Trois modèles de la famille BERT sont évalués (ModernCamemBERT-base, DrBERT, XLM-RoBERTa-base) dans une configuration baseline et une configuration optimisée (sliding window + class weights). Le meilleur modèle, XLM-RoBERTa-base optimisé, atteint un F1-score de **0.761** sur le set de test (évaluation seqeval). L'analyse d'erreurs approfondie révèle que 98.5% des documents dépassent la fenêtre contextuelle standard de 512 tokens, justifiant l'introduction d'une stratégie de sliding window qui apporte un gain de +12 à +18 points de F1 selon le modèle.

---

## 1. Introduction

Les établissements hospitaliers universitaires comme l'Assistance Publique-Hôpitaux de Paris (AP-HP) accumulent des volumes considérables de données médicales non structurées. L'Entrepôt de Données de Santé (EDS) de l'AP-HP contient plus de 20 millions de dossiers patients, 190 millions de comptes-rendus médicaux et 140 millions de diagnostics. La majorité de ces informations cliniques sont rédigées en texte libre, rendant leur exploitation directe pour la recherche clinique difficile.

En oncologie particulièrement, les informations critiques — type histologique de la tumeur, localisation anatomique, degré de différenciation cellulaire — sont systématiquement présentes dans les comptes-rendus d'anatomopathologie mais rarement structurées en base de données. Un chercheur souhaitant identifier tous les patients atteints d'un carcinome canalaire infiltrant du sein de stade T2N1M0 ne peut pas formuler une requête SQL simple sur ces données textuelles.

Ce travail propose une approche de NER basée sur les modèles de langage pré-entraînés pour transformer automatiquement ces textes médicaux en données structurées exploitables, normalisées selon la Classification Internationale des Maladies pour l'Oncologie (ICD-O-3.1).

**Contributions principales** :
1. Évaluation comparative de trois modèles BERT sur le corpus oncologique français FRACCO
2. Identification et correction d'un problème de troncature affectant 98.5% des documents
3. Amélioration significative de la classe rare `differenciation` (F1 : 0.000 → 0.500) via les class weights
4. Pipeline complet et bibliothèque Python open source avec API REST

---

## 2. État de l'Art

### 2.1 Modèles de Langage pour le Français Médical

Le traitement automatique du langage naturel médical en français a connu des avancées significatives avec l'introduction des modèles de langage pré-entraînés.

* **CamemBERT** [Martin et al., 2020] est le premier modèle de type RoBERTa entraîné spécifiquement sur un large corpus français (138 Go de texte extrait du Common Crawl). Il établit de nouveaux records sur de nombreuses tâches NLP françaises et constitue la référence pour le traitement du français. Le modèle récent **ModernCamemBERT** [Almanach, 2025] étend cette approche avec un pré-entraînement sur 1 trillion de tokens de texte français de haute qualité et une fenêtre contextuelle étendue à 8192 tokens, basé sur l'architecture ModernBERT.
* **DrBERT** [Labrak et al., 2023] est un modèle BERT spécialisé pour le domaine médical français, pré-entraîné sur 7 Go de textes biomédicaux français incluant des articles scientifiques, des résumés PubMed traduits et des données cliniques. Il surpasse CamemBERT sur plusieurs tâches d'extraction d'information médicale en français.
* **XLM-RoBERTa** [Conneau et al., 2020] est un modèle multilingue pré-entraîné sur 2.5 To de données dans 100 langues via la Common Crawl. Malgré son caractère généraliste, il démontre des performances compétitives sur les tâches NLP spécialisées grâce à la richesse de son pré-entraînement.

### 2.2 NER Médical en Français

* **EDS-NLP** [Dura et al., 2023] est la bibliothèque open source développée par l'équipe data science de l'AP-HP pour le traitement de textes médicaux en français. Elle propose des composants spaCy pour l'extraction d'entités médicales, la détection de négation et la normalisation terminologique sur les données de l'EDS.
* **DEFT** (Défi Fouille de Textes) est une série de campagnes d'évaluation en NLP médical français. Les éditions 2019-2021 portent sur l'extraction d'informations depuis des cas cliniques en français et constituent la référence pour évaluer les systèmes d'extraction d'entités médicales.

### 2.3 Corpus Oncologiques Annotés

* **CANTEMIST** [Miranda-Escalada et al., 2020] est un corpus espagnol de 1301 cas cliniques oncologiques annotés selon ICD-O pour la morphologie tumorale. Il constitue la source originale du corpus FRACCO.
* **FRACCO** [2025] (French Annotated Corpus for Cancer Oncology) est la version française de CANTEMIST, obtenue par projection d'annotations cross-linguale via BERT. Il contient 1301 cas cliniques synthétiques en français avec 71 126 annotations couvrant morphologie, topographie, différenciation et expression CIM selon ICD-O-3.1. C'est le premier corpus annoté ouvert spécifiquement dédié à l'oncologie en français.

### 2.4 Gestion des Séquences Longues en NLP

La limite de 512 tokens des modèles BERT constitue un défi majeur pour les textes médicaux longs. Plusieurs approches ont été proposées : la troncature simple [Devlin et al., 2018], le sliding window avec agrégation des prédictions [Jiang et al., 2019], et les architectures à attention longue portée comme Longformer [Beltagy et al., 2020]. Pour la tâche NER, le sliding window avec stride constitue l'approche la plus simple et efficace, permettant de couvrir l'intégralité du document sans modification de l'architecture.

---

## 3. Données

### 3.1 Corpus FRACCO

Le corpus FRACCO (DOI: 10.5281/zenodo.17284817) est composé de 1301 cas cliniques oncologiques synthétiques en français. Chaque document décrit un patient fictif atteint d'un cancer, avec des informations sur le diagnostic, le traitement et le suivi.

**Statistiques du corpus** :

| Statistique | Valeur |
|---|---|
| Nombre de documents | 1 301 |
| Longueur moyenne (caractères) | 5 287 |
| Longueur minimale | 1 300 |
| Longueur maximale | 16 321 |
| Nombre total d'annotations | 71 126 |
| Documents avec spans discontinus | 572 (43.9%) |

**Distribution des labels** :

| Label | Nombre | Proportion |
|---|---|---|
| morphologie | 25 634 | 36.0% |
| expression_CIM | 25 634 | 36.0% |
| topographie | 18 805 | 26.4% |
| differenciation | 1 053 | 1.5% |

Le déséquilibre entre `morphologie` (25 634 exemples) et `differenciation` (1 053 exemples) représente un ratio de 1:24, constituant un défi majeur pour l'apprentissage.

### 3.2 Découverte des Spans Discontinus

Une analyse empirique du corpus révèle que 572 documents sur 1301 (43.9%) contiennent des annotations avec spans discontinus — c'est-à-dire des entités dont le texte n'est pas continu dans le document. Les tags BIO associés s'articulent ainsi : `O`, `B-morphologie`, `I-morphologie`, `B-topographie`, `I-topographie`, `B-differenciation`, `I-differenciation`, `B-expression_CIM`, `I-expression_CIM`.

---

## 4. Méthodologie

### 4.1 Modèles Évalués

Trois modèles de la famille BERT sont évalués :

| Modèle | Identifiant HuggingFace | Paramètres | Spécialisation |
|---|---|---|---|
| ModernCamemBERT-base | `almanach/moderncamembert-base` | ~140M | Français général |
| DrBERT | `Dr-BERT/DrBERT-7GB` | ~110M | Médical français |
| XLM-RoBERTa-base | `xlm-roberta-base` | ~270M | Multilingue |

### 4.2 Configuration Baseline

La configuration baseline consiste en un fine-tuning standard avec troncature des séquences à 512 tokens :

| Hyperparamètre | Valeur |
|---|---|
| Learning rate | 2e-5 |
| Epochs | 5 |
| Batch size | 8 |
| Max length | 512 tokens |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Optimizer | AdamW |

### 4.3 Configuration Optimisée

La configuration optimisée introduit deux améliorations majeures identifiées par l'analyse d'erreurs :

**Sliding Window** : Au lieu de tronquer les documents à 512 tokens, chaque document est découpé en chunks de 512 tokens avec un chevauchement de 256 tokens (stride = 50%). Pour un document de N tokens, le nombre de chunks est approximativement $N/256$. Les tokens dans la zone de chevauchement reçoivent le label `-100` (ignorés par la loss) pour éviter le double comptage.

```text
Document : [token_0 ... token_N]

Chunk 1  : [token_0   ... token_511]
Chunk 2  : [token_256 ... token_767]  (256 tokens de chevauchement)
Chunk 3  : [token_512 ... token_1023]
```
**Class Weights** : Pour compenser le déséquilibre de classes (ratio 1:24 entre `differenciation` et `morphologie`), une `CrossEntropyLoss` pondérée est utilisée avec des poids calculés par la formule :

$$weight(label) = \sqrt{\frac{max\_count}{count(label)}}$$

La racine carrée atténue le déséquilibre sans pénaliser agressivement la classe majoritaire O, évitant les hallucinations d'entités dans des contextes sains.

| Label | Count | Weight |
|---|---|---|
| O | — | 1.000 |
| morphologie / expression_CIM | 25 634 | 1.000 |
| topographie | 18 805 | 1.168 |
| differenciation | 1 053 | 4.937 |

### 4.4 ICDONormalizer

Le composant de normalisation ICD-O associe chaque entité détectée à son code ICD-O officiel via une recherche dans le CSV FRACCO. Deux stratégies sont implémentées :
* **Exact match** : recherche exacte du texte normalisé (minuscules, espaces supprimés) dans l'index de lookup.
* **Fuzzy match** : si pas de match exact, recherche par similarité de Levenshtein normalisée via `rapidfuzz` avec un seuil configurable (défaut : 0.8).

### 4.5 Évaluation

Deux niveaux d'évaluation sont utilisés :
* **Seqeval (niveau token)** : métrique standard HuggingFace qui évalue les séquences BIO au niveau token. C'est la métrique utilisée pendant l'entraînement.
* **Évaluation stricte par spans** : évaluation au niveau entité avec correspondance exacte des positions de début et fin. Cette métrique distingue trois types d'erreurs :
  * **TP** : span exact + label correct
  * **FN pur** : entité gold complètement manquée
  * **FP pur** : entité prédite sans correspondance gold
  * **Boundary error** : bon label mais frontières décalées

Un matching greedy par overlap décroissant évite le double comptage des boundary errors.

---

## 5. Résultats

### 5.1 Résultats Baseline (Val Set, seqeval)

| Modèle | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| ModernCamemBERT-base | 0.482 | 0.616 | 0.541 | 0.955 |
| XLM-RoBERTa-base | 0.580 | 0.756 | 0.656 | 0.962 |
| DrBERT | 0.651 | 0.773 | 0.707 | 0.974 |

DrBERT, le modèle spécialisé médical, surpasse les deux autres en configuration baseline. ModernCamemBERT obtient les performances les plus faibles malgré son pré-entraînement récent sur un large corpus français.

### 5.2 Impact du Sliding Window et des Class Weights (Val Set, seqeval)

| Modèle | F1 Baseline | F1 Optimisé | Gain |
|---|---|---|---|
| ModernCamemBERT-base | 0.541 | 0.719 | **+17.8 pts** |
| XLM-RoBERTa-base | 0.656 | 0.782 | **+12.6 pts** |
| DrBERT | 0.707 | 0.770 | **+6.3 pts** |

Le gain est inversement proportionnel à la performance baseline : ModernCamemBERT, qui bénéficiait le moins de la troncature, est le plus amélioré. DrBERT, dont le vocabulaire médical lui permettait de mieux exploiter les 512 premiers tokens, progresse moins.

### 5.3 Résultats sur le Test Set (modèles optimisés uniquement)

| Modèle | Precision | Recall | F1 | Chunks |
|---|---|---|---|---|
| ModernCamemBERT-base | 0.645 | 0.754 | 0.696 | 620 |
| DrBERT | 0.700 | 0.801 | 0.747 | 539 |
| **XLM-RoBERTa-base** | **0.713** | **0.816** | **0.761** | 721 |

**XLM-RoBERTa-base optimisé est le meilleur modèle** sur le test set avec F1 = 0.761.

Le nombre de chunks différent selon le modèle (539 pour DrBERT vs 721 pour XLM-RoBERTa) s'explique par la tokenization : XLM-RoBERTa utilise un tokenizer SentencePiece universel qui fragmente plus agressivement le vocabulaire médical spécialisé (absent de son pré-entraînement multilingue), générant davantage de subtokens et donc davantage de chunks par document.

### 5.4 Analyse d'Erreurs par Label (XLM-RoBERTa optimisé, évaluation stricte)

| Label | Precision | Recall | F1 | TP | FP pur | FN pur | Boundary |
|---|---|---|---|---|---|---|---|
| differenciation | 0.581 | 0.439 | 0.500 | 36 | 18 | 38 | 8 |
| morphologie | 0.775 | 0.324 | 0.457 | 749 | 153 | 1 496 | 65 |
| topographie | 0.760 | 0.274 | 0.403 | 516 | 85 | 1 288 | 78 |
| expression_CIM | 0.039 | 0.012 | 0.019 | 20 | 162 | 1 280 | 334 |

**Observations clés** :
* **`differenciation`** : classe la plus rare (F1=0.000 en baseline), atteint F1=0.500 grâce aux class weights — l'amélioration la plus significative.
* **`expression_CIM`** : classe la plus difficile avec 334 boundary errors. Les expressions composites longues (ex: *"adénocarcinome pulmonaire non à petites cellules"*) ont des frontières floues qui complexifient l'alignement BIO.
* **FN purs massifs** : 4102 entités complètement manquées sur le test set en évaluation stricte, contre un F1=0.761 en seqeval. Cet écart s'explique par la sévérité de l'évaluation stricte qui exige une correspondance exacte des positions de caractères.

### 5.5 Comparaison Seqeval vs Évaluation Stricte

| Métrique | F1 (seqeval) | F1 (strict) | Écart |
|---|---|---|---|
| XLM-RoBERTa optimisé (global) | 0.761 | ~0.35* | ~0.41 |

*\* F1 strict calculé sur l'ensemble des labels — la faiblesse de expression_CIM (F1=0.019) tire fortement ce score vers le bas.*

Cet écart important illustre la différence entre l'évaluation au niveau token (seqeval, plus permissive) et l'évaluation au niveau entité (stricte, plus réaliste pour les applications cliniques). Pour un usage production, l'évaluation stricte est plus représentative de la qualité réelle du système.

### 5.6 Analyse par Longueur de Document

| Catégorie | Taux d'erreur moyen |
|---|---|
| Long (>8000 car.) | 74.6% |
| Moyen (3000-8000 car.) | 63.2% |
| Court (<3000 car.) | 53.6% |

Les documents longs restent les plus difficiles malgré le sliding window — les entités dans les zones de chevauchement reçoivent `-100` dans la loss et ne contribuent pas à l'apprentissage.

---

## 6. Discussion

### 6.1 Pourquoi XLM-RoBERTa Surpasse DrBERT

Le résultat contre-intuitif où XLM-RoBERTa-base (modèle multilingue généraliste) surpasse DrBERT (modèle médical français spécialisé) mérite une analyse.

En configuration baseline, DrBERT domine grâce à son vocabulaire médical spécialisé qui représente efficacement les termes oncologiques dans les 512 premiers tokens. Cependant, une fois le problème de troncature résolu par le sliding window, XLM-RoBERTa prend l'avantage grâce à sa capacité de généralisation supérieure issue d'un pré-entraînement sur 2.5 To de données multilingues.

Cette observation suggère que sur le corpus FRACCO — composé de textes synthétiques plutôt que de vrais comptes-rendus hospitaliers — la capacité de généralisation prime sur la spécialisation domaine. Sur des données AP-HP réelles, DrBERT pourrait retrouver son avantage.

### 6.2 Impact Asymétrique du Sliding Window

Le gain du sliding window est inversement corrélé à la performance baseline (+17.8 pts pour ModernCamemBERT vs +6.3 pts pour DrBERT). Cette asymétrie indique que DrBERT exploitait mieux les informations présentes dans les 512 premiers tokens, tandis que ModernCamemBERT en tirait peu de valeur. Avec le sliding window, tous les modèles accèdent aux mêmes informations et le biais lié à la longueur disparaît.

### 6.3 Efficacité des Class Weights

L'amélioration de `differenciation` de F1=0.000 à F1=0.500 démontre l'efficacité des class weights pour les classes très déséquilibrées. Cependant, la formule choisie (racine carrée) est conservatrice — une Focal Loss [Lin et al., 2017] ou des poids inversement proportionnels pourraient apporter des gains supplémentaires.

### 6.4 Le Problème des Boundary Errors

485 boundary errors sur XLM-RoBERTa optimisé (contre 1630 sur DrBERT baseline) révèlent un problème d'alignement des offsets de caractères lié à la tokenization SentencePiece. Les tokenizers basés sur SentencePiece absorbent souvent l'espace précédant un mot dans le premier subtoken, décalant les offsets de un caractère et causant des erreurs de frontière sur le premier token des entités.

---

## 7. Limitations

* **Corpus synthétique** : FRACCO est composé de textes synthétiques dérivés du corpus espagnol CANTEMIST par projection cross-linguale. Les performances sur de vrais comptes-rendus AP-HP pourraient être significativement différentes — le vocabulaire, le style rédactionnel et les abréviations hospitalières diffèrent des textes synthétiques.
* **Spans discontinus partiellement gérés** : 43.9% des annotations contiennent des spans discontinus, mais seul le premier sous-span est annoté pour l'entraînement NER. Une approche basée sur les spans [Lee et al., 2017] ou un schéma de tagging BIOUL permettrait de capturer ces entités discontinues.
* **Normalisation ICD-O limitée sur `expression_CIM`** : Les expressions composites longues dépassent souvent le seuil de similarité fuzzy (0.8), laissant un nombre significatif d'entités sans code ICD-O. Une approche d'entity linking basée sur des embeddings sémantiques améliorerait ce point.
* **Absence de test de significativité statistique** : La différence de F1 entre DrBERT optimisé (0.747) et XLM-RoBERTa optimisé (0.761) sur 130 documents de test n'a pas été testée statistiquement. Un test de McNemar ou un bootstrap test confirmerait si cet écart est significatif.
* **Hyperparamètres non optimisés** : Les mêmes hyperparamètres ont été utilisés pour les trois modèles. Un grid search sur le learning rate et le stride pourrait améliorer les performances, particulièrement pour DrBERT.
* **Fenêtre contextuelle** : Malgré le sliding window à 512 tokens, les documents très longs (>8000 caractères) conservent un taux d'erreur élevé (74.6%). ModernCamemBERT avec sa fenêtre de 8192 tokens pourrait mieux gérer ces cas, mais nécessite davantage de ressources GPU.

---

## 8. Perspectives

**Court terme** :
* **Focal Loss** : remplacer les class weights par une Focal Loss [Lin et al., 2017] pour mieux gérer le déséquilibre extrême de `differenciation`.
* **ONNX Runtime** : convertir XLM-RoBERTa en ONNX pour diviser la latence d'inférence par 2-3 sur CPU, essentiel pour un déploiement production à l'AP-HP.
* **Évaluation sur données réelles** : valider les performances sur un échantillon de comptes-rendus AP-HP pour mesurer le transfert du corpus synthétique vers les données hospitalières.

**Moyen terme** :
* **Span-based NER** : adopter une architecture basée sur les spans pour gérer nativement les entités discontinues (43.9% du corpus).
* **Entity linking sémantique** : remplacer le fuzzy match par une approche d'embedding sémantique (bi-encoder fine-tuné sur ICD-O) pour améliorer la normalisation des expressions complexes.
* **DVC + MLflow** : introduire le versioning des données et des modèles pour garantir la reproductibilité et la traçabilité, indispensables dans un contexte médical réglementé.

**Long terme** :
* **Intégration EDS-NLP** : contribuer le composant oncologique au pipeline EDS-NLP de l'AP-HP pour une utilisation sur l'EDS.
* **Validation clinique** : évaluation par des experts oncologiques pour mesurer l'utilité clinique réelle du système.
* **Détection de dérive** : intégrer des outils de monitoring (Evidently AI) pour détecter les dérives de distribution quand le système est déployé sur de nouveaux comptes-rendus.

---

## 9. Références

* Beltagy, I., Peters, M. E., & Cohan, A. (2020). *Longformer: The Long-Document Transformer*. arXiv:2004.05150.
* Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., ... & Stoyanov, V. (2020). *Unsupervised Cross-lingual Representation Learning at Scale*. ACL 2020.
* Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805.
* Dura, B., Petit, G., Neuraz, A., & Tannier, X. (2023). *EDS-NLP: A Natural Language Processing Library for French Clinical Notes*. JAMIA Open.
* FRACCO Annotation Toolkit (2025). *FRACCO: French Annotated Corpus for Cancer Oncology*. Zenodo. DOI: 10.5281/zenodo.17284817.
* Jiang, Z., Xu, F. F., Araki, J., & Neubig, G. (2019). *How Can We Know What a Language Model Knows?* arXiv:1911.12543.
* Labrak, Y., Bazoge, A., Morin, E., Gourraud, P. A., Rouvier, M., & Dufour, R. (2023). *DrBERT: A Robust Pre-trained Model in French for Biomedical and Clinical domains*. ACL 2023.
* Lee, K., He, L., Lewis, M., & Zettlemoyer, L. (2017). *End-to-end Neural Coreference Resolution*. EMNLP 2017.
* Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal Loss for Dense Object Detection*. ICCV 2017.
* Martin, L., Muller, B., Suárez, P. J. O., Dupont, Y., Romary, L., de la Clergerie, É., ... & Sagot, B. (2020). *CamemBERT: a Tasty French Language Model*. ACL 2020.
* Miranda-Escalada, A., Farré, E., & Krallinger, M. (2020). *Named Entity Recognition, Concept Normalization and Clinical Coding: Overview of the Track on Cancer Disease Named Entity Recognition at IberLEF 2020*. IberLEF 2020.

---
*Ce rapport a été rédigé dans le cadre d'un projet de NLP médical — Juillet 2026*
