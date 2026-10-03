# 🎰 Système de Prédiction Loto - Intelligence Artificielle & Statistiques

![Version](https://img.shields.io/badge/Version-3.1.0-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

Plateforme complète d'analyse et de prédiction pour le Loto français, utilisant des modèles de Machine Learning (Random Forest), des analyses mathématiques (Fibonacci) et des cycles temporels.

---

## 📋 Table des Matières
1. [Vue d'ensemble](#-vue-densemble)
2. [Caractéristiques Techniques](#-caractéristiques-techniques)
3. [Architecture du Système](#-architecture-du-système)
4. [Installation et Démarrage](#-installation-et-démarrage)
5. [Méthodes de Prédiction](#-méthodes-de-prédiction)
6. [Machine Learning & Entraînement](#-machine-learning--entraînement)
7. [Automatisation & Scraping](#-automatisation--scraping)
8. [API REST](#-api-rest)
9. [Dépannage](#-dépannage)

---

## 📋 Vue d'ensemble

Ce système analyse l'historique des tirages du Loto français avec plusieurs approches :
- **Modèles statistiques / ML** : Random Forest entraîné sur l'historique des tirages.
- **Analyse de cycles** : Détection de motifs temporels.
- **Mathématiques** : Pondération par suite de Fibonacci.
- **Automatisation** : Scraping et mise à jour sans intervention.

> ### ⚠️ Réalité statistique à lire avant tout
> Le Loto tire **5 numéros parmi 49 + 1 Chance parmi 10** avec un tambour physique.
> Chaque tirage est **indépendant** et les **19 068 840** combinaisons sont **strictement
> équiprobables**. Aucune méthode (ML, Fibonacci, cycles, fréquences) ne peut *prédire*
> un tirage ni créer un avantage : il n'y a pas de signal dans un bruit uniforme.
> L'espérance de gain d'une grille est **négative** (RTP ≈ 40 %). Cet outil est un
> projet d'**analyse et de pédagogie statistique**, pas une martingale.
> Pour le vérifier vous-même avec des chiffres, voir la section
> [Évaluation scientifique](#-évaluation-scientifique-mesurer-la-réalité).

---

## 🛠 Spécifications Techniques

- **Langage** : Python 3.13 (recommendé) / 3.8+
- **Framework Web** : Flask 3.0.0
- **Machine Learning** : Scikit-Learn, Joblib
- **Data Science** : Pandas, NumPy, Matplotlib
- **Automatisation** : Schedule, BeautifulSoup4 (Scraping)
- **Déploiement** : Supporté sur VPS (Windows Server) et Local

---

## 🚀 Installation et Démarrage

### 1. Installation (Local ou VPS)
```bash
# Entrer dans le dossier
cd loto

# Créer l'environnement virtuel et l'activer
python -m venv env
env\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Configuration (`.env`)
Créez un fichier `.env` à partir de `.env.example` :
```env
PORT=5000
HOST=0.0.0.0
VPS_IP=107.189.17.46
```

### 3. Lancer l'application
- **Windows (Double-clic)** : `start_loto.bat`
- **Ligne de commande** : `python app.py`

---

## 🧠 Méthodes de Prédiction

Le système propose 6 modes d'analyse :
1. **Machine Learning (Default)** : Utilise les modèles Random Forest.
2. **Fibonacci** : Pondération basée sur la suite mathématique (30% Fib / 70% ML).
3. **Cycle** : Analyse des patterns temporels récurrents.
4. **Frequency** : Statistiques pures (numéros les plus sortis).
5. **Optimized** : Combinaison multi-critères intelligente.
6. **All** : Fusion de toutes les méthodes pour une recommandation robuste.

---

## 📊 Machine Learning & Entraînement

### ⚠️ IMPORTANT : Entraînement Local Recommandé
L'entraînement étant gourmand en ressources, il est recommandé de l'effectuer sur votre PC puis de transférer les modèles vers le VPS.

**Pour entraîner localement :**
```bash
# Utiliser le script automatique
.\train_local.bat

# Ou manuellement
python train_local.py --csv tirages_loto.csv
```

**Modèles générés (`resultats_loto/models/`) :**
- `rf_number_model.joblib` & `rf_chance_model.joblib`
- `scaler_numbers.joblib` & `scaler_chance.joblib`

---

## 🔄 Automatisation & Scraping

Le système se met à jour tout seul :
- **Jours de tirage** : Lundi, Mercredi et Samedi à **23h00**.
- **Source** : scraping d'un site tiers (non officiel) en HTTP. ⚠️ Cette source n'offre
  aucune garantie d'intégrité ni de disponibilité ; voir le point de durcissement
  ci-dessous. Pour une donnée fiable, utilisez la source officielle FDJ en HTTPS et
  vérifiez la cohérence avant ingestion.
- **Sauvegardes** : Backups automatiques du fichier `tirages_loto.csv` avant chaque mise à jour.
- **Auto-Update** : Intégration directe dans le thread d'arrière-plan de Flask.

---

## 📡 API REST

| Méthode | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/test` | Test de connectivité |
| `GET` | `/api/predict` | Génère une prédiction rapide |
| `POST` | `/api/predict` | Configuration avancée des prédictions |
| `GET` | `/api/stats` | Statistiques globales des tirages |

**Exemple de requête POST :**
```json
{
  "method": "all",
  "combinations": 5
}
```

---

## 🔬 Évaluation scientifique (mesurer la réalité)

Le module `script/loto_evaluation.py` fournit l'évaluation honnête qui manquait au
projet. Il ne « prédit » rien : il **mesure** ce que valent réellement les méthodes.

```bash
python script/loto_evaluation.py --csv tirages_loto.csv
python script/loto_evaluation.py --csv tirages_loto.csv --mc-runs 500 --json rapport.json
```

Ce qu'il calcule :
- **Cotes exactes** de chaque rang (calcul hypergéométrique, pas d'estimation).
- **Espérance de gain / RTP** à partir d'une grille de gains paramétrable, et le
  **jackpot de rentabilité** (~27 M€) qui illustre pourquoi aucun seuil réaliste ne
  rend le jeu gagnant (partage du jackpot + variance colossale).
- **Backtest walk-forward** de chaque stratégie (aléatoire, fréquents, retardataires,
  récence, Fibonacci) : pour chaque tirage, la stratégie ne voit que le passé.
- **Référence Monte-Carlo** + test de significativité : aucune stratégie ne se
  distingue du hasard (p ≥ 0,01). C'est le résultat attendu, et pouvoir le mesurer
  est précisément la valeur ajoutée.

Tests : `python test_loto_evaluation.py` (verrouille les faits mathématiques).

---

## 🐛 Dépannage

- **Erreur scikit-learn/PyArrow** : Utilisez `start_loto.bat` qui corrige automatiquement les conflits de DLL.
- **Modèles absents** : Le système passera automatiquement en mode "Frequency" (statistiques) pour rester fonctionnel.
- **Port déjà utilisé** : Modifiez `PORT` dans votre `.env`.

---

## 🛡️ Sécurité & Avertissement

- **Jeu Responsable** : La loterie est un jeu de hasard. L'IA ne garantit pas de gain.
- **Gestion des Secrets** : Ne partagez jamais votre fichier `.env` ou vos clés API.

---
**Auteur** : Anissdev8897  
**Dernière mise à jour** : Janvier 2026
