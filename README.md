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

Ce système est conçu pour maximiser les probabilités de succès en combinant plusieurs approches analytiques :
- **IA prédictive** : Modèles entraînés sur l'historique complet des tirages.
- **Analyse de cycles** : Détection de patterns récurrents dans le temps.
- **Mathématiques** : Utilisation de suites de Fibonacci pour la pondération.
- **Automatisation** : Scraping et mise à jour transparente sans intervention humaine.

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
- **Source** : Scraping sécurisé depuis les résultats officiels.
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
