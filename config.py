#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration centrale du système de prédiction Loto
Toutes les constantes et paramètres sont centralisés ici
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Répertoire de base du projet
BASE_DIR = Path(__file__).parent

# ============================================================================
# CONSTANTES LOTO
# ============================================================================

# Paramètres de jeu
MAX_NUMBER = 49          # Numéro maximum (1-49)
MAX_CHANCE = 10          # Numéro chance maximum (1-10)
PROPOSE_SIZE = 5         # Nombre de numéros principaux à proposer

# Colonnes CSV
DATE_COL = 'Date'
NUMBER_COLS = [f'Numéro {i}' for i in range(1, 6)]  # Numéro 1 à Numéro 5
CHANCE_COL = 'Chance'

# Fichiers
CSV_FILE = BASE_DIR / "tirages_loto.csv"
MODELS_DIR = BASE_DIR / "resultats_loto" / "models"
OUTPUT_DIR = BASE_DIR / "resultats_loto"

# ============================================================================
# CONFIGURATION FLASK
# ============================================================================

# Port et host
FLASK_PORT = int(os.getenv('PORT', 5000))
FLASK_HOST = os.getenv('HOST', '0.0.0.0')
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

# Configuration VPS
VPS_IP = os.getenv('VPS_IP', '107.189.17.46')
VPS_DOMAIN = os.getenv('VPS_DOMAIN', 'kenopredictionia.fr')
VPS_PATH = os.getenv('VPS_PATH', '/inetpub/wwwroot/bottrading_website/loto')
VPS_USER = os.getenv('VPS_USER', 'root')

# CORS - Origines autorisées
ALLOWED_ORIGINS = [
    f'http://{VPS_IP}:{FLASK_PORT}',
    f'http://{VPS_IP}:80',
    f'http://{VPS_IP}',
    f'https://{VPS_IP}:{FLASK_PORT}',
    f'https://{VPS_IP}:80',
    f'https://{VPS_IP}',
    f'http://localhost:{FLASK_PORT}',
    f'http://127.0.0.1:{FLASK_PORT}',
    f'https://{VPS_DOMAIN}',
    f'http://{VPS_DOMAIN}',
]

# ============================================================================
# CONFIGURATION SCRAPING
# ============================================================================

# Jours et heures de scraping
SCRAPING_DAYS = ['lundi', 'mercredi', 'samedi']  # Lundi, Mercredi, Samedi
SCRAPING_HOUR = 23
SCRAPING_MINUTE = 0

# Source de données
FDJ_URL = "https://www.fdj.fr/jeux/jeux-de-tirage/loto/resultats"

# ============================================================================
# CONFIGURATION MACHINE LEARNING
# ============================================================================

# Paramètres d'entraînement
ML_WINDOW_SIZE = 50              # Fenêtre d'analyse pour les features
ML_N_ESTIMATORS = 200            # Nombre d'arbres pour Random Forest
ML_N_ITER = 50                   # Itérations pour recherche hyperparamètres
ML_CV = 5                        # Folds pour validation croisée
ML_MIN_SAMPLES = 100             # Minimum de tirages pour entraînement

# Répertoire des modèles
MODEL_FILES = {
    'number_model': MODELS_DIR / "rf_number_model.joblib",
    'chance_model': MODELS_DIR / "rf_chance_model.joblib",
    'scaler_numbers': MODELS_DIR / "scaler_numbers.joblib",
    'scaler_chance': MODELS_DIR / "scaler_chance.joblib",
}

# ============================================================================
# CONFIGURATION API (optionnel - pour futures fonctionnalités)
# ============================================================================

# OpenAI (optionnel)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini-2024-07-18')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')

# Grok AI (optionnel)
GROK_API_KEY = os.getenv('GROK_API_KEY', '')
GROK_MODEL = os.getenv('GROK_MODEL', 'grok-4-latest')
GROK_BASE_URL = os.getenv('GROK_BASE_URL', 'https://api.x.ai/v1')

# OpenRouter (optionnel)
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'anthropic/claude-opus-4.1')
OPENROUTER_BASE_URL = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')

# ============================================================================
# CONFIGURATION LOTO ANALYZER
# ============================================================================

def get_loto_config() -> Dict[str, Any]:
    """Retourne la configuration par défaut pour LotoAnalyzer"""
    return {
        'csv_path': str(CSV_FILE),
        'output_dir': str(OUTPUT_DIR),
        'date_col': DATE_COL,
        'ball_cols': NUMBER_COLS,
        'chance_col': CHANCE_COL,
        'max_number': MAX_NUMBER,
        'max_chance': MAX_CHANCE,
        'propose_size': PROPOSE_SIZE,
        'window_size_for_ml': ML_WINDOW_SIZE,
        'use_cycle_analysis': True,
        'use_fibonacci_inverse': True,
        'generate_optimized_grid': True,
        'use_ml': True,
    }

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Valide la configuration et affiche un avertissement en cas de problème"""
    issues = []
    
    # Vérifier les constantes Loto
    if MAX_NUMBER != 49:
        issues.append(f"MAX_NUMBER devrait être 49, actuellement {MAX_NUMBER}")
    if MAX_CHANCE != 10:
        issues.append(f"MAX_CHANCE devrait être 10, actuellement {MAX_CHANCE}")
    if PROPOSE_SIZE != 5:
        issues.append(f"PROPOSE_SIZE devrait être 5, actuellement {PROPOSE_SIZE}")
    
    # Vérifier les fichiers/dossiers
    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    if issues:
        print("⚠️  Avertissements de configuration:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    return True

# Valider au chargement du module
if __name__ != "__main__":
    validate_config()

