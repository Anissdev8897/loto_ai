#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend Flask pour l'interface web de prédiction Loto
Utilise les modèles de machine learning entraînés pour générer des prédictions
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

# Ajouter le répertoire script au path
current_dir = Path(__file__).parent
script_dir = current_dir / "script"
sys.path.insert(0, str(script_dir))

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LotoWebApp")

# Initialisation Flask
# Configuration pour servir les templates et fichiers statiques
app = Flask(__name__, 
            template_folder=str(current_dir), 
            static_folder=str(current_dir))

# Configuration CORS pour autoriser l'IP spécifiée et toutes les connexions
# Autoriser toutes les origines pour permettre l'accès depuis n'importe quelle IP
# Si vous voulez restreindre, décommentez les ALLOWED_ORIGINS ci-dessous
ALLOWED_ORIGINS = [
    'http://107.189.17.46:5000',
    'http://107.189.17.46:80',
    'http://107.189.17.46',
    'https://107.189.17.46:5000',
    'https://107.189.17.46:80',
    'https://107.189.17.46',
    'http://localhost:5000',
    'http://127.0.0.1:5000',
    'https://kenopredictionia.fr',
    'http://kenopredictionia.fr',
    'https://www.kenopredictionia.fr',
    'http://www.kenopredictionia.fr'
]

# Autoriser toutes les origines (pour le développement)
# En production, vous pouvez restreindre avec origins=ALLOWED_ORIGINS
CORS(app, origins="*", supports_credentials=True)

# Configuration Flask
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['JSON_AS_ASCII'] = False  # Pour supporter les caractères UTF-8 dans les réponses JSON
app.config['ALLOWED_HOSTS'] = ['107.189.17.46', '0.0.0.0', '127.0.0.1', 'localhost']

# Middleware pour logger toutes les requêtes et valider les IPs
@app.before_request
def log_request_info():
    """Log toutes les requêtes entrantes pour le débogage et validation IP"""
    client_ip = request.remote_addr
    origin = request.headers.get('Origin', 'N/A')
    
    logger.info(f"Requête: {request.method} {request.path}")
    logger.info(f"IP client: {client_ip}")
    logger.info(f"Origin: {origin}")
    logger.debug(f"Headers: {dict(request.headers)}")
    logger.debug(f"Args: {dict(request.args)}")
    if request.is_json:
        logger.debug(f"JSON: {request.json}")
    
    # Autoriser toutes les connexions (0.0.0.0 écoute sur toutes les interfaces)
    # La validation IP est gérée par le firewall/serveur

# Configuration
MODELS_DIR = current_dir / "resultats_loto" / "models"
CSV_FILE = current_dir / "tirages_loto.csv"
MAX_NUMBER = 49
MAX_CHANCE = 10
PROPOSE_SIZE = 5

# Variables globales pour les modèles
rf_number_model = None
rf_chance_model = None
scaler_numbers = None
scaler_chance = None
df_loto = None


def load_data():
    """Charge les données des tirages Loto depuis le CSV"""
    global df_loto
    try:
        if CSV_FILE.exists():
            df_loto = pd.read_csv(CSV_FILE, encoding='utf-8')
            logger.info(f"Données chargées: {len(df_loto)} tirages")
            return True
        else:
            logger.error(f"Fichier CSV non trouvé: {CSV_FILE}")
            return False
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}", exc_info=True)
        return False


def load_models():
    """Charge les modèles de machine learning"""
    global rf_number_model, rf_chance_model, scaler_numbers, scaler_chance
    
    try:
        # Vérifier que scikit-learn peut être importé
        try:
            # Workaround: désactiver temporairement l'import pyarrow si nécessaire
            import os
            original_env = os.environ.copy()
            # Essayer d'importer sklearn sans pyarrow si possible
            try:
                # Désactiver pyarrow si disponible mais problématique
                os.environ['ARROW_NO_IMPORT'] = '1'
            except:
                pass
            
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            logger.info("scikit-learn importé avec succès")
            
            # Restaurer l'environnement (ne pas le vider complètement)
            if 'ARROW_NO_IMPORT' in os.environ:
                del os.environ['ARROW_NO_IMPORT']
        except (ImportError, OSError) as import_error:
            logger.error(f"Impossible d'importer scikit-learn: {import_error}")
            logger.error("Solution: pip uninstall pyarrow -y && pip install --upgrade scikit-learn")
            logger.error("PyArrow peut causer des conflits - il n'est pas nécessaire pour notre code")
            return False
        
        number_model_path = MODELS_DIR / "rf_number_model.joblib"
        number_scaler_path = MODELS_DIR / "scaler_numbers.joblib"
        chance_model_path = MODELS_DIR / "rf_chance_model.joblib"
        chance_scaler_path = MODELS_DIR / "scaler_chance.joblib"
        
        # Charger les modèles de numéros
        if number_model_path.exists() and number_scaler_path.exists():
            try:
                rf_number_model = joblib.load(number_model_path)
                scaler_numbers = joblib.load(number_scaler_path)
                logger.info("Modèles de numéros chargés avec succès")
            except Exception as load_error:
                logger.error(f"Erreur lors du chargement des modèles de numéros: {load_error}")
                logger.error("Le problème peut venir d'une incompatibilité de versions de scikit-learn")
                logger.error("Solution: réentraîner les modèles avec la version actuelle de scikit-learn")
                rf_number_model = None
                scaler_numbers = None
        else:
            logger.warning("Modèles de numéros non trouvés - ils seront créés à la prochaine exécution")
        
        # Charger les modèles de numéro chance
        if chance_model_path.exists() and chance_scaler_path.exists():
            try:
                rf_chance_model = joblib.load(chance_model_path)
                scaler_chance = joblib.load(chance_scaler_path)
                logger.info("Modèles de numéro chance chargés avec succès")
            except Exception as load_error:
                logger.error(f"Erreur lors du chargement des modèles de chance: {load_error}")
                rf_chance_model = None
                scaler_chance = None
        else:
            logger.warning("Modèles de numéro chance non trouvés")
        
        return rf_number_model is not None and scaler_numbers is not None
    except Exception as e:
        logger.error(f"Erreur lors du chargement des modèles: {e}", exc_info=True)
        logger.error("L'application continuera sans les modèles ML")
        logger.error("Solution: installer/réinstaller pyarrow et scikit-learn")
        return False


def predict_from_frequencies(num_combinations: int = 5):
    """
    Prédiction basée uniquement sur les fréquences quand les modèles ML ne sont pas disponibles
    
    Args:
        num_combinations: Nombre de combinaisons à générer
        
    Returns:
        JSON response avec les prédictions
    """
    try:
        # Identifier les colonnes de numéros
        ball_cols = [col for col in df_loto.columns if 'Numéro' in col or 'numéro' in col]
        if not ball_cols:
            ball_cols = [f'Numéro {i}' for i in range(1, 6)]
        ball_cols = [col for col in ball_cols if col in df_loto.columns]
        
        # Calculer les fréquences
        from collections import Counter
        all_numbers = []
        for col in ball_cols:
            all_numbers.extend(df_loto[col].dropna().astype(int).tolist())
        
        freq = Counter(all_numbers)
        
        # Trier par fréquence décroissante
        sorted_numbers = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        top_numbers = [num for num, _ in sorted_numbers[:PROPOSE_SIZE]]
        
        # Générer des combinaisons
        combinations = []
        for i in range(min(num_combinations, 5)):
            # Prendre les numéros les plus fréquents
            combo = sorted_numbers[i*PROPOSE_SIZE:(i+1)*PROPOSE_SIZE]
            if len(combo) < PROPOSE_SIZE:
                combo = sorted_numbers[:PROPOSE_SIZE]
            numbers = sorted([num for num, _ in combo])
            
            # Numéro chance (fréquence)
            chance_num = None
            if 'Chance' in df_loto.columns:
                chance_freq = Counter(df_loto['Chance'].dropna().astype(int))
                if chance_freq:
                    chance_num = sorted(chance_freq.items(), key=lambda x: x[1], reverse=True)[0][0]
            
            combinations.append({
                'id': i + 1,
                'numbers': numbers,
                'chance': chance_num,
                'method': 'Fréquences (fallback)'
            })
        
        return jsonify({
            'success': True,
            'prediction': {
                'recommended_combination': combinations[0]['numbers'] if combinations else [],
                'top_numbers': top_numbers,
                'chance_number': combinations[0]['chance'] if combinations else None,
                'number_predictions': {num: count / len(all_numbers) for num, count in freq.items()},
                'all_top_numbers': [num for num, _ in sorted_numbers[:10]],
                'combinations': combinations,
                'warning': 'Modèles ML non disponibles - Prédiction basée sur les fréquences uniquement'
            }
        })
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction par fréquences: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """
    Prépare les features pour la prédiction à partir des dernières données.
    Utilise le même EnhancedEncoder que lors de l'entraînement pour garantir la cohérence.
    """
    try:
        if df is None or df.empty:
            return np.array([])
        
        # Importer EnhancedEncoder depuis auto_trainer pour utiliser le même encodage
        try:
            from auto_trainer import EnhancedEncoder
            encoder = EnhancedEncoder()
        except ImportError:
            logger.warning("EnhancedEncoder non disponible, utilisation de la méthode simplifiée")
            # Fallback vers méthode simplifiée
            return prepare_features_simple(df)
        
        # Identifier les colonnes de numéros
        ball_cols = [col for col in df.columns if 'Numéro' in col or 'numéro' in col]
        if not ball_cols:
            ball_cols = [f'Numéro {i}' for i in range(1, 6)]
        ball_cols = [col for col in ball_cols if col in df.columns]
        
        if not ball_cols:
            logger.warning("Aucune colonne de numéro trouvée")
            return np.array([])
        
        # Prendre les 50 derniers tirages pour calculer les features
        window_size = min(50, len(df))
        recent_df = df.tail(window_size).copy()
        
        # Utiliser le même encodeur que lors de l'entraînement
        # Features numériques (fréquences, patterns, statistiques)
        numeric_features = encoder.encode_number_features(recent_df, window_size)
        
        # Features temporelles (pour le prochain tirage = aujourd'hui)
        from datetime import datetime
        today_df = pd.DataFrame([{
            'Date': datetime.now().strftime('%d/%m/%Y')
        }])
        temporal_features = encoder.encode_temporal_features(today_df)
        
        # Combiner les features de la même manière que lors de l'entraînement
        if numeric_features.shape[0] > 0 and temporal_features.shape[0] > 0:
            # Prendre la dernière ligne des features numériques (représente les derniers tirages)
            numeric_last = numeric_features[-1:].reshape(1, -1)
            temporal_last = temporal_features[0:1].reshape(1, -1)
            combined = np.hstack([numeric_last, temporal_last])
        elif numeric_features.shape[0] > 0:
            combined = numeric_features[-1:].reshape(1, -1)
        elif temporal_features.shape[0] > 0:
            combined = temporal_features[0:1].reshape(1, -1)
        else:
            logger.warning("Impossible de générer des features")
            return np.array([])
        
        return combined
        
    except Exception as e:
        logger.error(f"Erreur lors de la préparation des features: {e}", exc_info=True)
        logger.warning("Fallback vers méthode simplifiée")
        return prepare_features_simple(df)


def prepare_features_simple(df: pd.DataFrame) -> np.ndarray:
    """
    Méthode simplifiée de préparation des features (fallback).
    Utilisée si EnhancedEncoder n'est pas disponible.
    """
    try:
        if df is None or df.empty:
            return np.array([])
        
        # Identifier les colonnes de numéros
        ball_cols = [col for col in df.columns if 'Numéro' in col or 'numéro' in col]
        if not ball_cols:
            ball_cols = [f'Numéro {i}' for i in range(1, 6)]
        ball_cols = [col for col in ball_cols if col in df.columns]
        
        if not ball_cols:
            logger.warning("Aucune colonne de numéro trouvée")
            return np.array([])
        
        # Prendre les 50 derniers tirages pour calculer les features
        window_size = min(50, len(df))
        recent_df = df.tail(window_size)
        
        # Extraire tous les numéros de la fenêtre récente
        all_numbers = []
        for col in ball_cols:
            if col in recent_df.columns:
                numbers = recent_df[col].dropna().astype(int).tolist()
                all_numbers.extend(numbers)
        
        # Calculer les fréquences
        from collections import Counter
        freq = Counter(all_numbers)
        
        # Créer les features (fréquence normalisée de chaque numéro)
        features = np.zeros(MAX_NUMBER)
        for num, count in freq.items():
            if 1 <= num <= MAX_NUMBER:
                features[num - 1] = count / (window_size * len(ball_cols))
        
        # Reshaper pour avoir la bonne dimension (1, MAX_NUMBER)
        return features.reshape(1, -1)
        
    except Exception as e:
        logger.error(f"Erreur lors de la préparation simplifiée des features: {e}", exc_info=True)
        return np.array([])


@app.route('/')
def index():
    """Route principale - sert l'interface HTML"""
    try:
        # Essayer de servir le fichier directement
        html_path = current_dir / 'loto_page.html'
        if html_path.exists():
            with open(html_path, 'r', encoding='utf-8') as f:
                return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
        else:
            return render_template('loto_page.html')
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la page HTML: {e}", exc_info=True)
        return f"Erreur: {str(e)}", 500


@app.route('/api/test', methods=['GET'])
def test():
    """Route de test pour vérifier que les routes API fonctionnent"""
    return jsonify({
        'success': True,
        'message': 'API fonctionne correctement',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predict', methods=['GET', 'POST'])
def predict():
    """API endpoint pour générer des prédictions avec différentes méthodes"""
    logger.info(f"=== Appel API /api/predict ===")
    logger.info(f"Méthode: {request.method}")
    logger.info(f"Headers: {dict(request.headers)}")
    try:
        # Récupérer les paramètres de la requête
        if request.method == 'POST':
            method = request.json.get('method', 'all') if request.is_json else 'all'
            num_combinations = request.json.get('combinations', 5) if request.is_json else 5
        else:
            method = 'all'
            num_combinations = 1
        if df_loto is None or df_loto.empty:
            return jsonify({
                'success': False,
                'error': 'Aucune donnée disponible'
            }), 400
        
        if rf_number_model is None or scaler_numbers is None:
            logger.warning("Modèles ML non disponibles - utilisation de la prédiction basée sur les fréquences")
            # Fallback: prédiction basée sur les fréquences uniquement
            try:
                return predict_from_frequencies(num_combinations)
            except Exception as fallback_error:
                logger.error(f"Erreur lors de la prédiction par fréquences: {fallback_error}", exc_info=True)
                return jsonify({
                    'success': False,
                    'error': 'Modèles ML non disponibles et prédiction par fréquences échouée. Veuillez installer scikit-learn: pip install --upgrade scikit-learn pyarrow'
                }), 500
        
        # Préparer les features
        features = prepare_features(df_loto)
        if features.size == 0:
            return jsonify({
                'success': False,
                'error': 'Impossible de préparer les features'
            }), 400
        
        # Normaliser les features
        features_scaled = scaler_numbers.transform(features)
        
        # Prédire les probabilités pour les numéros
        # Le modèle RandomForest retourne une liste où chaque élément correspond à un numéro
        # Chaque élément est un array [prob_classe_0, prob_classe_1]
        # prob_classe_1 est la probabilité que le numéro soit tiré
        try:
            probas_numbers = rf_number_model.predict_proba(features_scaled)
            
            # Convertir les probabilités en dictionnaire (comme dans loto_main.py)
            number_predictions = {}
            for i in range(MAX_NUMBER):
                num = i + 1
                # Vérifier si la classe existe dans le modèle
                if i < len(probas_numbers):
                    # Vérifier si la classe a des probabilités pour les deux classes
                    if len(probas_numbers[i]) > 1 and len(probas_numbers[i][0]) > 1:
                        # probas_numbers[i] est de la forme [[prob_0, prob_1]]
                        # On prend prob_1 qui est la probabilité que le numéro soit tiré
                        number_predictions[num] = float(probas_numbers[i][0][1])
                    elif len(probas_numbers[i]) > 0:
                        # Si une seule probabilité, prendre celle-là
                        number_predictions[num] = float(probas_numbers[i][0][0])
                    else:
                        number_predictions[num] = 0.0
                else:
                    number_predictions[num] = 0.0
        except Exception as e:
            logger.warning(f"Erreur lors de l'extraction des probabilités: {e}", exc_info=True)
            # Fallback: utiliser predict pour obtenir les prédictions binaires
            try:
                predictions = rf_number_model.predict(features_scaled)
                # Convertir les prédictions en probabilités (0.0 ou 1.0)
                number_predictions = {}
                for i in range(MAX_NUMBER):
                    num = i + 1
                    if i < len(predictions) and i < predictions.shape[1]:
                        number_predictions[num] = float(predictions[0][i])
                    else:
                        number_predictions[num] = 0.0
            except Exception as e2:
                logger.error(f"Erreur lors du fallback predict: {e2}")
                number_predictions = {i+1: 0.0 for i in range(MAX_NUMBER)}
        
        # Trier par probabilité décroissante et prendre les top 10
        sorted_numbers = sorted(number_predictions.items(), key=lambda x: x[1], reverse=True)
        top_numbers = [num for num, prob in sorted_numbers[:10]]
        
        # Générer une combinaison recommandée
        recommended_numbers = sorted(sorted_numbers[:PROPOSE_SIZE], key=lambda x: x[0])
        recommended_combination = [num for num, _ in recommended_numbers]
        
        # Prédire le numéro chance (même format que les numéros principaux)
        chance_prediction = None
        if rf_chance_model is not None and scaler_chance is not None:
            try:
                features_scaled_chance = scaler_chance.transform(features)
                probas_chance = rf_chance_model.predict_proba(features_scaled_chance)
                
                # Convertir les probabilités en dictionnaire
                chance_predictions = {}
                for i in range(MAX_CHANCE):
                    num = i + 1
                    # Vérifier si la classe existe dans le modèle
                    if i < len(probas_chance):
                        # Vérifier si la classe a des probabilités pour les deux classes
                        if len(probas_chance[i]) > 1 and len(probas_chance[i][0]) > 1:
                            # probas_chance[i] est de la forme [[prob_0, prob_1]]
                            chance_predictions[num] = float(probas_chance[i][0][1])
                        elif len(probas_chance[i]) > 0:
                            chance_predictions[num] = float(probas_chance[i][0][0])
                        else:
                            chance_predictions[num] = 0.0
                    else:
                        chance_predictions[num] = 0.0
                
                sorted_chance = sorted(chance_predictions.items(), key=lambda x: x[1], reverse=True)
                chance_prediction = sorted_chance[0][0] if sorted_chance else None
            except Exception as e:
                logger.warning(f"Erreur lors de la prédiction du numéro chance: {e}", exc_info=True)
                # Fallback: prédire directement
                try:
                    features_scaled_chance = scaler_chance.transform(features)
                    predictions = rf_chance_model.predict(features_scaled_chance)
                    # Extraire le numéro chance prédit (première position où la prédiction est 1)
                    if len(predictions.shape) > 1:
                        chance_idx = np.argmax(predictions[0])
                        chance_prediction = chance_idx + 1 if 1 <= chance_idx + 1 <= MAX_CHANCE else None
                    else:
                        chance_prediction = int(predictions[0]) if 1 <= int(predictions[0]) <= MAX_CHANCE else None
                except Exception as e2:
                    logger.error(f"Erreur lors du fallback pour le numéro chance: {e2}")
                    chance_prediction = None
        
        # Générer des combinaisons selon la méthode demandée
        combinations = []
        
        if method == 'all' or method == 'main':
            # Méthode principale avec ML
            for i in range(min(num_combinations, 5)):
                combo = sorted(sorted_numbers[:PROPOSE_SIZE], key=lambda x: x[0])
                combinations.append({
                    'id': i + 1,
                    'numbers': [num for num, _ in combo],
                    'chance': chance_prediction,
                    'method': 'Machine Learning'
                })
        elif method == 'fibonacci':
            # Méthode Fibonacci
            try:
                from script.fibonacci_weighting import apply_inverse_fibonacci_weights
                from collections import Counter
                
                # Identifier les colonnes de numéros
                ball_cols = [col for col in df_loto.columns if 'Numéro' in col or 'numéro' in col]
                if not ball_cols:
                    ball_cols = [f'Numéro {i}' for i in range(1, 6)]
                ball_cols = [col for col in ball_cols if col in df_loto.columns]
                
                # Extraire tous les numéros
                all_numbers = []
                for col in ball_cols:
                    if col in df_loto.columns:
                        numbers = df_loto[col].dropna().astype(int).tolist()
                        all_numbers.extend(numbers)
                
                # Calculer les fréquences et appliquer la pondération Fibonacci
                freq = Counter(all_numbers)
                fib_weights = apply_inverse_fibonacci_weights(freq, reverse_order=True)
                
                # Combiner avec les prédictions ML (30% Fibonacci, 70% ML)
                combined_predictions = {}
                for num in range(1, MAX_NUMBER + 1):
                    ml_score = number_predictions.get(num, 0.0)
                    fib_score = fib_weights.get(num, 0.0)
                    combined_predictions[num] = 0.7 * ml_score + 0.3 * fib_score
                
                sorted_fib = sorted(combined_predictions.items(), key=lambda x: x[1], reverse=True)
                
                for i in range(min(num_combinations, 5)):
                    combo = sorted(sorted_fib[:PROPOSE_SIZE], key=lambda x: x[0])
                    combinations.append({
                        'id': i + 1,
                        'numbers': [num for num, _ in combo],
                        'chance': chance_prediction,
                        'method': 'Fibonacci (avec ML)'
                    })
            except Exception as e:
                logger.warning(f"Erreur lors de la prédiction Fibonacci: {e}, utilisation de la méthode principale")
                combo = sorted(sorted_numbers[:PROPOSE_SIZE], key=lambda x: x[0])
                combinations.append({
                    'id': 1,
                    'numbers': [num for num, _ in combo],
                    'chance': chance_prediction,
                    'method': 'Machine Learning (fallback)'
                })
        elif method == 'cycle':
            # Méthode Cycle
            try:
                from script.cycle_analysis import calculate_adaptive_cycle_weights
                
                # Identifier les colonnes de numéros
                ball_cols = [col for col in df_loto.columns if 'Numéro' in col or 'numéro' in col]
                if not ball_cols:
                    ball_cols = [f'Numéro {i}' for i in range(1, 6)]
                ball_cols = [col for col in ball_cols if col in df_loto.columns]
                
                # Calculer les poids adaptatifs basés sur les cycles
                cycle_weights_data = calculate_adaptive_cycle_weights(
                    df_loto, ball_cols, window_sizes=[10, 20, 50],
                    chance_col='Chance' if 'Chance' in df_loto.columns else None,
                    min_main_num=1, max_main_num=49,
                    min_chance_num=1, max_chance_num=10
                )
                
                cycle_weights = cycle_weights_data.get('main_adaptive_weights', {})
                
                # Combiner avec les prédictions ML (40% Cycles, 60% ML)
                combined_predictions = {}
                for num in range(1, MAX_NUMBER + 1):
                    ml_score = number_predictions.get(num, 0.0)
                    cycle_score = cycle_weights.get(num, 0.0)
                    combined_predictions[num] = 0.6 * ml_score + 0.4 * cycle_score
                
                sorted_cycle = sorted(combined_predictions.items(), key=lambda x: x[1], reverse=True)
                
                for i in range(min(num_combinations, 5)):
                    combo = sorted(sorted_cycle[:PROPOSE_SIZE], key=lambda x: x[0])
                    combinations.append({
                        'id': i + 1,
                        'numbers': [num for num, _ in combo],
                        'chance': chance_prediction,
                        'method': 'Cycles (avec ML)'
                    })
            except Exception as e:
                logger.warning(f"Erreur lors de la prédiction par cycles: {e}, utilisation de la méthode principale")
                combo = sorted(sorted_numbers[:PROPOSE_SIZE], key=lambda x: x[0])
                combinations.append({
                    'id': 1,
                    'numbers': [num for num, _ in combo],
                    'chance': chance_prediction,
                    'method': 'Machine Learning (fallback)'
                })
        elif method == 'frequency':
            # Méthode Frequency (fréquences uniquement)
            try:
                from script.frequency_analysis import calculate_number_frequencies
                from collections import Counter
                
                # Identifier les colonnes de numéros
                ball_cols = [col for col in df_loto.columns if 'Numéro' in col or 'numéro' in col]
                if not ball_cols:
                    ball_cols = [f'Numéro {i}' for i in range(1, 6)]
                ball_cols = [col for col in ball_cols if col in df_loto.columns]
                
                # Calculer les fréquences
                freq_data = calculate_number_frequencies(
                    df_loto, ball_cols,
                    chance_col='Chance' if 'Chance' in df_loto.columns else None
                )
                
                freq_relative = freq_data.get('main_numbers_frequency_relative', {})
                
                # Trier par fréquence
                sorted_freq = sorted(freq_relative.items(), key=lambda x: x[1], reverse=True)
                
                for i in range(min(num_combinations, 5)):
                    combo = sorted_freq[i*PROPOSE_SIZE:(i+1)*PROPOSE_SIZE]
                    if len(combo) < PROPOSE_SIZE:
                        combo = sorted_freq[:PROPOSE_SIZE]
                    numbers = sorted([num for num, _ in combo])
                    
                    # Numéro chance par fréquence
                    chance_num = None
                    if 'chance_numbers_frequency_relative' in freq_data and freq_data['chance_numbers_frequency_relative']:
                        chance_freq = freq_data['chance_numbers_frequency_relative']
                        sorted_chance = sorted(chance_freq.items(), key=lambda x: x[1], reverse=True)
                        chance_num = sorted_chance[0][0] if sorted_chance else chance_prediction
                    
                    combinations.append({
                        'id': i + 1,
                        'numbers': numbers,
                        'chance': chance_num or chance_prediction,
                        'method': 'Fréquences'
                    })
            except Exception as e:
                logger.warning(f"Erreur lors de la prédiction par fréquences: {e}, utilisation de la méthode principale")
                combo = sorted(sorted_numbers[:PROPOSE_SIZE], key=lambda x: x[0])
                combinations.append({
                    'id': 1,
                    'numbers': [num for num, _ in combo],
                    'chance': chance_prediction,
                    'method': 'Machine Learning (fallback)'
                })
        elif method == 'optimized':
            # Méthode Optimisée (combinaison de toutes les méthodes)
            try:
                from script.fibonacci_weighting import apply_inverse_fibonacci_weights
                from script.cycle_analysis import calculate_adaptive_cycle_weights
                from script.frequency_analysis import calculate_number_frequencies
                from collections import Counter
                
                # Identifier les colonnes de numéros
                ball_cols = [col for col in df_loto.columns if 'Numéro' in col or 'numéro' in col]
                if not ball_cols:
                    ball_cols = [f'Numéro {i}' for i in range(1, 6)]
                ball_cols = [col for col in ball_cols if col in df_loto.columns]
                
                # Calculer toutes les métriques
                all_numbers = []
                for col in ball_cols:
                    if col in df_loto.columns:
                        numbers = df_loto[col].dropna().astype(int).tolist()
                        all_numbers.extend(numbers)
                
                freq = Counter(all_numbers)
                fib_weights = apply_inverse_fibonacci_weights(freq, reverse_order=True)
                
                cycle_weights_data = calculate_adaptive_cycle_weights(
                    df_loto, ball_cols, window_sizes=[10, 20, 50],
                    chance_col='Chance' if 'Chance' in df_loto.columns else None,
                    min_main_num=1, max_main_num=49,
                    min_chance_num=1, max_chance_num=10
                )
                cycle_weights = cycle_weights_data.get('main_adaptive_weights', {})
                
                freq_data = calculate_number_frequencies(
                    df_loto, ball_cols,
                    chance_col='Chance' if 'Chance' in df_loto.columns else None
                )
                freq_relative = freq_data.get('main_numbers_frequency_relative', {})
                
                # Combinaison optimisée : 35% ML + 25% Fibonacci + 25% Cycles + 15% Fréquences
                combined_predictions = {}
                for num in range(1, MAX_NUMBER + 1):
                    ml_score = number_predictions.get(num, 0.0)
                    fib_score = fib_weights.get(num, 0.0)
                    cycle_score = cycle_weights.get(num, 0.0)
                    freq_score = freq_relative.get(num, 0.0)
                    
                    combined_predictions[num] = (
                        0.35 * ml_score +
                        0.25 * fib_score +
                        0.25 * cycle_score +
                        0.15 * freq_score
                    )
                
                sorted_optimized = sorted(combined_predictions.items(), key=lambda x: x[1], reverse=True)
                
                for i in range(min(num_combinations, 5)):
                    combo = sorted(sorted_optimized[:PROPOSE_SIZE], key=lambda x: x[0])
                    combinations.append({
                        'id': i + 1,
                        'numbers': [num for num, _ in combo],
                        'chance': chance_prediction,
                        'method': 'Optimisée (ML + Fibonacci + Cycles + Fréquences)'
                    })
            except Exception as e:
                logger.warning(f"Erreur lors de la prédiction optimisée: {e}, utilisation de la méthode principale")
                combo = sorted(sorted_numbers[:PROPOSE_SIZE], key=lambda x: x[0])
                combinations.append({
                    'id': 1,
                    'numbers': [num for num, _ in combo],
                    'chance': chance_prediction,
                    'method': 'Machine Learning (fallback)'
                })
        else:
            # Utiliser la méthode principale par défaut
            combo = sorted(sorted_numbers[:PROPOSE_SIZE], key=lambda x: x[0])
            combinations.append({
                'id': 1,
                'numbers': [num for num, _ in combo],
                'chance': chance_prediction,
                'method': get_method_name(method)
            })
        
        return jsonify({
            'success': True,
            'prediction': {
                'recommended_combination': recommended_combination,
                'top_numbers': top_numbers[:PROPOSE_SIZE],
                'chance_number': chance_prediction,
                'number_predictions': number_predictions,
                'all_top_numbers': top_numbers,
                'combinations': combinations
            }
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}", exc_info=True)
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Traceback: {error_trace}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': error_trace if app.debug else None
        }), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """API endpoint pour obtenir des statistiques sur les données"""
    try:
        if df_loto is None or df_loto.empty:
            return jsonify({
                'success': False,
                'error': 'Aucune donnée disponible'
            }), 400
        
        # Identifier les colonnes de numéros
        ball_cols = [col for col in df_loto.columns if 'Numéro' in col or 'numéro' in col]
        if not ball_cols:
            ball_cols = [f'Numéro {i}' for i in range(1, 6)]
        ball_cols = [col for col in ball_cols if col in df_loto.columns]
        
        # Calculer les fréquences
        number_counts = {}
        for num in range(1, MAX_NUMBER + 1):
            count = 0
            for col in ball_cols:
                count += (df_loto[col] == num).sum()
            number_counts[num] = count
        
        # Calculer les fréquences pour le numéro chance si disponible
        chance_counts = {}
        if 'Chance' in df_loto.columns:
            for num in range(1, MAX_CHANCE + 1):
                chance_counts[num] = (df_loto['Chance'] == num).sum()
        
        return jsonify({
            'success': True,
            'stats': {
                'total_draws': len(df_loto),
                'number_frequencies': number_counts,
                'chance_frequencies': chance_counts if chance_counts else None
            }
        })
        
    except Exception as e:
        logger.error(f"Erreur lors du calcul des statistiques: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def get_method_name(method):
    """Retourne le nom d'une méthode"""
    names = {
        'all': 'Méthodes Combinées',
        'main': 'Machine Learning',
        'fibonacci': 'Fibonacci',
        'cycle': 'Cycles',
        'frequency': 'Fréquence',
        'optimized': 'Optimisée'
    }
    return names.get(method, 'Standard')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    models_loaded = rf_number_model is not None and scaler_numbers is not None
    data_loaded = df_loto is not None and not df_loto.empty
    
    return jsonify({
        'status': 'ok',
        'models_loaded': models_loaded,
        'data_loaded': data_loaded
    })


def initialize_auto_systems():
    """
    Initialise les systèmes automatiques de scraping et d'entraînement
    """
    try:
        # Vérifier d'abord que scikit-learn peut être importé
        try:
            # Workaround pour éviter les problèmes PyArrow
            import os
            original_env = os.environ.copy()
            try:
                os.environ['ARROW_NO_IMPORT'] = '1'
            except:
                pass
            
            import sklearn
            from sklearn.ensemble import RandomForestClassifier
            logger.info("scikit-learn disponible pour les systèmes automatiques")
            
            # Restaurer l'environnement
            if 'ARROW_NO_IMPORT' in os.environ:
                del os.environ['ARROW_NO_IMPORT']
        except (ImportError, OSError) as e:
            logger.error(f"scikit-learn non disponible: {e}")
            logger.error("Les systèmes automatiques seront désactivés")
            logger.error("Solution: pip uninstall pyarrow -y && pip install --upgrade scikit-learn")
            return False
        
        # Importer les modules d'auto-scraping et d'auto-training
        from auto_scraper import AutoScraper
        try:
            from auto_trainer import AutoTrainer
        except ImportError as e:
            logger.error(f"Impossible d'importer AutoTrainer: {e}")
            logger.error("Le réentraînement automatique sera désactivé")
            logger.error("Solution: pip install --upgrade scikit-learn pyarrow")
            # Continuer sans AutoTrainer si disponible
            AutoTrainer = None
        
        logger.info("=== Initialisation des systèmes automatiques ===")
        
        # Note: L'entraînement initial a déjà été fait avant le démarrage de Flask
        # Ici, on démarre seulement le scraper automatique en arrière-plan
        # qui pourra déclencher un réentraînement périodique si nécessaire
        
        # Démarrer le scraper automatique en arrière-plan
        scraper = AutoScraper(str(CSV_FILE))
        scraper.start()
        logger.info("Système de scraping automatique démarré")
        
        logger.info("=== Systèmes automatiques initialisés ===")
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation des systèmes automatiques: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    logger.info("Démarrage de l'application Flask...")
    
    # 1. Charger les données
    try:
        if not load_data():
            logger.warning("Les données n'ont pas pu être chargées")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}", exc_info=True)
    
    # 2. Vérifier si les modèles existent et sont compatibles, sinon les entraîner AVANT de charger
    models_dir = current_dir / "resultats_loto" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_files = [
        models_dir / "rf_number_model.joblib",
        models_dir / "scaler_numbers.joblib"
    ]
    
    models_exist = all(f.exists() for f in model_files)
    models_compatible = False
    
    # Vérifier si les modèles existants sont compatibles avec la version actuelle de scikit-learn
    if models_exist:
        try:
            import os
            original_env = os.environ.copy()
            try:
                os.environ['ARROW_NO_IMPORT'] = '1'
            except:
                pass
            
            import sklearn
            import joblib
            
            # Essayer de charger un modèle pour vérifier la compatibilité
            # Capturer les warnings pour détecter les incompatibilités de version
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    test_model = joblib.load(model_files[0])
                    # Vérifier si on a des warnings de version incompatible
                    version_warnings = [warning for warning in w 
                                      if 'version' in str(warning.message).lower() 
                                      or 'inconsistent' in str(warning.message).lower()
                                      or 'unpickle' in str(warning.message).lower()]
                    
                    if version_warnings:
                        logger.warning("⚠️ Warnings de version incompatible détectés lors du test de chargement")
                        for warning in version_warnings:
                            logger.warning(f"  - {warning.message}")
                        logger.info("Réentraînement nécessaire pour éviter ces warnings")
                        models_compatible = False
                    else:
                        models_compatible = True
                        logger.info("Modèles existants détectés et compatibles")
                except Exception as e:
                    # Si erreur de version incompatible, on devra réentraîner
                    error_msg = str(e).lower()
                    if 'version' in error_msg or 'inconsistent' in error_msg or 'unpickle' in error_msg:
                        logger.warning(f"⚠️ Modèles incompatibles détectés: {e}")
                        logger.warning("Les modèles ont été entraînés avec une version différente de scikit-learn")
                        logger.info("Réentraînement nécessaire pour éviter les warnings")
                        models_compatible = False
                    else:
                        # Autre erreur, on essaie quand même
                        models_compatible = True
                        logger.warning(f"Erreur lors du test de chargement (non fatale): {e}")
            
            if 'ARROW_NO_IMPORT' in os.environ:
                del os.environ['ARROW_NO_IMPORT']
        except Exception as e:
            logger.warning(f"Impossible de vérifier la compatibilité des modèles: {e}")
            models_compatible = True  # On suppose qu'ils sont OK pour continuer
    
    if not models_exist or not models_compatible:
        if not models_exist:
            logger.info("=== Modèles non trouvés, démarrage de l'entraînement AVANT le démarrage de Flask ===")
        else:
            logger.info("=== Modèles incompatibles détectés, réentraînement AVANT le démarrage de Flask ===")
            # Supprimer les anciens modèles incompatibles
            try:
                for model_file in model_files:
                    if model_file.exists():
                        model_file.unlink()
                        logger.info(f"Ancien modèle supprimé: {model_file.name}")
                # Supprimer aussi les modèles de chance
                chance_model = models_dir / "rf_chance_model.joblib"
                chance_scaler = models_dir / "scaler_chance.joblib"
                if chance_model.exists():
                    chance_model.unlink()
                if chance_scaler.exists():
                    chance_scaler.unlink()
            except Exception as e:
                logger.warning(f"Erreur lors de la suppression des anciens modèles: {e}")
        try:
            # Vérifier que scikit-learn peut être importé
            try:
                import os
                original_env = os.environ.copy()
                try:
                    os.environ['ARROW_NO_IMPORT'] = '1'
                except:
                    pass
                
                import sklearn
                from sklearn.ensemble import RandomForestClassifier
                
                if 'ARROW_NO_IMPORT' in os.environ:
                    del os.environ['ARROW_NO_IMPORT']
                
                logger.info("scikit-learn disponible pour l'entraînement")
                
                # Importer AutoTrainer
                from auto_trainer import AutoTrainer
                
                # Entraîner les modèles
                trainer = AutoTrainer(str(CSV_FILE))
                if trainer.load_data() and trainer.train_and_save():
                    logger.info("✅ Entraînement initial terminé avec succès AVANT le démarrage de Flask")
                else:
                    logger.warning("⚠️ Échec de l'entraînement initial")
            except (ImportError, OSError) as e:
                logger.error(f"scikit-learn non disponible pour l'entraînement: {e}")
                logger.warning("L'application démarrera sans les modèles ML (mode fallback)")
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement initial: {e}", exc_info=True)
                logger.warning("L'application démarrera sans les modèles ML (mode fallback)")
        except Exception as e:
            logger.error(f"Erreur lors de la vérification/préparation des modèles: {e}", exc_info=True)
            logger.warning("L'application démarrera sans les modèles ML (mode fallback)")
    else:
        logger.info("Modèles existants trouvés, pas d'entraînement nécessaire")
    
    # 3. Charger les modèles (après l'entraînement si nécessaire)
    try:
        if not load_models():
            logger.warning("Les modèles n'ont pas pu être chargés")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des modèles: {e}", exc_info=True)
    
    # 4. Initialiser les autres systèmes automatiques (scraping en arrière-plan)
    try:
        initialize_auto_systems()
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation des systèmes automatiques: {e}", exc_info=True)
        logger.warning("L'application continuera sans les systèmes automatiques")
    
    # Afficher toutes les routes disponibles pour le débogage
    logger.info("Routes disponibles:")
    for rule in app.url_map.iter_rules():
        logger.info(f"  {rule.rule} -> {rule.endpoint} [{', '.join(rule.methods)}]")
    
    # Démarrer l'application
    logger.info("Serveur Flask démarré sur http://0.0.0.0:5000")
    logger.info("Interface disponible sur http://107.189.17.46:5000")
    logger.info("API test disponible sur http://107.189.17.46:5000/api/test")
    logger.info("API predict disponible sur http://107.189.17.46:5000/api/predict")
    logger.info("Connexions autorisées depuis: 107.189.17.46")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

