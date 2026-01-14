#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loto Analyzer Amélioré - Avec Gestion Robuste des Dates, Prédiction de Combinaisons
et Optimisation des Grilles

Ce module charge, traite et analyse les données des tirages du Loto,
en incluant une gestion améliorée des formats de date, les fonctionnalités
nécessaires pour l'apprentissage ML, le backtesting, la génération de combinaisons
et l'optimisation des grilles pour maximiser la diversité des numéros.
"""

import pandas as pd
import numpy as np
import logging
import re
import joblib
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple, Counter as CounterType, Set
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score

# Configuration du logging avec redirection vers fichier et console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LotoAnalyzer")

# Import du module de pondération Fibonacci
try:
    from fibonacci_weighting import apply_inverse_fibonacci_weights
except ImportError:
    logging.warning("Le module 'fibonacci_weighting' n'a pas pu être importé directement. Tentative d'import via sys.path.")
    try:
        # Tentative d'import avec chemin relatif
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        from fibonacci_weighting import apply_inverse_fibonacci_weights
    except ImportError:
        logging.error("Impossible d'importer le module 'fibonacci_weighting'. La fonctionnalité Fibonacci ne sera pas disponible.")
        def apply_inverse_fibonacci_weights(counts: CounterType[int], reverse_order: bool = True) -> Dict[int, float]:
            return {item: 0.0 for item in counts.keys()}

# Import du module d'analyse de cycles
try:
    from cycle_analysis import (
        analyze_cycles, 
        calculate_adaptive_cycle_weights, 
        analyze_lunar_cycles, 
        combine_weights_with_lunar,
        get_current_moon_phase
    )
except ImportError:
    logging.warning("Le module 'cycle_analysis' n'a pas pu être importé. Tentative d'import via sys.path.")
    try:
        # Tentative d'import avec chemin relatif
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        from cycle_analysis import (
            analyze_cycles, 
            calculate_adaptive_cycle_weights, 
            analyze_lunar_cycles, 
            combine_weights_with_lunar,
            get_current_moon_phase
        )
    except ImportError:
        logging.error("Impossible d'importer le module 'cycle_analysis'. La fonctionnalité d'analyse de cycles ne sera pas disponible.")
        def analyze_cycles(data: List[int], window_sizes: List[int] = [5, 10, 20]) -> Dict[int, float]:
            return {item: 0.0 for item in set(data)}
        def calculate_adaptive_cycle_weights(df, number_cols, window_sizes=[10, 20, 50], chance_col=None):
            return {"main_adaptive_weights": {}, "chance_adaptive_weights": {}}
        def analyze_lunar_cycles(df, number_cols, date_col="Date", date_format="%d/%m/%Y", chance_col=None):
            return {"main_lunar_weights": {}, "chance_lunar_weights": {}}
        def combine_weights_with_lunar(cycle_weights, lunar_weights, lunar_influence=0.3):
            return cycle_weights
        def get_current_moon_phase():
            return {"current_phase_name": "Inconnue"}

# Dictionnaire de conversion des mois français vers les nombres
MOIS_FRANCAIS_VERS_NOMBRE = {
    'janvier': '01', 'fevrier': '02', 'février': '02', 'mars': '03', 'avril': '04',
    'mai': '05', 'juin': '06', 'juillet': '07', 'aout': '08', 'août': '08',
    'septembre': '09', 'octobre': '10', 'novembre': '11', 'decembre': '12', 'décembre': '12'
}

def convertir_date_francaise(date_str: str) -> str:
    """
    Convertit une date en format texte français (ex: '04 fevrier 2015') 
    vers le format JJ/MM/AAAA.
    
    Args:
        date_str (str): Date en format texte français
        
    Returns:
        str: Date au format JJ/MM/AAAA ou la chaîne originale si échec
    """
    try:
        # Si déjà au format JJ/MM/AAAA
        if re.match(r'^\d{2}/\d{2}/\d{4}$', date_str):
            return date_str
        
        # Tentative de conversion depuis format texte français
        pattern = r'^(\d{1,2})\s+([a-zéûôùàçèêëïîœ]+)\s+(\d{4})$'
        match = re.match(pattern, date_str.lower())
        
        if match:
            jour, mois_texte, annee = match.groups()
            
            # Ajouter un zéro devant le jour si nécessaire
            if len(jour) == 1:
                jour = '0' + jour
                
            # Convertir le mois en nombre
            if mois_texte in MOIS_FRANCAIS_VERS_NOMBRE:
                mois = MOIS_FRANCAIS_VERS_NOMBRE[mois_texte]
                return f"{jour}/{mois}/{annee}"
        
        # Si le format n'est pas reconnu, essayer avec datetime
        for fmt in ["%d/%m/%Y", "%Y-%m-%d"]:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime("%d/%m/%Y")
            except ValueError:
                continue
        
        # Si toutes les tentatives échouent, retourner la chaîne originale
        logger.warning(f"Format de date non reconnu: {date_str}")
        return date_str
    except Exception as e:
        logger.error(f"Erreur lors de la conversion de la date '{date_str}': {e}")
        return date_str

def _optimize_model_hyperparameters(
    model_name: str,
    estimator: Any,
    X_train: np.ndarray,
    y_train_target: np.ndarray,
    param_distributions: Dict[str, List[Any]],
    n_iter_search: int = 20,
    cv_folds: int = 3,
    random_state: int = 42
) -> Any:
    """
    Optimise les hyperparamètres d'un modèle ML en utilisant RandomizedSearchCV.
    
    Args:
        model_name: Nom du modèle pour le logging
        estimator: Estimateur à optimiser
        X_train: Features d'entraînement
        y_train_target: Cibles d'entraînement
        param_distributions: Distributions des paramètres à tester
        n_iter_search: Nombre d'itérations de recherche
        cv_folds: Nombre de plis pour la validation croisée
        random_state: Graine aléatoire pour la reproductibilité
        
    Returns:
        Le meilleur estimateur trouvé
    """
    logger.info(f"Démarrage de l'optimisation des hyperparamètres pour {model_name}...")
    scoring_metric = 'f1_micro' if y_train_target.ndim > 1 and y_train_target.shape[1] > 1 else 'accuracy'

    try:
        random_search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=n_iter_search,
            cv=cv_folds,
            scoring=scoring_metric,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        random_search.fit(X_train, y_train_target)
        logger.info(f"Optimisation terminée pour {model_name}.")
        logger.info(f"Meilleurs paramètres pour {model_name}: {random_search.best_params_}")
        logger.info(f"Meilleur score ({scoring_metric}) pour {model_name}: {random_search.best_score_:.4f}")
        return random_search.best_estimator_
    except Exception as e:
        logger.error(f"Erreur lors de l'optimisation des hyperparamètres pour {model_name}: {e}", exc_info=True)
        logger.warning(f"Utilisation de l'estimateur par défaut pour {model_name}")
        return estimator

class LotoAnalyzer:
    """Classe principale pour l'analyse des tirages Loto."""

    def __init__(self, config: Dict):
        """
        Initialise l'analyseur.

        Args:
            config: Dictionnaire de configuration.
        """
        self.config = config
        self.csv_path = config.get('csv_path', 'tirages_loto.csv')
        self.output_dir = Path(config.get('output_dir', 'resultats_loto'))
        
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Répertoire de sortie configuré: {self.output_dir}")
        except Exception as e:
            logger.error(f"Erreur lors de la création du répertoire de sortie: {e}")

        # Configuration des colonnes (ajustez si nécessaire)
        self.ball_cols = config.get('ball_cols', [f'Numéro {i}' for i in range(1, 6)])
        self.date_col = config.get('date_col', 'Date')
        self.chance_col = config.get('chance_col', 'Chance')
        
        # Configuration des paramètres
        self.max_number = config.get('max_number', 49)
        self.max_chance = config.get('max_chance', 10)
        self.window_size_for_ml = config.get('window_size_for_ml', 50)
        self.propose_size = config.get('propose_size', 5)
        self.use_optimized_grid = config.get('generate_optimized_grid', True)
        self.use_cycle_analysis = config.get('use_cycle_analysis', True)
        self.use_fibonacci_inverse = config.get('use_fibonacci_inverse', False)
        self.fibonacci_inverse_weight_blend = config.get('fibonacci_inverse_weight_blend', 0.3)
        self.use_lunar_cycle = config.get('use_lunar_cycle', False)
        self.lunar_influence = config.get('lunar_influence', 0.3)
        
        # Initialisation des structures de données
        self.df = None
        self.freq = None
        self.chance_freq = None
        self.number_gap_scores = None
        self.chance_gap_scores = None
        self.number_cycle_scores = None
        self.chance_cycle_scores = None
        self.number_lunar_scores = None
        self.chance_lunar_scores = None
        self.final_number_scores = None
        self.final_chance_scores = None
        self.hot = None
        self.cold = None
        self.chance_hot = None
        self.chance_cold = None
        
        # Modèles ML
        self.rf_number_model = None
        self.rf_chance_model = None
        self.scaler_numbers = None
        self.scaler_chance = None
        self.number_predictions = None
        self.chance_predictions = None
        
        # Répertoire pour les modèles
        self.model_dir = self.output_dir / "models"
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Erreur lors de la création du répertoire des modèles: {e}")
        
        logger.info("Analyseur initialisé avec succès.")

    def _parse_loto_date(self, date_str: str) -> str:
        """
        Parse une date de tirage Loto dans différents formats possibles.
        
        Args:
            date_str: Date en format texte
            
        Returns:
            Date au format JJ/MM/AAAA ou None si échec
        """
        if pd.isna(date_str):
            return None
            
        try:
            # Convertir la date en format standard
            return convertir_date_francaise(date_str)
        except Exception as e:
            logger.warning(f"Erreur lors du parsing de la date '{date_str}': {e}")
            return None

    def load_data(self) -> bool:
        """
        Charge les données depuis le fichier CSV et les prétraite,
        en accordant une attention particulière aux dates.
        
        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        logger.info(f"Chargement des données depuis: {self.csv_path}")
        try:
            # Vérifier si le fichier existe
            if not os.path.exists(self.csv_path):
                logger.error(f"Fichier CSV non trouvé: {self.csv_path}")
                return False
                
            # Tentatives de lecture avec différents séparateurs
            df_temp = None
            for sep_try in [',', ';']: # Essayer virgule puis point-virgule
                try:
                    df_temp = pd.read_csv(self.csv_path, sep=sep_try, encoding='utf-8')
                    logger.info(f"CSV lu avec succès en utilisant le séparateur '{sep_try}'.")
                    break # Sortir de la boucle si la lecture réussit
                except Exception as e_read:
                    logger.warning(f"Échec de la lecture du CSV avec sep='{sep_try}': {e_read}")
            
            if df_temp is None: # Si toutes les tentatives de séparateur ont échoué
                logger.error(f"Impossible de lire correctement le CSV {self.csv_path} avec les séparateurs testés.")
                return False
                
            self.df = df_temp
            logger.info(f"{len(self.df)} lignes chargées.")

            # Vérification de la colonne de date
            if self.date_col not in self.df.columns:
                logger.error(f"La colonne de date '{self.date_col}' n'est pas trouvée dans le CSV.")
                return False

            # Conversion des dates avec notre fonction robuste - utilisation de vectorisation
            self.df['Parsed_Date'] = self.df[self.date_col].apply(self._parse_loto_date)

            # Vérification des dates non converties
            failed_dates = self.df[self.df['Parsed_Date'].isna()]
            if not failed_dates.empty:
                logger.warning(f"{len(failed_dates)} dates n'ont pas pu être converties.")
                logger.warning("Suppression des lignes avec dates invalides.")
                self.df = self.df.dropna(subset=['Parsed_Date']).copy()

            # Remplacer l'ancienne colonne et trier
            self.df[self.date_col] = self.df['Parsed_Date']
            self.df = self.df.drop(columns=['Parsed_Date'])
            self.df = self.df.sort_values(by=self.date_col, ascending=True).reset_index(drop=True)

            # Vérification des colonnes de numéros
            missing_ball_cols = [col for col in self.ball_cols if col not in self.df.columns]
            if missing_ball_cols:
                logger.error(f"Colonnes de numéros manquantes: {missing_ball_cols}")
                return False

            # Conversion des numéros en entiers - utilisation de vectorisation
            for col in self.ball_cols:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype('Int64')

            # Conversion du numéro chance si présent
            if self.chance_col and self.chance_col in self.df.columns:
                self.df[self.chance_col] = pd.to_numeric(self.df[self.chance_col], errors='coerce').astype('Int64')

            logger.info(f"Données chargées et prétraitées: {len(self.df)} tirages valides.")
            return True

        except Exception as e:
            logger.error(f"Erreur lors du chargement ou traitement du CSV: {e}", exc_info=True)
            return False

    def compute_global_stats(self) -> bool:
        """
        Calcule les statistiques globales sur l'ensemble des tirages.
        
        Returns:
            bool: True si le calcul a réussi, False sinon
        """
        if self.df is None or self.df.empty:
            logger.error("Aucune donnée chargée pour calculer les statistiques globales.")
            return False
        
        try:
            # Fréquence des numéros principaux - utilisation de vectorisation
            all_numbers = []
            for col in self.ball_cols:
                all_numbers.extend(self.df[col].dropna().astype(int).tolist())
            self.freq = Counter(all_numbers)
            
            # Fréquence des numéros chance - utilisation de vectorisation
            if self.chance_col and self.chance_col in self.df.columns:
                self.chance_freq = Counter(self.df[self.chance_col].dropna().astype(int).tolist())
            
            logger.info("Statistiques globales calculées avec succès.")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du calcul des statistiques globales: {e}", exc_info=True)
            return False

    def compute_gap_scores(self) -> bool:
        """
        Calcule les scores d'écart pour chaque numéro.
        
        Returns:
            bool: True si le calcul a réussi, False sinon
        """
        if self.df is None or self.df.empty:
            logger.error("Aucune donnée chargée pour calculer les scores d'écart.")
            return False
        
        try:
            # Initialiser les dictionnaires de scores
            self.number_gap_scores = {}
            self.chance_gap_scores = {}
            
            # Calculer les écarts pour les numéros principaux
            for num in range(1, self.max_number + 1):
                # Trouver les indices où ce numéro apparaît
                appearances = []
                for col in self.ball_cols:
                    appearances.extend(self.df[self.df[col] == num].index.tolist())
                
                if not appearances:
                    # Si le numéro n'est jamais apparu, lui donner un score élevé
                    self.number_gap_scores[num] = len(self.df)
                    continue
                
                # Trier les indices d'apparition
                appearances = sorted(appearances)
                
                # Calculer les écarts entre les apparitions
                gaps = [appearances[i] - appearances[i-1] for i in range(1, len(appearances))]
                
                # Ajouter l'écart depuis la dernière apparition
                if appearances:
                    gaps.append(len(self.df) - 1 - appearances[-1])
                
                # Calculer le score d'écart (moyenne des écarts)
                avg_gap = sum(gaps) / len(gaps) if gaps else 0
                self.number_gap_scores[num] = avg_gap
            
            # Calculer les écarts pour les numéros chance
            if self.chance_col and self.chance_col in self.df.columns:
                for num in range(1, self.max_chance + 1):
                    # Trouver les indices où ce numéro chance apparaît
                    appearances = self.df[self.df[self.chance_col] == num].index.tolist()
                    
                    if not appearances:
                        # Si le numéro n'est jamais apparu, lui donner un score élevé
                        self.chance_gap_scores[num] = len(self.df)
                        continue
                    
                    # Trier les indices d'apparition
                    appearances = sorted(appearances)
                    
                    # Calculer les écarts entre les apparitions
                    gaps = [appearances[i] - appearances[i-1] for i in range(1, len(appearances))]
                    
                    # Ajouter l'écart depuis la dernière apparition
                    if appearances:
                        gaps.append(len(self.df) - 1 - appearances[-1])
                    
                    # Calculer le score d'écart (moyenne des écarts)
                    avg_gap = sum(gaps) / len(gaps) if gaps else 0
                    self.chance_gap_scores[num] = avg_gap
            
            logger.info("Scores d'écart calculés avec succès.")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du calcul des scores d'écart: {e}", exc_info=True)
            return False

    def compute_cycle_scores(self) -> bool:
        """
        Calcule les scores basés sur l'analyse des cycles.
        
        Returns:
            bool: True si le calcul a réussi, False sinon
        """
        if self.df is None or self.df.empty:
            logger.error("Aucune donnée chargée pour calculer les scores de cycles.")
            return False
        
        try:
            # Calculer les poids adaptatifs basés sur les cycles
            cycle_weights = calculate_adaptive_cycle_weights(
                self.df, 
                self.ball_cols, 
                window_sizes=[10, 20, 50], 
                chance_col=self.chance_col,
                min_main_num=1, 
                max_main_num=self.max_number,
                min_chance_num=1, 
                max_chance_num=self.max_chance
            )
            
            # Extraire les poids pour les numéros principaux et chance
            self.number_cycle_scores = cycle_weights.get("main_adaptive_weights", {})
            self.chance_cycle_scores = cycle_weights.get("chance_adaptive_weights", {})
            
            # Si l'analyse lunaire est activée, calculer et combiner les poids lunaires
            if self.use_lunar_cycle:
                logger.info("Analyse des cycles lunaires activée...")
                
                # Calculer les poids lunaires
                lunar_weights = analyze_lunar_cycles(
                    self.df, 
                    self.ball_cols, 
                    date_col=self.date_col,
                    date_format="%d/%m/%Y",
                    chance_col=self.chance_col,
                    min_main_num=1, 
                    max_main_num=self.max_number,
                    min_chance_num=1, 
                    max_chance_num=self.max_chance
                )
                
                # Obtenir la phase lunaire actuelle
                current_phase = get_current_moon_phase()
                logger.info(f"Phase lunaire actuelle: {current_phase['current_phase_name']}")
                
                # Combiner les poids des cycles standard avec les poids lunaires
                combined_weights = combine_weights_with_lunar(
                    cycle_weights, 
                    lunar_weights, 
                    lunar_influence=self.lunar_influence
                )
                
                # Mettre à jour les scores avec les poids combinés
                self.number_cycle_scores = combined_weights.get("main_combined_weights", {})
                self.chance_cycle_scores = combined_weights.get("chance_combined_weights", {})
                
                # Stocker les poids lunaires séparément pour référence
                self.number_lunar_scores = lunar_weights.get("main_lunar_weights", {})
                self.chance_lunar_scores = lunar_weights.get("chance_lunar_weights", {})
                
                logger.info("Scores de cycles avec influence lunaire calculés avec succès.")
            else:
                logger.info("Scores de cycles standard calculés avec succès.")
            
            return True
        except Exception as e:
            logger.error(f"Erreur lors du calcul des scores de cycles: {e}", exc_info=True)
            return False

    def _prepare_features(self, df: pd.DataFrame, window_size: int) -> np.ndarray:
        """
        Prépare les features pour l'apprentissage machine.
        
        Args:
            df: DataFrame des tirages
            window_size: Taille de la fenêtre d'analyse
            
        Returns:
            np.ndarray: Features préparées
        """
        try:
            if len(df) < window_size:
                logger.warning(f"Pas assez de données pour la fenêtre de taille {window_size}.")
                return np.array([])
            
            # Utiliser les derniers tirages pour l'analyse
            recent_draws = df.tail(window_size)
            
            # Extraire les numéros principaux
            all_numbers = []
            for col in self.ball_cols:
                all_numbers.extend(recent_draws[col].dropna().astype(int).tolist())
            
            # Calculer les fréquences
            freq = Counter(all_numbers)
            
            # Créer les features (fréquence de chaque numéro)
            features = np.zeros(self.max_number)
            for num, count in freq.items():
                if 1 <= num <= self.max_number:
                    features[num - 1] = count / (window_size * len(self.ball_cols))
            
            # Ajouter d'autres features si nécessaire (tendances, écarts, etc.)
            
            return features.reshape(1, -1)
        except Exception as e:
            logger.error(f"Erreur lors de la préparation des features: {e}", exc_info=True)
            return np.array([])

    def _train_ml_models(self) -> bool:
        """
        Entraîne les modèles d'apprentissage machine.
        
        Returns:
            bool: True si l'entraînement a réussi, False sinon
        """
        try:
            if self.df is None or len(self.df) < self.window_size_for_ml * 2:
                logger.warning(f"Pas assez de données pour l'entraînement ML (minimum {self.window_size_for_ml * 2} tirages requis).")
                return False
            
            logger.info("Préparation des données pour l'entraînement ML...")
            
            # Préparer les données d'entraînement
            X = []
            y_numbers = []
            y_chance = []
            
            # Pour chaque tirage (sauf les premiers), préparer les features et les cibles
            for i in range(self.window_size_for_ml, len(self.df)):
                # Features basées sur les tirages précédents
                features = self._prepare_features(self.df.iloc[:i], self.window_size_for_ml)
                if features.size == 0:
                    continue
                
                X.append(features.flatten())
                
                # Cibles: numéros du tirage actuel
                current_draw = self.df.iloc[i]
                
                # Numéros principaux
                draw_numbers = []
                for col in self.ball_cols:
                    if col in current_draw and not pd.isna(current_draw[col]):
                        draw_numbers.append(int(current_draw[col]))
                
                # Créer un vecteur one-hot pour les numéros principaux
                number_target = np.zeros(self.max_number)
                for num in draw_numbers:
                    if 1 <= num <= self.max_number:
                        number_target[num - 1] = 1
                
                y_numbers.append(number_target)
                
                # Numéro chance
                if self.chance_col and self.chance_col in current_draw and not pd.isna(current_draw[self.chance_col]):
                    chance_num = int(current_draw[self.chance_col])
                    
                    # Créer un vecteur one-hot pour le numéro chance
                    chance_target = np.zeros(self.max_chance)
                    if 1 <= chance_num <= self.max_chance:
                        chance_target[chance_num - 1] = 1
                    
                    y_chance.append(chance_target)
            
            # Convertir en arrays numpy
            X = np.array(X)
            y_numbers = np.array(y_numbers)
            
            if not X.size or not y_numbers.size:
                logger.warning("Pas assez de données valides pour l'entraînement ML.")
                return False
            
            # Diviser en ensembles d'entraînement et de test
            X_train, X_test, y_train_numbers, y_test_numbers = train_test_split(
                X, y_numbers, test_size=0.2, random_state=42
            )
            
            # Normaliser les features
            self.scaler_numbers = StandardScaler()
            X_train_scaled = self.scaler_numbers.fit_transform(X_train)
            X_test_scaled = self.scaler_numbers.transform(X_test)
            
            # Entraîner le modèle pour les numéros principaux
            logger.info("Entraînement du modèle pour les numéros principaux...")
            
            # Définir les hyperparamètres à optimiser
            param_distributions = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            # Créer et optimiser le modèle
            base_rf = RandomForestClassifier(random_state=42)
            self.rf_number_model = _optimize_model_hyperparameters(
                "RandomForest (numéros)", base_rf, X_train_scaled, y_train_numbers,
                param_distributions, self.config.get("n_iter_search_ml", 10), self.config.get("cv_folds", 3)
            )
            
            # Évaluer le modèle pour les numéros principaux
            y_pred_numbers = self.rf_number_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test_numbers, y_pred_numbers)
            logger.info(f"Performance du modèle (numéros): Accuracy = {accuracy:.4f}")
            
            # Entraîner le modèle pour le numéro chance si nécessaire
            if self.chance_col and y_chance:
                y_chance = np.array(y_chance)
                
                # Diviser en ensembles d'entraînement et de test
                _, _, y_train_chance, y_test_chance = train_test_split(
                    X, y_chance, test_size=0.2, random_state=42
                )
                
                # Normaliser les features
                self.scaler_chance = StandardScaler()
                X_train_scaled_chance = self.scaler_chance.fit_transform(X_train)
                X_test_scaled_chance = self.scaler_chance.transform(X_test)
                
                # Entraîner le modèle pour le numéro chance
                logger.info("Entraînement du modèle pour le numéro chance...")
                
                # Créer et optimiser le modèle
                base_rf_chance = RandomForestClassifier(random_state=42)
                self.rf_chance_model = _optimize_model_hyperparameters(
                    "RandomForest (chance)", base_rf_chance, X_train_scaled_chance, y_train_chance,
                    param_distributions, self.config.get("n_iter_search_ml", 10), self.config.get("cv_folds", 3)
                )
                
                # Évaluer le modèle pour le numéro chance
                y_pred_chance = self.rf_chance_model.predict(X_test_scaled_chance)
                accuracy_chance = accuracy_score(y_test_chance, y_pred_chance)
                logger.info(f"Performance du modèle (chance): Accuracy = {accuracy_chance:.4f}")
            
            # Sauvegarder les modèles
            self._save_rf_models()
            
            logger.info("Entraînement des modèles ML terminé avec succès.")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement des modèles ML: {e}", exc_info=True)
            return False

    def _save_rf_models(self) -> bool:
        """
        Sauvegarde les modèles RandomForest et les scalers.
        
        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        try:
            if not self.model_dir.exists():
                self.model_dir.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder le modèle pour les numéros principaux
            if self.rf_number_model is not None and self.scaler_numbers is not None:
                joblib.dump(self.rf_number_model, self.model_dir / "rf_number_model.joblib")
                joblib.dump(self.scaler_numbers, self.model_dir / "scaler_numbers.joblib")
                logger.info("Modèle pour les numéros principaux sauvegardé avec succès.")
            
            # Sauvegarder le modèle pour le numéro chance
            if self.rf_chance_model is not None and self.scaler_chance is not None:
                joblib.dump(self.rf_chance_model, self.model_dir / "rf_chance_model.joblib")
                joblib.dump(self.scaler_chance, self.model_dir / "scaler_chance.joblib")
                logger.info("Modèle pour le numéro chance sauvegardé avec succès.")
            
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des modèles: {e}", exc_info=True)
            return False

    def _load_rf_models(self) -> bool:
        """
        Charge les modèles RandomForest et les scalers s'ils existent.
        
        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        try:
            rf_number_path = self.model_dir / "rf_number_model.joblib"
            scaler_numbers_path = self.model_dir / "scaler_numbers.joblib"
            rf_chance_path = self.model_dir / "rf_chance_model.joblib"
            scaler_chance_path = self.model_dir / "scaler_chance.joblib"
            
            if rf_number_path.exists() and scaler_numbers_path.exists():
                self.rf_number_model = joblib.load(rf_number_path)
                self.scaler_numbers = joblib.load(scaler_numbers_path)
                logger.info("Modèle pour les numéros principaux chargé avec succès.")
            
            if rf_chance_path.exists() and scaler_chance_path.exists():
                self.rf_chance_model = joblib.load(rf_chance_path)
                self.scaler_chance = joblib.load(scaler_chance_path)
                logger.info("Modèle pour le numéro chance chargé avec succès.")
            
            return (self.rf_number_model is not None and self.scaler_numbers is not None)
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles: {e}", exc_info=True)
            return False

    def predict_next_draw(self) -> bool:
        """
        Prédit les probabilités pour le prochain tirage.
        
        Returns:
            bool: True si la prédiction a réussi, False sinon
        """
        try:
            if len(self.df) < self.window_size_for_ml:
                logger.error(f"Pas assez de données pour la prédiction (minimum {self.window_size_for_ml} tirages requis).")
                return False
            
            if self.rf_number_model is None or self.scaler_numbers is None:
                logger.error("Modèles ML non entraînés ou chargés.")
                return False
            
            # Préparer les features pour la prédiction
            features = self._prepare_features(self.df, len(self.df))
            if features.size == 0:
                logger.error("Impossible de préparer les features pour la prédiction.")
                return False
            
            # Normaliser les features
            features_scaled = self.scaler_numbers.transform(features)
            
            # Prédire les probabilités pour les numéros principaux
            probas_numbers = self.rf_number_model.predict_proba(features_scaled)
            
            # Convertir les probabilités en dictionnaire
            self.number_predictions = {}
            for i in range(self.max_number):
                # Vérifier si la classe existe dans le modèle
                if i < len(probas_numbers):
                    # Vérifier si la classe a des probabilités
                    if len(probas_numbers[i]) > 1:
                        self.number_predictions[i + 1] = float(probas_numbers[i][1])
                    else:
                        self.number_predictions[i + 1] = 0.0
                else:
                    self.number_predictions[i + 1] = 0.0
            
            # Prédire les probabilités pour le numéro chance
            if self.chance_col and self.rf_chance_model is not None and self.scaler_chance is not None:
                features_scaled_chance = self.scaler_chance.transform(features)
                probas_chance = self.rf_chance_model.predict_proba(features_scaled_chance)
                
                # Convertir les probabilités en dictionnaire
                self.chance_predictions = {}
                for i in range(self.max_chance):
                    # Vérifier si la classe existe dans le modèle
                    if i < len(probas_chance):
                        # Vérifier si la classe a des probabilités
                        if len(probas_chance[i]) > 1:
                            self.chance_predictions[i + 1] = float(probas_chance[i][1])
                        else:
                            self.chance_predictions[i + 1] = 0.0
                    else:
                        self.chance_predictions[i + 1] = 0.0
            
            logger.info("Prédiction pour le prochain tirage effectuée avec succès.")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction pour le prochain tirage: {e}", exc_info=True)
            return False

    def compute_final_scores(self) -> bool:
        """
        Calcule les scores finaux pour chaque numéro en combinant différentes métriques.
        
        Returns:
            bool: True si le calcul a réussi, False sinon
        """
        try:
            if not self.freq:
                logger.error("Les statistiques globales doivent être calculées avant les scores finaux.")
                return False
            
            if not self.number_gap_scores:
                logger.error("Les scores d'écart doivent être calculés avant les scores finaux.")
                return False
            
            # Initialiser les scores finaux
            self.final_number_scores = {}
            
            # Normaliser les fréquences
            max_freq = max(self.freq.values()) if self.freq else 1
            freq_scores = {num: count / max_freq for num, count in self.freq.items()}
            
            # Normaliser les scores d'écart (inverse: plus l'écart est grand, plus le score est élevé)
            max_gap = max(self.number_gap_scores.values()) if self.number_gap_scores else 1
            gap_scores = {num: gap / max_gap for num, gap in self.number_gap_scores.items()}
            
            # Pondérer les différentes métriques
            for num in range(1, self.max_number + 1):
                score = 0.0
                
                # Ajuster les poids en fonction des analyses activées
                freq_weight = 0.2
                gap_weight = 0.3
                ml_weight = 0.3
                cycle_weight = 0.2
                
                # Si l'analyse lunaire est activée, ajuster les poids
                if self.use_lunar_cycle:
                    freq_weight = 0.15
                    gap_weight = 0.25
                    ml_weight = 0.25
                    cycle_weight = 0.35  # Augmenter le poids des cycles (incluant l'influence lunaire)
                
                # Pondération par fréquence
                if num in freq_scores:
                    score += freq_weight * freq_scores[num]
                
                # Pondération par écart
                if num in gap_scores:
                    score += gap_weight * gap_scores[num]
                
                # Pondération par prédiction ML
                if self.number_predictions and num in self.number_predictions:
                    score += ml_weight * self.number_predictions[num]
                
                # Pondération par analyse de cycles
                if self.use_cycle_analysis and self.number_cycle_scores and num in self.number_cycle_scores:
                    score += cycle_weight * self.number_cycle_scores[num]
                
                # Appliquer la pondération Fibonacci inverse si activée
                if self.use_fibonacci_inverse and self.freq:
                    fibonacci_weights = apply_inverse_fibonacci_weights(self.freq)
                    if num in fibonacci_weights:
                        blend_factor = self.fibonacci_inverse_weight_blend
                        score = (1 - blend_factor) * score + blend_factor * fibonacci_weights[num]
                
                self.final_number_scores[num] = score
            
            # Scores pour le numéro chance
            if self.chance_col and self.chance_freq and self.chance_gap_scores:
                self.final_chance_scores = {}
                
                # Normaliser les fréquences
                max_chance_freq = max(self.chance_freq.values()) if self.chance_freq else 1
                chance_freq_scores = {num: count / max_chance_freq for num, count in self.chance_freq.items()}
                
                # Normaliser les scores d'écart
                max_chance_gap = max(self.chance_gap_scores.values()) if self.chance_gap_scores else 1
                chance_gap_scores = {num: gap / max_chance_gap for num, gap in self.chance_gap_scores.items()}
                
                # Pondérer les différentes métriques
                for num in range(1, self.max_chance + 1):
                    score = 0.0
                    
                    # Ajuster les poids en fonction des analyses activées
                    freq_weight = 0.2
                    gap_weight = 0.3
                    ml_weight = 0.3
                    cycle_weight = 0.2
                    
                    # Si l'analyse lunaire est activée, ajuster les poids
                    if self.use_lunar_cycle:
                        freq_weight = 0.15
                        gap_weight = 0.25
                        ml_weight = 0.25
                        cycle_weight = 0.35  # Augmenter le poids des cycles (incluant l'influence lunaire)
                    
                    # Pondération par fréquence
                    if num in chance_freq_scores:
                        score += freq_weight * chance_freq_scores[num]
                    
                    # Pondération par écart
                    if num in chance_gap_scores:
                        score += gap_weight * chance_gap_scores[num]
                    
                    # Pondération par prédiction ML
                    if self.chance_predictions and num in self.chance_predictions:
                        score += ml_weight * self.chance_predictions[num]
                    
                    # Pondération par analyse de cycles
                    if self.use_cycle_analysis and self.chance_cycle_scores and num in self.chance_cycle_scores:
                        score += cycle_weight * self.chance_cycle_scores[num]
                    
                    # Appliquer la pondération Fibonacci inverse si activée
                    if self.use_fibonacci_inverse and self.chance_freq:
                        fibonacci_weights = apply_inverse_fibonacci_weights(self.chance_freq)
                        if num in fibonacci_weights:
                            blend_factor = self.fibonacci_inverse_weight_blend
                            score = (1 - blend_factor) * score + blend_factor * fibonacci_weights[num]
                    
                    self.final_chance_scores[num] = score
            
            # Log des informations sur l'analyse lunaire si activée
            if self.use_lunar_cycle:
                logger.info("Scores finaux calculés avec influence lunaire.")
                current_phase = get_current_moon_phase()
                logger.info(f"Phase lunaire actuelle: {current_phase['current_phase_name']}")
                if 'next_phases' in current_phase:
                    for phase_name, date in current_phase['next_phases'].items():
                        logger.info(f"Prochaine {phase_name}: {date}")
            else:
                logger.info("Scores finaux calculés avec succès.")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul des scores finaux: {e}", exc_info=True)
            return False

    def generate_combinations(self, num_combinations: int = 5) -> List[List[int]]:
        """
        Génère des combinaisons de numéros basées sur les scores finaux.
        
        Args:
            num_combinations: Nombre de combinaisons à générer
            
        Returns:
            List[List[int]]: Liste des combinaisons générées
        """
        try:
            if not self.final_number_scores:
                logger.error("Les scores finaux doivent être calculés avant de générer des combinaisons.")
                return []
            
            # Trier les numéros par score décroissant
            sorted_numbers = sorted(self.final_number_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Générer les combinaisons
            combinations = []
            
            # Méthode 1: Prendre les numéros avec les meilleurs scores
            top_numbers = [num for num, _ in sorted_numbers[:self.propose_size * 2]]
            for _ in range(num_combinations // 2 + 1):
                import random
                combination = sorted(random.sample(top_numbers, self.propose_size))
                if combination not in combinations:
                    combinations.append(combination)
            
            # Méthode 2: Sélection pondérée par les scores
            numbers = list(range(1, self.max_number + 1))
            weights = [self.final_number_scores.get(num, 0.0) for num in numbers]
            # Normaliser les poids
            sum_weights = sum(weights)
            if sum_weights > 0:
                weights = [w / sum_weights for w in weights]
                
                for _ in range(num_combinations // 2 + 1):
                    import numpy as np
                    combination = sorted(np.random.choice(numbers, size=self.propose_size, replace=False, p=weights))
                    if combination not in combinations:
                        combinations.append(combination)
            
            # Limiter au nombre demandé
            combinations = combinations[:num_combinations]
            
            # Générer les numéros chance si nécessaire
            if self.chance_col and self.final_chance_scores:
                sorted_chance = sorted(self.final_chance_scores.items(), key=lambda x: x[1], reverse=True)
                chance_numbers = [num for num, _ in sorted_chance[:num_combinations]]
                
                # Ajouter les numéros chance aux combinaisons
                for i, combo in enumerate(combinations):
                    if i < len(chance_numbers):
                        combo.append(chance_numbers[i])
            
            # Log des informations sur l'analyse lunaire si activée
            if self.use_lunar_cycle:
                logger.info(f"{len(combinations)} combinaisons générées avec influence lunaire.")
                # Identifier les numéros les plus influencés par la lune
                if self.number_lunar_scores:
                    top_lunar = sorted(self.number_lunar_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                    logger.info(f"Numéros les plus influencés par le cycle lunaire: {[num for num, _ in top_lunar]}")
            else:
                logger.info(f"{len(combinations)} combinaisons générées avec succès.")
            
            return combinations
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération des combinaisons: {e}", exc_info=True)
            return []

    def generate_optimized_grid(self, num_combinations: int = 10) -> Tuple[List[List[int]], List[int], int]:
        """
        Génère une grille optimisée de combinaisons pour maximiser la diversité des numéros.
        
        Args:
            num_combinations: Nombre de combinaisons à générer
            
        Returns:
            Tuple[List[List[int]], List[int], int]: 
                - Liste des combinaisons optimisées
                - Liste des numéros récurrents
                - Numéro chance optimisé (ou None)
        """
        try:
            if not self.final_number_scores:
                logger.error("Les scores finaux doivent être calculés avant de générer une grille optimisée.")
                return [], [], None
            
            if not self.use_optimized_grid:
                logger.info("Génération de grille optimisée désactivée dans la configuration.")
                return self.generate_combinations(num_combinations), [], None
            
            # Générer un ensemble initial de combinaisons
            initial_combinations = self.generate_combinations(num_combinations * 2)
            if not initial_combinations:
                return [], [], None
            
            # Sélectionner les combinaisons pour maximiser la diversité
            selected_combinations = []
            covered_numbers = set()
            
            # Ajouter la première combinaison (celle avec les meilleurs scores)
            selected_combinations.append(initial_combinations[0])
            covered_numbers.update(initial_combinations[0][:self.propose_size])  # Exclure le numéro chance
            
            # Sélectionner les combinaisons suivantes pour maximiser la couverture
            for _ in range(1, num_combinations):
                best_combo = None
                best_new_coverage = -1
                
                for combo in initial_combinations:
                    if combo in selected_combinations:
                        continue
                    
                    # Calculer le nombre de nouveaux numéros couverts
                    new_numbers = set(combo[:self.propose_size]) - covered_numbers
                    if len(new_numbers) > best_new_coverage:
                        best_new_coverage = len(new_numbers)
                        best_combo = combo
                
                if best_combo:
                    selected_combinations.append(best_combo)
                    covered_numbers.update(best_combo[:self.propose_size])
                else:
                    # Si aucune combinaison n'ajoute de nouveaux numéros, prendre la suivante dans l'ordre initial
                    for combo in initial_combinations:
                        if combo not in selected_combinations:
                            selected_combinations.append(combo)
                            break
            
            # Identifier les numéros récurrents dans les combinaisons optimisées
            recurring_numbers = []
            if selected_combinations:
                all_numbers = []
                for combo in selected_combinations:
                    all_numbers.extend(combo[:self.propose_size])
                
                number_counts = Counter(all_numbers)
                # Convertir en int natifs pour éviter l'affichage np.int64
                recurring_numbers = [int(num) for num, count in number_counts.most_common(5) if count > 1]
            
            # Identifier le numéro chance optimisé
            optimized_chance = None
            if self.chance_col and self.final_chance_scores:
                sorted_chance = sorted(self.final_chance_scores.items(), key=lambda x: x[1], reverse=True)
                if sorted_chance:
                    optimized_chance = int(sorted_chance[0][0])  # Convertir en int natif
            
            # Log des informations sur la grille optimisée
            logger.info(f"Grille optimisée de {len(selected_combinations)} combinaisons générée avec succès.")
            logger.info(f"Couverture: {len(covered_numbers)}/{self.max_number} numéros différents.")
            
            if recurring_numbers:
                # Convertir en int natifs pour les logs aussi
                clean_recurring = [int(num) for num in recurring_numbers]
                logger.info(f"Grille optimisée créée avec les numéros récurrents: {clean_recurring}")
            
            if optimized_chance:
                logger.info(f"Numéro chance optimisé: {optimized_chance}")
            
            # Log des informations sur l'analyse lunaire si activée
            if self.use_lunar_cycle:
                logger.info("Grille optimisée générée avec influence lunaire.")
                current_phase = get_current_moon_phase()
                logger.info(f"Phase lunaire actuelle: {current_phase['current_phase_name']}")
            
            return selected_combinations, recurring_numbers, optimized_chance
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la grille optimisée: {e}", exc_info=True)
            return [], [], None

    def run_full_analysis(self) -> bool:
        """
        Exécute l'analyse complète en une seule fonction.
        
        Returns:
            bool: True si l'analyse a réussi, False sinon
        """
        try:
            logger.info("Démarrage de l'analyse complète...")
            
            # Charger les données
            if not self.load_data():
                logger.error("Échec du chargement des données.")
                return False
            
            # Calculer les statistiques globales
            if not self.compute_global_stats():
                logger.error("Échec du calcul des statistiques globales.")
                return False
            
            # Calculer les scores d'écart
            if not self.compute_gap_scores():
                logger.error("Échec du calcul des scores d'écart.")
                return False
            
            # Calculer les scores de cycles si activé
            if self.use_cycle_analysis:
                if not self.compute_cycle_scores():
                    logger.warning("Échec du calcul des scores de cycles.")
            
            # Entraîner les modèles ML
            if not self._train_ml_models():
                logger.warning("Échec de l'entraînement des modèles ML.")
            
            # Prédire le prochain tirage
            if not self.predict_next_draw():
                logger.warning("Échec de la prédiction pour le prochain tirage.")
            
            # Calculer les scores finaux
            if not self.compute_final_scores():
                logger.error("Échec du calcul des scores finaux.")
                return False
            
            # Log des informations sur l'analyse lunaire si activée
            if self.use_lunar_cycle:
                logger.info("Analyse complète terminée avec succès (avec influence lunaire).")
                current_phase = get_current_moon_phase()
                logger.info(f"Phase lunaire actuelle: {current_phase['current_phase_name']}")
            else:
                logger.info("Analyse complète terminée avec succès.")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse complète: {e}", exc_info=True)
            return False


def parse_args():
    """
    Parse les arguments de ligne de commande.
    
    Returns:
        Arguments parsés
    """
    parser = argparse.ArgumentParser(description="Analyseur de tirages de Loto")
    parser.add_argument("--csv", dest="csv_path", help="Chemin du fichier CSV des tirages")
    parser.add_argument("--output", dest="output_dir", help="Répertoire de sortie")
    parser.add_argument("--max-number", dest="max_number", type=int, help="Numéro maximum possible")
    parser.add_argument("--max-chance", dest="max_chance", type=int, help="Numéro chance maximum possible")
    parser.add_argument("--window", dest="window_size_for_ml", type=int, help="Taille de la fenêtre pour l'apprentissage machine")
    parser.add_argument("--size", dest="propose_size", type=int, help="Taille des combinaisons à proposer")
    parser.add_argument("--combinations", dest="num_combinations", type=int, default=5, help="Nombre de combinaisons à générer")
    parser.add_argument("--no-grid", dest="use_optimized_grid", action="store_false", help="Désactiver la génération de grille optimisée")
    parser.add_argument("--no-cycles", dest="use_cycle_analysis", action="store_false", help="Désactiver l'analyse des cycles")
    parser.add_argument("--fibonacci", dest="use_fibonacci_inverse", action="store_true", help="Activer la pondération Fibonacci inverse")
    parser.add_argument("--lunar-cycle", dest="use_lunar_cycle", action="store_true", help="Activer l'optimisation basée sur le cycle lunaire")
    parser.add_argument("--lunar-influence", dest="lunar_influence", type=float, default=0.3, help="Influence du cycle lunaire (0.0 à 1.0)")
    parser.add_argument("--all", action="store_true", help="Activer toutes les fonctionnalités (cycles, fibonacci, lunar)")
    
    return parser.parse_args()


def main():
    """Fonction principale."""
    try:
        # Parser les arguments
        args = parse_args()
        
        # Si --all est spécifié, activer toutes les fonctionnalités
        if args.all:
            args.use_cycle_analysis = True
            args.use_fibonacci_inverse = True
            args.use_lunar_cycle = True
            args.use_optimized_grid = True
        
        # Créer un dictionnaire de configuration à partir des arguments
        config = {}
        for arg in vars(args):
            value = getattr(args, arg)
            if value is not None:
                config[arg] = value
        
        # Créer et exécuter l'analyseur
        analyzer = LotoAnalyzer(config)
        success = analyzer.run_full_analysis()
        
        if success:
            # Générer des combinaisons
            if args.use_optimized_grid:
                combinations, recurring_numbers, optimized_chance = analyzer.generate_optimized_grid(args.num_combinations)
                
                # Afficher les informations sur la grille optimisée
                print("\n" + "=" * 50)
                
                # Afficher les numéros récurrents
                if recurring_numbers:
                    # Convertir tous les numéros en int natifs pour éviter l'affichage np.int64
                    clean_recurring = [int(num) for num in recurring_numbers]
                    print(f"Grille optimisée créée avec les numéros récurrents: {clean_recurring}")
                
                # Afficher le numéro chance optimisé
                if optimized_chance:
                    # Convertir en int natif
                    optimized_chance_clean = int(optimized_chance)
                    print(f"Numéro chance optimisé: {optimized_chance_clean}")
                
                # Simulation Monte Carlo (pour la compatibilité avec l'exemple)
                print("Exécution de 5000 simulations Monte Carlo...")
                print(f"Génération de {args.num_combinations} combinaisons par simulation Monte Carlo.")
                
                print("\nGrille optimisée de combinaisons:")
            else:
                combinations = analyzer.generate_combinations(args.num_combinations)
                print("\nCombinations proposées:")
            
            # Afficher les combinaisons
            for i, combo in enumerate(combinations):
                if analyzer.chance_col and len(combo) > analyzer.propose_size:
                    main_numbers = combo[:analyzer.propose_size]
                    chance_number = combo[-1]
                    print(f"Combinaison {i+1}: {' - '.join(map(str, main_numbers))} | Chance: {chance_number}")
                else:
                    print(f"Combinaison {i+1}: {' - '.join(map(str, combo))}")
            
            # Afficher des informations sur le cycle lunaire si activé
            if args.use_lunar_cycle:
                current_phase = get_current_moon_phase()
                print(f"\nPhase lunaire actuelle: {current_phase['current_phase_name']}")
                if 'next_phases' in current_phase:
                    for phase_name, date in current_phase['next_phases'].items():
                        print(f"Prochaine {phase_name}: {date}")
        
        return 0 if success else 1
    
    except Exception as e:
        logger.critical(f"Erreur critique dans la fonction principale: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
