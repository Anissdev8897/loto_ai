#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyseur de tirages de Loto - Version améliorée
Ce script analyse les statistiques des tirages de loto et propose des combinaisons.
Il intègre des fonctionnalités avancées comme l'analyse de fréquence, l'apprentissage
machine, et la génération de combinaisons optimisées.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Set, Counter as CounterType, Optional, Union, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from pathlib import Path
import openpyxl
from openpyxl.chart import Reference, BarChart
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import joblib
import traceback
import random
import locale  # Pour gérer les langues (dates)

# Configuration du logging (déplacée en haut)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("LotoAnalyzer") # Initialisation de l'instance du logger après basicConfig

# Import du module de pondération Fibonacci
try:
    from fibonacci_weighting import apply_inverse_fibonacci_weights
except ImportError:
    logging.warning("Le module 'fibonacci_weighting' n'a pas pu être importé directement. Tentative d'import via sys.path.")
    try:
        from fibonacci_weighting import apply_inverse_fibonacci_weights
    except ImportError:
        logging.error("Impossible d'importer le module 'fibonacci_weighting'. La fonctionnalité Fibonacci ne sera pas disponible.")
        def apply_inverse_fibonacci_weights(counts: CounterType[int], reverse_order: bool = True) -> Dict[int, float]:
            return {item: 0.0 for item in counts.keys()}

# Dictionnaire de conversion des mois français vers les nombres (pour compatibilité avec les autres modules)
MOIS_FRANCAIS_VERS_NOMBRE = {
    'janvier': '01', 'fevrier': '02', 'février': '02', 'mars': '03', 'avril': '04',
    'mai': '05', 'juin': '06', 'juillet': '07', 'aout': '08', 'août': '08',
    'septembre': '09', 'octobre': '10', 'novembre': '11', 'decembre': '12', 'décembre': '12'
}

# Constantes par défaut
DEFAULT_CONFIG = {
    "csv_file": "tirages_loto.csv",
    "window_draws": 1335,
    "num_hot": 5,
    "num_cold": 5,
    "propose_size": 5,
    "output_dir": "resultats_loto",
    "max_number": 49,
    "max_chance": 10,
    "chance_analysis": True,
    "use_fibonacci_inverse": True,
    "fibonacci_inverse_weight_blend": 0.3,
    "use_ml": True,
    "prediction_weight": 0.7,
    "test_size": 0.2,
    "cv_folds": 3,
    "n_iter_search_ml": 10,
    "model_dir": "models_loto",
    "score_weight_prediction": 0.5,
    "score_weight_gap": 0.3,
    "score_weight_frequency": 0.2,
    "score_weight_clustering": 0.0,
    "monte_carlo_simulations": 5000,
    "max_wheeling_combinations": 50,
    "wheeling_num_count": 10,
    "window_size_for_ml": 5,
    "combinations_to_generate": 5,
    "date_col": "Date",  # Ajouté pour compatibilité avec les autres modules
    "ball_cols": None    # Sera initialisé dans __init__
}

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


class LotoAnalyzer:
    """Classe principale pour l'analyse des tirages de loto."""

    def __init__(self, config: Dict = None):
        """
        Initialise l'analyseur avec la configuration spécifiée.
        
        Args:
            config: Dictionnaire de configuration optionnel qui remplace les valeurs par défaut
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        self.df = None
        # Initialisation des colonnes de numéros (compatibilité avec les autres modules)
        if self.config.get("ball_cols") is None:
            self.ball_cols = [f'Numéro {i}' for i in range(1, self.config.get("propose_size", 5) + 1)]
        else:
            self.ball_cols = self.config["ball_cols"]
            
        # Initialisation de la colonne de date et de chance
        self.date_col = self.config.get("date_col", "Date")
        self.chance_col = "Chance" if self.config.get("chance_analysis", True) else None
        
        # Chemins de sortie
        self.output_dir = Path(self.config["output_dir"])
        self.model_dir = Path(self.config["output_dir"]) / self.config.get("model_dir", "models_loto")

        # Modèles ML et scalers
        self.rf_number_model: Optional[RandomForestClassifier] = None
        self.rf_chance_model: Optional[RandomForestClassifier] = None
        self.scaler_numbers: Optional[StandardScaler] = None
        self.scaler_chance: Optional[StandardScaler] = None

        # Statistiques et analyses
        self.freq: Optional[CounterType] = None
        self.var_monthly: Optional[Dict] = None
        self.chance_freq: Optional[CounterType] = None
        self.window_counts: Optional[CounterType] = None
        self.chance_window_counts: Optional[CounterType] = None
        self.hot: Optional[List[int]] = None
        self.cold: Optional[List[int]] = None
        self.chance_hot: Optional[List[int]] = None
        self.chance_cold: Optional[List[int]] = None
        self.parity_percentages: Optional[Dict[str, float]] = None
        self.sum_stats: Optional[Dict[str, float]] = None
        self.sum_ranges: Optional[Dict[str, Dict[str, float]]] = None
        self.sequence_percentages: Optional[Dict[str, float]] = None
        self.most_common_sums: Optional[List[int]] = None
        self.most_common_sequences: Optional[List[Tuple[List[int], int]]] = None
        self.number_gap_scores: Optional[Dict[int, int]] = None
        self.chance_gap_scores: Optional[Dict[int, int]] = None
        self.number_predictions: Optional[Dict[int, float]] = None
        self.chance_predictions: Optional[Dict[int, float]] = None
        self.final_number_scores: Optional[Dict[int, float]] = None
        self.final_chance_scores: Optional[Dict[int, float]] = None

        # Création des répertoires nécessaires
        for directory in [self.output_dir, self.model_dir]:
            if not directory.exists():
                directory.mkdir(parents=True)
                logger.info(f"Répertoire créé: {directory}")

        # Définir la locale au démarrage
        self._set_locale()
        # Charger les modèles ML s'ils existent
        self._load_rf_models()

    def _set_locale(self):
        """Tente de configurer la locale sur 'fr_FR.UTF-8' ou 'french'."""
        try:
            locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')
            logger.info("Locale définie sur 'fr_FR.UTF-8'.")
        except locale.Error:
            try:
                locale.setlocale(locale.LC_TIME, 'french')
                logger.info("Locale définie sur 'french'.")
            except locale.Error:
                logger.warning("Impossible de définir la locale française. La conversion des dates avec mois en lettres pourrait échouer.")

    def _parse_loto_date(self, date_str: Any) -> pd.Timestamp:
        """
        Tente de convertir une chaîne en date, en essayant plusieurs formats,
        y compris le format français 'JJ Mois AAAA'.
        
        Args:
            date_str: La date à convertir (chaîne, datetime, etc.)
            
        Returns:
            pd.Timestamp: Date convertie ou NaT si échec
        """
        if pd.isna(date_str):
            return pd.NaT
        if isinstance(date_str, (datetime, pd.Timestamp)):
            return pd.Timestamp(date_str)
        if not isinstance(date_str, str):
            logger.warning(f"Format de date non-string '{date_str}'. Tentative de conversion.")
            date_str = str(date_str)

        formats_to_try = [
            '%d/%m/%Y',        # JJ/MM/AAAA
            '%Y-%m-%d',        # AAAA-MM-JJ (ISO)
            '%d %B %Y',        # JJ Mois_Complet AAAA (Nécessite locale FR)
            '%A %d %B %Y',     # Jour_Semaine JJ Mois_Complet AAAA
            '%d %b %Y'         # JJ Mois_Abrégé AAAA
        ]

        mois_francais = {
            'janvier': 'January', 'février': 'February', 'mars': 'March', 'avril': 'April',
            'mai': 'May', 'juin': 'June', 'juillet': 'July', 'août': 'August',
            'septembre': 'September', 'octobre': 'October', 'novembre': 'November', 'décembre': 'December'
        }
        date_str_lower = date_str.lower()
        for fr, en in mois_francais.items():
            date_str_lower = date_str_lower.replace(fr, en)

        if date_str_lower != date_str.lower():
             try:
                 return pd.to_datetime(date_str_lower, errors='raise')
             except (ValueError, TypeError):
                 pass

        for fmt in formats_to_try:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except (ValueError, TypeError):
                continue

        try:
            return pd.to_datetime(date_str, dayfirst=True, errors='raise')
        except (ValueError, TypeError):
            logger.warning(f"Format de date inattendu '{date_str}'. Impossible de convertir.")
            return pd.NaT

    def load_data(self) -> bool:
        """
        Charge et prétraite les données depuis le fichier CSV (ou lien).
        
        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        csv_path = self.config["csv_file"]
        try:
            if not csv_path.startswith(('http://', 'https://')) and not os.path.exists(csv_path):
                logger.error(f"Fichier CSV local introuvable: {csv_path}")
                return False

            logger.info(f"Chargement des données depuis {csv_path}")
            
            # Tentatives de lecture avec différents séparateurs
            df_temp = None
            for sep_try in [',', ';']: # Essayer virgule puis point-virgule
                try:
                    df_temp = pd.read_csv(csv_path, sep=sep_try)
                    logger.info(f"CSV lu avec succès en utilisant le séparateur '{sep_try}'.")
                    break # Sortir de la boucle si la lecture réussit
                except Exception as e_read:
                    logger.warning(f"Échec de la lecture du CSV avec sep='{sep_try}': {e_read}")
            
            if df_temp is None: # Si toutes les tentatives de séparateur ont échoué
                logger.error(f"Impossible de lire correctement le CSV {csv_path} avec les séparateurs testés.")
                return False
            
            self.df = df_temp
            logger.info(f"Colonnes brutes lues: {self.df.columns.tolist()}")

            # Nettoyer les noms de colonnes (supprimer les espaces superflus)
            self.df.columns = self.df.columns.str.strip()
            logger.info(f"Colonnes après nettoyage des espaces: {self.df.columns.tolist()}")

            # Vérification de la colonne date
            if self.date_col not in self.df.columns:
                logger.error(f"La colonne de date configurée '{self.date_col}' est introuvable. Colonnes disponibles: {self.df.columns.tolist()}")
                return False
            self.df[self.date_col] = self.df[self.date_col].apply(self._parse_loto_date)
            initial_rows = len(self.df)
            self.df.dropna(subset=[self.date_col], inplace=True)
            if len(self.df) < initial_rows:
                logger.warning(f"{initial_rows - len(self.df)} lignes avec dates invalides supprimées.")
            if self.df.empty:
                logger.error("DataFrame vide après suppression des dates invalides.")
                return False

            # Vérification des colonnes de numéros
            missing_ball_cols = [col for col in self.ball_cols if col not in self.df.columns]
            if missing_ball_cols:
                logger.error(f"Colonnes de numéros configurées manquantes: {missing_ball_cols}. Colonnes disponibles: {self.df.columns.tolist()}")
                return False
            if len(self.ball_cols) != self.config.get("propose_size", 5):
                 logger.error(f"Le nombre de colonnes dans 'ball_cols' ({len(self.ball_cols)}) ne correspond pas à 'propose_size' ({self.config.get('propose_size',5)}).")
                 return False

            for col in self.ball_cols:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype('Int64')
            
            initial_rows = len(self.df)
            self.df.dropna(subset=self.ball_cols, inplace=True)
            if len(self.df) < initial_rows:
                logger.warning(f"{initial_rows - len(self.df)} lignes avec numéros principaux manquants supprimées.")
            if self.df.empty:
                logger.error("DataFrame vide après suppression des numéros principaux manquants.")
                return False

            # Colonne Chance
            if self.chance_col and self.chance_col in self.df.columns:
                logger.info(f"Analyse du numéro Chance activée pour la colonne '{self.chance_col}'.")
                self.df[self.chance_col] = pd.to_numeric(self.df[self.chance_col], errors='coerce').astype('Int64')
                # Pas de dropna sur chance_col ici, car certains jeux peuvent ne pas avoir de numéro chance
            elif self.chance_col:
                logger.warning(f"Colonne Chance configurée '{self.chance_col}' non trouvée dans le CSV. Analyse Chance désactivée.")
                self.chance_col = None
            else:
                logger.info("Analyse du numéro Chance non activée ou colonne non configurée.")

            # Tri et réinitialisation de l'index
            self.df.sort_values(by=self.date_col, ascending=True, inplace=True)
            self.df.reset_index(drop=True, inplace=True)

            logger.info(f"Données chargées et prétraitées: {len(self.df)} tirages valides.")
            return True

        except Exception as e:
            logger.error(f"Erreur majeure lors du chargement des données: {e}", exc_info=True)
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
            # Fréquence des numéros principaux
            all_numbers = []
            for col in self.ball_cols:
                all_numbers.extend(self.df[col].dropna().astype(int).tolist())
            self.freq = Counter(all_numbers)
            
            # Fréquence des numéros chance
            if self.chance_col and self.chance_col in self.df.columns:
                self.chance_freq = Counter(self.df[self.chance_col].dropna().astype(int).tolist())
            
            logger.info("Statistiques globales calculées avec succès.")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du calcul des statistiques globales: {e}", exc_info=True)
            return False

    def compute_window_stats(self) -> bool:
        """
        Calcule les statistiques sur une fenêtre des derniers tirages.
        
        Returns:
            bool: True si le calcul a réussi, False sinon
        """
        if self.df is None or self.df.empty:
            logger.error("Aucune donnée chargée pour calculer les statistiques de fenêtre.")
            return False
        
        try:
            window_size = min(self.config["window_draws"], len(self.df))
            window_df = self.df.tail(window_size)
            
            # Fréquence des numéros principaux dans la fenêtre
            window_numbers = []
            for col in self.ball_cols:
                window_numbers.extend(window_df[col].dropna().astype(int).tolist())
            self.window_counts = Counter(window_numbers)
            
            # Fréquence des numéros chance dans la fenêtre
            if self.chance_col and self.chance_col in self.df.columns:
                self.chance_window_counts = Counter(window_df[self.chance_col].dropna().astype(int).tolist())
            
            logger.info(f"Statistiques de fenêtre calculées avec succès (fenêtre de {window_size} tirages).")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du calcul des statistiques de fenêtre: {e}", exc_info=True)
            return False

    def identify_hot_cold(self) -> bool:
        """
        Identifie les numéros chauds (fréquents) et froids (rares) dans la fenêtre.
        
        Returns:
            bool: True si l'identification a réussi, False sinon
        """
        if not self.window_counts:
            logger.error("Les statistiques de fenêtre doivent être calculées avant d'identifier les numéros chauds/froids.")
            return False
        
        try:
            # Numéros principaux
            num_hot = min(self.config["num_hot"], len(self.window_counts))
            num_cold = min(self.config["num_cold"], len(self.window_counts))
            
            self.hot = [num for num, _ in self.window_counts.most_common(num_hot)]
            self.cold = [num for num, _ in self.window_counts.most_common()[:-num_cold-1:-1]]
            
            # Numéros chance
            if self.chance_window_counts:
                chance_num_hot = min(self.config["num_hot"], len(self.chance_window_counts))
                chance_num_cold = min(self.config["num_cold"], len(self.chance_window_counts))
                
                self.chance_hot = [num for num, _ in self.chance_window_counts.most_common(chance_num_hot)]
                self.chance_cold = [num for num, _ in self.chance_window_counts.most_common()[:-chance_num_cold-1:-1]]
            
            logger.info(f"Numéros chauds identifiés: {self.hot}")
            logger.info(f"Numéros froids identifiés: {self.cold}")
            if self.chance_hot:
                logger.info(f"Numéros chance chauds identifiés: {self.chance_hot}")
                logger.info(f"Numéros chance froids identifiés: {self.chance_cold}")
            
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'identification des numéros chauds/froids: {e}", exc_info=True)
            return False

    def analyze_parity(self) -> bool:
        """
        Analyse la distribution des numéros pairs et impairs dans les tirages.
        
        Returns:
            bool: True si l'analyse a réussi, False sinon
        """
        if self.df is None or self.df.empty:
            logger.error("Aucune donnée chargée pour analyser la parité.")
            return False
        
        try:
            parity_counts = Counter()
            
            for _, row in self.df.iterrows():
                even_count = 0
                odd_count = 0
                
                for col in self.ball_cols:
                    if pd.notna(row[col]):
                        num = int(row[col])
                        if num % 2 == 0:
                            even_count += 1
                        else:
                            odd_count += 1
                
                parity_pattern = f"{even_count}E-{odd_count}O"
                parity_counts[parity_pattern] += 1
            
            total_draws = len(self.df)
            self.parity_percentages = {pattern: (count / total_draws) * 100 
                                      for pattern, count in parity_counts.items()}
            
            logger.info("Analyse de parité terminée.")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de parité: {e}", exc_info=True)
            return False

    def analyze_sum(self) -> bool:
        """
        Analyse la somme des numéros dans les tirages.
        
        Returns:
            bool: True si l'analyse a réussi, False sinon
        """
        if self.df is None or self.df.empty:
            logger.error("Aucune donnée chargée pour analyser les sommes.")
            return False
        
        try:
            sums = []
            
            for _, row in self.df.iterrows():
                row_sum = sum(int(row[col]) for col in self.ball_cols if pd.notna(row[col]))
                sums.append(row_sum)
            
            sums_series = pd.Series(sums)
            
            self.sum_stats = {
                "mean": sums_series.mean(),
                "median": sums_series.median(),
                "min": sums_series.min(),
                "max": sums_series.max(),
                "std": sums_series.std()
            }
            
            # Définir des plages de somme
            min_sum = int(self.sum_stats["min"])
            max_sum = int(self.sum_stats["max"])
            range_size = (max_sum - min_sum) // 5  # Diviser en 5 plages
            
            sum_ranges_count = Counter()
            for s in sums:
                for i in range(5):
                    lower = min_sum + i * range_size
                    upper = min_sum + (i + 1) * range_size if i < 4 else max_sum + 1
                    if lower <= s < upper:
                        sum_ranges_count[f"{lower}-{upper-1}"] += 1
                        break
            
            total_draws = len(self.df)
            self.sum_ranges = {
                range_str: {
                    "count": count,
                    "percentage": (count / total_draws) * 100
                }
                for range_str, count in sum_ranges_count.items()
            }
            
            # Sommes les plus fréquentes
            sum_counter = Counter(sums)
            self.most_common_sums = [sum_val for sum_val, _ in sum_counter.most_common(10)]
            
            logger.info("Analyse des sommes terminée.")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des sommes: {e}", exc_info=True)
            return False

    def analyze_sequences(self) -> bool:
        """
        Analyse les séquences de numéros consécutifs dans les tirages.
        
        Returns:
            bool: True si l'analyse a réussi, False sinon
        """
        if self.df is None or self.df.empty:
            logger.error("Aucune donnée chargée pour analyser les séquences.")
            return False
        
        try:
            sequence_counts = Counter()
            all_sequences = []
            
            for _, row in self.df.iterrows():
                numbers = sorted([int(row[col]) for col in self.ball_cols if pd.notna(row[col])])
                
                max_seq_length = 0
                current_seq_length = 1
                
                for i in range(1, len(numbers)):
                    if numbers[i] == numbers[i-1] + 1:
                        current_seq_length += 1
                    else:
                        max_seq_length = max(max_seq_length, current_seq_length)
                        current_seq_length = 1
                
                max_seq_length = max(max_seq_length, current_seq_length)
                sequence_counts[max_seq_length] += 1
                
                # Collecter les séquences spécifiques
                i = 0
                while i < len(numbers) - 1:
                    seq_start = i
                    while i + 1 < len(numbers) and numbers[i + 1] == numbers[i] + 1:
                        i += 1
                    
                    seq_length = i - seq_start + 1
                    if seq_length > 1:
                        all_sequences.append((numbers[seq_start:i+1], seq_length))
                    
                    i += 1
            
            total_draws = len(self.df)
            self.sequence_percentages = {length: (count / total_draws) * 100 
                                        for length, count in sequence_counts.items()}
            
            # Séquences les plus communes
            sequence_counter = Counter()
            for seq, _ in all_sequences:
                sequence_counter[tuple(seq)] += 1
            
            self.most_common_sequences = [(list(seq), count) 
                                         for seq, count in sequence_counter.most_common(10)]
            
            logger.info("Analyse des séquences terminée.")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des séquences: {e}", exc_info=True)
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
            for num in range(1, self.config["max_number"] + 1):
                gaps = self._get_number_gaps_at_draw(num, len(self.df) - 1)
                if gaps:
                    # Plus l'écart est grand, plus le score est élevé
                    self.number_gap_scores[num] = gaps[-1]
            
            # Calculer les écarts pour les numéros chance
            if self.chance_col and self.chance_col in self.df.columns:
                for num in range(1, self.config["max_chance"] + 1):
                    gaps = self._get_chance_gaps_at_draw(num, len(self.df) - 1)
                    if gaps:
                        self.chance_gap_scores[num] = gaps[-1]
            
            logger.info("Scores d'écart calculés avec succès.")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du calcul des scores d'écart: {e}", exc_info=True)
            return False

    def _get_number_gaps_at_draw(self, number: int, draw_index: int) -> List[int]:
        """
        Calcule les écarts d'un numéro principal jusqu'à un tirage donné.
        
        Args:
            number: Le numéro à analyser
            draw_index: L'indice du tirage jusqu'auquel calculer les écarts
            
        Returns:
            List[int]: Liste des écarts
        """
        if draw_index < 0 or draw_index >= len(self.df):
            return []
        
        gaps = []
        last_seen = -1
        
        for i in range(draw_index + 1):
            row = self.df.iloc[i]
            if any(pd.notna(row[col]) and int(row[col]) == number for col in self.ball_cols):
                if last_seen != -1:
                    gaps.append(i - last_seen - 1)
                last_seen = i
        
        # Ajouter l'écart actuel si le numéro a déjà été vu
        if last_seen != -1:
            gaps.append(draw_index - last_seen)
        
        return gaps

    def _get_chance_gaps_at_draw(self, number: int, draw_index: int) -> List[int]:
        """
        Calcule les écarts d'un numéro chance jusqu'à un tirage donné.
        
        Args:
            number: Le numéro chance à analyser
            draw_index: L'indice du tirage jusqu'auquel calculer les écarts
            
        Returns:
            List[int]: Liste des écarts
        """
        if not self.chance_col or draw_index < 0 or draw_index >= len(self.df):
            return []
        
        gaps = []
        last_seen = -1
        
        for i in range(draw_index + 1):
            row = self.df.iloc[i]
            if pd.notna(row[self.chance_col]) and int(row[self.chance_col]) == number:
                if last_seen != -1:
                    gaps.append(i - last_seen - 1)
                last_seen = i
        
        # Ajouter l'écart actuel si le numéro a déjà été vu
        if last_seen != -1:
            gaps.append(draw_index - last_seen)
        
        return gaps

    def _prepare_features(self, df: pd.DataFrame, target_index: int) -> np.ndarray:
        """
        Prépare les features pour l'apprentissage machine.
        
        Args:
            df: DataFrame des tirages
            target_index: Indice du tirage cible
            
        Returns:
            np.ndarray: Features préparées
        """
        window_size = self.config["window_size_for_ml"]
        start = max(0, target_index - window_size)
        
        if start >= target_index:
            return np.array([])
        
        subset = df.iloc[start:target_index]
        features = subset[self.ball_cols].values.flatten()
        
        # S'assurer que la taille est constante
        expected_size = len(self.ball_cols) * window_size
        if len(features) < expected_size:
            features = np.pad(features, (0, expected_size - len(features)), 'constant', constant_values=0)
        elif len(features) > expected_size:
            features = features[:expected_size]
        
        return features.reshape(1, -1)

    def _prepare_dataset_for_ml(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prépare le dataset complet pour l'apprentissage machine.
        
        Returns:
            Tuple: (X, y_numbers, y_chance)
        """
        window_size = self.config["window_size_for_ml"]
        X_list, y_numbers_list, y_chance_list = [], [], []
        
        for i in range(window_size, len(self.df)):
            features = self._prepare_features(self.df, i)
            if features.size > 0:
                X_list.append(features.flatten())
                y_numbers_list.append(self._prepare_multilabel_targets(
                    self.df.iloc[[i]], self.config["max_number"], self.ball_cols).flatten())
                
                if self.chance_col and self.chance_col in self.df.columns:
                    y_chance_list.append(int(self.df.iloc[i][self.chance_col]) 
                                        if pd.notna(self.df.iloc[i][self.chance_col]) else 0)
                else:
                    y_chance_list.append(0)
        
        return (np.array(X_list), np.array(y_numbers_list), 
                np.array(y_chance_list) if y_chance_list else np.array([]))

    def _prepare_multilabel_targets(self, df_row: pd.DataFrame, max_num: int, cols: List[str]) -> np.ndarray:
        """
        Prépare les cibles multilabel pour l'apprentissage machine.
        
        Args:
            df_row: Ligne du DataFrame contenant les numéros
            max_num: Numéro maximum possible
            cols: Colonnes contenant les numéros
            
        Returns:
            np.ndarray: Cibles multilabel
        """
        targets = np.zeros((1, max_num), dtype=int)
        
        for col in cols:
            if col in df_row.columns and pd.notna(df_row[col].iloc[0]):
                num = int(df_row[col].iloc[0])
                if 1 <= num <= max_num:
                    targets[0, num - 1] = 1
        
        return targets

    def _train_ml_models(self) -> bool:
        """
        Entraîne les modèles d'apprentissage machine.
        
        Returns:
            bool: True si l'entraînement a réussi, False sinon
        """
        if not self.config["use_ml"]:
            logger.info("L'apprentissage machine est désactivé dans la configuration.")
            return False
        
        if self.df is None or len(self.df) < self.config["window_size_for_ml"] + 10:
            logger.warning("Données insuffisantes pour l'entraînement ML.")
            return False
        
        try:
            logger.info("Préparation des données pour l'entraînement ML...")
            X, y_numbers, y_chance = self._prepare_dataset_for_ml()
            
            if X.size == 0 or y_numbers.size == 0:
                logger.error("Échec de la préparation des données pour l'entraînement ML.")
                return False
            
            # Split train/test
            test_size = self.config["test_size"]
            X_train, X_test, y_train_numbers, y_test_numbers = train_test_split(
                X, y_numbers, test_size=test_size, random_state=42)
            
            # Normalisation
            self.scaler_numbers = StandardScaler()
            X_train_scaled = self.scaler_numbers.fit_transform(X_train)
            X_test_scaled = self.scaler_numbers.transform(X_test)
            
            # Entraînement du modèle pour les numéros principaux
            logger.info("Entraînement du modèle pour les numéros principaux...")
            param_distributions = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            self.rf_number_model = _optimize_model_hyperparameters(
                "RandomForest (Numéros)", 
                RandomForestClassifier(random_state=42),
                X_train_scaled, y_train_numbers,
                param_distributions,
                n_iter_search=self.config["n_iter_search_ml"],
                cv_folds=self.config["cv_folds"]
            )
            
            # Évaluation du modèle pour les numéros principaux
            y_pred_numbers = self.rf_number_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test_numbers, y_pred_numbers)
            logger.info(f"Précision du modèle pour les numéros principaux: {accuracy:.4f}")
            
            # Entraînement du modèle pour le numéro chance
            if self.chance_col and y_chance.size > 0:
                logger.info("Entraînement du modèle pour le numéro chance...")
                y_train_chance, y_test_chance = train_test_split(
                    y_chance, test_size=test_size, random_state=42)
                
                self.scaler_chance = StandardScaler()
                X_train_scaled_chance = self.scaler_chance.fit_transform(X_train)
                X_test_scaled_chance = self.scaler_chance.transform(X_test)
                
                self.rf_chance_model = _optimize_model_hyperparameters(
                    "RandomForest (Chance)",
                    RandomForestClassifier(random_state=42),
                    X_train_scaled_chance, y_train_chance,
                    param_distributions,
                    n_iter_search=self.config["n_iter_search_ml"],
                    cv_folds=self.config["cv_folds"]
                )
                
                # Évaluation du modèle pour le numéro chance
                y_pred_chance = self.rf_chance_model.predict(X_test_scaled_chance)
                accuracy_chance = accuracy_score(y_test_chance, y_pred_chance)
                logger.info(f"Précision du modèle pour le numéro chance: {accuracy_chance:.4f}")
            
            # Sauvegarde des modèles
            self._save_rf_models()
            
            logger.info("Entraînement des modèles ML terminé avec succès.")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement des modèles ML: {e}", exc_info=True)
            return False

    def _save_rf_models(self) -> None:
        """Sauvegarde les modèles ML et les scalers."""
        logger.info("Sauvegarde des modèles ML...")
        
        try:
            if self.rf_number_model:
                joblib.dump(self.rf_number_model, self.model_dir / 'rf_number_model.joblib')
                logger.info("Modèle des numéros principaux sauvegardé.")
            
            if self.scaler_numbers:
                joblib.dump(self.scaler_numbers, self.model_dir / 'scaler_numbers.joblib')
                logger.info("Scaler des numéros principaux sauvegardé.")
            
            if self.rf_chance_model:
                joblib.dump(self.rf_chance_model, self.model_dir / 'rf_chance_model.joblib')
                logger.info("Modèle du numéro chance sauvegardé.")
            
            if self.scaler_chance:
                joblib.dump(self.scaler_chance, self.model_dir / 'scaler_chance.joblib')
                logger.info("Scaler du numéro chance sauvegardé.")
        
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des modèles ML: {e}", exc_info=True)

    def _load_rf_models(self) -> bool:
        """
        Charge les modèles ML et les scalers.
        
        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        logger.info("Tentative de chargement des modèles ML...")
        
        try:
            number_model_path = self.model_dir / 'rf_number_model.joblib'
            number_scaler_path = self.model_dir / 'scaler_numbers.joblib'
            chance_model_path = self.model_dir / 'rf_chance_model.joblib'
            chance_scaler_path = self.model_dir / 'scaler_chance.joblib'
            
            if number_model_path.exists() and number_scaler_path.exists():
                self.rf_number_model = joblib.load(number_model_path)
                self.scaler_numbers = joblib.load(number_scaler_path)
                logger.info("Modèle et scaler des numéros principaux chargés.")
            
            if chance_model_path.exists() and chance_scaler_path.exists():
                self.rf_chance_model = joblib.load(chance_model_path)
                self.scaler_chance = joblib.load(chance_scaler_path)
                logger.info("Modèle et scaler du numéro chance chargés.")
            
            return (self.rf_number_model is not None)
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles ML: {e}", exc_info=True)
            return False

    def predict_next_draw(self) -> bool:
        """
        Prédit les probabilités pour le prochain tirage.
        
        Returns:
            bool: True si la prédiction a réussi, False sinon
        """
        if self.df is None or self.df.empty:
            logger.error("Aucune donnée chargée pour la prédiction.")
            return False
        
        try:
            # Prédiction basée sur ML si disponible
            if self.config["use_ml"] and self.rf_number_model and self.scaler_numbers:
                logger.info("Utilisation des modèles ML pour la prédiction...")
                
                # Préparer les features pour la prédiction
                features = self._prepare_features(self.df, len(self.df))
                if features.size > 0:
                    # Normaliser les features
                    features_scaled = self.scaler_numbers.transform(features)
                    
                    # Prédire les probabilités pour chaque numéro
                    probas = self.rf_number_model.predict_proba(features_scaled)
                    self.number_predictions = {i + 1: probas[0][i][1] for i in range(self.config["max_number"])}
                    
                    # Prédire les probabilités pour le numéro chance
                    if self.chance_col and self.rf_chance_model and self.scaler_chance:
                        features_scaled_chance = self.scaler_chance.transform(features)
                        chance_probas = self.rf_chance_model.predict_proba(features_scaled_chance)
                        self.chance_predictions = {i + 1: chance_probas[0][i] 
                                                 for i in range(len(chance_probas[0]))}
                else:
                    logger.warning("Impossible de préparer les features pour la prédiction ML.")
                    return self._predict_from_frequencies_only()
            else:
                # Prédiction basée uniquement sur les fréquences
                return self._predict_from_frequencies_only()
            
            logger.info("Prédiction terminée avec succès.")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}", exc_info=True)
            return False

    def _predict_from_frequencies_only(self) -> bool:
        """
        Prédit les probabilités basées uniquement sur les fréquences.
        
        Returns:
            bool: True si la prédiction a réussi, False sinon
        """
        try:
            if not self.freq:
                logger.error("Les fréquences globales doivent être calculées avant la prédiction.")
                return False
            
            # Normaliser les fréquences pour obtenir des probabilités
            total_numbers = sum(self.freq.values())
            self.number_predictions = {num: count / total_numbers 
                                      for num, count in self.freq.items()}
            
            # Appliquer la pondération Fibonacci si activée
            if self.config["use_fibonacci_inverse"]:
                self._apply_fibonacci_weighting()
            
            # Numéro chance
            if self.chance_col and self.chance_freq:
                total_chance = sum(self.chance_freq.values())
                self.chance_predictions = {num: count / total_chance 
                                         for num, count in self.chance_freq.items()}
            
            logger.info("Prédiction basée sur les fréquences terminée.")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction basée sur les fréquences: {e}", exc_info=True)
            return False

    def _apply_fibonacci_weighting(self) -> None:
        """Applique la pondération Fibonacci aux prédictions."""
        if not self.number_predictions:
            logger.warning("Aucune prédiction à pondérer avec Fibonacci.")
            return
        
        try:
            # Créer un Counter à partir des prédictions
            freq_counter = Counter({num: self.freq.get(num, 0) for num in self.number_predictions})
            
            # Appliquer la pondération Fibonacci inverse
            fibonacci_weights = apply_inverse_fibonacci_weights(freq_counter, reverse_order=True)
            
            # Mélanger les poids
            blend_factor = self.config["fibonacci_inverse_weight_blend"]
            for num in self.number_predictions:
                orig_weight = self.number_predictions[num]
                fib_weight = fibonacci_weights.get(num, 0.0)
                self.number_predictions[num] = (1 - blend_factor) * orig_weight + blend_factor * fib_weight
            
            logger.info(f"Pondération Fibonacci appliquée avec facteur de mélange {blend_factor}.")
        
        except Exception as e:
            logger.error(f"Erreur lors de l'application de la pondération Fibonacci: {e}", exc_info=True)

    def compute_scores(self) -> bool:
        """
        Calcule les scores finaux pour chaque numéro.
        
        Returns:
            bool: True si le calcul a réussi, False sinon
        """
        if not self.number_predictions:
            logger.error("Les prédictions doivent être calculées avant les scores.")
            return False
        
        try:
            # Poids des différentes composantes
            pred_weight = self.config["score_weight_prediction"]
            gap_weight = self.config["score_weight_gap"]
            freq_weight = self.config["score_weight_frequency"]
            
            # Normaliser les scores d'écart
            if self.number_gap_scores:
                max_gap = max(self.number_gap_scores.values()) if self.number_gap_scores else 1
                norm_gap_scores = {num: score / max_gap for num, score in self.number_gap_scores.items()}
            else:
                norm_gap_scores = {}
            
            # Normaliser les fréquences
            if self.freq:
                max_freq = max(self.freq.values()) if self.freq else 1
                norm_freq = {num: count / max_freq for num, count in self.freq.items()}
            else:
                norm_freq = {}
            
            # Calculer les scores finaux pour les numéros principaux
            self.final_number_scores = {}
            for num in range(1, self.config["max_number"] + 1):
                pred_score = self.number_predictions.get(num, 0)
                gap_score = norm_gap_scores.get(num, 0)
                freq_score = norm_freq.get(num, 0)
                
                final_score = (pred_weight * pred_score + 
                              gap_weight * gap_score + 
                              freq_weight * freq_score)
                
                self.final_number_scores[num] = final_score
            
            # Calculer les scores finaux pour le numéro chance
            if self.chance_col and self.chance_predictions:
                self.final_chance_scores = {}
                
                if self.chance_gap_scores:
                    max_chance_gap = max(self.chance_gap_scores.values()) if self.chance_gap_scores else 1
                    norm_chance_gap = {num: score / max_chance_gap for num, score in self.chance_gap_scores.items()}
                else:
                    norm_chance_gap = {}
                
                if self.chance_freq:
                    max_chance_freq = max(self.chance_freq.values()) if self.chance_freq else 1
                    norm_chance_freq = {num: count / max_chance_freq for num, count in self.chance_freq.items()}
                else:
                    norm_chance_freq = {}
                
                for num in range(1, self.config["max_chance"] + 1):
                    pred_score = self.chance_predictions.get(num, 0)
                    gap_score = norm_chance_gap.get(num, 0)
                    freq_score = norm_chance_freq.get(num, 0)
                    
                    final_score = (pred_weight * pred_score + 
                                  gap_weight * gap_score + 
                                  freq_weight * freq_score)
                    
                    self.final_chance_scores[num] = final_score
            
            logger.info("Scores finaux calculés avec succès.")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors du calcul des scores finaux: {e}", exc_info=True)
            return False

    def _run_monte_carlo_simulation(self, num_simulations: int = 5000) -> Dict[str, List[int]]:
        """
        Exécute une simulation Monte Carlo pour générer des combinaisons.
        
        Args:
            num_simulations: Nombre de simulations à exécuter
            
        Returns:
            Dict[str, List[int]]: Résultats de la simulation
        """
        if not self.final_number_scores:
            logger.error("Les scores finaux doivent être calculés avant la simulation Monte Carlo.")
            return {}
        
        try:
            logger.info(f"Exécution de {num_simulations} simulations Monte Carlo...")
            
            # Préparer les poids pour l'échantillonnage
            numbers = list(range(1, self.config["max_number"] + 1))
            weights = [self.final_number_scores.get(num, 0) for num in numbers]
            
            # Normaliser les poids
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            
            # Préparer les poids pour le numéro chance
            chance_numbers = list(range(1, self.config["max_chance"] + 1))
            chance_weights = None
            if self.chance_col and self.final_chance_scores:
                chance_weights = [self.final_chance_scores.get(num, 0) for num in chance_numbers]
                total_chance_weight = sum(chance_weights)
                if total_chance_weight > 0:
                    chance_weights = [w / total_chance_weight for w in chance_weights]
            
            # Exécuter les simulations
            combinations = []
            chance_nums = []
            
            for _ in range(num_simulations):
                # Tirer une combinaison valide
                while True:
                    combo = sorted(np.random.choice(
                        numbers, size=self.config["propose_size"], 
                        replace=False, p=weights
                    ).tolist())
                    
                    if self._is_combination_valid_loto(combo):
                        combinations.append(combo)
                        break
                
                # Tirer un numéro chance
                if chance_weights:
                    chance_num = np.random.choice(chance_numbers, p=chance_weights)
                    chance_nums.append(chance_num)
            
            # Compter les combinaisons les plus fréquentes
            combo_counter = Counter(tuple(combo) for combo in combinations)
            top_combos = [list(combo) for combo, _ in combo_counter.most_common(self.config["combinations_to_generate"])]
            
            # Compter les numéros chance les plus fréquents
            chance_counter = Counter(chance_nums)
            top_chance = [num for num, _ in chance_counter.most_common(len(top_combos))]
            
            return {
                "combinations": top_combos,
                "chance_numbers": top_chance
            }
        
        except Exception as e:
            logger.error(f"Erreur lors de la simulation Monte Carlo: {e}", exc_info=True)
            return {}

    def _generate_wheeling_combinations(self) -> List[List[int]]:
        """
        Génère des combinaisons en utilisant une méthode de wheeling.
        
        Returns:
            List[List[int]]: Combinaisons générées
        """
        if not self.final_number_scores:
            logger.error("Les scores finaux doivent être calculés avant le wheeling.")
            return []
        
        try:
            logger.info("Génération de combinaisons par wheeling...")
            
            # Sélectionner les meilleurs numéros
            top_numbers = sorted([num for num, _ in sorted(
                self.final_number_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:self.config["wheeling_num_count"]]])
            
            if len(top_numbers) < self.config["propose_size"]:
                logger.warning(f"Pas assez de numéros pour le wheeling. Utilisation des {len(top_numbers)} disponibles.")
                return []
            
            # Générer toutes les combinaisons possibles
            from itertools import combinations
            all_combos = list(combinations(top_numbers, self.config["propose_size"]))
            
            # Limiter le nombre de combinaisons
            max_combos = min(len(all_combos), self.config["max_wheeling_combinations"])
            selected_combos = all_combos[:max_combos]
            
            # Convertir en listes
            result = [list(combo) for combo in selected_combos]
            
            logger.info(f"{len(result)} combinaisons générées par wheeling.")
            return result
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération de combinaisons par wheeling: {e}", exc_info=True)
            return []

    def _is_combination_valid_loto(self, combo: List[int]) -> bool:
        """
        Vérifie si une combinaison est valide selon les règles du Loto.
        
        Args:
            combo: Combinaison à vérifier
            
        Returns:
            bool: True si la combinaison est valide, False sinon
        """
        # Vérifier la taille
        if len(combo) != self.config["propose_size"]:
            return False
        
        # Vérifier les doublons
        if len(set(combo)) != len(combo):
            return False
        
        # Vérifier la plage des numéros
        for num in combo:
            if num < 1 or num > self.config["max_number"]:
                return False
        
        return True

    def _calculate_combination_rank(self, combo_with_chance: List[int], actual_draw: Dict) -> Optional[str]:
        """
        Calcule le rang d'une combinaison par rapport à un tirage réel.
        
        Args:
            combo_with_chance: Combinaison avec numéro chance
            actual_draw: Tirage réel
            
        Returns:
            Optional[str]: Rang de la combinaison ou None
        """
        if not combo_with_chance or not actual_draw:
            return None
        
        try:
            # Séparer les numéros principaux et le numéro chance
            combo = combo_with_chance[:-1] if len(combo_with_chance) > self.config["propose_size"] else combo_with_chance
            combo_chance = combo_with_chance[-1] if len(combo_with_chance) > self.config["propose_size"] else None
            
            # Extraire les numéros du tirage réel
            actual_numbers = []
            for col in self.ball_cols:
                if col in actual_draw and pd.notna(actual_draw[col]):
                    actual_numbers.append(int(actual_draw[col]))
            
            actual_chance = None
            if self.chance_col and self.chance_col in actual_draw and pd.notna(actual_draw[self.chance_col]):
                actual_chance = int(actual_draw[self.chance_col])
            
            # Compter les correspondances
            correct_numbers = len(set(combo) & set(actual_numbers))
            correct_chance = 1 if combo_chance == actual_chance else 0
            
            # Déterminer le rang
            if correct_numbers == 5 and correct_chance == 1:
                return "5+1"
            elif correct_numbers == 5:
                return "5"
            elif correct_numbers == 4 and correct_chance == 1:
                return "4+1"
            elif correct_numbers == 4:
                return "4"
            elif correct_numbers == 3 and correct_chance == 1:
                return "3+1"
            elif correct_numbers == 3:
                return "3"
            elif correct_numbers == 2 and correct_chance == 1:
                return "2+1"
            elif correct_numbers == 2:
                return "2"
            elif correct_numbers == 1 and correct_chance == 1:
                return "1+1"
            elif correct_chance == 1:
                return "0+1"
            else:
                return None
        
        except Exception as e:
            logger.error(f"Erreur lors du calcul du rang: {e}", exc_info=True)
            return None

    def generate_multiple_combinations(self, count: int = 5) -> List[List[int]]:
        """
        Génère plusieurs combinaisons de numéros.
        
        Args:
            count: Nombre de combinaisons à générer
            
        Returns:
            List[List[int]]: Combinaisons générées
        """
        if not self.final_number_scores:
            logger.error("Les scores finaux doivent être calculés avant de générer des combinaisons.")
            return []
        
        try:
            # Méthode 1: Simulation Monte Carlo
            monte_carlo_results = self._run_monte_carlo_simulation(
                self.config["monte_carlo_simulations"])
            
            if monte_carlo_results and "combinations" in monte_carlo_results:
                mc_combos = monte_carlo_results["combinations"]
                if len(mc_combos) >= count:
                    return mc_combos[:count]
            
            # Méthode 2: Wheeling
            wheeling_combos = self._generate_wheeling_combinations()
            if wheeling_combos and len(wheeling_combos) >= count:
                return wheeling_combos[:count]
            
            # Méthode 3: Sélection directe des meilleurs numéros
            logger.info("Génération de combinaisons par sélection directe...")
            
            top_numbers = sorted([num for num, _ in sorted(
                self.final_number_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:self.config["max_number"]]])
            
            result = []
            for i in range(count):
                # Sélectionner des numéros avec un peu de variation
                offset = i * 2
                combo = sorted(top_numbers[offset:offset + self.config["propose_size"]])
                
                # S'assurer que la combinaison est complète
                while len(combo) < self.config["propose_size"]:
                    for num in top_numbers:
                        if num not in combo:
                            combo.append(num)
                            break
                    combo = sorted(combo)
                
                if self._is_combination_valid_loto(combo):
                    result.append(combo)
            
            return result[:count]
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération de combinaisons: {e}", exc_info=True)
            return []

    def propose_chance_numbers(self, count: int = 5) -> List[int]:
        """
        Propose des numéros chance.
        
        Args:
            count: Nombre de numéros chance à proposer
            
        Returns:
            List[int]: Numéros chance proposés
        """
        if not self.chance_col or not self.final_chance_scores:
            logger.warning("Analyse du numéro chance non disponible.")
            return [random.randint(1, self.config["max_chance"]) for _ in range(count)]
        
        try:
            # Trier les numéros chance par score
            sorted_chance = sorted(
                self.final_chance_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Sélectionner les meilleurs
            result = [num for num, _ in sorted_chance[:count]]
            
            # Compléter si nécessaire
            while len(result) < count:
                for num in range(1, self.config["max_chance"] + 1):
                    if num not in result:
                        result.append(num)
                        break
            
            return result
        
        except Exception as e:
            logger.error(f"Erreur lors de la proposition de numéros chance: {e}", exc_info=True)
            return [random.randint(1, self.config["max_chance"]) for _ in range(count)]

    def track_performance(self, generated_combinations: List[List[int]], history_file: str = None) -> Dict:
        """
        Évalue la performance des combinaisons générées sur l'historique des tirages.
        
        Args:
            generated_combinations: Combinaisons à évaluer
            history_file: Fichier d'historique optionnel
            
        Returns:
            Dict: Résultats de l'évaluation
        """
        if not generated_combinations:
            logger.error("Aucune combinaison fournie pour l'évaluation.")
            return {}
        
        try:
            # Charger l'historique si un fichier est fourni
            history_df = None
            if history_file:
                try:
                    history_df = pd.read_csv(history_file)
                except Exception as e:
                    logger.error(f"Erreur lors du chargement du fichier d'historique: {e}")
            
            # Utiliser le DataFrame actuel si aucun historique n'est fourni
            if history_df is None:
                history_df = self.df
            
            if history_df is None or history_df.empty:
                logger.error("Aucune donnée disponible pour l'évaluation.")
                return {}
            
            # Évaluer chaque combinaison sur chaque tirage
            results = {
                "combinations": generated_combinations,
                "evaluations": []
            }
            
            for _, draw in history_df.iterrows():
                draw_date = draw[self.date_col] if self.date_col in draw else "Inconnu"
                draw_numbers = [int(draw[col]) for col in self.ball_cols if pd.notna(draw[col])]
                draw_chance = int(draw[self.chance_col]) if self.chance_col in draw and pd.notna(draw[self.chance_col]) else None
                
                draw_eval = {
                    "date": draw_date,
                    "actual_numbers": draw_numbers,
                    "actual_chance": draw_chance,
                    "combination_results": []
                }
                
                for i, combo in enumerate(generated_combinations):
                    combo_with_chance = combo.copy()
                    if draw_chance is not None:
                        combo_with_chance.append(draw_chance)
                    
                    rank = self._calculate_combination_rank(combo_with_chance, draw)
                    correct_numbers = len(set(combo) & set(draw_numbers))
                    
                    draw_eval["combination_results"].append({
                        "combination_index": i,
                        "correct_numbers": correct_numbers,
                        "rank": rank
                    })
                
                results["evaluations"].append(draw_eval)
            
            # Calculer les statistiques globales
            total_evaluations = len(results["evaluations"])
            if total_evaluations > 0:
                stats = {
                    "total_draws": total_evaluations,
                    "combinations_stats": []
                }
                
                for i, _ in enumerate(generated_combinations):
                    combo_stats = {
                        "combination_index": i,
                        "avg_correct_numbers": sum(eval["combination_results"][i]["correct_numbers"] 
                                                for eval in results["evaluations"]) / total_evaluations,
                        "ranks": {}
                    }
                    
                    # Compter les rangs
                    rank_counter = Counter()
                    for eval in results["evaluations"]:
                        rank = eval["combination_results"][i]["rank"]
                        if rank:
                            rank_counter[rank] += 1
                    
                    combo_stats["ranks"] = dict(rank_counter)
                    stats["combinations_stats"].append(combo_stats)
                
                results["statistics"] = stats
            
            return results
        
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation des performances: {e}", exc_info=True)
            return {}

    def run_analysis(self) -> bool:
        """
        Exécute l'analyse complète des tirages.
        
        Returns:
            bool: True si l'analyse a réussi, False sinon
        """
        try:
            # Étape 1: Charger les données
            if not self.load_data():
                logger.error("Échec du chargement des données. Analyse annulée.")
                return False
            
            # Étape 2: Calculer les statistiques globales
            if not self.compute_global_stats():
                logger.error("Échec du calcul des statistiques globales. Analyse annulée.")
                return False
            
            # Étape 3: Calculer les statistiques de fenêtre
            if not self.compute_window_stats():
                logger.error("Échec du calcul des statistiques de fenêtre. Analyse annulée.")
                return False
            
            # Étape 4: Identifier les numéros chauds et froids
            if not self.identify_hot_cold():
                logger.error("Échec de l'identification des numéros chauds/froids. Analyse annulée.")
                return False
            
            # Étape 5: Analyser la parité
            if not self.analyze_parity():
                logger.warning("Échec de l'analyse de parité. Poursuite de l'analyse.")
            
            # Étape 6: Analyser les sommes
            if not self.analyze_sum():
                logger.warning("Échec de l'analyse des sommes. Poursuite de l'analyse.")
            
            # Étape 7: Analyser les séquences
            if not self.analyze_sequences():
                logger.warning("Échec de l'analyse des séquences. Poursuite de l'analyse.")
            
            # Étape 8: Calculer les scores d'écart
            if not self.compute_gap_scores():
                logger.warning("Échec du calcul des scores d'écart. Poursuite de l'analyse.")
            
            # Étape 9: Entraîner les modèles ML si nécessaire
            if self.config["use_ml"] and not self._load_rf_models():
                logger.info("Aucun modèle ML trouvé. Entraînement de nouveaux modèles...")
                if not self._train_ml_models():
                    logger.warning("Échec de l'entraînement des modèles ML. Poursuite de l'analyse sans ML.")
            
            # Étape 10: Prédire les probabilités pour le prochain tirage
            if not self.predict_next_draw():
                logger.error("Échec de la prédiction. Analyse annulée.")
                return False
            
            # Étape 11: Calculer les scores finaux
            if not self.compute_scores():
                logger.error("Échec du calcul des scores finaux. Analyse annulée.")
                return False
            
            logger.info("Analyse terminée avec succès.")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse: {e}", exc_info=True)
            return False

    def save_results(self, output_file: str = None) -> str:
        """
        Sauvegarde les résultats de l'analyse dans un fichier texte.
        
        Args:
            output_file: Nom du fichier de sortie optionnel
            
        Returns:
            str: Chemin du fichier de sortie
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"resultats_loto_{timestamp}.txt"
        else:
            output_file = Path(output_file)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(" RÉSULTATS DE L'ANALYSE LOTO \n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Date de l'analyse: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Fichier de données: {self.config['csv_file']}\n")
                f.write(f"Nombre de tirages analysés: {len(self.df) if self.df is not None else 0}\n\n")
                
                # Numéros chauds et froids
                f.write("-" * 80 + "\n")
                f.write("NUMÉROS CHAUDS ET FROIDS\n")
                f.write("-" * 80 + "\n")
                
                if self.hot:
                    f.write(f"Numéros chauds: {', '.join(map(str, self.hot))}\n")
                if self.cold:
                    f.write(f"Numéros froids: {', '.join(map(str, self.cold))}\n")
                
                if self.chance_hot:
                    f.write(f"Numéros chance chauds: {', '.join(map(str, self.chance_hot))}\n")
                if self.chance_cold:
                    f.write(f"Numéros chance froids: {', '.join(map(str, self.chance_cold))}\n")
                
                f.write("\n")
                
                # Analyse de parité
                if self.parity_percentages:
                    f.write("-" * 80 + "\n")
                    f.write("ANALYSE DE PARITÉ\n")
                    f.write("-" * 80 + "\n")
                    
                    for pattern, percentage in sorted(self.parity_percentages.items()):
                        f.write(f"{pattern} (pairs-impairs): {percentage:.2f}%\n")
                    
                    f.write("\n")
                
                # Analyse de somme
                if self.sum_stats:
                    f.write("-" * 80 + "\n")
                    f.write("ANALYSE DES SOMMES\n")
                    f.write("-" * 80 + "\n")
                    
                    f.write(f"Moyenne: {self.sum_stats['mean']:.2f}\n")
                    f.write(f"Médiane: {self.sum_stats['median']:.2f}\n")
                    f.write(f"Minimum: {self.sum_stats['min']}\n")
                    f.write(f"Maximum: {self.sum_stats['max']}\n")
                    f.write(f"Écart-type: {self.sum_stats['std']:.2f}\n\n")
                    
                    if self.sum_ranges:
                        f.write("Distribution des plages de somme:\n")
                        for range_str, stats in sorted(self.sum_ranges.items()):
                            f.write(f"  {range_str}: {stats['percentage']:.2f}%\n")
                    
                    f.write("\n")
                
                # Analyse de séquence
                if self.sequence_percentages:
                    f.write("-" * 80 + "\n")
                    f.write("ANALYSE DES SÉQUENCES\n")
                    f.write("-" * 80 + "\n")
                    
                    f.write("Longueur maximale de séquence dans un tirage:\n")
                    for length, percentage in sorted(self.sequence_percentages.items()):
                        f.write(f"  {length} numéros consécutifs: {percentage:.2f}%\n")
                    
                    f.write("\n")
                
                # Combinaisons proposées
                f.write("-" * 80 + "\n")
                f.write("COMBINAISONS PROPOSÉES\n")
                f.write("-" * 80 + "\n")
                
                combinations = self.generate_multiple_combinations(self.config["combinations_to_generate"])
                chance_numbers = self.propose_chance_numbers(len(combinations))
                
                for i, combo in enumerate(combinations):
                    chance = chance_numbers[i] if i < len(chance_numbers) else None
                    if chance:
                        f.write(f"Combinaison {i+1}: {' - '.join(map(str, combo))} | Chance: {chance}\n")
                    else:
                        f.write(f"Combinaison {i+1}: {' - '.join(map(str, combo))}\n")
                
                f.write("\n")
                
                # Statistiques supplémentaires
                f.write("-" * 80 + "\n")
                f.write("STATISTIQUES SUPPLÉMENTAIRES\n")
                f.write("-" * 80 + "\n")
                
                if self.freq:
                    f.write("Top 10 des numéros les plus fréquents:\n")
                    for num, count in sorted(self.freq.items(), key=lambda x: x[1], reverse=True)[:10]:
                        f.write(f"  Numéro {num}: {count} occurrences\n")
                    
                    f.write("\n")
                
                if self.chance_freq:
                    f.write("Fréquence des numéros chance:\n")
                    for num, count in sorted(self.chance_freq.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"  Numéro {num}: {count} occurrences\n")
                    
                    f.write("\n")
            
            logger.info(f"Résultats sauvegardés dans {output_file}")
            return str(output_file)
        
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des résultats: {e}", exc_info=True)
            return ""

    def export_to_excel(self, output_file: str = None) -> str:
        """
        Exporte les résultats de l'analyse dans un fichier Excel.
        
        Args:
            output_file: Nom du fichier Excel de sortie optionnel
            
        Returns:
            str: Chemin du fichier Excel
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"resultats_loto_{timestamp}.xlsx"
        else:
            output_file = Path(output_file)
        
        try:
            # Créer un nouveau classeur Excel
            wb = openpyxl.Workbook()
            
            # Feuille des combinaisons proposées
            ws_combos = wb.active
            ws_combos.title = "Combinaisons"
            
            # En-têtes
            ws_combos.append(["Combinaison", "Numéro 1", "Numéro 2", "Numéro 3", "Numéro 4", "Numéro 5", "Numéro Chance"])
            
            # Données
            combinations = self.generate_multiple_combinations(self.config["combinations_to_generate"])
            chance_numbers = self.propose_chance_numbers(len(combinations))
            
            for i, combo in enumerate(combinations):
                row = [f"Combinaison {i+1}"]
                for num in combo:
                    row.append(num)
                
                # Ajouter le numéro chance
                if i < len(chance_numbers):
                    row.append(chance_numbers[i])
                else:
                    row.append(None)
                
                ws_combos.append(row)
            
            # Mise en forme
            for col in range(1, 8):
                ws_combos.column_dimensions[get_column_letter(col)].width = 15
            
            # Style des en-têtes
            header_font = Font(bold=True)
            header_fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")
            header_alignment = Alignment(horizontal="center")
            
            for cell in ws_combos[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = header_alignment
            
            # Feuille des statistiques
            ws_stats = wb.create_sheet("Statistiques")
            
            # En-têtes
            ws_stats.append(["Numéro", "Fréquence", "Score", "Écart actuel", "Statut"])
            
            # Données
            for num in range(1, self.config["max_number"] + 1):
                freq = self.freq.get(num, 0) if self.freq else 0
                score = self.final_number_scores.get(num, 0) if self.final_number_scores else 0
                gap = self.number_gap_scores.get(num, 0) if self.number_gap_scores else 0
                
                status = ""
                if self.hot and num in self.hot:
                    status = "Chaud"
                elif self.cold and num in self.cold:
                    status = "Froid"
                
                ws_stats.append([num, freq, score, gap, status])
            
            # Mise en forme
            for col in range(1, 6):
                ws_stats.column_dimensions[get_column_letter(col)].width = 15
            
            # Style des en-têtes
            for cell in ws_stats[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = header_alignment
            
            # Graphique de fréquence
            if self.freq:
                chart_sheet = wb.create_sheet("Graphiques")
                
                # Préparer les données pour le graphique
                chart_data = [["Numéro", "Fréquence"]]
                for num in range(1, self.config["max_number"] + 1):
                    chart_data.append([num, self.freq.get(num, 0)])
                
                # Ajouter les données à la feuille
                for row in chart_data:
                    chart_sheet.append(row)
                
                # Créer le graphique
                chart = BarChart()
                chart.title = "Fréquence des numéros"
                chart.x_axis.title = "Numéro"
                chart.y_axis.title = "Fréquence"
                
                data = Reference(chart_sheet, min_col=2, min_row=1, max_row=len(chart_data), max_col=2)
                cats = Reference(chart_sheet, min_col=1, min_row=2, max_row=len(chart_data))
                
                chart.add_data(data, titles_from_data=True)
                chart.set_categories(cats)
                
                chart_sheet.add_chart(chart, "E5")
            
            # Sauvegarder le fichier
            wb.save(output_file)
            
            logger.info(f"Résultats exportés vers Excel: {output_file}")
            return str(output_file)
        
        except Exception as e:
            logger.error(f"Erreur lors de l'exportation vers Excel: {e}", exc_info=True)
            return ""

    def _convert_to_int_list(self, number_list: List[Any]) -> List[int]:
        """
        Convertit une liste de nombres en liste d'entiers.
        
        Args:
            number_list: Liste de nombres à convertir
            
        Returns:
            List[int]: Liste d'entiers
        """
        return [int(num) for num in number_list if pd.notna(num)]

    def _print_results(self) -> None:
        """Affiche les résultats de l'analyse dans la console."""
        print("=" * 80)
        print(" RÉSULTATS DE L'ANALYSE LOTO ")
        print("=" * 80)
        print()
        
        print(f"Nombre de tirages analysés: {len(self.df) if self.df is not None else 0}")
        print()
        
        # Numéros chauds et froids
        print("-" * 80)
        print("NUMÉROS CHAUDS ET FROIDS")
        print("-" * 80)
        
        if self.hot:
            print(f"Numéros chauds: {', '.join(map(str, self.hot))}")
        if self.cold:
            print(f"Numéros froids: {', '.join(map(str, self.cold))}")
        
        if self.chance_hot:
            print(f"Numéros chance chauds: {', '.join(map(str, self.chance_hot))}")
        if self.chance_cold:
            print(f"Numéros chance froids: {', '.join(map(str, self.chance_cold))}")
        
        print()
        
        # Combinaisons proposées
        print("-" * 80)
        print("COMBINAISONS PROPOSÉES")
        print("-" * 80)
        
        combinations = self.generate_multiple_combinations(self.config["combinations_to_generate"])
        chance_numbers = self.propose_chance_numbers(len(combinations))
        
        for i, combo in enumerate(combinations):
            chance = chance_numbers[i] if i < len(chance_numbers) else None
            if chance:
                print(f"Combinaison {i+1}: {' - '.join(map(str, combo))} | Chance: {chance}")
            else:
                print(f"Combinaison {i+1}: {' - '.join(map(str, combo))}")
        
        print()


def main():
    """Fonction principale du script."""
    parser = argparse.ArgumentParser(description="Analyseur de tirages de Loto")
    
    parser.add_argument("--csv", dest="csv_file", default="tirages_loto.csv",
                        help="Chemin vers le fichier CSV des tirages")
    
    parser.add_argument("--window", dest="window_draws", type=int, default=1335,
                        help="Nombre de tirages à considérer pour l'analyse de fenêtre")
    
    parser.add_argument("--hot", dest="num_hot", type=int, default=5,
                        help="Nombre de numéros chauds à identifier")
    
    parser.add_argument("--cold", dest="num_cold", type=int, default=5,
                        help="Nombre de numéros froids à identifier")
    
    parser.add_argument("--size", dest="propose_size", type=int, default=5,
                        help="Taille des combinaisons à proposer")
    
    parser.add_argument("--output", dest="output_dir", default="resultats_loto",
                        help="Répertoire de sortie pour les résultats")
    
    parser.add_argument("--max-number", dest="max_number", type=int, default=49,
                        help="Numéro maximum possible")
    
    parser.add_argument("--max-chance", dest="max_chance", type=int, default=10,
                        help="Numéro chance maximum possible")
    
    parser.add_argument("--no-chance", dest="chance_analysis", action="store_false",
                        help="Désactiver l'analyse du numéro chance")
    
    parser.add_argument("--no-ml", dest="use_ml", action="store_false",
                        help="Désactiver l'apprentissage machine")
    
    parser.add_argument("--no-excel", dest="export_excel", action="store_false",
                        help="Désactiver l'exportation vers Excel")
    
    parser.add_argument("--combinations", dest="combinations_to_generate", type=int, default=5,
                        help="Nombre de combinaisons à générer")
    
    args = parser.parse_args()
    
    # Créer la configuration
    config = {
        "csv_file": args.csv_file,
        "window_draws": args.window_draws,
        "num_hot": args.num_hot,
        "num_cold": args.num_cold,
        "propose_size": args.propose_size,
        "output_dir": args.output_dir,
        "max_number": args.max_number,
        "max_chance": args.max_chance,
        "chance_analysis": args.chance_analysis,
        "use_ml": args.use_ml,
        "combinations_to_generate": args.combinations_to_generate
    }
    
    # Créer et exécuter l'analyseur
    analyzer = LotoAnalyzer(config)
    
    if analyzer.run_analysis():
        # Afficher les résultats
        analyzer._print_results()
        
        # Sauvegarder les résultats
        output_file = analyzer.save_results()
        print(f"\nRésultats sauvegardés dans: {output_file}")
        
        # Exporter vers Excel
        if args.export_excel:
            excel_file = analyzer.export_to_excel()
            if excel_file:
                print(f"Résultats exportés vers Excel: {excel_file}")
    else:
        print("L'analyse a échoué. Consultez les logs pour plus d'informations.")


if __name__ == "__main__":
    main()
