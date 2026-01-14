#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module d'apprentissage incrémental pour l'analyseur de tirages de Loto
Ce module implémente un mécanisme qui s'adapte automatiquement à chaque nouveau tirage
sans avoir à refaire l'entraînement complet.

Version améliorée avec gestion robuste des dates au format JJ/MM/AAAA,
optimisation des performances avec mini-batch et joblib, et logs détaillés.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import Counter
from pathlib import Path
import traceback
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
import pickle
import time

# Configuration du logging avec redirection vers fichier et console
logger = logging.getLogger("LotoIncrementalLearning")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("log.txt"),
            logging.StreamHandler()
        ]
    )

# Dictionnaire de conversion des mois français vers les nombres
MOIS_FRANCAIS_VERS_NOMBRE = {
    'janvier': '01', 'fevrier': '02', 'février': '02', 'mars': '03', 'avril': '04',
    'mai': '05', 'juin': '06', 'juillet': '07', 'aout': '08', 'août': '08',
    'septembre': '09', 'octobre': '10', 'novembre': '11', 'decembre': '12', 'décembre': '12'
}

# Gestion de l'import de LotoAnalyzer pour l'annotation de type et l'utilisation
LOTO_ANALYZER_TYPE_HINT = 'LotoAnalyzer'
try:
    from improved_loto_analyzer_improved import LotoAnalyzer
except ImportError:
    try:
        from loto_analyzer_improved import LotoAnalyzer
    except ImportError:
        try:
            # Tentative d'import avec chemin relatif
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.append(current_dir)
            from improved_loto_analyzer_improved import LotoAnalyzer
        except ImportError:
            try:
                from loto_analyzer_improved import LotoAnalyzer
            except ImportError:
                logger.error("Impossible d'importer 'LotoAnalyzer'. "
                      "Assurez-vous que le fichier est accessible.")
                # Définition d'un type factice pour permettre au code de charger sans erreur de nom immédiate.
                LotoAnalyzer = type('LotoAnalyzer', (object,), {})


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


def train_model_batch(batch_data: Tuple[np.ndarray, np.ndarray, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fonction pour entraîner un modèle sur un batch de données, utilisée pour le parallélisme.
    
    Args:
        batch_data: Tuple contenant (X_train, y_train, params)
        
    Returns:
        Dict: Résultats de l'entraînement
    """
    try:
        X_train, y_train, params = batch_data
        model_type = params.get('model_type', 'RandomForest')
        random_state = params.get('random_state', 42)
        
        if model_type == 'RandomForest':
            model = RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                random_state=random_state
            )
            model.fit(X_train, y_train)
            return {'model': model, 'success': True}
        else:
            return {'success': False, 'error': f"Type de modèle non pris en charge: {model_type}"}
    except Exception as e:
        return {'success': False, 'error': str(e)}


class LotoIncrementalLearning:
    """Classe pour l'apprentissage incrémental des prédictions Loto."""
    
    def __init__(self, analyzer):
        """
        Initialise le module d'apprentissage incrémental.
        
        Args:
            analyzer: Instance de LotoAnalyzer (loto_analyzer_improved.py)
        """
        try:
            if not isinstance(analyzer, LotoAnalyzer):
                raise TypeError("L'argument 'analyzer' doit être une instance valide de LotoAnalyzer.")
                
            self.analyzer = analyzer
            self.df = getattr(analyzer, 'df', pd.DataFrame())
            self.ball_cols = getattr(analyzer, 'ball_cols', [])
            self.chance_col = getattr(analyzer, 'chance_col', None)
            self.config = getattr(analyzer, 'config', {})
            
            analyzer_output_dir = getattr(analyzer, 'output_dir', Path("resultats_loto"))
            self.models_dir = Path(analyzer_output_dir) / self.config.get("model_dir", "models_loto")
            self.models_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Répertoire des modèles incrémentaux: {self.models_dir}")
            
            self.plots_dir = self.models_dir / "plots_incremental" # Nom de dossier distinct
            self.plots_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Répertoire des plots incrémentaux: {self.plots_dir}")
            
            # Les modèles et scalers sont gérés par l'instance LotoAnalyzer principale.
            # LotoIncrementalLearning va appeler les méthodes de LotoAnalyzer pour
            # entraîner, charger et sauvegarder les modèles.

            self.performance_history: Dict[str, List[Any]] = {
                'dates': [],
                'accuracy_numbers': [],
                'f1_numbers': [],
                'accuracy_chance': [],
                'f1_chance': [],
                'training_time': []  # Nouveau: temps d'entraînement
            }
            
            # Paramètres pour l'optimisation
            self.use_mini_batch = self.config.get("use_mini_batch", True)
            self.batch_size = self.config.get("batch_size", 50)
            self.use_parallel = self.config.get("use_parallel", True)
            self.max_workers = self.config.get("max_workers", os.cpu_count() or 2)
            self.use_pickle_cache = self.config.get("use_pickle_cache", True)
            self.cache_dir = self.models_dir / "cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Configuration d'optimisation: mini-batch={self.use_mini_batch}, "
                       f"parallélisme={self.use_parallel}, cache pickle={self.use_pickle_cache}")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de LotoIncrementalLearning: {e}", exc_info=True)
            raise
        
    def load_models(self) -> bool:
        """
        Charge les modèles ML en utilisant la méthode de LotoAnalyzer.
        
        Returns:
            bool: True si le chargement a réussi, False sinon.
        """
        logger.info("Tentative de chargement des modèles via LotoAnalyzer...")
        try:
            if hasattr(self.analyzer, '_load_rf_models'):
                # Vérifier si des modèles en cache sont disponibles
                if self.use_pickle_cache:
                    cache_path = self.cache_dir / "models_cache.pkl"
                    if cache_path.exists():
                        try:
                            start_time = time.time()
                            with open(cache_path, 'rb') as f:
                                cache_data = pickle.load(f)
                                self.analyzer.rf_number_model = cache_data.get('rf_number_model')
                                self.analyzer.scaler_numbers = cache_data.get('scaler_numbers')
                                self.analyzer.rf_chance_model = cache_data.get('rf_chance_model')
                                self.analyzer.scaler_chance = cache_data.get('scaler_chance')
                            logger.info(f"Modèles chargés depuis le cache en {time.time() - start_time:.2f} secondes")
                            return True
                        except Exception as e_cache:
                            logger.warning(f"Échec du chargement depuis le cache: {e_cache}. Tentative via LotoAnalyzer.")
                
                # Appelle la méthode de chargement de l'analyseur
                return self.analyzer._load_rf_models() 
            else:
                logger.error("L'instance LotoAnalyzer n'a pas de méthode '_load_rf_models'.")
                return False
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles: {e}", exc_info=True)
            return False
        
    def train_initial_models(self) -> bool:
        """
        Assure que les modèles initiaux sont entraînés si nécessaire.
        Cette méthode s'appuie sur la logique d'entraînement de LotoAnalyzer.
            
        Returns:
            bool: True si les modèles sont prêts (déjà entraînés ou entraînement réussi), False sinon.
        """
        logger.info("Vérification/Entraînement des modèles initiaux via LotoAnalyzer...")
        try:
            # Si les modèles de l'analyzer ne sont pas déjà chargés/entraînés
            if not self.analyzer.rf_number_model:
                logger.info("Modèles non initialisés dans LotoAnalyzer. Tentative d'entraînement initial.")
                min_hist_for_ml = self.config.get("min_history", 50)
                if len(self.analyzer.df) < min_hist_for_ml:
                    logger.warning(f"Données insuffisantes pour l'entraînement initial des modèles ML (minimum {min_hist_for_ml} tirages requis).")
                    return False

                # Vérifier si un cache est disponible
                if self.use_pickle_cache:
                    cache_path = self.cache_dir / "initial_models_cache.pkl"
                    if cache_path.exists():
                        try:
                            start_time = time.time()
                            with open(cache_path, 'rb') as f:
                                cache_data = pickle.load(f)
                                self.analyzer.rf_number_model = cache_data.get('rf_number_model')
                                self.analyzer.scaler_numbers = cache_data.get('scaler_numbers')
                                self.analyzer.rf_chance_model = cache_data.get('rf_chance_model')
                                self.analyzer.scaler_chance = cache_data.get('scaler_chance')
                            logger.info(f"Modèles initiaux chargés depuis le cache en {time.time() - start_time:.2f} secondes")
                            return True
                        except Exception as e_cache:
                            logger.warning(f"Échec du chargement des modèles initiaux depuis le cache: {e_cache}. Tentative d'entraînement.")

                # Appeler la méthode d'entraînement ML de l'analyseur principal.
                # Elle gère l'optimisation et la sauvegarde.
                if hasattr(self.analyzer, '_train_ml_models') and callable(self.analyzer._train_ml_models):
                    start_time = time.time()
                    success = self.analyzer._train_ml_models()
                    training_time = time.time() - start_time
                    
                    if success:
                        logger.info(f"Modèles initiaux entraînés avec succès via LotoAnalyzer en {training_time:.2f} secondes.")
                        
                        # Sauvegarder dans le cache
                        if self.use_pickle_cache:
                            try:
                                cache_data = {
                                    'rf_number_model': self.analyzer.rf_number_model,
                                    'scaler_numbers': self.analyzer.scaler_numbers,
                                    'rf_chance_model': self.analyzer.rf_chance_model,
                                    'scaler_chance': self.analyzer.scaler_chance
                                }
                                with open(self.cache_dir / "initial_models_cache.pkl", 'wb') as f:
                                    pickle.dump(cache_data, f)
                                logger.info("Modèles initiaux sauvegardés dans le cache.")
                            except Exception as e_cache:
                                logger.warning(f"Échec de la sauvegarde des modèles initiaux dans le cache: {e_cache}")
                        
                        return True
                    else:
                        logger.error("Échec de l'entraînement initial des modèles ML via LotoAnalyzer.")
                        return False
                else:
                    logger.error("L'instance LotoAnalyzer n'a pas de méthode '_train_ml_models'.")
                    return False
            else:
                logger.info("Modèles initiaux déjà présents dans LotoAnalyzer.")
                return True
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement des modèles initiaux: {e}", exc_info=True)
            return False
            
    def update_models_with_new_data(self, training_df: pd.DataFrame) -> bool:
        """
        Met à jour (ré-entraîne) les modèles en utilisant le DataFrame fourni.
        S'appuie sur la logique d'entraînement de LotoAnalyzer, avec optimisations.
        
        Args:
            training_df: DataFrame contenant toutes les données à utiliser pour le ré-entraînement.
            
        Returns:
            bool: True si la mise à jour a réussi, False sinon.
        """
        try:
            if training_df is None or training_df.empty:
                logger.warning("DataFrame d'entraînement vide fourni pour update_models_with_new_data.")
                return False
                
            logger.info(f"Mise à jour (ré-entraînement) des modèles avec {len(training_df)} tirages.")
            start_time = time.time()

            # Vérifier et convertir les dates si nécessaire
            if 'Date' in training_df.columns:
                # Créer une copie pour 
(Content truncated due to size limit. Use line ranges to read in chunks)