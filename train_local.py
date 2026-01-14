#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'entraînement local des modèles Loto
Permet d'entraîner les modèles sur un PC local puis de les transférer vers le VPS
Optimisé pour une utilisation locale avec plus de ressources
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Ajouter le répertoire script au path
current_dir = Path(__file__).parent
script_dir = current_dir / "script"
sys.path.insert(0, str(script_dir))

# Configuration du logging
log_file = current_dir / "train_local.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LocalTrainer")

# Import des modules d'analyse
try:
    from loto_main import LotoAnalyzer
except ImportError:
    logger.warning("Module loto_main non disponible, utilisation de l'encodeur de base")

# Workaround PyArrow si nécessaire
try:
    os.environ['ARROW_NO_IMPORT'] = '1'
except:
    pass


class EnhancedEncoder:
    """Encodeur amélioré pour les features de prédiction Loto"""
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_scalers = {}
        
    def encode_temporal_features(self, df: pd.DataFrame) -> np.ndarray:
        """Encode les features temporelles"""
        try:
            df['Date_obj'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
            features = []
            
            if 'Date_obj' in df.columns:
                features.append(df['Date_obj'].dt.dayofweek.values.reshape(-1, 1))
                features.append(df['Date_obj'].dt.month.values.reshape(-1, 1))
                features.append(df['Date_obj'].dt.day.values.reshape(-1, 1))
                features.append(df['Date_obj'].dt.isocalendar().week.values.reshape(-1, 1))
            
            if features:
                return np.hstack(features)
            else:
                return np.array([]).reshape(len(df), 0)
        except Exception as e:
            logger.warning(f"Erreur lors de l'encodage temporel: {e}")
            return np.array([]).reshape(len(df), 0)
    
    def encode_number_features(self, df: pd.DataFrame, window_size: int = 50) -> np.ndarray:
        """Encode les features numériques (fréquences, patterns, etc.)"""
        try:
            ball_cols = [col for col in df.columns if 'Numéro' in col or 'numéro' in col]
            if not ball_cols:
                ball_cols = [f'Numéro {i}' for i in range(1, 6)]
            ball_cols = [col for col in ball_cols if col in df.columns]
            
            if not ball_cols:
                return np.array([]).reshape(len(df), 0)
            
            features_list = []
            
            for idx in range(len(df)):
                if idx < window_size:
                    recent_df = df.iloc[:idx+1]
                else:
                    recent_df = df.iloc[idx-window_size+1:idx+1]
                
                all_numbers = []
                for col in ball_cols:
                    if col in recent_df.columns:
                        numbers = recent_df[col].dropna().astype(int).tolist()
                        all_numbers.extend(numbers)
                
                freq = Counter(all_numbers)
                
                # Créer les features (fréquence de chaque numéro de 1 à 49)
                number_features = np.zeros(49)
                for num, count in freq.items():
                    if 1 <= num <= 49:
                        number_features[num - 1] = count / len(recent_df) if len(recent_df) > 0 else 0
                
                # Ajouter d'autres features
                if all_numbers:
                    number_features = np.append(number_features, [
                        np.mean(all_numbers) / 49,
                        np.std(all_numbers) / 49 if len(all_numbers) > 1 else 0
                    ])
                else:
                    number_features = np.append(number_features, [0, 0])
                
                # Parité
                parities = [n % 2 for n in all_numbers]
                if parities:
                    number_features = np.append(number_features, np.mean(parities))
                else:
                    number_features = np.append(number_features, 0)
                
                features_list.append(number_features)
            
            return np.array(features_list)
            
        except Exception as e:
            logger.warning(f"Erreur lors de l'encodage numérique: {e}", exc_info=True)
            return np.array([]).reshape(len(df), 0)


class LocalTrainer:
    """Classe pour l'entraînement local des modèles Loto"""
    
    def __init__(self, csv_file: str = "tirages_loto.csv", models_dir: str = None):
        """Initialise l'entraîneur local"""
        self.csv_file = Path(csv_file)
        
        if models_dir:
            self.models_dir = Path(models_dir)
        else:
            self.models_dir = current_dir / "resultats_loto" / "models"
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.encoder = EnhancedEncoder()
        self.df = None
        
        # Modèles
        self.rf_number_model = None
        self.rf_chance_model = None
        self.scaler_numbers = None
        self.scaler_chance = None
        
    def load_data(self) -> bool:
        """Charge les données depuis le CSV"""
        try:
            if not self.csv_file.exists():
                logger.error(f"Fichier CSV introuvable: {self.csv_file}")
                return False
            
            logger.info(f"Chargement des données depuis {self.csv_file}...")
            self.df = pd.read_csv(self.csv_file, encoding='utf-8')
            logger.info(f"✅ Données chargées: {len(self.df)} tirages")
            
            # Vérifier les colonnes nécessaires
            required_cols = ['Date']
            ball_cols = [col for col in self.df.columns if 'Numéro' in col]
            if len(ball_cols) < 5:
                logger.error("Colonnes de numéros insuffisantes")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}", exc_info=True)
            return False
    
    def prepare_features(self, idx: int, window_size: int = 50) -> np.ndarray:
        """Prépare les features pour un tirage donné"""
        try:
            if idx < window_size:
                recent_df = self.df.iloc[:idx]
            else:
                recent_df = self.df.iloc[idx-window_size:idx]
            
            # Features numériques
            numeric_features = self.encoder.encode_number_features(recent_df, window_size)
            
            # Features temporelles
            if idx < len(self.df):
                temporal_df = self.df.iloc[[idx]]
                temporal_features = self.encoder.encode_temporal_features(temporal_df)
                
                if temporal_features.shape[0] > 0:
                    temporal_features = temporal_features[0:1]
                    
                    if numeric_features.shape[0] > 0:
                        combined = np.hstack([numeric_features[-1:], temporal_features])
                    else:
                        combined = temporal_features
                else:
                    combined = numeric_features[-1:] if numeric_features.shape[0] > 0 else np.array([]).reshape(1, -1)
            else:
                combined = numeric_features[-1:] if numeric_features.shape[0] > 0 else np.array([]).reshape(1, -1)
            
            return combined
            
        except Exception as e:
            logger.warning(f"Erreur lors de la préparation des features: {e}")
            return np.array([]).reshape(1, -1)
    
    def train_models(self, n_estimators: int = 200, n_iter: int = 50, cv: int = 5) -> bool:
        """
        Entraîne les modèles sur toutes les données disponibles
        Optimisé pour un PC local avec plus de ressources
        """
        try:
            if self.df is None or len(self.df) < 100:
                logger.error("Pas assez de données pour l'entraînement (minimum 100 tirages)")
                return False
            
            logger.info("=" * 60)
            logger.info("DÉMARRAGE DE L'ENTRAÎNEMENT LOCAL")
            logger.info("=" * 60)
            logger.info(f"Nombre de tirages: {len(self.df)}")
            logger.info(f"Estimateurs: {n_estimators}, Itérations: {n_iter}, CV: {cv}")
            logger.info(f"Modèles seront sauvegardés dans: {self.models_dir}")
            logger.info("=" * 60)
            
            window_size = 50
            X = []
            y_numbers = []
            y_chance = []
            
            # Préparer les données pour chaque tirage
            logger.info("Préparation des features...")
            for i in range(window_size, len(self.df)):
                features = self.prepare_features(i, window_size)
                if features.size == 0 or features.shape[1] == 0:
                    continue
                
                X.append(features.flatten())
                
                # Cibles
                current_draw = self.df.iloc[i]
                
                # Numéros principaux (one-hot encoding)
                ball_cols = [col for col in self.df.columns if 'Numéro' in col]
                draw_numbers = []
                for col in ball_cols:
                    if col in current_draw and pd.notna(current_draw[col]):
                        draw_numbers.append(int(current_draw[col]))
                
                number_target = np.zeros(49)
                for num in draw_numbers:
                    if 1 <= num <= 49:
                        number_target[num - 1] = 1
                
                y_numbers.append(number_target)
                
                # Numéro chance
                if 'Chance' in current_draw and pd.notna(current_draw['Chance']):
                    chance_num = int(current_draw['Chance'])
                    chance_target = np.zeros(10)
                    if 1 <= chance_num <= 10:
                        chance_target[chance_num - 1] = 1
                    y_chance.append(chance_target)
            
            if len(X) < 50:
                logger.error(f"Pas assez de données valides pour l'entraînement: {len(X)}")
                return False
            
            # Convertir en arrays numpy
            X = np.array(X)
            y_numbers = np.array(y_numbers)
            
            logger.info(f"✅ Données préparées: {len(X)} échantillons, {X.shape[1]} features")
            
            # Diviser en train/test
            X_train, X_test, y_train_numbers, y_test_numbers = train_test_split(
                X, y_numbers, test_size=0.2, random_state=42
            )
            
            # Normaliser
            logger.info("Normalisation des features...")
            self.scaler_numbers = StandardScaler()
            X_train_scaled = self.scaler_numbers.fit_transform(X_train)
            X_test_scaled = self.scaler_numbers.transform(X_test)
            
            # Entraîner le modèle pour les numéros
            logger.info("=" * 60)
            logger.info("ENTRAÎNEMENT DU MODÈLE POUR LES NUMÉROS PRINCIPAUX")
            logger.info("=" * 60)
            
            param_distributions = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30, 40],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            base_rf = RandomForestClassifier(random_state=42, n_jobs=-1, verbose=1)
            random_search = RandomizedSearchCV(
                base_rf, param_distributions, n_iter=n_iter, cv=cv,
                random_state=42, n_jobs=-1, verbose=2, scoring='accuracy'
            )
            
            logger.info("Recherche des meilleurs hyperparamètres en cours...")
            random_search.fit(X_train_scaled, y_train_numbers)
            self.rf_number_model = random_search.best_estimator_
            
            logger.info(f"✅ Meilleurs paramètres: {random_search.best_params_}")
            logger.info(f"✅ Meilleur score (CV): {random_search.best_score_:.4f}")
            
            # Évaluer
            logger.info("Évaluation du modèle sur les données de test...")
            y_pred = self.rf_number_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test_numbers, y_pred)
            logger.info(f"✅ Précision du modèle numéros: {accuracy:.4f}")
            
            # Entraîner le modèle pour le numéro chance
            if y_chance and len(y_chance) >= 50:
                logger.info("=" * 60)
                logger.info("ENTRAÎNEMENT DU MODÈLE POUR LE NUMÉRO CHANCE")
                logger.info("=" * 60)
                y_chance = np.array(y_chance)
                
                # Diviser
                _, _, y_train_chance, y_test_chance = train_test_split(
                    X[:len(y_chance)], y_chance, test_size=0.2, random_state=42
                )
                
                self.scaler_chance = StandardScaler()
                X_train_chance_scaled = self.scaler_chance.fit_transform(
                    self.scaler_numbers.transform(X_train[:len(y_train_chance)])
                )
                X_test_chance_scaled = self.scaler_chance.transform(
                    self.scaler_numbers.transform(X_test[:len(y_test_chance)])
                )
                
                base_rf_chance = RandomForestClassifier(random_state=42, n_jobs=-1, verbose=1)
                random_search_chance = RandomizedSearchCV(
                    base_rf_chance, param_distributions, n_iter=n_iter, cv=cv,
                    random_state=42, n_jobs=-1, verbose=2, scoring='accuracy'
                )
                
                logger.info("Recherche des meilleurs hyperparamètres pour le numéro chance...")
                random_search_chance.fit(X_train_chance_scaled, y_train_chance)
                self.rf_chance_model = random_search_chance.best_estimator_
                
                logger.info(f"✅ Meilleurs paramètres chance: {random_search_chance.best_params_}")
                logger.info(f"✅ Meilleur score chance (CV): {random_search_chance.best_score_:.4f}")
                
                y_pred_chance = self.rf_chance_model.predict(X_test_chance_scaled)
                accuracy_chance = accuracy_score(y_test_chance, y_pred_chance)
                logger.info(f"✅ Précision du modèle chance: {accuracy_chance:.4f}")
            else:
                logger.warning("Pas assez de données pour entraîner le modèle chance")
            
            logger.info("=" * 60)
            logger.info("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement: {e}", exc_info=True)
            return False
    
    def save_models(self) -> bool:
        """Sauvegarde les modèles entraînés"""
        try:
            # Sauvegarder les anciens modèles si ils existent
            backup_dir = self.models_dir / "backup"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.rf_number_model and self.scaler_numbers:
                # Backup
                if (self.models_dir / "rf_number_model.joblib").exists():
                    shutil.copy(
                        self.models_dir / "rf_number_model.joblib",
                        backup_dir / f"rf_number_model_backup_{timestamp}.joblib"
                    )
                if (self.models_dir / "scaler_numbers.joblib").exists():
                    shutil.copy(
                        self.models_dir / "scaler_numbers.joblib",
                        backup_dir / f"scaler_numbers_backup_{timestamp}.joblib"
                    )
                
                # Sauvegarder
                joblib.dump(self.rf_number_model, self.models_dir / "rf_number_model.joblib")
                joblib.dump(self.scaler_numbers, self.models_dir / "scaler_numbers.joblib")
                logger.info(f"✅ Modèles de numéros sauvegardés dans {self.models_dir}")
            
            if self.rf_chance_model and self.scaler_chance:
                # Backup
                if (self.models_dir / "rf_chance_model.joblib").exists():
                    shutil.copy(
                        self.models_dir / "rf_chance_model.joblib",
                        backup_dir / f"rf_chance_model_backup_{timestamp}.joblib"
                    )
                if (self.models_dir / "scaler_chance.joblib").exists():
                    shutil.copy(
                        self.models_dir / "scaler_chance.joblib",
                        backup_dir / f"scaler_chance_backup_{timestamp}.joblib"
                    )
                
                # Sauvegarder
                joblib.dump(self.rf_chance_model, self.models_dir / "rf_chance_model.joblib")
                joblib.dump(self.scaler_chance, self.models_dir / "scaler_chance.joblib")
                logger.info(f"✅ Modèles de chance sauvegardés dans {self.models_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}", exc_info=True)
            return False
    
    def train_and_save(self, n_estimators: int = 200, n_iter: int = 50, cv: int = 5) -> bool:
        """Charge les données, entraîne les modèles et les sauvegarde"""
        logger.info("=== DÉMARRAGE DE L'ENTRAÎNEMENT LOCAL ===")
        start_time = datetime.now()
        
        if not self.load_data():
            return False
        
        if not self.train_models(n_estimators=n_estimators, n_iter=n_iter, cv=cv):
            return False
        
        if not self.save_models():
            return False
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"=== ENTRAÎNEMENT TERMINÉ EN {duration/60:.2f} MINUTES ===")
        logger.info(f"Modèles sauvegardés dans: {self.models_dir}")
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entraînement local des modèles Loto')
    parser.add_argument('--csv', default='tirages_loto.csv', help='Fichier CSV des tirages')
    parser.add_argument('--models-dir', default=None, help='Répertoire pour sauvegarder les modèles')
    parser.add_argument('--n-estimators', type=int, default=200, help='Nombre d\'estimateurs pour RandomForest')
    parser.add_argument('--n-iter', type=int, default=50, help='Nombre d\'itérations pour la recherche d\'hyperparamètres')
    parser.add_argument('--cv', type=int, default=5, help='Nombre de folds pour la validation croisée')
    
    args = parser.parse_args()
    
    trainer = LocalTrainer(csv_file=args.csv, models_dir=args.models_dir)
    success = trainer.train_and_save(
        n_estimators=args.n_estimators,
        n_iter=args.n_iter,
        cv=args.cv
    )
    
    if success:
        print("\n✅ Entraînement terminé avec succès!")
        print(f"Modèles disponibles dans: {trainer.models_dir}")
        sys.exit(0)
    else:
        print("\n❌ Erreur lors de l'entraînement. Consultez train_local.log pour plus de détails.")
        sys.exit(1)

