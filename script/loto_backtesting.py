#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de Backtesting Amélioré pour LotoAnalyzer

Ce module fournit un cadre robuste pour évaluer la performance historique
des prédictions générées par LotoAnalyzer. Il simule l'application des
stratégies de prédiction sur des données passées pour mesurer leur
efficacité et identifier des axes d'amélioration.

Version améliorée avec gestion robuste des erreurs, logs détaillés,
et optimisation des performances.
"""

# Ajouté : Permet d'utiliser les noms de types avant leur définition complète
from __future__ import annotations

import os
import sys
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime
from collections import Counter
# Ajouté : TYPE_CHECKING pour les imports conditionnels (évite import circulaire)
from typing import Dict, List, Tuple, Set, Optional, Any, TYPE_CHECKING, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Configuration du logging avec redirection vers fichier et console
logger = logging.getLogger("LotoBacktesting")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("log.txt"),
            logging.StreamHandler()
        ]
    )

# Ajouté : Importe LotoAnalyzer uniquement pour la vérification de type
# Remplacez 'loto_analyzer_improved' par le nom réel de votre module
if TYPE_CHECKING:
    from loto_analyzer_improved import LotoAnalyzer

class LotoBacktesting:
    """
    Implémente le processus de backtesting pour évaluer les prédictions Loto.
    
    Cette classe permet de simuler l'application des stratégies de prédiction
    sur des données historiques pour mesurer leur efficacité et identifier
    des axes d'amélioration.
    """

    # Modifié : Utilise LotoAnalyzer directement comme type hint
    def __init__(self, analyzer: 'LotoAnalyzer'):
        """
        Initialise le module de backtesting.

        Args:
            analyzer: Une instance de la classe LotoAnalyzer.
            
        Raises:
            ValueError: Si l'instance de LotoAnalyzer est None.
            TypeError: Si l'objet fourni n'est pas une instance de LotoAnalyzer.
        """
        try:
            if analyzer is None:
                raise ValueError("L'instance de LotoAnalyzer ne peut pas être None.")

            # Vérification dynamique du type
            analyzer_class_name = analyzer.__class__.__name__
            if not analyzer_class_name.endswith('LotoAnalyzer'):
                logger.warning(f"L'objet fourni ({analyzer_class_name}) pourrait ne pas être une instance de LotoAnalyzer.")

            self.analyzer = analyzer
            self.df = analyzer.df.copy() if hasattr(analyzer, 'df') and analyzer.df is not None else pd.DataFrame()
            
            if self.df.empty:
                logger.warning("Le DataFrame de l'analyseur est vide.")
            
            self.ball_cols = getattr(analyzer, 'ball_cols', [])
            self.chance_col = getattr(analyzer, 'chance_col', None)
            self.config = getattr(analyzer, 'config', {})

            # Création des répertoires de sortie
            self.output_dir = Path(getattr(analyzer, 'output_dir', 'resultats_loto')) / "backtesting"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Création du répertoire pour les graphiques
            self.plots_dir = self.output_dir / "plots"
            self.plots_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Répertoire de backtesting prêt : {self.output_dir}")
            logger.info(f"Répertoire de graphiques prêt : {self.plots_dir}")

            # Initialisation des structures de données pour les résultats
            self.backtest_results: List[Dict] = []
            self.global_stats: Dict = {}
            self.error_analysis: Dict = {}
            
            # Paramètres de performance
            self.use_parallel = self.config.get("use_parallel", False)
            self.batch_size = self.config.get("batch_size", 10)
            
            logger.info("Module de backtesting initialisé avec succès.")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du module de backtesting: {e}", exc_info=True)
            raise

    def _convert_to_int_list(self, number_list: List[Any]) -> List[int]:
        """
        Convertit une liste de nombres en liste d'entiers Python.
        
        Args:
            number_list: Liste de nombres à convertir.
            
        Returns:
            Liste d'entiers Python.
        """
        try:
            return [int(num) for num in number_list if pd.notna(num)]
        except Exception as e:
            logger.error(f"Erreur lors de la conversion en liste d'entiers: {e}", exc_info=True)
            return []

    def _calculate_combination_rank(self, combo_with_chance: List[int], actual_draw_dict: Dict) -> Optional[str]:
        """
        Calcule le rang d'une combinaison Loto.
        
        Args:
            combo_with_chance: Combinaison avec numéro chance.
            actual_draw_dict: Dictionnaire du tirage réel.
            
        Returns:
            Rang de la combinaison ou None en cas d'erreur.
        """
        try:
            if hasattr(self.analyzer, '_calculate_combination_rank') and callable(getattr(self.analyzer, '_calculate_combination_rank')):
                return self.analyzer._calculate_combination_rank(combo_with_chance, actual_draw_dict)
            else:
                # Implémentation de secours si la méthode n'existe pas dans l'analyseur
                logger.warning("Méthode '_calculate_combination_rank' non trouvée dans l'analyseur. Utilisation de l'implémentation de secours.")
                return self._fallback_calculate_rank(combo_with_chance, actual_draw_dict)
        except Exception as e:
            logger.error(f"Erreur lors du calcul du rang : {e}", exc_info=True)
            return None

    def _fallback_calculate_rank(self, combo_with_chance: List[int], actual_draw_dict: Dict) -> Optional[str]:
        """
        Implémentation de secours pour calculer le rang d'une combinaison Loto.
        
        Args:
            combo_with_chance: Combinaison avec numéro chance.
            actual_draw_dict: Dictionnaire du tirage réel.
            
        Returns:
            Rang de la combinaison ou None en cas d'erreur.
        """
        try:
            # Extraction des numéros principaux et du numéro chance
            if not combo_with_chance or len(combo_with_chance) < 1:
                return None
                
            combo_nums = combo_with_chance[:-1] if len(combo_with_chance) > 1 else combo_with_chance
            combo_chance = combo_with_chance[-1] if len(combo_with_chance) > 1 else None
            
            # Extraction des numéros réels
            actual_nums = []
            actual_chance = None
            
            for col in self.ball_cols:
                if col in actual_draw_dict and pd.notna(actual_draw_dict[col]):
                    actual_nums.append(int(actual_draw_dict[col]))
            
            if self.chance_col and self.chance_col in actual_draw_dict and pd.notna(actual_draw_dict[self.chance_col]):
                actual_chance = int(actual_draw_dict[self.chance_col])
            
            # Calcul du rang
            correct_nums = len(set(combo_nums) & set(actual_nums))
            correct_chance = 1 if combo_chance == actual_chance else 0
            
            # Détermination du rang selon les règles du Loto
            if correct_nums == 5 and correct_chance == 1:
                return "5+1"
            elif correct_nums == 5:
                return "5"
            elif correct_nums == 4 and correct_chance == 1:
                return "4+1"
            elif correct_nums == 4:
                return "4"
            elif correct_nums == 3 and correct_chance == 1:
                return "3+1"
            elif correct_nums == 3:
                return "3"
            elif correct_nums == 2 and correct_chance == 1:
                return "2+1"
            elif correct_nums == 2:
                return "2"
            elif correct_nums == 1 and correct_chance == 1:
                return "1+1"
            elif correct_chance == 1:
                return "0+1"
            else:
                return None
        except Exception as e:
            logger.error(f"Erreur dans l'implémentation de secours du calcul de rang: {e}", exc_info=True)
            return None

    def _convert_rank_to_numeric(self, rank_str: Optional[str]) -> int:
        """
        Convertit un rang (str) en score numérique pour la comparaison.
        
        Args:
            rank_str: Rang sous forme de chaîne.
            
        Returns:
            Score numérique correspondant au rang.
        """
        try:
            rank_scores = {
                "5+1": 100, "5": 90, "4+1": 80, "4": 70,
                "3+1": 60, "3": 50, "2+1": 40, "2": 30,
                "1+1": 20, "0+1": 10
            }
            return rank_scores.get(rank_str, 0) if rank_str else 0
        except Exception as e:
            logger.error(f"Erreur lors de la conversion du rang en score numérique: {e}", exc_info=True)
            return 0

    def run_backtesting(self, min_history: int = 100, step_size: int = 1) -> Optional[Dict]:
        """
        Exécute le backtesting sur les données historiques.
        
        Args:
            min_history: Nombre minimum de tirages historiques requis.
            step_size: Pas d'incrémentation pour le backtesting.
            
        Returns:
            Dictionnaire des résultats de backtesting ou None en cas d'erreur.
        """
        start_time = time.time()
        logger.info(f"Démarrage du backtesting : min_history={min_history}, step_size={step_size}")

        try:
            if self.df.empty:
                logger.error("Le DataFrame est vide. Impossible de procéder au backtesting.")
                return None
                
            if len(self.df) < min_history + 1:
                logger.error(f"Données insuffisantes : {len(self.df)} tirages, {min_history + 1} requis.")
                return None

            self.backtest_results = []
            
            # Vérification des colonnes nécessaires
            if not self.ball_cols:
                logger.error("Aucune colonne de boules définie. Impossible de procéder au backtesting.")
                return None
                
            if 'Date' not in self.df.columns:
                logger.warning("Colonne 'Date' non trouvée dans le DataFrame. Les dates seront remplacées par des indices.")

            # Préparation pour le traitement par lots si activé
            total_draws = len(self.df) - min_history
            batch_count = (total_draws + self.batch_size - 1) // self.batch_size if self.use_parallel else 1
            
            if batch_count > 1:
                logger.info(f"Traitement par lots activé: {batch_count} lots de {self.batch_size} tirages")

            for batch_idx in range(batch_count):
                batch_start = min_history + batch_idx * self.batch_size
                batch_end = min(len(self.df), batch_start + self.batch_size)
                
                if batch_count > 1:
                    logger.info(f"Traitement du lot {batch_idx+1}/{batch_count}: tirages {batch_start+1} à {batch_end}")
                
                for i in range(batch_start, batch_end, step_size):
                    logger.info(f"Backtesting : Évaluation du tirage {i + 1}/{len(self.df)}")

                    # Préparation des données d'entraînement et du tirage à évaluer
                    train_df = self.df.iloc[:i].copy()
                    actual_draw_row = self.df.iloc[i]

                    # Configuration de l'analyseur temporaire
                    temp_analyzer_config = self.config.copy()
                    temp_analyzer_config["window_draws"] = min(self.config.get("window_draws", 1328), len(train_df))
                    
                    # Import dynamique de LotoAnalyzer
                    try:
                        # Essayer d'abord improved_loto_analyzer_improved
                        try:
                            from improved_loto_analyzer_improved import LotoAnalyzer as TempAnalyzer
                            logger.debug("Utilisation de improved_loto_analyzer_improved.LotoAnalyzer")
                        except ImportError:
                            # Sinon, essayer loto_analyzer_improved
                            try:
                                from loto_analyzer_improved import LotoAnalyzer as TempAnalyzer
                                logger.debug("Utilisation de loto_analyzer_improved.LotoAnalyzer")
                            except ImportError:
                                # Dernier recours: utiliser le même module que l'analyseur principal
                                module_name = self.analyzer.__class__.__module__
                                if module_name:
                                    module = __import__(module_name, fromlist=['LotoAnalyzer'])
                                    TempAnalyzer = getattr(module, 'LotoAnalyzer')
                                    logger.debug(f"Utilisation de {module_name}.LotoAnalyzer")
                                else:
                                    raise ImportError("Impossible de déterminer le module de LotoAnalyzer")
                    except Exception as e:
                        logger.error(f"Erreur lors de l'import de LotoAnalyzer: {e}", exc_info=True)
                        continue

                    # Création et configuration de l'analyseur temporaire
                    try:
                        temp_analyzer = TempAnalyzer(temp_analyzer_config)
                        temp_analyzer.df = train_df
                        
                        # Copie des attributs essentiels si nécessaire
                        if not hasattr(temp_analyzer, 'ball_cols') or not temp_analyzer.ball_cols:
                            temp_analyzer.ball_cols = self.ball_cols
                        if not hasattr(temp_analyzer, 'chance_col') or not temp_analyzer.chance_col:
                            temp_analyzer.chance_col = self.chance_col
                    except Exception as e:
                        logger.error(f"Erreur lors de la création de l'analyseur temporaire: {e}", exc_info=True)
                        continue

                    # Exécution de l'analyse
                    try:
                        if not temp_analyzer.run_analysis():
                            logger.warning(f"L'analyse temporaire a échoué pour le tirage {i}. Skip.")
                            continue
                    except Exception as e:
                        logger.error(f"Erreur lors de l'exécution de l'analyse temporaire: {e}", exc_info=True)
                        continue

                    # Extraction des numéros réels
                    try:
                        actual_numbers = self._convert_to_int_list(actual_draw_row[self.ball_cols].tolist())
                        actual_chance = int(actual_draw_row[self.chance_col]) if self.chance_col and pd.notna(actual_draw_row[self.chance_col]) else None
                    except Exception as e:
                        logger.error(f"Erreur lors de l'extraction des numéros réels: {e}", exc_info=True)
                        continue

                    # Génération des prédictions
                    try:
                        predicted_combinations = temp_analyzer.generate_multiple_combinations(
                            count=self.config.get("combinations_to_generate", 5)
                        )
                    except Exception as e:
                        logger.error(f"Erreur lors de la generation des predictions: {e}", exc_info=True)
                        continue

            # NOTE (audit C2) : la suite de run_backtesting (scoring/agregation) a ete
            # perdue par une troncature de fichier. Utilisez script/loto_evaluation.py
            # (walk-forward + Monte-Carlo) comme evaluateur de reference.
            logger.error("run_backtesting incomplet (fichier tronque) : voir script/loto_evaluation.py")
            return None
        except Exception as e:
            logger.error(f"Erreur lors du backtesting: {e}", exc_info=True)
            return None
