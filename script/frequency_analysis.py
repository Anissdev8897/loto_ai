#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module pour l'analyse détaillée des fréquences et des écarts des numéros du Loto.
Ce module permet d'analyser les tendances statistiques des tirages du Loto
et inclut une pondération adaptative basée sur la fréquence et les cycles.
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional
import logging
import os
import time
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FrequencyAnalysis")

def calculate_number_frequencies(df: pd.DataFrame, number_cols: List[str], chance_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Calcule les fréquences absolues et relatives de chaque numéro et numéro chance.

    Args:
        df (pd.DataFrame): DataFrame contenant l'historique des tirages.
                           Doit contenir les colonnes de numéros et, si applicable, la colonne chance.
        number_cols (List[str]): Liste des noms des colonnes pour les numéros principaux.
        chance_col (Optional[str]): Nom de la colonne pour le numéro chance.

    Returns:
        Dict[str, Any]: Un dictionnaire contenant :
            - 'main_numbers_frequency_absolute': Counter des fréquences absolues des numéros principaux.
            - 'main_numbers_frequency_relative': Dict des fréquences relatives des numéros principaux.
            - 'chance_numbers_frequency_absolute': Counter des fréquences absolues des numéros chance (si applicable).
            - 'chance_numbers_frequency_relative': Dict des fréquences relatives des numéros chance (si applicable).
    """
    logger.info("Calcul des fréquences des numéros...")
    
    try:
        # Vérification des données d'entrée
        if df.empty:
            logger.warning("DataFrame vide fourni pour le calcul des fréquences")
            return {
                "main_numbers_frequency_absolute": Counter(),
                "main_numbers_frequency_relative": {},
                "chance_numbers_frequency_absolute": None,
                "chance_numbers_frequency_relative": None,
            }
        
        # Vérification des colonnes
        missing_cols = [col for col in number_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Colonnes manquantes dans le DataFrame: {missing_cols}")
            # Utiliser uniquement les colonnes disponibles
            number_cols = [col for col in number_cols if col in df.columns]
            if not number_cols:
                logger.error("Aucune colonne de numéros valide trouvée")
                return {
                    "main_numbers_frequency_absolute": Counter(),
                    "main_numbers_frequency_relative": {},
                    "chance_numbers_frequency_absolute": None,
                    "chance_numbers_frequency_relative": None,
                }
        
        # Extraction et conversion des numéros principaux
        all_main_numbers = []
        for col in number_cols:
            try:
                # Utilisation de vectorisation pour améliorer les performances
                numbers = df[col].dropna().astype(int).tolist()
                all_main_numbers.extend(numbers)
            except Exception as e:
                logger.warning(f"Erreur lors de l'extraction des numéros de la colonne {col}: {e}")
        
        # Calcul des fréquences pour les numéros principaux
        main_freq_abs = Counter(all_main_numbers)
        total_main_draws = len(all_main_numbers)
        
        if total_main_draws == 0:
            logger.warning("Aucun numéro principal valide trouvé")
            main_freq_rel = {}
        else:
            main_freq_rel = {num: count / total_main_draws for num, count in main_freq_abs.items()}
        
        results = {
            "main_numbers_frequency_absolute": main_freq_abs,
            "main_numbers_frequency_relative": main_freq_rel,
            "chance_numbers_frequency_absolute": None,
            "chance_numbers_frequency_relative": None,
        }
        
        # Traitement du numéro chance si applicable
        if chance_col and chance_col in df.columns:
            try:
                all_chance_numbers = df[chance_col].dropna().astype(int).tolist()
                chance_freq_abs = Counter(all_chance_numbers)
                total_chance_draws = len(all_chance_numbers)
                
                if total_chance_draws == 0:
                    logger.warning("Aucun numéro chance valide trouvé")
                    chance_freq_rel = {}
                else:
                    chance_freq_rel = {num: count / total_chance_draws for num, count in chance_freq_abs.items()}
                
                results["chance_numbers_frequency_absolute"] = chance_freq_abs
                results["chance_numbers_frequency_relative"] = chance_freq_rel
            except Exception as e:
                logger.warning(f"Erreur lors du traitement des numéros chance: {e}")
        
        logger.info(f"Fréquences calculées pour {total_main_draws} numéros principaux")
        return results
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul des fréquences: {e}")
        return {
            "main_numbers_frequency_absolute": Counter(),
            "main_numbers_frequency_relative": {},
            "chance_numbers_frequency_absolute": None,
            "chance_numbers_frequency_relative": None,
        }

def calculate_gaps(df: pd.DataFrame, number_cols: List[str], chance_col: Optional[str] = None,
                   min_main_num: int = 1, max_main_num: int = 49,
                   min_chance_num: int = 1, max_chance_num: int = 10) -> Dict[str, Any]:
    """
    Calcule les écarts de sortie pour chaque numéro.
    L'écart est le nombre de tirages depuis la dernière apparition d'un numéro.

    Args:
        df (pd.DataFrame): DataFrame des tirages, trié du plus ancien au plus récent.
        number_cols (List[str]): Colonnes des numéros principaux.
        chance_col (Optional[str]): Colonne du numéro chance.
        min_main_num (int): Numéro principal minimum possible.
        max_main_num (int): Numéro principal maximum possible.
        min_chance_num (int): Numéro chance minimum possible.
        max_chance_num (int): Numéro chance maximum possible.

    Returns:
        Dict[str, Any]: Dictionnaire contenant :
            - 'main_current_gaps': Dict des écarts actuels pour les numéros principaux.
            - 'main_gap_stats': Dict des statistiques d'écarts (moy, med, max, min) pour les numéros principaux.
            - 'chance_current_gaps': Dict des écarts actuels pour les numéros chance.
            - 'chance_gap_stats': Dict des statistiques d'écarts pour les numéros chance.
    """
    logger.info("Calcul des écarts entre les tirages...")
    
    try:
        # Vérification des données d'entrée
        if df.empty:
            logger.warning("DataFrame vide fourni pour le calcul des écarts")
            return {
                "main_current_gaps": {}, "main_gap_stats": {},
                "chance_current_gaps": {}, "chance_gap_stats": {}
            }
        
        # Vérification des colonnes
        missing_cols = [col for col in number_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Colonnes manquantes dans le DataFrame: {missing_cols}")
            # Utiliser uniquement les colonnes disponibles
            number_cols = [col for col in number_cols if col in df.columns]
            if not number_cols:
                logger.error("Aucune colonne de numéros valide trouvée")
                return {
                    "main_current_gaps": {}, "main_gap_stats": {},
                    "chance_current_gaps": {}, "chance_gap_stats": {}
                }
        
        results = {
            "main_current_gaps": {}, "main_gap_stats": {},
            "chance_current_gaps": {}, "chance_gap_stats": {}
        }
        
        # --- Analyse des numéros principaux ---
        all_possible_main_numbers = list(range(min_main_num, max_main_num + 1))
        last_seen_main = {num: -1 for num in all_possible_main_numbers}
        gaps_history_main = {num: [] for num in all_possible_main_numbers}
        
        # Utilisation de vectorisation pour améliorer les performances
        # Préparation des données pour un traitement plus rapide
        numbers_by_row = []
        for _, row in df.iterrows():
            current_draw_numbers = set()
            for col in number_cols:
                if pd.notna(row[col]):
                    try:
                        current_draw_numbers.add(int(row[col]))
                    except (ValueError, TypeError):
                        # Ignorer les valeurs non convertibles
                        pass
            numbers_by_row.append(current_draw_numbers)
        
        # Calcul des écarts
        for index, current_draw_numbers in enumerate(numbers_by_row):
            for num in all_possible_main_numbers:
                if num in current_draw_numbers:
                    if last_seen_main[num] != -1:  # Si ce n'est pas la première fois qu'on le voit
                        gaps_history_main[num].append(index - last_seen_main[num] - 1)
                    last_seen_main[num] = index
        
        # Calcul des écarts actuels
        current_index = len(df) - 1
        for num in all_possible_main_numbers:
            if last_seen_main[num] != -1:  # Si le numéro a déjà été vu
                results["main_current_gaps"][num] = current_index - last_seen_main[num]
            else:  # Si le numéro n'a jamais été vu (improbable avec un historique long mais géré)
                results["main_current_gaps"][num] = len(df)
        
        # Calcul des statistiques d'écarts
        for num in all_possible_main_numbers:
            if gaps_history_main[num]:
                # Utilisation de numpy pour des calculs plus rapides
                gaps_array = np.array(gaps_history_main[num])
                results["main_gap_stats"][num] = {
                    "mean": float(np.mean(gaps_array)),
                    "median": float(np.median(gaps_array)),
                    "max": int(np.max(gaps_array)),
                    "min": int(np.min(gaps_array)),
                    "std": float(np.std(gaps_array)),
                    "count": len(gaps_array)
                }
            else:
                results["main_gap_stats"][num] = {
                    "mean": None, "median": None, "max": None, "min": None, "std": None, "count": 0
                }
        
        # --- Analyse des numéros Chance (si applicable) ---
        if chance_col and chance_col in df.columns:
            all_possible_chance_numbers = list(range(min_chance_num, max_chance_num + 1))
            last_seen_chance = {num: -1 for num in all_possible_chance_numbers}
            gaps_history_chance = {num: [] for num in all_possible_chance_numbers}
            
            # Préparation des données pour un traitement plus rapide
            chance_numbers_by_row = []
            for _, row in df.iterrows():
                current_chance_num = None
                if pd.notna(row[chance_col]):
                    try:
                        current_chance_num = int(row[chance_col])
                    except (ValueError, TypeError):
                        # Ignorer les valeurs non convertibles
                        pass
                chance_numbers_by_row.append(current_chance_num)
            
            # Calcul des écarts
            for index, current_chance_num in enumerate(chance_numbers_by_row):
                for num in all_possible_chance_numbers:
                    if num == current_chance_num:
                        if last_seen_chance[num] != -1:
                            gaps_history_chance[num].append(index - last_seen_chance[num] - 1)
                        last_seen_chance[num] = index
            
            # Calcul des écarts actuels
            current_index = len(df) - 1
            for num in all_possible_chance_numbers:
                if last_seen_chance[num] != -1:
                    results["chance_current_gaps"][num] = current_index - last_seen_chance[num]
                else:
                    results["chance_current_gaps"][num] = len(df)
            
            # Calcul des statistiques d'écarts
            for num in all_possible_chance_numbers:
                if gaps_history_chance[num]:
                    # Utilisation de numpy pour des calculs plus rapides
                    gaps_array = np.array(gaps_history_chance[num])
                    results["chance_gap_stats"][num] = {
                        "mean": float(np.mean(gaps_array)),
                        "median": float(np.median(gaps_array)),
                        "max": int(np.max(gaps_array)),
                        "min": int(np.min(gaps_array)),
                        "std": float(np.std(gaps_array)),
                        "count": len(gaps_array)
                    }
                else:
                    results["chance_gap_stats"][num] = {
                        "mean": None, "median": None, "max": None, "min": None, "std": None, "count": 0
                    }
        
        logger.info(f"Écarts calculés pour {len(all_possible_main_numbers)} numéros principaux")
        return results
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul des écarts: {e}")
        return {
            "main_current_gaps": {}, "main_gap_stats": {},
            "chance_current_gaps": {}, "chance_gap_stats": {}
        }

def analyze_returns(df: pd.DataFrame, number_cols: List[str], chance_col: Optional[str] = None,
                    min_main_num: int = 1, max_main_num: int = 49,
                    min_chance_num: int = 1, max_chance_num: int = 10) -> Dict[str, Any]:
    """
    Analyse les "retours" des numéros : combien de tirages s'écoulent typiquement
    avant qu'un numéro ne réapparaisse après une sortie.
    Ceci est similaire aux écarts, mais se concentre sur les intervalles entre apparitions successives.

    Args:
        df (pd.DataFrame): DataFrame des tirages, trié du plus ancien au plus récent.
        number_cols (List[str]): Colonnes des numéros principaux.
        chance_col (Optional[str]): Colonne du numéro chance.
        min_main_num (int), max_main_num (int): Range des numéros principaux.
        min_chance_num (int), max_chance_num (int): Range des numéros chance.

    Returns:
        Dict[str, Any]: Dictionnaire contenant :
            - 'main_returns_stats': Stats sur les retours des numéros principaux.
            - 'chance_returns_stats': Stats sur les retours des numéros chance.
    """
    logger.info("Analyse des retours des numéros...")
    
    try:
        # Les statistiques de "retours" sont essentiellement les statistiques des écarts (gaps_history)
        # calculées dans la fonction calculate_gaps.
        # La fonction calculate_gaps calcule déjà l'historique des écarts entre les apparitions.
        # On peut donc réutiliser cette logique.
        
        gap_data = calculate_gaps(df, number_cols, chance_col, min_main_num, max_main_num, min_chance_num, max_chance_num)
        
        logger.info("Analyse des retours terminée")
        return {
            "main_returns_stats": gap_data["main_gap_stats"],
            "chance_returns_stats": gap_data["chance_gap_stats"]
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse des retours: {e}")
        return {
            "main_returns_stats": {},
            "chance_returns_stats": {}
        }

# NOTE (audit C2) : la fonction calculate_adaptive_weights, laissee incomplete par un
# marqueur de troncature ecrit dans le fichier, a ete retiree. Elle n'etait importee
# nulle part (le fibonacci_weighting.calculate_adaptive_weights est une autre fonction).
