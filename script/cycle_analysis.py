#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module pour l'analyse des cycles de tirage des numéros du Loto.
Permet d'identifier les numéros sur-représentés ou sous-représentés
sur différentes fenêtres de temps et d'appliquer une pondération adaptative
basée sur les cycles détectés, y compris les cycles lunaires.
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple
import logging
import os
import time
from datetime import datetime, timedelta
import math

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CycleAnalysis")

def calculate_frequencies_in_window(df_window: pd.DataFrame,
                                    number_cols: List[str],
                                    chance_col: Optional[str] = None) -> Dict[str, Counter]:
    """
    Calcule les fréquences des numéros dans une fenêtre de tirages donnée.
    
    Args:
        df_window (pd.DataFrame): DataFrame contenant les tirages dans la fenêtre d'analyse
        number_cols (List[str]): Liste des colonnes contenant les numéros principaux
        chance_col (Optional[str]): Colonne contenant le numéro chance, si applicable
        
    Returns:
        Dict[str, Counter]: Dictionnaire contenant les compteurs de fréquence pour les numéros
                           principaux et les numéros chance (si applicable)
    """
    try:
        # Vérification des données d'entrée
        if df_window.empty:
            logger.warning("DataFrame vide fourni pour le calcul des fréquences")
            return {"main_numbers": Counter(), "chance_numbers": Counter()}
        
        # Vérification des colonnes
        missing_cols = [col for col in number_cols if col not in df_window.columns]
        if missing_cols:
            logger.warning(f"Colonnes manquantes dans le DataFrame: {missing_cols}")
            # Utiliser uniquement les colonnes disponibles
            number_cols = [col for col in number_cols if col in df_window.columns]
            if not number_cols:
                logger.error("Aucune colonne de numéros valide trouvée")
                return {"main_numbers": Counter(), "chance_numbers": Counter()}
        
        # Utilisation de vectorisation pour améliorer les performances
        main_numbers_in_window = []
        for col in number_cols:
            try:
                # Conversion en une seule opération pour optimiser
                numbers = df_window[col].dropna().astype(int).tolist()
                main_numbers_in_window.extend(numbers)
            except Exception as e:
                logger.warning(f"Erreur lors de l'extraction des numéros de la colonne {col}: {e}")
        
        main_freq = Counter(main_numbers_in_window)
        result = {"main_numbers": main_freq}
        
        # Traitement du numéro chance si applicable
        if chance_col and chance_col in df_window.columns:
            try:
                chance_numbers_in_window = df_window[chance_col].dropna().astype(int).tolist()
                result["chance_numbers"] = Counter(chance_numbers_in_window)
            except Exception as e:
                logger.warning(f"Erreur lors du traitement des numéros chance: {e}")
                result["chance_numbers"] = Counter()
        
        return result
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul des fréquences dans la fenêtre: {e}")
        return {"main_numbers": Counter(), "chance_numbers": Counter()}

def analyze_cycles(df: pd.DataFrame,
                   number_cols: List[str],
                   window_size: int,
                   chance_col: Optional[str] = None,
                   min_main_num: int = 1, max_main_num: int = 49,
                   min_chance_num: int = 1, max_chance_num: int = 10) -> Dict[str, Any]:
    """
    Analyse les cycles de tirage en examinant les fréquences des numéros
    dans une fenêtre glissante de `window_size` tirages.
    Compare la fréquence observée à la fréquence théorique attendue.

    Args:
        df (pd.DataFrame): DataFrame des tirages, trié du plus ancien au plus récent.
        number_cols (List[str]): Colonnes des numéros principaux.
        window_size (int): Taille de la fenêtre d'analyse (nombre de tirages).
        chance_col (Optional[str]): Colonne du numéro chance.
        min_main_num, max_main_num: Range des numéros principaux.
        min_chance_num, max_chance_num: Range des numéros chance.

    Returns:
        Dict[str, Any]: Dictionnaire contenant les analyses de cycle pour les
                        numéros principaux et chance. Pour chaque numéro, indique
                        sa fréquence observée dans la dernière fenêtre, la fréquence
                        attendue, et un ratio obs/attendu.
    """
    logger.info(f"Analyse des cycles avec une fenêtre de {window_size} tirages...")
    
    try:
        # Vérification des données d'entrée
        if df.empty:
            logger.warning("DataFrame vide fourni pour l'analyse des cycles")
            return {
                "main_cycle_analysis": {},
                "chance_cycle_analysis": {},
                "window_size_used": 0
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
                    "main_cycle_analysis": {},
                    "chance_cycle_analysis": {},
                    "window_size_used": 0
                }
        
        # Validation et ajustement de la taille de la fenêtre
        if window_size <= 0 or window_size > len(df):
            logger.warning(f"Taille de fenêtre invalide ({window_size}), ajustée à {len(df)}")
            # Prend tous les tirages si la window_size est invalide ou trop grande
            df_window = df
            actual_window_size = len(df)
        else:
            df_window = df.tail(window_size)
            actual_window_size = window_size
        
        if actual_window_size == 0:
            logger.warning("Aucun tirage disponible pour l'analyse")
            return {
                "main_cycle_analysis": {},
                "chance_cycle_analysis": {},
                "window_size_used": 0
            }
        
        # Calcul des fréquences dans la fenêtre
        frequencies_in_window = calculate_frequencies_in_window(df_window, number_cols, chance_col)
        
        # Analyse pour les numéros principaux
        main_cycle_analysis = {}
        # Nombre total de numéros principaux tirés dans la fenêtre
        total_main_numbers_drawn_in_window = actual_window_size * len(number_cols)
        # Fréquence attendue pour un numéro principal si parfaitement aléatoire
        total_distinct_main_numbers = max_main_num - min_main_num + 1
        expected_main_freq_per_num = total_main_numbers_drawn_in_window / total_distinct_main_numbers
        
        # Utilisation de vectorisation pour améliorer les performances
        for num in range(min_main_num, max_main_num + 1):
            observed_freq = frequencies_in_window["main_numbers"].get(num, 0)
            ratio = observed_freq / expected_main_freq_per_num if expected_main_freq_per_num > 0 else 0
            main_cycle_analysis[num] = {
                "observed_freq": observed_freq,
                "expected_freq": round(expected_main_freq_per_num, 2),
                "ratio_obs_exp": round(ratio, 2)
            }
        
        # Analyse pour les numéros chance
        chance_cycle_analysis = {}
        if chance_col and "chance_numbers" in frequencies_in_window:
            total_chance_numbers_drawn_in_window = actual_window_size  # 1 numéro chance par tirage
            total_distinct_chance_numbers = max_chance_num - min_chance_num + 1
            expected_chance_freq_per_num = total_chance_numbers_drawn_in_window / total_distinct_chance_numbers
            
            for num in range(min_chance_num, max_chance_num + 1):
                observed_freq = frequencies_in_window["chance_numbers"].get(num, 0)
                ratio = observed_freq / expected_chance_freq_per_num if expected_chance_freq_per_num > 0 else 0
                chance_cycle_analysis[num] = {
                    "observed_freq": observed_freq,
                    "expected_freq": round(expected_chance_freq_per_num, 2),
                    "ratio_obs_exp": round(ratio, 2)
                }
        
        logger.info(f"Analyse des cycles terminée avec succès (fenêtre de {actual_window_size} tirages)")
        return {
            "main_cycle_analysis": main_cycle_analysis,
            "chance_cycle_analysis": chance_cycle_analysis,
            "window_size_used": actual_window_size
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse des cycles: {e}")
        return {
            "main_cycle_analysis": {},
            "chance_cycle_analysis": {},
            "window_size_used": 0
        }

def analyze_multiple_windows(df: pd.DataFrame,
                            number_cols: List[str],
                            window_sizes: List[int],
                            chance_col: Optional[str] = None,
                            min_main_num: int = 1, max_main_num: int = 49,
                            min_chance_num: int = 1, max_chance_num: int = 10) -> Dict[str, Any]:
    """
    Analyse les cycles sur plusieurs fenêtres de temps différentes.
    
    Args:
        df (pd.DataFrame): DataFrame des tirages, trié du plus ancien au plus récent.
        number_cols (List[str]): Colonnes des numéros principaux.
        window_sizes (List[int]): Liste des tailles de fenêtres à analyser.
        chance_col (Optional[str]): Colonne du numéro chance.
        min_main_num, max_main_num: Range des numéros principaux.
        min_chance_num, max_chance_num: Range des numéros chance.
        
    Returns:
        Dict[str, Any]: Dictionnaire contenant les analyses de cycle pour chaque fenêtre.
    """
    logger.info(f"Analyse des cycles sur {len(window_sizes)} fenêtres différentes...")
    
    try:
        results = {}
        
        for window_size in window_sizes:
            results[f"window_{window_size}"] = analyze_cycles(
                df, number_cols, window_size, chance_col,
                min_main_num, max_main_num, min_chance_num, max_chance_num
            )
        
        logger.info(f"Analyse multi-fenêtres terminée avec succès")
        return results
    
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse multi-fenêtres: {e}")
        return {}

def calculate_adaptive_cycle_weights(df: pd.DataFrame,
                                    number_cols: List[str],
                                    window_sizes: List[int] = [10, 20, 50],
                                    chance_col: Optional[str] = None,
                                    min_main_num: int = 1, max_main_num: int = 49,
                                    min_chance_num: int = 1, max_chance_num: int = 10,
                                    recent_weight: float = 0.5,
                                    medium_weight: float = 0.3,
                                    long_weight: float = 0.2) -> Dict[str, Any]:
    """
    Calcule une pondération adaptative basée sur les cycles de tirage
    en combinant les analyses de plusieurs fenêtres temporelles.
    
    Args:
        df (pd.DataFrame): DataFrame des tirages, trié du plus ancien au plus récent.
        number_cols (List[str]): Colonnes des numéros principaux.
        window_sizes (List[int]): Liste des tailles de fenêtres à analyser (court, moyen, long terme).
        chance_col (Optional[str]): Colonne du numéro chance.
        min_main_num, max_main_num: Range des numéros principaux.
        min_chance_num, max_chance_num: Range des numéros chance.
        recent_weight, medium_weight, long_weight: Poids relatifs des fenêtres.
        
    Returns:
        Dict[str, Any]: Dictionnaire contenant les poids adaptatifs pour chaque numéro.
    """
    logger.info("Calcul des pondérations adaptatives basées sur les cycles...")
    
    try:
        # Vérification des poids
        total_weight = recent_weight + medium_weight + long_weight
        if not np.isclose(total_weight, 1.0):
            logger.warning(f"La somme des poids ({total_weight}) n'est pas égale à 1. Normalisation appliquée.")
            recent_weight /= total_weight
            medium_weight /= total_weight
            long_weight /= total_weight
        
        # Vérification du nombre de fenêtres
        if len(window_sizes) < 3:
            logger.warning(f"Nombre insuffisant de fenêtres ({len(window_sizes)}), ajout de fenêtres par défaut.")
            # Assurer qu'il y a au moins 3 fenêtres (court, moyen, long terme)
            if len(window_sizes) == 0:
                window_sizes = [10, 20, 50]
            elif len(window_sizes) == 1:
                window_sizes = [window_sizes[0], window_sizes[0]*2, window_sizes[0]*5]
            elif len(window_sizes) == 2:
                window_sizes = [window_sizes[0], window_sizes[1], max(window_sizes)*2]
        
        # Tri des fenêtres par taille croissante
        window_sizes = sorted(window_sizes)
        
        # Analyse des cycles pour chaque fenêtre
        multi_window_analysis = analyze_multiple_windows(
            df, number_cols, window_sizes, chance_col,
            min_main_num, max_main_num, min_chance_num, max_chance_num
        )
        
        # Initialisation des résultats
        results = {
            "main_adaptive_weights": {},
            "chance_adaptive_weights": {},
            "window_sizes_used": window_sizes
        }
        
        # Attribution des poids aux fenêtres
        window_weights = {}
        if len(window_sizes) >= 3:
            window_weights[f"window_{window_sizes[0]}"] = recent_weight
            window_weights[f"window_{window_sizes[1]}"] = medium_weight
            window_weights[f"window_{window_sizes[2]}"] = long_weight
        else:
            # Répartition équitable si moins de 3 fenêtres
            equal_weight = 1.0 / len(window_sizes)
            for i, size in enumerate(window_sizes):
                window_weights[f"window_{size}"] = equal_weight
        
        # Calcul des poids adaptatifs pour les numéros principaux
        for num in range(min_main_num, max_main_num + 1):
            weighted_ratio = 0.0
            
            for window_key, weight in window_weights.items():
                if window_key in multi_window_analysis and "main_cycle_analysis" in multi_window_analysis[window_key]:
                    if num in multi_window_analysis[window_key]["main_cycle_analysis"]:
                        ratio = multi_window_analysis[window_key]["main_cycle_analysis"][num]["ratio_obs_exp"]
                        weighted_ratio += weight * ratio
            
            # Normalisation du poids (1.0 = fréquence attendue)
            results["main_adaptive_weights"][num] = weighted_ratio
        
        # Calcul des poids adaptatifs pour les numéros chance
        if chance_col:
            for num in range(min_chance_num, max_chance_num + 1):
                weighted_ratio = 0.0
                
                for window_key, weight in window_weights.items():
                    if window_key in multi_window_analysis and "chance_cycle_analysis" in multi_window_analysis[window_key]:
                        if num in multi_window_analysis[window_key]["chance_cycle_analysis"]:
                            ratio = multi_window_analysis[window_key]["chance_cycle_analysis"][num]["ratio_obs_exp"]
                            weighted_ratio += weight * ratio
                
                # Normalisation du poids (1.0 = fréquence attendue)
                results["chance_adaptive_weights"][num] = weighted_ratio
        
        # Normalisation finale des poids pour qu'ils soient entre 0 et 1
        # Numéros principaux
        main_weights = results["main_adaptive_weights"]
        if main_weights:
            max_weight = max(main_weights.values())
            min_weight = min(main_weights.values())
            weight_range = max_weight - min_weight
            
            if weight_range > 0:
                for num in main_weights:
                    main_weights[num] = (main_weights[num] - min_weight) / weight_range
        
        # Numéros chance
        chance_weights = results["chance_adaptive_weights"]
        if chance_weights:
            max_weight = max(chance_weights.values())
            min_weight = min(chance_weights.values())
            weight_range = max_weight - min_weight
            
            if weight_range > 0:
                for num in chance_weights:
                    chance_weights[num] = (chance_weights[num] - min_weight) / weight_range
        
        logger.info("Pondérations adaptatives calculées avec succès")
        return results
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul des pondérations adaptatives: {e}")
        return {
            "main_adaptive_weights": {},
            "chance_adaptive_weights": {},
            "window_sizes_used": []
        }

def calculate_moon_phase(date_str: str, date_format: str = "%d/%m/%Y") -> float:
    """
    Calcule la phase lunaire pour une date donnée.
    
    Args:
        date_str (str): Date au format spécifié
        date_format (str): Format de la date
        
    Returns:
        float: Phase lunaire (0 = nouvelle lune, 0.5 = pleine lune, 0.25 et 0.75 = quartiers)
    """
    try:
        # Conversion de la date
        date_obj = datetime.strptime(date_str, date_format)
        
        # Calcul de la phase lunaire
        # Algorithme simplifié basé sur le cycle lunaire moyen de 29.53 jours
        # Référence: 6 janvier 2000 = nouvelle lune
        reference_date = datetime(2000, 1, 6)
        days_since_reference = (date_obj - reference_date).days
        lunar_cycle = 29.53  # Durée moyenne d'un cycle lunaire en jours
        
        # Calcul de la phase (0 à 1, où 0 et 1 = nouvelle lune, 0.5 = pleine lune)
        phase = (days_since_reference % lunar_cycle) / lunar_cycle
        
        return phase
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul de la phase lunaire pour {date_str}: {e}")
        return 0.0

def get_moon_phase_name(phase: float) -> str:
    """
    Convertit une valeur numérique de phase lunaire en nom descriptif.
    
    Args:
        phase (float): Phase lunaire (0 à 1)
        
    Returns:
        str: Nom de la phase lunaire
    """
    if phase < 0.03 or phase > 0.97:
        return "Nouvelle lune"
    elif 0.03 <= phase < 0.22:
        return "Premier croissant"
    elif 0.22 <= phase < 0.28:
        return "Premier quartier"
    elif 0.28 <= phase < 0.47:
        return "Gibbeuse croissante"
    elif 0.47 <= phase < 0.53:
        return "Pleine lune"
    elif 0.53 <= phase < 0.72:
        return "Gibbeuse décroissante"
    elif 0.72 <= phase < 0.78:
        return "Dernier quartier"
    else:  # 0.78 <= phase < 0.97
        return "Dernier croissant"

def analyze_lunar_cycles(df: pd.DataFrame, 
                         number_cols: List[str],
                         date_col: str = "Date",
                         date_format: str = "%d/%m/%Y",
                         chance_col: Optional[str] = None,
                         min_main_num: int = 1, max_main_num: int = 49,
                         min_chance_num: int = 1, max_chance_num: int = 10) -> Dict[str, Any]:
    """
    Analyse la corrélation entre les phases lunaires et les numéros tirés.
    
    Args:
        df (pd.DataFrame): DataFrame des tirages
        number_cols (List[str]): Colonnes des numéros principaux
        date_col (str): Colonne contenant la date
        date_format (str): Format de la date
        chance_col (Optional[str]): Colonne du numéro chance
        min_main_num, max_main_num: Range des numéros principaux
        min_chance_num, max_chance_num: Range des numéros chance
        
    Returns:
        Dict[str, Any]: Analyse des corrélations entre phases lunaires et numéros
    """
    logger.info("Analyse des cycles lunaires en cours...")
    
    try:
        # Vérification des données d'entrée
        if df.empty:
            logger.warning("DataFrame vide fourni pour l'analyse des cycles lunaires")
            return {
                "main_lunar_weights": {},
                "chance_lunar_weights": {},
                "phase_distribution": {}
            }
        
        # Vérification des colonnes
        required_cols = [date_col] + number_cols
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Colonnes manquantes dans le DataFrame: {missing_cols}")
            if date_col in missing_cols:
                logger.error(f"Colonne de date '{date_col}' manquante, impossible de calculer les phases lunaires")
                return {
                    "main_lunar_weights": {},
                    "chance_lunar_weights": {},
                    "phase_distribution": {}
                }
        
        # Calcul des phases lunaires pour chaque tirage
        logger.info("Calcul des phases lunaires pour chaque tirage...")
        
        # Initialisation des structures de données
        phase_distribution = {
            "Nouvelle lune": {"count": 0, "numbers": Counter()},
            "Premier croissant": {"count": 0, "numbers": Counter()},
            "Premier quartier": {"count": 0, "numbers": Counter()},
            "Gibbeuse croissante": {"count": 0, "numbers": Counter()},
            "Pleine lune": {"count": 0, "numbers": Counter()},
            "Gibbeuse décroissante": {"count": 0, "numbers": Counter()},
            "Dernier quartier": {"count": 0, "numbers": Counter()},
            "Dernier croissant": {"count": 0, "numbers": Counter()}
        }
        
        # Analyse des tirages par phase lunaire
        for _, row in df.iterrows():
            try:
                # Calcul de la phase lunaire
                date_str = row[date_col]
                phase = calculate_moon_phase(date_str, date_format)
                phase_name = get_moon_phase_name(phase)
                
                # Comptage des phases
                phase_distribution[phase_name]["count"] += 1
                
                # Extraction et comptage des numéros pour cette phase
                for col in number_cols:
                    if col in row and not pd.isna(row[col]):
                        num = int(row[col])
                        phase_distribution[phase_name]["numbers"][num] += 1
                
                # Traitement du numéro chance si applicable
                if chance_col and chance_col in row and not pd.isna(row[chance_col]):
                    chance_num = int(row[chance_col])
                    # Ajout d'un compteur spécifique pour les numéros chance si nécessaire
                    if "chance_numbers" not in phase_distribution[phase_name]:
                        phase_distribution[phase_name]["chance_numbers"] = Counter()
                    phase_distribution[phase_name]["chance_numbers"][chance_num] += 1
            
            except Exception as e:
                logger.warning(f"Erreur lors du traitement d'un tirage pour l'analyse lunaire: {e}")
                continue
        
        # Calcul des poids lunaires pour chaque numéro
        main_lunar_weights = {}
        chance_lunar_weights = {}
        
        # Nombre total de tirages
        total_draws = len(df)
        
        # Calcul des poids pour les numéros principaux
        for num in range(min_main_num, max_main_num + 1):
            # Initialisation du poids
            weight = 0.0
            
            # Calcul du poids basé sur la corrélation avec les phases lunaires
            for phase_name, data in phase_distribution.items():
                if data["count"] > 0:
                    # Fréquence observée du numéro dans cette phase
                    observed_freq = data["numbers"].get(num, 0) / data["count"]
                    
                    # Fréquence attendue du numéro (distribution uniforme)
                    expected_freq = 1 / (max_main_num - min_main_num + 1)
                    
                    # Ratio observé/attendu
                    ratio = observed_freq / expected_freq if expected_freq > 0 else 0
                    
                    # Pondération par l'importance de la phase (plus de poids aux phases extrêmes)
                    phase_importance = 1.0
                    if phase_name in ["Nouvelle lune", "Pleine lune"]:
                        phase_importance = 1.5  # Plus d'importance aux phases extrêmes
                    
                    # Contribution de cette phase au poids total
                    phase_contribution = ratio * phase_importance * (data["count"] / total_draws)
                    weight += phase_contribution
            
            main_lunar_weights[num] = weight
        
        # Calcul des poids pour les numéros chance
        if chance_col:
            for num in range(min_chance_num, max_chance_num + 1):
                # Initialisation du poids
                weight = 0.0
                
                # Calcul du poids basé sur la corrélation avec les phases lunaires
                for phase_name, data in phase_distribution.items():
                    if data["count"] > 0 and "chance_numbers" in data:
                        # Fréquence observée du numéro chance dans cette phase
                        observed_freq = data["chance_numbers"].get(num, 0) / data["count"]
                        
                        # Fréquence attendue du numéro chance (distribution uniforme)
                        expected_freq = 1 / (max_chance_num - min_chance_num + 1)
                        
                        # Ratio observé/attendu
                        ratio = observed_freq / expected_freq if expected_freq > 0 else 0
                        
                        # Pondération par l'importance de la phase
                        phase_importance = 1.0
                        if phase_name in ["Nouvelle lune", "Pleine lune"]:
                            phase_importance = 1.5
                        
                        # Contribution de cette phase au poids total
                        phase_contribution = ratio * phase_importance * (data["count"] / total_draws)
                        weight += phase_contribution
                
                chance_lunar_weights[num] = weight
        
        # Normalisation des poids lunaires (0 à 1)
        if main_lunar_weights:
            max_weight = max(main_lunar_weights.values())
            min_weight = min(main_lunar_weights.values())
            weight_range = max_weight - min_weight
            
            if weight_range > 0:
                for num in main_lunar_weights:
                    main_lunar_weights[num] = (main_lunar_weights[num] - min_weight) / weight_range
        
        if chance_lunar_weights:
            max_weight = max(chance_lunar_weights.values())
            min_weight = min(chance_lunar_weights.values())
            weight_range = max_weight - min_weight
            
            if weight_range > 0:
                for num in chance_lunar_weights:
                    chance_lunar_weights[num] = (chance_lunar_weights[num] - min_weight) / weight_range
        
        logger.info("Analyse des cycles lunaires terminée avec succès")
        return {
            "main_lunar_weights": main_lunar_weights,
            "chance_lunar_weights": chance_lunar_weights,
            "phase_distribution": phase_distribution
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse des cycles lunaires: {e}")
        return {
            "main_lunar_weights": {},
            "chance_lunar_weights": {},
            "phase_distribution": {}
        }

def get_current_moon_phase() -> Dict[str, Any]:
    """
    Calcule la phase lunaire actuelle.
    
    Returns:
        Dict[str, Any]: Informations sur la phase lunaire actuelle
    """
    try:
        # Date actuelle
        current_date = datetime.now()
        date_str = current_date.strftime("%d/%m/%Y")
        
        # Calcul de la phase
        phase = calculate_moon_phase(date_str)
        phase_name = get_moon_phase_name(phase)
        
        # Calcul des dates des prochaines phases importantes
        next_phases = {}
        current_day = current_date
        
        # Recherche des 4 prochaines phases importantes (nouvelle lune, premier quartier, pleine lune, dernier quartier)
        important_phases = [0.0, 0.25, 0.5, 0.75]  # Phases importantes (0=nouvelle lune, 0.5=pleine lune)
        
        for _ in range(30):  # Recherche sur les 30 prochains jours
            current_day += timedelta(days=1)
            day_str = current_day.strftime("%d/%m/%Y")
            day_phase = calculate_moon_phase(day_str)
            
            # Vérification si c'est une phase importante
            for target_phase in important_phases:
                # Tolérance de 0.02 (environ 14 heures) pour capturer la phase
                if abs(day_phase - target_phase) < 0.02 or abs(day_phase - target_phase - 1) < 0.02:
                    phase_name = "Nouvelle lune" if target_phase == 0.0 else \
                                "Premier quartier" if target_phase == 0.25 else \
                                "Pleine lune" if target_phase == 0.5 else \
                                "Dernier quartier"
                    
                    if phase_name not in next_phases:
                        next_phases[phase_name] = current_day.strftime("%d/%m/%Y")
                        # Si on a trouvé les 4 phases, on arrête
                        if len(next_phases) == 4:
                            break
            
            if len(next_phases) == 4:
                break
        
        return {
            "current_phase": phase,
            "current_phase_name": phase_name,
            "next_phases": next_phases
        }
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul de la phase lunaire actuelle: {e}")
        return {
            "current_phase": 0.0,
            "current_phase_name": "Inconnue",
            "next_phases": {}
        }

def combine_weights_with_lunar(cycle_weights: Dict[str, Any], 
                              lunar_weights: Dict[str, Any],
                              lunar_influence: float = 0.3) -> Dict[str, Any]:
    """
    Combine les poids des cycles standard avec les poids lunaires.
    
    Args:
        cycle_weights (Dict[str, Any]): Poids des cycles standard
        lunar_weights (Dict[str, Any]): Poids des cycles lunaires
        lunar_influence (float): Influence relative des cycles lunaires (0 à 1)
        
    Returns:
        Dict[str, Any]: Poids combinés
    """
    logger.info(f"Combinaison des poids avec influence lunaire de {lunar_influence}...")
    
    try:
        # Vérification des données d'entrée
        if not cycle_weights or "main_adaptive_weights" not in cycle_weights:
            logger.warning("Poids des cycles standard manquants ou invalides")
            return cycle_weights
        
        if not lunar_weights or "main_lunar_weights" not in lunar_weights:
            logger.warning("Poids lunaires manquants ou invalides")
            return cycle_weights
        
        # Vérification de l'influence lunaire
        if lunar_influence < 0 or lunar_influence > 1:
            logger.warning(f"Influence lunaire invalide ({lunar_influence}), ajustée à 0.3")
            lunar_influence = 0.3
        
        # Poids des cycles standard
        standard_influence = 1.0 - lunar_influence
        
        # Initialisation des résultats
        combined_weights = {
            "main_combined_weights": {},
            "chance_combined_weights": {},
            "window_sizes_used": cycle_weights.get("window_sizes_used", []),
            "lunar_influence": lunar_influence
        }
        
        # Combinaison des poids pour les numéros principaux
        main_cycle_weights = cycle_weights.get("main_adaptive_weights", {})
        main_lunar_weights = lunar_weights.get("main_lunar_weights", {})
        
        for num in main_cycle_weights:
            cycle_weight = main_cycle_weights.get(num, 0.0)
            lunar_weight = main_lunar_weights.get(num, 0.0)
            
            # Combinaison pondérée
            combined_weight = (standard_influence * cycle_weight) + (lunar_influence * lunar_weight)
            combined_weights["main_combined_weights"][num] = combined_weight
        
        # Combinaison des poids pour les numéros chance
        chance_cycle_weights = cycle_weights.get("chance_adaptive_weights", {})
        chance_lunar_weights = lunar_weights.get("chance_lunar_weights", {})
        
        for num in chance_cycle_weights:
            cycle_weight = chance_cycle_weights.get(num, 0.0)
            lunar_weight = chance_lunar_weights.get(num, 0.0)
            
            # Combinaison pondérée
            combined_weight = (standard_influence * cycle_weight) + (lunar_influence * lunar_weight)
            combined_weights["chance_combined_weights"][num] = combined_weight
        
        # Normalisation finale des poids combinés (0 à 1)
        main_combined = combined_weights["main_combined_weights"]
        if main_combined:
            max_weight = max(main_combined.values())
            min_weight = min(main_combined.values())
            weight_range = max_weight - min_weight
            
            if weight_range > 0:
                for num in main_combined:
                    main_combined[num] = (main_combined[num] - min_weight) / weight_range
        
        chance_combined = combined_weights["chance_combined_weights"]
        if chance_combined:
            max_weight = max(chance_combined.values())
            min_weight = min(chance_combined.values())
            weight_range = max_weight - min_weight
            
            if weight_range > 0:
                for num in chance_combined:
                    chance_combined[num] = (chance_combined[num] - min_weight) / weight_range
        
        logger.info("Combinaison des poids avec influence lunaire terminée avec succès")
        return combined_weights
    
    except Exception as e:
        logger.error(f"Erreur lors de la combinaison des poids avec influence lunaire: {e}")
        return cycle_weights

if __name__ == '__main__':
    # Exemple d'utilisation
    try:
        logger.info("Démarrage de l'analyse des cycles...")
        start_time = time.time()
        
        # Création d'un DataFrame d'exemple
        data = {
            'Jour': ['Lundi', 'Mercredi', 'Samedi', 'Lundi', 'Mercredi', 'Samedi', 'Lundi'],
            'Date': ['01/01/2023', '03/01/2023', '06/01/2023', '08/01/2023', '10/01/2023', '13/01/2023', '15/01/2023'],
            'Numéro 1': [10, 20, 5, 10, 15, 5, 20],
            'Numéro 2': [12, 22, 8, 13, 18, 8, 22],
            'Numéro 3': [15, 25, 12, 18, 22, 12, 25],
            'Numéro 4': [30, 35, 22, 33, 38, 22, 30],
            'Numéro 5': [40, 45, 30, 42, 48, 30, 40],
            'Chance': [1, 5, 8, 1, 7, 8, 5]
        }
        
        df = pd.DataFrame(data)
        number_cols = ['Numéro 1', 'Numéro 2', 'Numéro 3', 'Numéro 4', 'Numéro 5']
        chance_col = 'Chance'
        
        # Analyse des cycles standard
        cycle_weights = calculate_adaptive_cycle_weights(
            df, number_cols, window_sizes=[3, 5, 7], chance_col=chance_col
        )
        
        # Analyse des cycles lunaires
        lunar_weights = analyze_lunar_cycles(
            df, number_cols, date_col='Date', chance_col=chance_col
        )
        
        # Combinaison des poids
        combined_weights = combine_weights_with_lunar(
            cycle_weights, lunar_weights, lunar_influence=0.3
        )
        
        # Affichage des résultats
        logger.info(f"Phase lunaire actuelle: {get_current_moon_phase()['current_phase_name']}")
        logger.info(f"Poids combinés pour les numéros principaux: {combined_weights['main_combined_weights']}")
        
        end_time = time.time()
        logger.info(f"Analyse terminée en {end_time - start_time:.2f} secondes")
    
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du script: {e}")
