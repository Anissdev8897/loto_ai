#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de pondération Fibonacci pour l'analyse des tirages du Loto.
Ce module fournit des fonctions pour appliquer une pondération inverse de Fibonacci
aux fréquences des numéros, ce qui peut être utile pour l'analyse des tirages.
Il intègre également une couche de pondération adaptative basée sur la fréquence et les cycles.
"""

import logging
import numpy as np
import os
import time
from collections import Counter
from typing import Dict, List, Union, Any, Optional, Tuple

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FibonacciWeighting")

def fibonacci(n: int) -> int:
    """
    Calcule le n-ième nombre de Fibonacci de manière optimisée.
    
    Args:
        n (int): Position dans la séquence de Fibonacci (commence à 0)
        
    Returns:
        int: Le n-ième nombre de Fibonacci
    """
    try:
        # Validation de l'entrée
        if not isinstance(n, int):
            logger.warning(f"Valeur non entière fournie à fibonacci(): {n}")
            n = int(n)
        
        if n < 0:
            logger.warning(f"Valeur négative fournie à fibonacci(): {n}")
            return 0
        elif n == 0:
            return 0
        elif n == 1:
            return 1
        
        # Utilisation d'une approche itérative pour de meilleures performances
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul de fibonacci({n}): {e}")
        return 0

def apply_inverse_fibonacci_weights(counts: Counter, reverse_order: bool = True) -> Dict[int, float]:
    """
    Applique une pondération inverse de Fibonacci aux compteurs.
    
    Cette fonction prend un Counter (dictionnaire de compteurs) et applique
    une pondération inverse basée sur la séquence de Fibonacci. Les éléments
    sont d'abord triés par fréquence, puis une pondération est appliquée
    en fonction de leur rang.
    
    Args:
        counts (Counter): Dictionnaire de compteurs (numéro -> fréquence)
        reverse_order (bool): Si True, les éléments les plus fréquents reçoivent
                             les poids les plus faibles (pondération inverse).
                             Si False, les éléments les plus fréquents reçoivent
                             les poids les plus élevés.
    
    Returns:
        Dict[int, float]: Dictionnaire des poids (numéro -> poids)
    """
    logger.info("Application de la pondération inverse de Fibonacci...")
    
    try:
        # Vérification des données d'entrée
        if not counts:
            logger.warning("Counter vide fourni à apply_inverse_fibonacci_weights")
            return {}
        
        # Trier les éléments par fréquence
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=not reverse_order)
        
        # Calculer les poids de Fibonacci
        weights = {}
        for i, (item, _) in enumerate(sorted_items):
            try:
                # Utiliser i+2 pour éviter les premiers nombres de Fibonacci qui sont petits
                fib_value = fibonacci(i + 2)
                # Inverser pour que les éléments moins fréquents aient des poids plus élevés
                weights[item] = 1.0 / fib_value if fib_value > 0 else 0.0
            except Exception as e:
                logger.warning(f"Erreur lors du calcul du poids pour l'élément {item}: {e}")
                weights[item] = 0.0
        
        # Normaliser les poids pour qu'ils somment à 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            logger.warning("Somme des poids nulle, normalisation impossible")
        
        logger.info(f"Pondération inverse de Fibonacci appliquée à {len(counts)} éléments")
        return weights
    
    except Exception as e:
        logger.error(f"Erreur lors de l'application de la pondération inverse de Fibonacci: {e}")
        return {}

def apply_weighted_blend(frequency_weights: Dict[int, float], 
                         fibonacci_weights: Dict[int, float], 
                         blend_factor: float = 0.3) -> Dict[int, float]:
    """
    Combine les poids de fréquence et les poids de Fibonacci avec un facteur de mélange.
    
    Args:
        frequency_weights (Dict[int, float]): Poids basés sur la fréquence
        fibonacci_weights (Dict[int, float]): Poids basés sur Fibonacci
        blend_factor (float): Facteur de mélange entre 0 et 1.
                             0 = uniquement fréquence, 1 = uniquement Fibonacci
    
    Returns:
        Dict[int, float]: Poids combinés
    """
    logger.info(f"Combinaison des poids avec un facteur de mélange de {blend_factor}...")
    
    try:
        # Vérification des données d'entrée
        if not frequency_weights and not fibonacci_weights:
            logger.warning("Dictionnaires de poids vides fournis à apply_weighted_blend")
            return {}
        
        if not frequency_weights:
            logger.warning("Dictionnaire de poids de fréquence vide, utilisation uniquement des poids de Fibonacci")
            return fibonacci_weights
        
        if not fibonacci_weights:
            logger.warning("Dictionnaire de poids de Fibonacci vide, utilisation uniquement des poids de fréquence")
            return frequency_weights
        
        # S'assurer que le facteur de mélange est entre 0 et 1
        blend_factor = max(0.0, min(1.0, blend_factor))
        
        # Combiner les poids
        combined_weights = {}
        all_keys = set(frequency_weights.keys()) | set(fibonacci_weights.keys())
        
        for key in all_keys:
            freq_weight = frequency_weights.get(key, 0.0)
            fib_weight = fibonacci_weights.get(key, 0.0)
            combined_weights[key] = (1 - blend_factor) * freq_weight + blend_factor * fib_weight
        
        logger.info(f"Combinaison des poids terminée pour {len(all_keys)} éléments")
        return combined_weights
    
    except Exception as e:
        logger.error(f"Erreur lors de la combinaison des poids: {e}")
        return frequency_weights or fibonacci_weights or {}

def calculate_adaptive_weights(counts: Counter, 
                              cycle_data: Optional[Dict[int, Dict[str, float]]] = None,
                              frequency_weight: float = 0.4,
                              fibonacci_weight: float = 0.3,
                              cycle_weight: float = 0.3) -> Dict[int, float]:
    """
    Calcule une pondération adaptative basée sur la fréquence, Fibonacci et les cycles.
    
    Args:
        counts (Counter): Dictionnaire de compteurs (numéro -> fréquence)
        cycle_data (Optional[Dict]): Données de cycle pour chaque numéro
        frequency_weight (float): Poids accordé à la fréquence (0-1)
        fibonacci_weight (float): Poids accordé à la pondération Fibonacci (0-1)
        cycle_weight (float): Poids accordé aux cycles (0-1)
        
    Returns:
        Dict[int, float]: Poids adaptatifs pour chaque numéro
    """
    logger.info("Calcul des pondérations adaptatives...")
    
    try:
        # Vérification des poids
        total_weight = frequency_weight + fibonacci_weight + cycle_weight
        if not np.isclose(total_weight, 1.0):
            logger.warning(f"La somme des poids ({total_weight}) n'est pas égale à 1. Normalisation appliquée.")
            frequency_weight /= total_weight
            fibonacci_weight /= total_weight
            cycle_weight /= total_weight
        
        # Vérification des données d'entrée
        if not counts:
            logger.warning("Counter vide fourni à calculate_adaptive_weights")
            return {}
        
        # Calcul des poids de fréquence normalisés
        total_count = sum(counts.values())
        if total_count > 0:
            frequency_weights = {num: count / total_count for num, count in counts.items()}
        else:
            logger.warning("Somme des compteurs nulle, impossible de calculer les poids de fréquence")
            frequency_weights = {}
        
        # Calcul des poids de Fibonacci
        fibonacci_weights = apply_inverse_fibonacci_weights(counts)
        
        # Initialisation des poids de cycle
        cycle_weights = {}
        
        # Calcul des poids de cycle si les données sont disponibles
        if cycle_data:
            # Extraction des ratios observés/attendus
            ratios = {}
            for num, data in cycle_data.items():
                if "ratio_obs_exp" in data:
                    ratios[num] = data["ratio_obs_exp"]
            
            if ratios:
                # Normalisation des ratios
                max_ratio = max(ratios.values())
                min_ratio = min(ratios.values())
                ratio_range = max_ratio - min_ratio
                
                if ratio_range > 0:
                    cycle_weights = {num: (ratio - min_ratio) / ratio_range for num, ratio in ratios.items()}
                else:
                    # Si tous les ratios sont égaux, attribuer un poids égal
                    cycle_weights = {num: 1.0 / len(ratios) for num in ratios}
            else:
                logger.warning("Aucun ratio observé/attendu trouvé dans les données de cycle")
        
        # Combinaison des poids
        adaptive_weights = {}
        all_keys = set(frequency_weights.keys()) | set(fibonacci_weights.keys()) | set(cycle_weights.keys())
        
        for key in all_keys:
            freq_component = frequency_weights.get(key, 0.0) * frequency_weight
            fib_component = fibonacci_weights.get(key, 0.0) * fibonacci_weight
            cycle_component = cycle_weights.get(key, 0.0) * cycle_weight
            
            adaptive_weights[key] = freq_component + fib_component + cycle_component
        
        # Normalisation finale
        total_adaptive_weight = sum(adaptive_weights.values())
        if total_adaptive_weight > 0:
            adaptive_weights = {k: v / total_adaptive_weight for k, v in adaptive_weights.items()}
        
        logger.info(f"Pondérations adaptatives calculées pour {len(adaptive_weights)} éléments")
        return adaptive_weights
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul des pondérations adaptatives: {e}")
        return {}

def apply_fibonacci_sequence_weights(numbers: List[int], max_weight: float = 1.0) -> Dict[int, float]:
    """
    Applique des poids basés sur la séquence de Fibonacci directement à une liste de numéros.
    
    Args:
        numbers (List[int]): Liste de numéros à pondérer
        max_weight (float): Poids maximum à attribuer
        
    Returns:
        Dict[int, float]: Dictionnaire des poids (numéro -> poids)
    """
    logger.info(f"Application des poids de séquence Fibonacci à {len(numbers)} numéros...")
    
    try:
        if not numbers:
            logger.warning("Liste de numéros vide fournie à apply_fibonacci_sequence_weights")
            return {}
        
        # Trier les numéros
        sorted_numbers = sorted(numbers)
        
        # Calculer les poids de Fibonacci
        weights = {}
        for i, num in enumerate(sorted_numbers):
            fib_value = fibonacci(i + 2)  # Commencer à partir de fib(2) = 1
            weights[num] = fib_value
        
        # Normaliser les poids
        max_fib = max(weights.values())
        if max_fib > 0:
            weights = {k: (v / max_fib) * max_weight for k, v in weights.items()}
        
        logger.info(f"Poids de séquence Fibonacci appliqués avec succès")
        return weights
    
    except Exception as e:
        logger.error(f"Erreur lors de l'application des poids de séquence Fibonacci: {e}")
        return {}

# Exemple d'utilisation
if __name__ == "__main__":
    try:
        logger.info("Démarrage du module de pondération Fibonacci...")
        start_time = time.time()
        
        # Exemple de compteurs (numéro -> fréquence)
        example_counts = Counter({1: 10, 2: 8, 3: 15, 4: 5, 5: 12})
        
        # Exemple de données de cycle
        example_cycle_data = {
            1: {"observed_freq": 3, "expected_freq": 2.5, "ratio_obs_exp": 1.2},
            2: {"observed_freq": 2, "expected_freq": 2.5, "ratio_obs_exp": 0.8},
            3: {"observed_freq": 4, "expected_freq": 2.5, "ratio_obs_exp": 1.6},
            4: {"observed_freq": 1, "expected_freq": 2.5, "ratio_obs_exp": 0.4},
            5: {"observed_freq": 3, "expected_freq": 2.5, "ratio_obs_exp": 1.2}
        }
        
        # Appliquer la pondération inverse de Fibonacci
        fib_weights = apply_inverse_fibonacci_weights(example_counts)
        
        print("Compteurs originaux:")
        for num, count in sorted(example_counts.items()):
            print(f"Numéro {num}: {count} occurrences")
        
        print("\nPoids de Fibonacci inverse:")
        for num, weight in sorted(fib_weights.items()):
            print(f"Numéro {num}: {weight:.4f}")
        
        # Créer des poids de fréquence normalisés
        total_count = sum(example_counts.values())
        freq_weights = {num: count / total_count for num, count in example_counts.items()}
        
        # Combiner les poids
        combined = apply_weighted_blend(freq_weights, fib_weights, 0.3)
        
        print("\nPoids combinés (30% Fibonacci, 70% fréquence):")
        for num, weight in sorted(combined.items()):
            print(f"Numéro {num}: {weight:.4f}")
        
        # Calculer les poids adaptatifs
        adaptive_weights = calculate_adaptive_weights(
            example_counts, 
            example_cycle_data,
            frequency_weight=0.4,
            fibonacci_weight=0.3,
            cycle_weight=0.3
        )
        
        print("\nPoids adaptatifs (40% fréquence, 30% Fibonacci, 30% cycle):")
        for num, weight in sorted(adaptive_weights.items()):
            print(f"Numéro {num}: {weight:.4f}")
        
        # Appliquer des poids basés sur la séquence de Fibonacci
        sequence_weights = apply_fibonacci_sequence_weights([1, 2, 3, 4, 5])
        
        print("\nPoids basés sur la séquence de Fibonacci:")
        for num, weight in sorted(sequence_weights.items()):
            print(f"Numéro {num}: {weight:.4f}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Exécution terminée en {elapsed_time:.2f} secondes")
        print(f"\nExécution terminée en {elapsed_time:.2f} secondes")
    
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du module: {e}")
        print(f"Une erreur est survenue: {e}")
