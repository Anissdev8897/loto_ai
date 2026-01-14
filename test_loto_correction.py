#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test pour valider la correction du conflit de nommage
dans loto_main.py et vérifier que la génération de combinaisons
optimisées fonctionne correctement.
"""

import os
import sys
import logging
from pathlib import Path

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TestLotoCorrection")

# Ajouter le répertoire du projet au path
current_dir = os.path.dirname(os.path.abspath(__file__))
projet_loto_dir = os.path.join(current_dir, "projet_loto")
if projet_loto_dir not in sys.path:
    sys.path.append(projet_loto_dir)

try:
    # Importer le module corrigé
    logger.info("Importation du module loto_main corrigé...")
    sys.path.insert(0, projet_loto_dir)
    from projet_loto.loto_main import LotoAnalyzer
    
    # Créer une configuration de test
    logger.info("Création d'une configuration de test...")
    config = {
        'csv_path': 'projet_loto/tirages_loto.csv',
        'output_dir': 'test_output',
        'max_number': 49,
        'max_chance': 10,
        'window_size_for_ml': 50,
        'propose_size': 5,
        'use_optimized_grid': True,  # Utilisation du nouvel attribut
        'use_cycle_analysis': True,
        'use_fibonacci_inverse': False,
        'use_lunar_cycle': False
    }
    
    # Créer le répertoire de sortie
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # Instancier l'analyseur
    logger.info("Instanciation de l'analyseur Loto...")
    analyzer = LotoAnalyzer(config)
    
    # Vérifier que l'attribut use_optimized_grid est bien défini
    logger.info(f"Valeur de l'attribut use_optimized_grid: {analyzer.use_optimized_grid}")
    
    # Vérifier que la méthode generate_optimized_grid existe et est bien une méthode
    logger.info("Vérification de la méthode generate_optimized_grid...")
    if hasattr(analyzer, 'generate_optimized_grid') and callable(getattr(analyzer, 'generate_optimized_grid')):
        logger.info("La méthode generate_optimized_grid existe et est bien une méthode.")
    else:
        logger.error("La méthode generate_optimized_grid n'existe pas ou n'est pas une méthode.")
        sys.exit(1)
    
    # Charger les données et exécuter l'analyse
    logger.info("Chargement des données et exécution de l'analyse...")
    if not os.path.exists(config['csv_path']):
        logger.warning(f"Le fichier CSV {config['csv_path']} n'existe pas. Utilisation d'un fichier fictif pour le test.")
        # Créer un DataFrame minimal pour le test
        import pandas as pd
        import numpy as np
        
        # Créer un DataFrame de test avec des données aléatoires
        np.random.seed(42)
        n_rows = 100
        
        # Générer des dates
        dates = pd.date_range(start='2020-01-01', periods=n_rows, freq='W')
        dates = [d.strftime('%d/%m/%Y') for d in dates]
        
        # Générer des numéros aléatoires
        data = {
            'Date': dates,
            'Numéro 1': np.random.randint(1, 50, n_rows),
            'Numéro 2': np.random.randint(1, 50, n_rows),
            'Numéro 3': np.random.randint(1, 50, n_rows),
            'Numéro 4': np.random.randint(1, 50, n_rows),
            'Numéro 5': np.random.randint(1, 50, n_rows),
            'Chance': np.random.randint(1, 11, n_rows)
        }
        
        # Créer le DataFrame
        df = pd.DataFrame(data)
        
        # Sauvegarder dans un fichier temporaire
        temp_csv = 'temp_tirages_loto.csv'
        df.to_csv(temp_csv, index=False)
        config['csv_path'] = temp_csv
        logger.info(f"Fichier CSV temporaire créé: {temp_csv}")
    
    # Exécuter l'analyse
    success = analyzer.load_data()
    if success:
        logger.info("Données chargées avec succès.")
        analyzer.compute_global_stats()
        analyzer.compute_gap_analysis()
        analyzer.compute_cycle_analysis()
        
        # Calculer les scores finaux
        analyzer.compute_final_scores()
        
        # Tester la génération de combinaisons optimisées
        logger.info("Test de la génération de combinaisons optimisées...")
        try:
            combinations = analyzer.generate_optimized_grid(5)
            logger.info(f"Génération réussie! {len(combinations)} combinaisons générées.")
            
            # Afficher les combinaisons
            for i, combo in enumerate(combinations):
                if len(combo) > analyzer.propose_size:
                    main_numbers = combo[:analyzer.propose_size]
                    chance_number = combo[-1]
                    logger.info(f"Combinaison {i+1}: {' - '.join(map(str, main_numbers))} | Chance: {chance_number}")
                else:
                    logger.info(f"Combinaison {i+1}: {' - '.join(map(str, combo))}")
            
            logger.info("Test réussi! La correction du conflit de nommage a fonctionné.")
        except Exception as e:
            logger.error(f"Erreur lors de la génération des combinaisons: {e}", exc_info=True)
            sys.exit(1)
    else:
        logger.error("Échec du chargement des données.")
        sys.exit(1)

except Exception as e:
    logger.error(f"Erreur lors du test: {e}", exc_info=True)
    sys.exit(1)
