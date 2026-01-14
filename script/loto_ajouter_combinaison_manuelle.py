#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour ajouter manuellement une combinaison au modèle Loto

Ce script permet d'ajouter une combinaison de tirage manuellement au modèle
d'apprentissage incrémental sans avoir à modifier le fichier CSV des tirages.
Il met à jour les modèles prédictifs et génère une nouvelle prédiction.
"""

import os
import sys
import logging
import argparse
import traceback
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Configuration du logging avec redirection vers fichier et console
logger = logging.getLogger("AjoutCombinaisonManuelle")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("log.txt"),
            logging.StreamHandler()
        ]
    )

def parse_arguments():
    """
    Parse les arguments de ligne de commande.
    
    Returns:
        Namespace: Arguments parsés
        
    Raises:
        ArgumentTypeError: Si les arguments ne respectent pas les contraintes
    """
    parser = argparse.ArgumentParser(description="Ajouter manuellement une combinaison au modèle Loto")
    
    parser.add_argument("--csv", type=str, default="tirages_loto.csv",
                        help="Chemin vers le fichier CSV des tirages (pour initialiser l'analyseur)")
    
    parser.add_argument("--output", type=str, default="resultats_loto",
                        help="Répertoire de sortie pour les résultats")
    
    parser.add_argument("--numeros", type=str, required=True,
                        help="Numéros du tirage séparés par des virgules (ex: 1,15,23,34,45)")
    
    parser.add_argument("--chance", type=int, required=True,
                        help="Numéro chance du tirage")
    
    parser.add_argument("--date", type=str, default=None,
                        help="Date du tirage au format JJ/MM/AAAA (par défaut: date actuelle)")
    
    parser.add_argument("--jour", type=str, default="Lundi",
                        choices=["Lundi", "Mercredi", "Samedi", "lundi", "mercredi", "samedi"],
                        help="Jour du tirage (ex: Lundi, Mercredi, Samedi)")
    
    parser.add_argument("--max-number", type=int, default=49,
                        help="Numéro maximum possible dans le tirage (par défaut: 49)")
    
    parser.add_argument("--max-chance", type=int, default=10,
                        help="Numéro chance maximum possible (par défaut: 10)")
    
    parser.add_argument("--save-history", action="store_true",
                        help="Sauvegarder l'historique des combinaisons ajoutées manuellement")
    
    return parser.parse_args()

def validate_numbers(numeros, max_number, expected_count=5):
    """
    Valide les numéros fournis.
    
    Args:
        numeros (list): Liste des numéros à valider
        max_number (int): Numéro maximum autorisé
        expected_count (int): Nombre de numéros attendus
        
    Returns:
        bool: True si les numéros sont valides, False sinon
        
    Raises:
        ValueError: Si les numéros ne sont pas valides
    """
    try:
        # Vérifier le nombre de numéros
        if len(numeros) != expected_count:
            raise ValueError(f"Nombre de numéros invalide: {len(numeros)}, attendu: {expected_count}")
        
        # Vérifier que les numéros sont uniques
        if len(set(numeros)) != len(numeros):
            raise ValueError(f"Les numéros doivent être uniques: {numeros}")
        
        # Vérifier que les numéros sont dans la plage valide
        for num in numeros:
            if num < 1 or num > max_number:
                raise ValueError(f"Numéro invalide: {num}, doit être entre 1 et {max_number}")
        
        return True
    except ValueError as e:
        logger.error(str(e))
        return False

def validate_chance_number(chance, max_chance=10):
    """
    Valide le numéro chance fourni.
    
    Args:
        chance (int): Numéro chance à valider
        max_chance (int): Numéro chance maximum autorisé
        
    Returns:
        bool: True si le numéro chance est valide, False sinon
    """
    try:
        if chance < 1 or chance > max_chance:
            raise ValueError(f"Numéro chance invalide: {chance}, doit être entre 1 et {max_chance}")
        return True
    except ValueError as e:
        logger.error(str(e))
        return False

def parse_date(date_str):
    """
    Parse une date au format JJ/MM/AAAA.
    
    Args:
        date_str (str): Date au format JJ/MM/AAAA
        
    Returns:
        datetime: Objet datetime correspondant à la date
        
    Raises:
        ValueError: Si le format de date est invalide
    """
    if not date_str:
        return datetime.now()
    
    try:
        # Essayer plusieurs formats de date courants
        for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"]:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # Si aucun format ne correspond
        raise ValueError(f"Format de date invalide: {date_str}, doit être au format JJ/MM/AAAA")
    except Exception as e:
        logger.error(f"Erreur lors du parsing de la date: {str(e)}")
        raise

def save_combination_history(output_dir, new_draw):
    """
    Sauvegarde l'historique des combinaisons ajoutées manuellement.
    
    Args:
        output_dir (Path): Répertoire de sortie
        new_draw (dict): Nouvelle combinaison
        
    Returns:
        bool: True si la sauvegarde a réussi, False sinon
    """
    try:
        history_file = output_dir / "manual_combinations_history.json"
        
        # Convertir les objets datetime en chaînes
        new_draw_copy = new_draw.copy()
        if isinstance(new_draw_copy.get('Date'), datetime):
            new_draw_copy['Date'] = new_draw_copy['Date'].strftime("%d/%m/%Y")
        
        # Charger l'historique existant ou créer un nouveau
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []
        
        # Ajouter la nouvelle combinaison avec un horodatage
        entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'combination': new_draw_copy
        }
        history.append(entry)
        
        # Sauvegarder l'historique mis à jour
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Historique des combinaisons manuelles mis à jour: {history_file}")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de l'historique: {str(e)}")
        return False

def main():
    """
    Fonction principale du script.
    
    Cette fonction:
    1. Parse et valide les arguments
    2. Initialise l'analyseur et le module d'apprentissage
    3. Ajoute la nouvelle combinaison au modèle
    4. Génère une nouvelle prédiction
    
    Returns:
        int: Code de retour (0 pour succès, 1 pour erreur)
    """
    start_time = datetime.now()
    logger.info("Démarrage du script d'ajout manuel de combinaison")
    
    try:
        # Parser les arguments
        args = parse_arguments()
        logger.info(f"Arguments: {args}")
        
        # Vérifier que les numéros sont valides
        try:
            numeros = [int(n.strip()) for n in args.numeros.split(",")]
        except ValueError:
            logger.error(f"Format de numéros invalide: {args.numeros}, doit être des entiers séparés par des virgules")
            return 1
        
        if not validate_numbers(numeros, args.max_number):
            return 1
        
        # Vérifier que le numéro chance est valide
        if not validate_chance_number(args.chance, args.max_chance):
            return 1
        
        # Préparer la date
        try:
            date = parse_date(args.date)
        except ValueError as e:
            logger.error(str(e))
            return 1
        
        # Créer le répertoire de sortie
        output_dir = Path(args.output)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Répertoire de sortie: {output_dir}")
        except Exception as e:
            logger.error(f"Erreur lors de la création du répertoire de sortie: {str(e)}")
            return 1
        
        # Vérifier que le fichier CSV existe
        csv_path = Path(args.csv)
        if not csv_path.exists():
            logger.error(f"Fichier CSV introuvable: {csv_path}")
            return 1
        
        # Configuration de l'analyseur
        config = {
            "csv_file": str(csv_path),
            "window_draws": 50,
            "num_hot": 5,
            "num_cold": 5,
            "propose_size": 5,
            "output_dir": str(output_dir),
            "max_number": args.max_number,
            "max_chance": args.max_chance,
            "chance_analysis": True
        }
        
        # Importer les modules personnalisés
        try:
            # Essayer d'abord improved_loto_analyzer_improved
            try:
                from improved_loto_analyzer_improved import LotoAnalyzer
                logger.info("Module 'improved_loto_analyzer_improved.LotoAnalyzer' importé avec succès")
            except ImportError:
                # Sinon, essayer loto_analyzer_improved
                from loto_analyzer_improved import LotoAnalyzer
                logger.info("Module 'loto_analyzer_improved.LotoAnalyzer' importé avec succès")
            
            # Importer le module d'apprentissage incrémental
            from loto_incremental_learning import LotoIncrementalLearning
            logger.info("Module 'loto_incremental_learning.LotoIncrementalLearning' importé avec succès")
        except ImportError as e:
            logger.error(f"Erreur lors de l'import des modules: {str(e)}")
            logger.error("Assurez-vous que les modules loto_analyzer_improved.py et loto_incremental_learning.py sont disponibles")
            return 1
        
        # Initialiser l'analyseur
        try:
            analyzer = LotoAnalyzer(config)
            logger.info("Analyseur initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de l'analyseur: {str(e)}")
            return 1
        
        # Charger les données
        try:
            if not analyzer.load_data():
                logger.error("Impossible de charger les données. Arrêt du programme.")
                return 1
            logger.info(f"Données chargées avec succès: {len(analyzer.df)} tirages")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {str(e)}")
            return 1
        
        # Initialiser le module d'apprentissage incrémental
        try:
            learner = LotoIncrementalLearning(analyzer)
            logger.info("Module d'apprentissage incrémental initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du module d'apprentissage: {str(e)}")
            return 1
        
        # Charger les modèles existants
        try:
            if not learner.load_models():
                logger.warning("Aucun modèle existant trouvé. Entraînement initial des modèles...")
                if not learner.train_initial_models():
                    logger.error("Échec de l'entraînement initial des modèles. Arrêt du programme.")
                    return 1
                logger.info("Modèles initiaux entraînés avec succès")
            else:
                logger.info("Modèles existants chargés avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement/entraînement des modèles: {str(e)}")
            return 1
        
        # Créer un dictionnaire avec la nouvelle combinaison
        new_draw = {
            'Jour': args.jour.capitalize(),
            'Date': date,
            'Numéro 1': numeros[0],
            'Numéro 2': numeros[1],
            'Numéro 3': numeros[2],
            'Numéro 4': numeros[3],
            'Numéro 5': numeros[4],
            'Chance': args.chance
        }
        
        # Sauvegarder l'historique si demandé
        if args.save_history:
            save_combination_history(output_dir, new_draw)
        
        # Mettre à jour les modèles avec la nouvelle combinaison
        try:
            if learner.update_models(new_draw):
                logger.info("Modèles mis à jour avec succès avec la nouvelle combinaison")
                
                # Afficher la combinaison ajoutée
                date_str = date.strftime("%d/%m/%Y")
                logger.info(f"Combinaison ajoutée: {args.jour} {date_str} - Numéros: {numeros} - Chance: {args.chance}")
                
                # Générer une prédiction pour le prochain tirage
                predicted_numbers, predicted_chance, proba_numbers, proba_chance = learner.predict_next_draw()
                
                logger.info(f"Nouvelle prédiction pour le prochain tirage:")
                logger.info(f"  Numéros: {predicted_numbers}")
                logger.info(f"  Chance: {predicted_chance}")
                
                # Afficher les probabilités si disponibles
                if proba_numbers is not None:
                    logger.info(f"  Probabilités des numéros: {[f'{p:.4f}' for p in proba_numbers]}")
                if proba_chance is not None:
                    logger.info(f"  Probabilité du numéro chance: {proba_chance:.4f}")
                
                # Sauvegarder la prédiction dans un fichier
                prediction_file = output_dir / "derniere_prediction.json"
                prediction_data = {
                    "date_prediction": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "numeros": predicted_numbers,
                    "chance": predicted_chance,
                    "derniere_combinaison_ajoutee": {
                        "date": date_str,
                        "jour": args.jour,
                        "numeros": numeros,
                        "chance": args.chance
                    }
                }
                
                with open(prediction_file, 'w', encoding='utf-8') as f:
                    json.dump(prediction_data, f, indent=4, ensure_ascii=False)
                
                logger.info(f"Prédiction sauvegardée dans: {prediction_file}")
                
                # Afficher le temps d'exécution
                elapsed_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Opération terminée en {elapsed_time:.2f} secondes")
                
                return 0
            else:
                logger.error("Échec de la mise à jour des modèles avec la nouvelle combinaison")
                return 1
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des modèles: {str(e)}")
            return 1
        
    except Exception as e:
        logger.error(f"Erreur lors de l'ajout de la combinaison: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
