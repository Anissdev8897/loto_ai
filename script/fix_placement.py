#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module utilitaire pour corriger le placement des méthodes dans le fichier loto_analyzer_improved.py.
Ce script identifie et déplace les méthodes mal placées dans la classe LotoAnalyzer.
"""

import sys
import os
import re
import logging
import shutil
from typing import List, Tuple, Optional

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FixPlacement")

def find_class_boundaries(lines: List[str]) -> Tuple[int, int, str]:
    """
    Trouve les limites de la classe LotoAnalyzer et son niveau d'indentation.
    
    Args:
        lines (List[str]): Lignes du fichier source
        
    Returns:
        Tuple[int, int, str]: Index de début de classe, index de fin de classe, indentation de classe
    """
    logger.info("Recherche des limites de la classe LotoAnalyzer...")
    
    class_start_line_index = -1
    class_end_line_index = -1
    class_indent = "    "  # Indentation par défaut
    
    try:
        # Trouver le début de la classe
        for i, line in enumerate(lines):
            if line.strip().startswith("class LotoAnalyzer"):
                class_start_line_index = i
                logger.info(f"Début de classe trouvé à la ligne {i+1}")
                
                # Trouver l'indentation réelle
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() and not lines[j].strip().startswith("#"):
                        match = re.match(r"^(\s+)", lines[j])
                        if match:
                            class_indent = match.group(1)
                            logger.debug(f"Indentation de classe détectée: '{class_indent}'")
                        break
                break
        
        if class_start_line_index == -1:
            logger.error("Classe LotoAnalyzer non trouvée dans le fichier")
            return -1, -1, class_indent
        
        # Trouver la fin de la classe (première ligne non indentée après le début de la classe)
        for i in range(class_start_line_index + 1, len(lines)):
            if lines[i].strip() and not lines[i].startswith(class_indent) and not lines[i].strip().startswith("#"):
                class_end_line_index = i
                logger.info(f"Fin de classe trouvée à la ligne {i+1}")
                break
        
        # Si la fin n'est pas trouvée, supposer que la classe va jusqu'à la fin du fichier
        if class_end_line_index == -1:
            class_end_line_index = len(lines)
            logger.warning("Fin de classe non trouvée explicitement, utilisation de la fin du fichier")
        
        return class_start_line_index, class_end_line_index, class_indent
    
    except Exception as e:
        logger.error(f"Erreur lors de la recherche des limites de classe: {e}")
        return -1, -1, class_indent

def find_method_boundaries(lines: List[str], method_name: str, start_index: int, end_index: int, indent: str) -> Tuple[int, int]:
    """
    Trouve les limites d'une méthode spécifique dans une plage de lignes.
    
    Args:
        lines (List[str]): Lignes du fichier source
        method_name (str): Nom de la méthode à rechercher
        start_index (int): Index de début de recherche
        end_index (int): Index de fin de recherche
        indent (str): Indentation de la classe
        
    Returns:
        Tuple[int, int]: Index de début et de fin de la méthode
    """
    logger.info(f"Recherche des limites de la méthode '{method_name}'...")
    
    method_start_index = -1
    method_end_index = -1
    method_pattern = re.compile(f"^{re.escape(indent)}def {re.escape(method_name)}\\(")
    
    try:
        # Trouver le début de la méthode
        for i in range(start_index, min(end_index, len(lines))):
            if re.match(method_pattern, lines[i]):
                method_start_index = i
                logger.info(f"Début de méthode '{method_name}' trouvé à la ligne {i+1}")
                break
        
        if method_start_index == -1:
            logger.warning(f"Méthode '{method_name}' non trouvée dans la plage spécifiée")
            return -1, -1
        
        # Trouver la fin de la méthode (première ligne avec indentation inférieure ou égale à celle de la classe)
        method_body_indent = indent + "    "  # Indentation du corps de la méthode
        for i in range(method_start_index + 1, min(end_index, len(lines))):
            line = lines[i]
            if line.strip() and not line.startswith(method_body_indent) and not line.strip().startswith("#"):
                method_end_index = i - 1
                logger.info(f"Fin de méthode '{method_name}' trouvée à la ligne {i}")
                break
        
        # Si la fin n'est pas trouvée, supposer que la méthode va jusqu'à la fin de la classe
        if method_end_index == -1:
            method_end_index = end_index - 1
            logger.warning(f"Fin de méthode '{method_name}' non trouvée explicitement, utilisation de la fin de la classe")
        
        return method_start_index, method_end_index
    
    except Exception as e:
        logger.error(f"Erreur lors de la recherche des limites de méthode '{method_name}': {e}")
        return -1, -1

def find_misplaced_methods(lines: List[str], class_end_index: int) -> Tuple[int, int, List[str]]:
    """
    Trouve les méthodes mal placées après la fin de la classe.
    
    Args:
        lines (List[str]): Lignes du fichier source
        class_end_index (int): Index de fin de la classe
        
    Returns:
        Tuple[int, int, List[str]]: Index de début et de fin des méthodes mal placées, et liste des méthodes à déplacer
    """
    logger.info("Recherche des méthodes mal placées...")
    
    misplaced_start_index = -1
    misplaced_end_index = -1
    methods_to_move = []
    
    try:
        # Trouver le début des méthodes mal placées
        for i in range(class_end_index, len(lines)):
            if lines[i].strip().startswith("def _run_monte_carlo_simulation("):
                misplaced_start_index = i
                logger.info(f"Début des méthodes mal placées trouvé à la ligne {i+1}")
                break
        
        if misplaced_start_index == -1:
            logger.warning("Aucune méthode mal placée trouvée")
            return -1, -1, []
        
        # Trouver la fin des méthodes mal placées (avant track_performance ou avant le bloc main)
        for i in range(misplaced_start_index, len(lines)):
            if lines[i].strip().startswith("def track_performance("):
                misplaced_end_index = i - 1
                logger.info(f"Fin des méthodes mal placées trouvée à la ligne {i} (avant track_performance)")
                break
            elif lines[i].strip().startswith("if __name__ == \"__main__\":"):
                misplaced_end_index = i - 1
                logger.info(f"Fin des méthodes mal placées trouvée à la ligne {i} (avant le bloc main)")
                break
        
        if misplaced_end_index == -1:
            misplaced_end_index = len(lines) - 1
            logger.warning("Fin des méthodes mal placées non trouvée explicitement, utilisation de la fin du fichier")
        
        # Extraire les méthodes à déplacer
        if misplaced_start_index != -1 and misplaced_end_index != -1:
            # Déterminer l'indentation des méthodes mal placées
            match = re.match(r"^(\s*)", lines[misplaced_start_index])
            indent_level = match.group(1) if match else ""
            
            # Extraire les méthodes
            in_methods_to_move = True
            for i in range(misplaced_start_index, misplaced_end_index + 1):
                line = lines[i]
                if in_methods_to_move:
                    # Supprimer l'indentation actuelle pour la réappliquer plus tard
                    if line.startswith(indent_level):
                        methods_to_move.append(line[len(indent_level):])
                    else:
                        methods_to_move.append(line)
        
        return misplaced_start_index, misplaced_end_index, methods_to_move
    
    except Exception as e:
        logger.error(f"Erreur lors de la recherche des méthodes mal placées: {e}")
        return -1, -1, []

def find_main_block(lines: List[str], start_index: int) -> int:
    """
    Trouve le début du bloc if __name__ == "__main__".
    
    Args:
        lines (List[str]): Lignes du fichier source
        start_index (int): Index de début de recherche
        
    Returns:
        int: Index du début du bloc main
    """
    logger.info("Recherche du bloc main...")
    
    try:
        for i in range(start_index, len(lines)):
            if lines[i].strip().startswith("if __name__ == \"__main__\":"):
                logger.info(f"Bloc main trouvé à la ligne {i+1}")
                return i
        
        logger.warning("Bloc main non trouvé")
        return -1
    
    except Exception as e:
        logger.error(f"Erreur lors de la recherche du bloc main: {e}")
        return -1

def fix_method_placement(file_path: str) -> bool:
    """
    Corrige le placement des méthodes dans le fichier spécifié.
    
    Args:
        file_path (str): Chemin du fichier à corriger
        
    Returns:
        bool: True si la correction a réussi, False sinon
    """
    logger.info(f"Début de la correction du placement des méthodes dans {file_path}")
    
    # Vérifier que le fichier existe
    if not os.path.exists(file_path):
        logger.error(f"Le fichier {file_path} n'existe pas")
        return False
    
    # Créer une sauvegarde du fichier original
    backup_path = f"{file_path}.bak"
    try:
        shutil.copy2(file_path, backup_path)
        logger.info(f"Sauvegarde créée: {backup_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la création de la sauvegarde: {e}")
        return False
    
    try:
        # Lire le contenu du fichier
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Trouver les limites de la classe
        class_start_index, class_end_index, class_indent = find_class_boundaries(lines)
        if class_start_index == -1 or class_end_index == -1:
            logger.error("Impossible de trouver les limites de la classe")
            return False
        
        # Trouver les méthodes mal placées
        misplaced_start_index, misplaced_end_index, methods_to_move = find_misplaced_methods(lines, class_end_index)
        if misplaced_start_index == -1 or misplaced_end_index == -1 or not methods_to_move:
            logger.warning("Aucune méthode mal placée trouvée ou erreur lors de l'extraction")
            return False
        
        # Trouver le bloc main
        main_block_start_index = find_main_block(lines, misplaced_end_index + 1)
        if main_block_start_index == -1:
            logger.warning("Bloc main non trouvé, utilisation de la fin du fichier")
            main_block_start_index = len(lines)
        
        # Indenter correctement les méthodes à déplacer
        indented_methods = []
        for line in methods_to_move:
            if line.strip():  # Si la ligne n'est pas vide
                indented_methods.append(class_indent + line)
            else:
                indented_methods.append(line)
        
        # Ajouter une ligne vide avant les méthodes si nécessaire
        if indented_methods and lines[class_end_index-1].strip():
            indented_methods.insert(0, '\n')
        
        # Construire le contenu corrigé
        corrected_lines = (
            lines[:class_end_index] +  # Partie 1: Définition de classe jusqu'au point d'insertion
            indented_methods +         # Partie 2: Les méthodes à insérer (maintenant correctement indentées)
            lines[class_end_index:misplaced_start_index] +  # Partie 3: Code entre la fin de classe et les méthodes mal placées
            lines[main_block_start_index:]  # Partie 4: Le bloc if __name__ == "__main__": et la suite
        )
        
        # Écrire le contenu corrigé dans le fichier
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(corrected_lines)
        
        logger.info(f"Correction réussie: {len(methods_to_move)} lignes déplacées dans la classe")
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors de la correction du placement des méthodes: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Restaurer la sauvegarde en cas d'erreur
        try:
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, file_path)
                logger.info(f"Restauration de la sauvegarde après erreur")
        except Exception as restore_error:
            logger.error(f"Erreur lors de la restauration de la sauvegarde: {restore_error}")
        
        return False

if __name__ == "__main__":
    try:
        logger.info("Démarrage du script de correction de placement...")
        
        # Définir le chemin du fichier à corriger
        file_path = "/home/ubuntu/projet_loto/loto_analyzer_improved.py"
        
        # Vérifier si un chemin de fichier est fourni en argument
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
            logger.info(f"Utilisation du chemin de fichier fourni en argument: {file_path}")
        else:
            logger.info(f"Utilisation du chemin de fichier par défaut: {file_path}")
        
        # Corriger le placement des méthodes
        success = fix_method_placement(file_path)
        
        if success:
            print(f"Correction réussie du placement des méthodes dans {file_path}")
            logger.info("Script terminé avec succès")
        else:
            print(f"Échec de la correction du placement des méthodes dans {file_path}")
            logger.error("Script terminé avec des erreurs")
    
    except Exception as e:
        logger.critical(f"Erreur critique: {e}")
        print(f"Une erreur critique est survenue: {e}")
        import traceback
        logger.critical(traceback.format_exc())
        print(traceback.format_exc())
