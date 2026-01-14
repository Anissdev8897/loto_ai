#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour extraire et comparer les méthodes entre deux versions d'un fichier Python.
"""

import re
import sys
import os

def extract_methods(file_path):
    """
    Extrait toutes les définitions de méthodes d'un fichier Python.
    
    Args:
        file_path (str): Chemin vers le fichier Python
        
    Returns:
        list: Liste des signatures de méthodes
    """
    methods = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # Recherche des définitions de fonctions et méthodes
            # Capture le nom et les paramètres
            pattern = r'def\s+([a-zA-Z0-9_]+)\s*\((.*?)\):'
            matches = re.findall(pattern, content, re.DOTALL)
            
            for name, params in matches:
                # Nettoyer les paramètres (enlever les espaces superflus)
                params = re.sub(r'\s+', ' ', params.strip())
                methods.append(f"def {name}({params}):")
                
        return methods
    except Exception as e:
        print(f"Erreur lors de l'extraction des méthodes de {file_path}: {str(e)}")
        return []

def compare_methods(original_methods, corrected_methods):
    """
    Compare deux listes de méthodes pour identifier les différences.
    
    Args:
        original_methods (list): Liste des méthodes originales
        corrected_methods (list): Liste des méthodes corrigées
        
    Returns:
        tuple: (méthodes manquantes, méthodes ajoutées, méthodes communes)
    """
    # Extraction des noms de méthodes (sans les paramètres)
    original_names = [re.match(r'def\s+([a-zA-Z0-9_]+)', method).group(1) for method in original_methods]
    corrected_names = [re.match(r'def\s+([a-zA-Z0-9_]+)', method).group(1) for method in corrected_methods]
    
    # Identification des méthodes manquantes et ajoutées
    missing_methods = [method for method, name in zip(original_methods, original_names) if name not in corrected_names]
    added_methods = [method for method, name in zip(corrected_methods, corrected_names) if name not in original_names]
    
    # Identification des méthodes communes
    common_methods = [name for name in original_names if name in corrected_names]
    
    return missing_methods, added_methods, common_methods

def main():
    """
    Fonction principale pour comparer les méthodes entre deux fichiers.
    """
    if len(sys.argv) != 3:
        print("Usage: python extract_methods.py <fichier_original> <fichier_corrige>")
        return
    
    original_file = sys.argv[1]
    corrected_file = sys.argv[2]
    
    if not os.path.exists(original_file) or not os.path.exists(corrected_file):
        print("Un des fichiers spécifiés n'existe pas.")
        return
    
    original_methods = extract_methods(original_file)
    corrected_methods = extract_methods(corrected_file)
    
    missing_methods, added_methods, common_methods = compare_methods(original_methods, corrected_methods)
    
    print(f"\nComparaison entre {os.path.basename(original_file)} et {os.path.basename(corrected_file)}:")
    print(f"Nombre de méthodes dans le fichier original: {len(original_methods)}")
    print(f"Nombre de méthodes dans le fichier corrigé: {len(corrected_methods)}")
    print(f"Nombre de méthodes communes: {len(common_methods)}")
    
    if missing_methods:
        print("\nMéthodes présentes dans l'original mais absentes dans la version corrigée:")
        for method in missing_methods:
            print(f"  - {method}")
    else:
        print("\nAucune méthode n'a été supprimée.")
    
    if added_methods:
        print("\nMéthodes ajoutées dans la version corrigée:")
        for method in added_methods:
            print(f"  + {method}")
    
    # Écrire un résumé dans un fichier
    output_file = f"comparaison_{os.path.basename(original_file).split('.')[0]}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Comparaison entre {os.path.basename(original_file)} et {os.path.basename(corrected_file)}:\n")
        f.write(f"Nombre de méthodes dans le fichier original: {len(original_methods)}\n")
        f.write(f"Nombre de méthodes dans le fichier corrigé: {len(corrected_methods)}\n")
        f.write(f"Nombre de méthodes communes: {len(common_methods)}\n\n")
        
        if missing_methods:
            f.write("Méthodes présentes dans l'original mais absentes dans la version corrigée:\n")
            for method in missing_methods:
                f.write(f"  - {method}\n")
        else:
            f.write("Aucune méthode n'a été supprimée.\n")
        
        if added_methods:
            f.write("\nMéthodes ajoutées dans la version corrigée:\n")
            for method in added_methods:
                f.write(f"  + {method}\n")
    
    print(f"\nRésumé écrit dans {output_file}")

if __name__ == "__main__":
    main()
