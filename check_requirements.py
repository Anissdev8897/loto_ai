#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour vérifier si les packages de requirements.txt sont déjà installés.
Utilisé par start_loto.bat pour éviter de réinstaller les packages déjà présents.
"""
import subprocess
import sys
from pathlib import Path


def check_requirements(requirements_file='requirements.txt'):
    """
    Vérifie quels packages de requirements.txt sont déjà installés.
    
    Args:
        requirements_file: Chemin vers le fichier requirements.txt
        
    Returns:
        tuple: (liste des packages manquants, liste de tous les packages requis)
    """
    req_file = Path(requirements_file)
    if not req_file.exists():
        print(f"[ERREUR] Le fichier {requirements_file} est introuvable.")
        sys.exit(1)
    
    # Lire les packages requis
    required_packages = []
    with open(req_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Ignorer les lignes vides et les commentaires
            if line and not line.startswith('#'):
                # Extraire le nom du package (sans version si présente)
                pkg_name = line.split('=')[0].split('<')[0].split('>')[0].split('!')[0].strip()
                if pkg_name:
                    required_packages.append(pkg_name)
    
    # Obtenir la liste des packages installés
    try:
        installed_output = subprocess.check_output(
            ['pip', 'freeze'],
            encoding='utf-8',
            stderr=subprocess.DEVNULL
        )
        # Extraire les noms des packages installés
        installed_packages = {
            pkg.split('==')[0].lower() for pkg in installed_output.strip().split('\n')
            if '==' in pkg
        }
    except subprocess.CalledProcessError:
        print("[ERREUR] Impossible d'obtenir la liste des packages installés.")
        sys.exit(1)
    
    # Vérifier quels packages sont manquants
    missing_packages = []
    installed_packages_set = set(installed_packages)
    
    for pkg in required_packages:
        if pkg.lower() not in installed_packages_set:
            missing_packages.append(pkg)
            print(f"[MANQUANT] {pkg}")
        else:
            print(f"[OK] {pkg} déjà installé")
    
    return missing_packages, required_packages


if __name__ == '__main__':
    missing, required = check_requirements()
    
    if missing:
        print(f"\n[INFO] {len(missing)} package(s) manquant(s) sur {len(required)} total")
        sys.exit(1)  # Code de sortie 1 = packages manquants
    else:
        print(f"\n[SUCCES] Toutes les dépendances sont déjà installées ({len(required)} package(s)).")
        sys.exit(0)  # Code de sortie 0 = tout est installé

