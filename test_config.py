#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test pour vérifier que toutes les configurations sont correctes pour le Loto
"""

import sys
from pathlib import Path

# Ajouter le répertoire au path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_constants():
    """Teste que les constantes sont correctes pour le Loto"""
    print("=" * 60)
    print("TEST DES CONSTANTES LOTO")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # Importer config
    try:
        import config
        print("[OK] Module config importe avec succes")
    except ImportError as e:
        print(f"[ERREUR] Erreur lors de l'import de config: {e}")
        return False
    
    # Tester MAX_NUMBER
    if config.MAX_NUMBER == 49:
        print(f"[OK] MAX_NUMBER = {config.MAX_NUMBER} (correct pour Loto)")
    else:
        errors.append(f"MAX_NUMBER = {config.MAX_NUMBER} (devrait etre 49)")
        print(f"[ERREUR] MAX_NUMBER = {config.MAX_NUMBER} (devrait etre 49)")
    
    # Tester MAX_CHANCE
    if config.MAX_CHANCE == 10:
        print(f"[OK] MAX_CHANCE = {config.MAX_CHANCE} (correct pour Loto)")
    else:
        errors.append(f"MAX_CHANCE = {config.MAX_CHANCE} (devrait etre 10)")
        print(f"[ERREUR] MAX_CHANCE = {config.MAX_CHANCE} (devrait etre 10)")
    
    # Tester PROPOSE_SIZE
    if config.PROPOSE_SIZE == 5:
        print(f"[OK] PROPOSE_SIZE = {config.PROPOSE_SIZE} (correct pour Loto)")
    else:
        errors.append(f"PROPOSE_SIZE = {config.PROPOSE_SIZE} (devrait etre 5)")
        print(f"[ERREUR] PROPOSE_SIZE = {config.PROPOSE_SIZE} (devrait etre 5)")
    
    # Tester les fichiers
    print("\n" + "=" * 60)
    print("TEST DES FICHIERS ET RÉPERTOIRES")
    print("=" * 60)
    
    if config.CSV_FILE.exists():
        print(f"[OK] CSV trouve: {config.CSV_FILE}")
    else:
        warnings.append(f"CSV non trouve: {config.CSV_FILE}")
        print(f"[AVERTISSEMENT] CSV non trouve: {config.CSV_FILE}")
    
    if config.MODELS_DIR.exists():
        print(f"[OK] Repertoire modeles existe: {config.MODELS_DIR}")
        
        # Vérifier les fichiers de modèles
        required_models = [
            config.MODEL_FILES['number_model'],
            config.MODEL_FILES['scaler_numbers'],
        ]
        
        for model_file in required_models:
            if model_file.exists():
                print(f"  [OK] {model_file.name}")
            else:
                warnings.append(f"Modele manquant: {model_file.name}")
                print(f"  [AVERTISSEMENT] {model_file.name} manquant")
    else:
        warnings.append(f"Repertoire modeles non trouve: {config.MODELS_DIR}")
        print(f"[AVERTISSEMENT] Repertoire modeles non trouve: {config.MODELS_DIR}")
    
    # Tester app.py
    print("\n" + "=" * 60)
    print("TEST DE APP.PY")
    print("=" * 60)
    
    try:
        # Lire app.py et vérifier les constantes
        app_py = current_dir / "app.py"
        if app_py.exists():
            content = app_py.read_text(encoding='utf-8')
            
            if 'MAX_NUMBER = 49' in content or 'MAX_NUMBER=49' in content:
                print("[OK] app.py: MAX_NUMBER = 49")
            else:
                if 'MAX_NUMBER = 50' in content:
                    errors.append("app.py contient MAX_NUMBER = 50 (devrait etre 49)")
                    print("[ERREUR] app.py: MAX_NUMBER = 50 (devrait etre 49)")
                else:
                    warnings.append("app.py: MAX_NUMBER non verifie")
                    print("[AVERTISSEMENT] app.py: MAX_NUMBER non verifie")
            
            if 'MAX_CHANCE = 10' in content or 'MAX_CHANCE=10' in content:
                print("[OK] app.py: MAX_CHANCE = 10")
            else:
                if 'MAX_CHANCE = 12' in content or 'MAX_CHANCE = 2' in content:
                    errors.append("app.py contient MAX_CHANCE incorrect")
                    print("[ERREUR] app.py: MAX_CHANCE incorrect (devrait etre 10)")
                else:
                    warnings.append("app.py: MAX_CHANCE non verifie")
                    print("[AVERTISSEMENT] app.py: MAX_CHANCE non verifie")
            
            if 'PROPOSE_SIZE = 5' in content or 'PROPOSE_SIZE=5' in content:
                print("[OK] app.py: PROPOSE_SIZE = 5")
            else:
                warnings.append("app.py: PROPOSE_SIZE non verifie")
                print("[AVERTISSEMENT] app.py: PROPOSE_SIZE non verifie")
            
            # Vérifier qu'il n'y a pas de références aux étoiles
            if 'E1' in content or 'E2' in content or 'etoile' in content.lower():
                warnings.append("app.py contient des references aux etoiles (devrait etre Chance)")
                print("[AVERTISSEMENT] app.py: References aux etoiles detectees")
            else:
                print("[OK] app.py: Pas de references aux etoiles")
        else:
            warnings.append("app.py non trouvé")
            print("⚠️  app.py non trouvé")
    except Exception as e:
        warnings.append(f"Erreur lors de la vérification de app.py: {e}")
        print(f"⚠️  Erreur lors de la vérification de app.py: {e}")
    
    # Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    
    if errors:
        print(f"[ERREUR] {len(errors)} erreur(s) detectee(s):")
        for error in errors:
            print(f"  - {error}")
        return False
    
    if warnings:
        print(f"[AVERTISSEMENT] {len(warnings)} avertissement(s):")
        for warning in warnings:
            print(f"  - {warning}")
        print("\n[OK] Configuration de base correcte pour le Loto")
        print("[AVERTISSEMENT] Certains avertissements peuvent necessiter votre attention")
        return True
    
    print("[OK] Toutes les configurations sont correctes pour le Loto!")
    return True


def test_scripts():
    """Teste que les scripts principaux fonctionnent"""
    print("\n" + "=" * 60)
    print("TEST DES SCRIPTS")
    print("=" * 60)
    
    scripts_to_check = [
        "app.py",
        "auto_scraper.py",
        "auto_trainer.py",
        "train_local.py",
    ]
    
    for script_name in scripts_to_check:
        script_path = current_dir / script_name
        if script_path.exists():
            try:
                # Essayer de l'importer (sans l'exécuter)
                spec = __import__(script_name.replace('.py', ''), fromlist=[''])
                print(f"[OK] {script_name} peut etre importe")
            except Exception as e:
                print(f"[AVERTISSEMENT] {script_name} existe mais ne peut pas etre importe: {e}")
        else:
            print(f"[AVERTISSEMENT] {script_name} non trouve")


if __name__ == "__main__":
    print("\n")
    success = test_constants()
    test_scripts()
    print("\n" + "=" * 60)
    
    if success:
        print("[OK] Tests termines avec succes")
        sys.exit(0)
    else:
        print("[ERREUR] Des erreurs ont ete detectees")
        sys.exit(1)

