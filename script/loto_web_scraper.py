#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de Web Scraping pour récupérer l'historique des tirages du Loto
depuis le site loto.akroweb.fr et les sauvegarder dans un fichier CSV.

Ce script est compatible avec les modules d'analyse loto_analyzer_improved.py,
frequency_analysis.py et cycle_analysis.py.
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import csv
from datetime import datetime
import locale
import logging
import time
import os
import re
from typing import List, Dict, Any, Optional, Union

# Configuration du logging
# Création d'un logger pour la console et un fichier log.txt
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LotoWebScraper")

# Dictionnaire de conversion des mois français vers les nombres
MOIS_FRANCAIS_VERS_NOMBRE = {
    'janvier': '01', 'fevrier': '02', 'février': '02', 'mars': '03', 'avril': '04',
    'mai': '05', 'juin': '06', 'juillet': '07', 'aout': '08', 'août': '08',
    'septembre': '09', 'octobre': '10', 'novembre': '11', 'decembre': '12', 'décembre': '12'
}

def convertir_date_francaise(date_str: str) -> str:
    """
    Convertit une date en format texte français (ex: '04 fevrier 2015') 
    vers le format JJ/MM/AAAA.
    
    Args:
        date_str (str): Date en format texte français
        
    Returns:
        str: Date au format JJ/MM/AAAA ou la chaîne originale si échec
    """
    # Vérification que date_str est bien une chaîne
    if not isinstance(date_str, str):
        logger.warning(f"La date fournie n'est pas une chaîne: {date_str}")
        return str(date_str)
    
    # Si déjà au format JJ/MM/AAAA
    if re.match(r'^\d{2}/\d{2}/\d{4}$', date_str):
        return date_str
    
    # Tentative de conversion depuis format texte français
    pattern = r'^(\d{1,2})\s+([a-zéûôùàçèêëïîœ]+)\s+(\d{4})$'
    match = re.match(pattern, date_str.lower())
    
    if match:
        jour, mois_texte, annee = match.groups()
        
        # Ajouter un zéro devant le jour si nécessaire
        if len(jour) == 1:
            jour = '0' + jour
            
        # Convertir le mois en nombre
        if mois_texte in MOIS_FRANCAIS_VERS_NOMBRE:
            mois = MOIS_FRANCAIS_VERS_NOMBRE[mois_texte]
            return f"{jour}/{mois}/{annee}"
        else:
            logger.warning(f"Mois non reconnu: {mois_texte}")
    
    # Si le format n'est pas reconnu, essayer avec datetime
    try:
        for fmt in ["%d/%m/%Y", "%Y-%m-%d"]:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime("%d/%m/%Y")
            except ValueError:
                continue
    except Exception as e:
        logger.warning(f"Erreur lors de la conversion de date: {e}")
    
    # Si toutes les tentatives échouent, retourner la chaîne originale
    logger.warning(f"Format de date non reconnu: {date_str}")
    return date_str

def scrap_loto_numbers() -> pd.DataFrame:
    """
    Fonction de scraping des tirages du loto depuis le site loto.akroweb.fr.
    
    Returns:
        pd.DataFrame: DataFrame contenant l'historique des tirages avec les colonnes:
                     'Jour', 'Date', 'Numéro 1', 'Numéro 2', 'Numéro 3', 'Numéro 4', 'Numéro 5', 'Chance'
    """
    my_list = []
    
    # Ajout d'un délai pour éviter de surcharger le serveur
    time.sleep(2)
    
    # URL du site source
    loto_url = "http://loto.akroweb.fr/loto-historique-tirages/"
    logger.info(f"Récupération des données depuis: {loto_url}")
    
    try:
        # Récupération de la page avec timeout et retry
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Utilisation d'un User-Agent pour éviter les blocages
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                page = requests.get(loto_url, timeout=15, headers=headers)
                page.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
                break
            except requests.exceptions.RequestException as e:
                retry_count += 1
                wait_time = 2 ** retry_count  # Attente exponentielle
                logger.warning(f"Tentative {retry_count}/{max_retries} échouée: {e}. Nouvelle tentative dans {wait_time}s...")
                time.sleep(wait_time)
                
                if retry_count >= max_retries:
                    logger.error(f"Échec après {max_retries} tentatives: {e}")
                    return pd.DataFrame()
    except Exception as e:
        logger.error(f"Erreur inattendue lors de la requête HTTP: {e}")
        return pd.DataFrame()
    
    # Parsing du HTML
    try:
        soup = BeautifulSoup(page.text, 'html.parser')
        body = soup.find('table')
        
        if not body:
            logger.error("Aucune table trouvée sur la page. La structure du site a peut-être changé.")
            return pd.DataFrame()
        
        tirage_line = body.find_all('tr')
        
        if not tirage_line:
            logger.error("Aucune ligne de tirage trouvée dans la table.")
            return pd.DataFrame()
        
        logger.info(f"Nombre de lignes trouvées: {len(tirage_line)}")
        
        # Extraction des données
        for idx, value in enumerate(tirage_line):
            try:
                my_dict = {}
                res = value.text.split('\n')
                
                # Vérification que nous avons suffisamment d'éléments
                if len(res) < 11:
                    logger.debug(f"Ligne {idx} ignorée: nombre d'éléments insuffisant ({len(res)})")
                    continue
                
                # Extraction du jour et de la date
                my_dict['Jour'] = res[2].strip()
                
                # Conversion de la date au format JJ/MM/AAAA
                date_str = res[3].strip()
                my_dict['Date'] = convertir_date_francaise(date_str)
                
                # Extraction des numéros principaux
                all_numbers_valid = True
                for i, val in enumerate(res[5:10]):
                    try:
                        num_val = int(val.strip())
                        # Vérification que le numéro est dans la plage valide (1-49 pour le Loto)
                        if 1 <= num_val <= 49:
                            my_dict[f'Numéro {i+1}'] = num_val
                        else:
                            logger.warning(f"Numéro {i+1} hors plage valide: {num_val}")
                            all_numbers_valid = False
                            break
                    except ValueError:
                        logger.warning(f"Valeur non numérique pour le numéro {i+1}: {val}")
                        all_numbers_valid = False
                        break
                
                if not all_numbers_valid:
                    continue
                
                # Vérification que tous les numéros principaux ont été extraits
                if len([k for k in my_dict.keys() if k.startswith('Numéro')]) != 5:
                    logger.debug(f"Ligne {idx} ignorée: nombre de numéros principaux incorrect")
                    continue
                
                # Extraction du numéro chance
                try:
                    chance_val = int(res[10].strip())
                    # Vérification que le numéro chance est dans la plage valide (1-10 pour le Loto)
                    if 1 <= chance_val <= 10:
                        my_dict['Chance'] = chance_val
                    else:
                        logger.warning(f"Numéro chance hors plage valide: {chance_val}")
                        my_dict['Chance'] = None
                except ValueError:
                    logger.warning(f"Valeur non numérique pour le numéro chance: {res[10]}")
                    my_dict['Chance'] = None  # Permettre des lignes sans numéro chance
                
                my_list.append(my_dict)
            except Exception as e:
                logger.warning(f"Erreur lors du traitement de la ligne {idx}: {e}")
                continue
    except Exception as e:
        logger.error(f"Erreur lors du parsing HTML: {e}")
        return pd.DataFrame()
    
    if not my_list:
        logger.error("Aucune donnée n'a pu être extraite.")
        return pd.DataFrame()
    
    # Création du DataFrame
    df = pd.DataFrame(my_list)
    
    # Tri par date (du plus ancien au plus récent)
    try:
        df['Date_obj'] = pd.to_datetime(df['Date'], format="%d/%m/%Y", errors='coerce')
        # Vérification des dates invalides
        invalid_dates = df[df['Date_obj'].isna()]
        if not invalid_dates.empty:
            logger.warning(f"{len(invalid_dates)} dates invalides trouvées et seront ignorées dans le tri")
        
        df = df.sort_values(by='Date_obj').reset_index(drop=True)
        df = df.drop(columns=['Date_obj'])
    except Exception as e:
        logger.warning(f"Erreur lors du tri par date: {e}. Le DataFrame ne sera pas trié.")
    
    logger.info(f"Extraction réussie: {len(df)} tirages récupérés.")
    return df

def save_to_csv(df: pd.DataFrame, csv_filename: str = "tirages_loto.csv") -> bool:
    """
    Sauvegarde le DataFrame des tirages dans un fichier CSV.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les tirages.
        csv_filename (str): Nom du fichier CSV de sortie.
        
    Returns:
        bool: True si la sauvegarde a réussi, False sinon.
    """
    if df.empty:
        logger.error("Impossible de sauvegarder un DataFrame vide.")
        return False
    
    try:
        # Création d'une sauvegarde si le fichier existe déjà
        if os.path.exists(csv_filename):
            backup_filename = f"{os.path.splitext(csv_filename)[0]}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            try:
                os.rename(csv_filename, backup_filename)
                logger.info(f"Sauvegarde de l'ancien fichier créée: {backup_filename}")
            except Exception as e:
                logger.warning(f"Impossible de créer une sauvegarde: {e}")
        
        # Définition de l'ordre des colonnes pour compatibilité
        column_order = ['Jour', 'Date']
        for i in range(1, 6):
            column_order.append(f'Numéro {i}')
        column_order.append('Chance')
        
        # Réorganisation des colonnes si elles existent toutes
        if all(col in df.columns for col in column_order):
            df = df[column_order]
        else:
            missing_cols = [col for col in column_order if col not in df.columns]
            logger.warning(f"Colonnes manquantes dans le DataFrame: {missing_cols}")
        
        # Vérification des valeurs manquantes
        missing_values = df.isna().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Valeurs manquantes dans le DataFrame: {missing_values[missing_values > 0].to_dict()}")
        
        # Sauvegarde au format CSV
        df.to_csv(csv_filename, index=False, encoding='utf-8')
        logger.info(f"Fichier CSV '{csv_filename}' créé avec succès.")
        
        # Vérification que le fichier a bien été créé
        if os.path.exists(csv_filename):
            file_size = os.path.getsize(csv_filename)
            logger.info(f"Taille du fichier CSV: {file_size} octets")
            if file_size == 0:
                logger.error(f"Le fichier CSV '{csv_filename}' est vide!")
                return False
            return True
        else:
            logger.error(f"Le fichier CSV '{csv_filename}' n'a pas été créé!")
            return False
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du fichier CSV: {e}")
        return False

def main():
    """Fonction principale pour exécuter le scraping et la sauvegarde."""
    logger.info("Démarrage du scraping des tirages du Loto...")
    
    try:
        # Scraping des données
        start_time = time.time()
        df_tirage = scrap_loto_numbers()
        elapsed_time = time.time() - start_time
        
        if df_tirage.empty:
            logger.error("Échec du scraping. Aucune donnée récupérée.")
            return False
        
        logger.info(f"Scraping terminé en {elapsed_time:.2f} secondes.")
        
        # Affichage des premières lignes pour vérification
        logger.info(f"Aperçu des données récupérées ({len(df_tirage)} tirages au total):")
        print(df_tirage[['Jour', 'Date', 'Numéro 1', 'Numéro 2', 'Numéro 3', 'Numéro 4', 'Numéro 5', 'Chance']].head())
        
        # Vérification des doublons
        duplicates = df_tirage.duplicated(subset=['Date']).sum()
        if duplicates > 0:
            logger.warning(f"{duplicates} tirages en double détectés.")
            # On garde la première occurrence de chaque date
            df_tirage = df_tirage.drop_duplicates(subset=['Date'], keep='first')
            logger.info(f"Doublons supprimés. {len(df_tirage)} tirages uniques restants.")
        
        # Sauvegarde au format CSV
        csv_filename = "tirages_loto.csv"
        success = save_to_csv(df_tirage, csv_filename)
        
        if success:
            logger.info(f"Opération terminée avec succès. Données sauvegardées dans {csv_filename}.")
            print(f"Opération terminée avec succès. {len(df_tirage)} tirages sauvegardés dans {csv_filename}.")
        else:
            logger.error("Échec de l'opération de sauvegarde.")
            print("Échec de l'opération de sauvegarde.")
        
        return success
    except Exception as e:
        logger.error(f"Erreur inattendue dans la fonction principale: {e}")
        print(f"Erreur inattendue: {e}")
        return False

if __name__ == "__main__":
    # Exécution du script en tant que programme principal
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interruption manuelle du script.")
        print("\nInterruption manuelle du script.")
    except Exception as e:
        logger.critical(f"Erreur critique: {e}")
        print(f"Erreur critique: {e}")
