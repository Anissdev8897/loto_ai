#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de scraping automatique et réentraînement des modèles Loto
Système qui scrape automatiquement les jours de tirage (lundi, mercredi, samedi)
et réentraîne les modèles avec un encodeur amélioré
"""

import os
import sys
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
import schedule
import pandas as pd

# Ajouter le répertoire script au path
current_dir = Path(__file__).parent
script_dir = current_dir / "script"
sys.path.insert(0, str(script_dir))

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("auto_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AutoScraper")

# Import des modules de scraping et d'entraînement
try:
    from loto_web_scraper import scrap_loto_numbers, save_to_csv
except ImportError as e:
    logger.error(f"Impossible d'importer loto_web_scraper: {e}")
    scrap_loto_numbers = None
    save_to_csv = None


class AutoScraper:
    """Classe pour le scraping automatique et la mise à jour des modèles"""
    
    def __init__(self, csv_file: str = "tirages_loto.csv"):
        """
        Initialise le système de scraping automatique
        
        Args:
            csv_file: Chemin vers le fichier CSV des tirages
        """
        self.csv_file = Path(csv_file)
        self.is_running = False
        self.scraper_thread = None
        
    def should_scrape_today(self) -> bool:
        """
        Vérifie si on doit scraper aujourd'hui
        
        Returns:
            bool: True si on doit scraper (lundi, mercredi ou samedi)
        """
        today = datetime.now()
        day_name = today.strftime('%A').lower()
        
        # Jours de tirage Loto
        draw_days = ['monday', 'wednesday', 'saturday']  # lundi, mercredi, samedi
        
        return day_name in draw_days
    
    def scrape_and_update(self) -> bool:
        """
        Scrape les données et met à jour le CSV
        
        Returns:
            bool: True si le scraping a réussi
        """
        if not scrap_loto_numbers or not save_to_csv:
            logger.error("Modules de scraping non disponibles")
            return False
        
        try:
            logger.info("Début du scraping automatique...")
            
            # Scraper les données
            df_new = scrap_loto_numbers()
            
            if df_new.empty:
                logger.error("Aucune donnée récupérée lors du scraping")
                return False
            
            # Charger les données existantes si le fichier existe
            if self.csv_file.exists():
                try:
                    df_existing = pd.read_csv(self.csv_file, encoding='utf-8')
                    logger.info(f"Chargement de {len(df_existing)} tirages existants")
                    
                    # Convertir les dates pour comparaison
                    df_existing['Date_obj'] = pd.to_datetime(
                        df_existing['Date'], format='%d/%m/%Y', errors='coerce'
                    )
                    df_new['Date_obj'] = pd.to_datetime(
                        df_new['Date'], format='%d/%m/%Y', errors='coerce'
                    )
                    
                    # Fusionner en gardant seulement les nouveaux tirages (qui n'existent pas déjà)
                    existing_dates = set(df_existing['Date_obj'].dropna())
                    df_new_unique = df_new[~df_new['Date_obj'].isin(existing_dates)]
                    
                    if df_new_unique.empty:
                        logger.info("Aucun nouveau tirage détecté dans les données scrapées")
                        # Vérifier s'il manque des tirages dans le CSV (comparer avec les données scrapées)
                        existing_count = len(df_existing)
                        scraped_count = len(df_new)
                        
                        if scraped_count > existing_count:
                            missing_count = scraped_count - existing_count
                            logger.warning(f"⚠️ {missing_count} tirages potentiellement manquants détectés")
                            logger.info("Les données scrapées contiennent plus de tirages que le CSV")
                            logger.info("Vérification des dates manquantes...")
                            
                            # Trouver les dates manquantes
                            scraped_dates = set(df_new['Date_obj'].dropna())
                            missing_dates = scraped_dates - existing_dates
                            
                            if missing_dates:
                                logger.info(f"Dates manquantes détectées: {len(missing_dates)}")
                                # Utiliser toutes les données scrapées (plus complètes)
                                logger.info("Remplacement du CSV par les données scrapées complètes")
                                df_final = df_new.copy()
                                df_final = df_final.drop(columns=['Date_obj'])
                                df_final['Date_obj'] = pd.to_datetime(
                                    df_final['Date'], format='%d/%m/%Y', errors='coerce'
                                )
                                df_final = df_final.sort_values('Date_obj').reset_index(drop=True)
                                df_final = df_final.drop(columns=['Date_obj'])
                            else:
                                logger.info("Pas de tirages manquants détectés")
                                return False
                        else:
                            return False
                    else:
                        logger.info(f"{len(df_new_unique)} nouveaux tirages détectés")
                        
                        # Combiner les DataFrames
                        df_combined = pd.concat([df_existing, df_new_unique], ignore_index=True)
                        df_combined = df_combined.drop(columns=['Date_obj'])
                        
                        # Trier par date
                        df_combined['Date_obj'] = pd.to_datetime(
                            df_combined['Date'], format='%d/%m/%Y', errors='coerce'
                        )
                        df_combined = df_combined.sort_values('Date_obj').reset_index(drop=True)
                        df_combined = df_combined.drop(columns=['Date_obj'])
                        
                        df_final = df_combined
                except Exception as e:
                    logger.warning(f"Erreur lors du chargement des données existantes: {e}")
                    logger.info("Utilisation des données scrapées complètes comme fallback")
                    df_final = df_new
            else:
                logger.info("Aucun fichier CSV existant, création avec les données scrapées")
                df_final = df_new
            
            # Sauvegarder
            success = save_to_csv(df_final, str(self.csv_file))
            
            if success:
                logger.info(f"Scraping terminé avec succès. {len(df_final)} tirages au total")
                return True
            else:
                logger.error("Échec de la sauvegarde")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors du scraping automatique: {e}", exc_info=True)
            return False
    
    def schedule_scraping(self):
        """Programme les scrapings automatiques"""
        # Jours de tirage Loto : lundi, mercredi, samedi à 23h00 (après le tirage)
        schedule.every().monday.at("23:00").do(self.scrape_and_train)
        schedule.every().wednesday.at("23:00").do(self.scrape_and_train)
        schedule.every().saturday.at("23:00").do(self.scrape_and_train)
        
        logger.info("Scraping automatique programmé pour les lundis, mercredis et samedis à 23h00")
    
    def scrape_and_train(self):
        """Scrape les données et réentraîne les modèles"""
        logger.info("=== Début du cycle de scraping et réentraînement ===")
        
        try:
            # 1. Scraper les nouvelles données
            scrap_success = self.scrape_and_update()
            
            # 2. Réentraîner les modèles avec l'encodeur amélioré (même sans nouvelles données)
            logger.info("Démarrage du réentraînement avec encodeur amélioré...")
            try:
                from auto_trainer import AutoTrainer
                trainer = AutoTrainer(str(self.csv_file))
                if trainer.train_and_save():
                    logger.info("=== Cycle terminé avec succès ===")
                    # Recharger les modèles dans l'application Flask si elle tourne
                    try:
                        # Import conditionnel pour éviter les dépendances circulaires
                        import app
                        if hasattr(app, 'load_models'):
                            app.load_models()
                            logger.info("Modèles rechargés dans l'application Flask")
                    except:
                        pass
                else:
                    logger.error("=== Échec du réentraînement ===")
            except Exception as e:
                logger.error(f"Erreur lors du réentraînement: {e}", exc_info=True)
                
            if not scrap_success:
                logger.warning("Aucune nouvelle donnée récupérée, réentraînement effectué avec données existantes")
        except Exception as e:
            logger.error(f"Erreur dans scrape_and_train: {e}", exc_info=True)
    
    def run_scheduler(self):
        """Exécute le scheduler en continu"""
        self.is_running = True
        logger.info("Démarrage du scheduler de scraping automatique...")
        
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Vérifier chaque minute
    
    def start(self):
        """Démarre le scraping automatique dans un thread séparé"""
        if self.scraper_thread and self.scraper_thread.is_alive():
            logger.warning("Le scraper est déjà en cours d'exécution")
            return
        
        # Programmer les scrapings
        self.schedule_scraping()
        
        # Vérifier si on doit scraper maintenant (au démarrage si c'est un jour de tirage)
        if self.should_scrape_today():
            current_time = datetime.now().time()
            # Si on est après 23h00, scraper immédiatement
            if current_time >= datetime.strptime("23:00", "%H:%M").time():
                logger.info("Scraping immédiat car c'est un jour de tirage après 23h00")
                self.scrape_and_train()
        
        # Démarrer le thread du scheduler
        self.scraper_thread = threading.Thread(target=self.run_scheduler, daemon=True)
        self.scraper_thread.start()
        logger.info("Scraper automatique démarré en arrière-plan")
    
    def stop(self):
        """Arrête le scraping automatique"""
        self.is_running = False
        logger.info("Arrêt du scraper automatique demandé")


if __name__ == "__main__":
    scraper = AutoScraper()
    scraper.start()
    
    try:
        # Maintenir le programme en vie
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interruption demandée")
        scraper.stop()

