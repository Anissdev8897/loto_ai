#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de visualisation pour l'analyseur Loto
Ce module implémente des composants de visualisation avancés pour présenter
les résultats des analyses de tirages Loto.

Version améliorée avec gestion robuste des dates au format JJ/MM/AAAA,
optimisation des performances et gestion d'erreurs renforcée.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from datetime import datetime
import re
import time
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import Counter
from pathlib import Path
import traceback


# Configuration du logging avec redirection vers fichier et console
logger = logging.getLogger("LotoVisualization")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("log.txt"),
            logging.StreamHandler()
        ]
    )

# Dictionnaire de conversion des mois français vers les nombres
MOIS_FRANCAIS_VERS_NOMBRE = {
    'janvier': '01', 'fevrier': '02', 'février': '02', 'mars': '03', 'avril': '04',
    'mai': '05', 'juin': '06', 'juillet': '07', 'aout': '08', 'août': '08',
    'septembre': '09', 'octobre': '10', 'novembre': '11', 'decembre': '12', 'décembre': '12'
}

# Gestion de l'import de LotoAnalyzer pour l'annotation de type et l'utilisation
LOTO_ANALYZER_TYPE_HINT = 'LotoAnalyzer'
try:
    from improved_loto_analyzer_improved import LotoAnalyzer
    logger.info("Module 'improved_loto_analyzer_improved.LotoAnalyzer' importé avec succès")
except ImportError:
    try:
        from loto_analyzer_improved import LotoAnalyzer
        logger.info("Module 'loto_analyzer_improved.LotoAnalyzer' importé avec succès")
    except ImportError:
        logger.error("Impossible d'importer 'LotoAnalyzer'. "
              "Assurez-vous que le fichier est accessible et ne contient pas d'erreurs d'importation cyclique.")
        # Définition d'un type factice pour permettre au code de charger sans erreur de nom immédiate.
        LotoAnalyzer = type('LotoAnalyzer', (object,), {})


def convertir_date_francaise(date_str: str) -> str:
    """
    Convertit une date en format texte français (ex: '04 fevrier 2015') 
    vers le format JJ/MM/AAAA.
    
    Args:
        date_str (str): Date en format texte français
        
    Returns:
        str: Date au format JJ/MM/AAAA ou la chaîne originale si échec
    """
    try:
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
        
        # Si le format n'est pas reconnu, essayer avec datetime
        for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"]:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime("%d/%m/%Y")
            except ValueError:
                continue
        
        # Si toutes les tentatives échouent, retourner la chaîne originale
        logger.warning(f"Format de date non reconnu: {date_str}")
        return date_str
    except Exception as e:
        logger.error(f"Erreur lors de la conversion de la date '{date_str}': {e}", exc_info=True)
        return date_str


class LotoVisualization:
    """
    Classe pour la visualisation des analyses Loto.
    
    Cette classe fournit des méthodes pour générer différents types de visualisations
    à partir des résultats d'analyse de tirages Loto, notamment:
    - Distribution des fréquences des numéros
    - Analyse de parité
    - Analyse des sommes
    - Analyse de séries temporelles
    """
    
    def __init__(self, analyzer: Any, output_dir: Optional[str] = None):
        """
        Initialise le module de visualisation.
        
        Args:
            analyzer: Instance de LotoAnalyzer.
            output_dir: Répertoire de sortie pour les visualisations (optionnel).
            
        Raises:
            TypeError: Si l'argument analyzer n'est pas une instance valide de LotoAnalyzer.
        """
        try:
            # Vérification du type d'analyzer
            if not hasattr(analyzer, '__class__') or not hasattr(analyzer.__class__, '__name__'):
                raise TypeError("L'argument 'analyzer' n'est pas un objet valide.")
                
            analyzer_class_name = analyzer.__class__.__name__
            if not analyzer_class_name.endswith('LotoAnalyzer'):
                logger.warning(f"L'objet fourni ({analyzer_class_name}) pourrait ne pas être une instance de LotoAnalyzer.")
            
            self.analyzer = analyzer
            
            # Accès sécurisé aux attributs de l'analyzer
            self.df = getattr(analyzer, 'df', pd.DataFrame())
            if self.df.empty:
                logger.warning("Le DataFrame de l'analyseur est vide.")
                
            self.ball_cols = getattr(analyzer, 'ball_cols', [])
            if not self.ball_cols:
                logger.warning("Aucune colonne de boules définie dans l'analyseur.")
                
            self.chance_col = getattr(analyzer, 'chance_col', None)
            self.config = getattr(analyzer, 'config', {})
            
            # Configuration du répertoire de sortie
            analyzer_output_dir = getattr(analyzer, 'output_dir', Path("resultats_loto"))
            
            if output_dir:
                self.output_dir = Path(output_dir)
            else:
                self.output_dir = Path(analyzer_output_dir) / "visualizations"
                
            # Création des répertoires nécessaires
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Répertoire de visualisations prêt: {self.output_dir}")
            except Exception as e:
                logger.error(f"Impossible de créer le répertoire de visualisations {self.output_dir}: {e}", exc_info=True)
                # Utiliser un répertoire temporaire ou local comme fallback
                self.output_dir = Path(".")
                logger.info(f"Utilisation du répertoire courant comme fallback: {self.output_dir.absolute()}")

            # Configuration du style des graphiques
            self.setup_plot_style()
            
            # Paramètres de performance
            self.use_cache = True
            self.cache_dir = self.output_dir / "cache"
            if self.use_cache:
                try:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Répertoire de cache prêt: {self.cache_dir}")
                except Exception as e:
                    logger.warning(f"Impossible de créer le répertoire de cache: {e}")
                    self.use_cache = False
            
            logger.info("Module de visualisation initialisé avec succès.")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du module de visualisation: {e}", exc_info=True)
            raise
    
    def setup_plot_style(self):
        """
        Configure le style des graphiques pour une présentation optimale.
        """
        try:
            # Essayer d'utiliser un style moderne de seaborn
            try:
                plt.style.use('seaborn-v0_8-whitegrid')
            except OSError:
                try:
                    plt.style.use('seaborn-whitegrid')
                except OSError:
                    logger.warning("Style 'seaborn-whitegrid' non trouvé. Utilisation du style par défaut.")
                    plt.style.use('default')
                
            # Configuration des paramètres de police et de taille
            mpl.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
            mpl.rcParams['font.size'] = 10
            mpl.rcParams['axes.titlesize'] = 14
            mpl.rcParams['axes.labelsize'] = 12
            mpl.rcParams['xtick.labelsize'] = 9
            mpl.rcParams['ytick.labelsize'] = 10
            mpl.rcParams['legend.fontsize'] = 10
            
            # Palette de couleurs personnalisée pour une meilleure lisibilité
            self.colors = {
                'primary': '#1f77b4',  # Bleu
                'secondary': '#ff7f0e',  # Orange
                'tertiary': '#2ca02c',  # Vert
                'quaternary': '#d62728',  # Rouge
                'hot': '#d62728',  # Rouge pour les numéros chauds
                'cold': '#1f77b4',  # Bleu pour les numéros froids
                'positive': '#2ca02c',  # Vert pour les valeurs positives
                'negative': '#d62728',  # Rouge pour les valeurs négatives
                'neutral': '#7f7f7f',  # Gris pour les valeurs neutres
                'highlight': '#9467bd'  # Violet pour les éléments à mettre en évidence
            }
            
            logger.debug("Style des graphiques configuré avec succès.")
        except Exception as e:
            logger.error(f"Erreur lors de la configuration du style des graphiques: {e}", exc_info=True)
            # Définir des couleurs par défaut en cas d'erreur
            self.colors = {
                'primary': 'blue', 'secondary': 'orange', 'tertiary': 'green',
                'quaternary': 'red', 'hot': 'red', 'cold': 'blue',
                'positive': 'green', 'negative': 'red', 'neutral': 'gray',
                'highlight': 'purple'
            }

    def _save_plot(self, fig: plt.Figure, filename: str) -> Optional[str]:
        """
        Sauvegarde une figure Matplotlib dans le répertoire de sortie.
        
        Args:
            fig: Figure Matplotlib à sauvegarder.
            filename: Nom du fichier pour la sauvegarde.
            
        Returns:
            Chemin du fichier sauvegardé ou None en cas d'erreur.
        """
        try:
            if not fig or not filename:
                logger.warning("Figure ou nom de fichier manquant pour la sauvegarde.")
                if fig: plt.close(fig)
                return None
                
            filepath = self.output_dir / filename
            
            # Sauvegarde avec gestion d'erreurs
            try:
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Graphique sauvegardé: {filepath}")
                return str(filepath)
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde du graphique {filename}: {e}", exc_info=True)
                # Tentative de sauvegarde avec un format alternatif
                try:
                    alt_filepath = self.output_dir / f"{Path(filename).stem}_alt.png"
                    fig.savefig(alt_filepath, dpi=200, format='png')
                    plt.close(fig)
                    logger.info(f"Graphique sauvegardé avec format alternatif: {alt_filepath}")
                    return str(alt_filepath)
                except Exception as e2:
                    logger.error(f"Échec de la sauvegarde alternative: {e2}", exc_info=True)
                    if fig: plt.close(fig)
                    return None
        except Exception as e:
            logger.error(f"Erreur inattendue lors de la sauvegarde du graphique: {e}", exc_info=True)
            if 'fig' in locals() and fig: plt.close(fig)
            return None

    def _check_cache(self, plot_type: str, timestamp: str = None) -> Optional[str]:
        """
        Vérifie si un graphique est déjà en cache.
        
        Args:
            plot_type: Type de graphique à vérifier.
            timestamp: Horodatage spécifique à vérifier (optionnel).
            
        Returns:
            Chemin du fichier en cache ou None si non trouvé.
        """
        if not self.use_cache:
            return None
            
        try:
            # Si un timestamp spécifique est fourni, vérifier ce fichier
            if timestamp:
                cache_file = self.cache_dir / f"{plot_type}_{timestamp}.png"
                if cache_file.exists():
                    logger.info(f"Graphique trouvé en cache: {cache_file}")
                    return str(cache_file)
                return None
            
            # Sinon, chercher le fichier le plus récent du type demandé
            cache_files = list(self.cache_dir.glob(f"{plot_type}_*.png"))
            if cache_files:
                # Trier par date de modification (la plus récente d'abord)
                cache_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                # Vérifier si le cache est récent (moins de 24h)
                cache_age = time.time() - cache_files[0].stat().st_mtime
                if cache_age < 86400:  # 24 heures en secondes
                    logger.info(f"Graphique récent trouvé en cache: {cache_files[0]}")
                    return str(cache_files[0])
            
            return None
        except Exception as e:
            logger.error(f"Erreur lors de la vérification du cache: {e}", exc_info=True)
            return None

    def _save_to_cache(self, fig: plt.Figure, plot_type: str, timestamp: str) -> Optional[str]:
        """
        Sauvegarde une figure dans le cache.
        
        Args:
            fig: Figure Matplotlib à sauvegarder.
            plot_type: Type de graphique.
            timestamp: Horodatage pour le nom du fichier.
            
        Returns:
            Chemin du fichier en cache ou None en cas d'erreur.
        """
        if not self.use_cache:
            return None
            
        try:
            cache_file = self.cache_dir / f"{plot_type}_{timestamp}.png"
            fig.savefig(cache_file, dpi=300, bbox_inches='tight')
            logger.debug(f"Graphique sauvegardé en cache: {cache_file}")
            return str(cache_file)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde en cache: {e}", exc_info=True)
            return None

    def plot_frequency_distribution(self, force_refresh: bool = False) -> Optional[str]:
        """
        Génère un graphique de la distribution des fréquences des numéros et du numéro chance.
        
        Args:
            force_refresh: Force la régénération du graphique même si un cache existe.
            
        Returns:
            Chemin du fichier graphique généré ou None en cas d'erreur.
        """
        start_time = time.time()
        logger.info("Génération du graphique de distribution des fréquences...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Vérifier le cache si force_refresh est False
        if not force_refresh:
            cache_path = self._check_cache("frequency_distribution")
            if cache_path:
                return cache_path
        
        try:
            # Vérifier la disponibilité des données
            if not hasattr(self.analyzer, 'freq') or not self.analyzer.freq:
                logger.warning("Données de fréquence des numéros principaux non disponibles (attribut 'freq').")
                return None

            # Création de la figure avec deux sous-graphiques
            fig, axs = plt.subplots(2, 1, figsize=(14, 12), tight_layout=True)
            # NOTE (audit C2) : la fin de plot_frequency_distribution a ete perdue (troncature).
            try:
                plt.close(fig)
            except Exception:
                pass
            return None
        except Exception as e:
            logger.error(f"Erreur lors de la generation du graphique: {e}", exc_info=True)
            return None
