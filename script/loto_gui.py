#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface graphique pour l'analyseur de tirages de Loto

Version améliorée avec gestion robuste des dates au format JJ/MM/AAAA,
compatibilité avec les scripts adaptés, logs détaillés et gestion d'erreurs renforcée.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import subprocess
from pathlib import Path
import webbrowser
import glob  # Pour la recherche de fichiers
import re
import logging
from datetime import datetime
import traceback
import json
import time

# Configuration du logging avec redirection vers fichier et console
logger = logging.getLogger("LotoGUI")
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
        for fmt in ["%d/%m/%Y", "%Y-%m-%d"]:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime("%d/%m/%Y")
            except ValueError:
                continue
        
        # Si toutes les tentatives échouent, retourner la chaîne originale
        logger.warning(f"Format de date non reconnu: {date_str}")
        return date_str
    except Exception as e:
        logger.error(f"Erreur lors de la conversion de la date '{date_str}': {e}")
        return date_str

class LotoAnalyzerGUI:
    """Interface graphique pour l'analyseur de tirages de Loto."""
    
    def __init__(self, root):
        """
        Initialise l'interface graphique.
        
        Args:
            root: Fenêtre principale Tkinter
        """
        try:
            logger.info("Initialisation de l'interface graphique LotoAnalyzerGUI")
            self.root = root
            self.root.title("Analyseur de Loto - Interface Graphique")
            self.root.geometry("800x650")  # Légèrement plus grand pour les nouvelles options
            self.root.minsize(800, 650)
            
            # Gestion des erreurs non capturées
            self.root.report_callback_exception = self.handle_exception
            
            # Variables de configuration
            self.csv_file_var = tk.StringVar()
            self.window_var = tk.StringVar(value="1328")  # Valeur par défaut LotoAnalyzer
            self.hot_var = tk.StringVar(value="5")
            self.cold_var = tk.StringVar(value="5")
            self.size_var = tk.StringVar(value="5")  # propose_size
            self.max_number_var = tk.StringVar(value="49")  # max_number
            self.max_chance_var = tk.StringVar(value="10")  # max_chance
            
            # Nouvelles variables pour les options avancées
            self.chance_analysis_var = tk.BooleanVar(value=True)
            self.fibonacci_inverse_var = tk.BooleanVar(value=False)
            self.train_ml_var = tk.BooleanVar(value=True)  # Entraîner ML par défaut
            self.incremental_var = tk.BooleanVar(value=False)  # Apprentissage incrémental
            self.backtesting_var = tk.BooleanVar(value=False)  # Backtesting
            self.visualize_var = tk.BooleanVar(value=True)  # Générer visualisations
            self.excel_var = tk.BooleanVar(value=False)  # Export Excel
            self.output_dir_var = tk.StringVar(value="resultats_loto")
            
            # Fichiers de résultats (chemins complets)
            self.results_txt_file = None
            self.excel_file = None
            self.main_plot_file = None  # Plot de fréquence/parité/somme
            self.backtesting_report_file = None
            self.backtesting_plot_files = []
            self.incremental_plot_files = []
            
            # Processus en cours
            self.running_process = None
            
            # Créer les widgets
            self.create_widgets()
            
            # Configurer le style
            self.configure_style()
            
            logger.info("Interface graphique initialisée avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de l'interface graphique: {e}", exc_info=True)
            messagebox.showerror("Erreur d'initialisation", 
                                f"Une erreur est survenue lors de l'initialisation: {str(e)}")
    
    def handle_exception(self, exc_type, exc_value, exc_traceback):
        """
        Gère les exceptions non capturées dans l'interface graphique.
        
        Args:
            exc_type: Type de l'exception
            exc_value: Valeur de l'exception
            exc_traceback: Traceback de l'exception
        """
        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        logger.error(f"Exception non capturée dans l'interface graphique: {error_msg}")
        messagebox.showerror("Erreur", f"Une erreur inattendue est survenue:\n{str(exc_value)}")
    
    def configure_style(self):
        """Configure le style de l'interface graphique."""
        try:
            style = ttk.Style()
            
            # Essayer de configurer un thème moderne si disponible
            available_themes = style.theme_names()
            if 'clam' in available_themes:
                style.theme_use('clam')
            elif 'vista' in available_themes:
                style.theme_use('vista')
            
            # Configurer le style du bouton d'action principal
            style.configure("Accent.TButton", 
                            font=('Helvetica', 10, 'bold'),
                            background="#4CAF50",
                            foreground="white")
        except Exception as e:
            logger.warning(f"Erreur lors de la configuration du style: {e}")
    
    def create_widgets(self):
        """Crée tous les widgets de l'interface."""
        try:
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            notebook = ttk.Notebook(main_frame)
            notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            config_frame = ttk.Frame(notebook, padding="10")
            notebook.add(config_frame, text="Configuration")
            
            results_frame = ttk.Frame(notebook, padding="10")
            notebook.add(results_frame, text="Résultats & Fichiers")
            
            about_frame = ttk.Frame(notebook, padding="10")
            notebook.add(about_frame, text="À propos")
            
            self.setup_config_tab(config_frame)
            self.setup_results_tab(results_frame)
            self.setup_about_tab(about_frame)
            
            # Barre d'état
            status_frame = ttk.Frame(main_frame, relief=tk.SUNKEN, padding=(2, 2, 2, 2))
            status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
            
            self.status_var = tk.StringVar(value="Prêt")
            status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W)
            status_label.pack(fill=tk.X)
            
            logger.debug("Widgets créés avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de la création des widgets: {e}", exc_info=True)
            raise
    
    def setup_config_tab(self, parent):
        """
        Configure l'onglet de configuration.
        
        Args:
            parent: Widget parent pour l'onglet
        """
        try:
            # Cadre pour le fichier CSV
            file_frame = ttk.LabelFrame(parent, text="Fichier de données", padding="10")
            file_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(file_frame, text="Fichier CSV:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
            ttk.Entry(file_frame, textvariable=self.csv_file_var, width=50).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
            ttk.Button(file_frame, text="Parcourir...", command=self.browse_csv).grid(row=0, column=2, sticky=tk.E, padx=5, pady=5)
            
            # Cadre pour les paramètres de base
            params_frame = ttk.LabelFrame(parent, text="Paramètres de base", padding="10")
            params_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(params_frame, text="Fenêtre d'analyse:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
            ttk.Entry(params_frame, textvariable=self.window_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
            ttk.Label(params_frame, text="tirages").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
            
            ttk.Label(params_frame, text="Numéros chauds:").grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
            ttk.Entry(params_frame, textvariable=self.hot_var, width=10).grid(row=0, column=4, sticky=tk.W, padx=5, pady=5)
            
            ttk.Label(params_frame, text="Numéros froids:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
            ttk.Entry(params_frame, textvariable=self.cold_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
            
            ttk.Label(params_frame, text="Taille combinaison:").grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)
            ttk.Entry(params_frame, textvariable=self.size_var, width=10).grid(row=1, column=4, sticky=tk.W, padx=5, pady=5)
            
            ttk.Label(params_frame, text="Numéro max:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
            ttk.Entry(params_frame, textvariable=self.max_number_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
            
            ttk.Label(params_frame, text="Numéro chance max:").grid(row=2, column=3, sticky=tk.W, padx=5, pady=5)
            ttk.Entry(params_frame, textvariable=self.max_chance_var, width=10).grid(row=2, column=4, sticky=tk.W, padx=5, pady=5)
            
            ttk.Label(params_frame, text="Répertoire de sortie:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
            ttk.Entry(params_frame, textvariable=self.output_dir_var, width=30).grid(row=3, column=1, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=5)
            ttk.Button(params_frame, text="...", command=self.browse_output_dir, width=3).grid(row=3, column=4, sticky=tk.W, padx=5, pady=5)
            
            # Cadre pour les options avancées
            options_frame = ttk.LabelFrame(parent, text="Options avancées", padding="10")
            options_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Checkbutton(options_frame, text="Analyser numéros chance", variable=self.chance_analysis_var).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Checkbutton(options_frame, text="Utiliser Fibonacci inversé", variable=self.fibonacci_inverse_var).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Checkbutton(options_frame, text="Entraîner modèles ML", variable=self.train_ml_var).grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
            
            ttk.Checkbutton(options_frame, text="Apprentissage incrémental", variable=self.incremental_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
            ttk.Checkbutton(options_frame, text="Activer Backtesting", variable=self.backtesting_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
            ttk.Checkbutton(options_frame, text="Générer visualisations", variable=self.visualize_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
            ttk.Checkbutton(options_frame, text="Exporter vers Excel", variable=self.excel_var).grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)

            # Boutons d'action
            action_frame = ttk.Frame(parent, padding="10")
            action_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Button(action_frame, text="Lancer l'analyse", command=self.run_analysis, style="Accent.TButton").pack(side=tk.RIGHT, padx=5)
            ttk.Button(action_frame, text="Réinitialiser", command=self.reset_config).pack(side=tk.RIGHT, padx=5)
            
            logger.debug("Onglet de configuration configuré avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de la configuration de l'onglet de configuration: {e}", exc_info=True)
            raise
    
    def setup_results_tab(self, parent):
        """
        Configure l'onglet de résultats et de fichiers.
        
        Args:
            parent: Widget parent pour l'onglet
        """
        try:
            # Zone de texte pour les résultats du log
            results_label = ttk.Label(parent, text="Journal de l'analyse:")
            results_label.pack(anchor=tk.W, padx=5, pady=5)
            
            self.results_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD, height=15)  # Hauteur ajustée
            self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            self.results_text.config(state=tk.DISABLED)
            
            # Cadre pour les fichiers générés
            files_frame = ttk.LabelFrame(parent, text="Fichiers générés", padding="10")
            files_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Utiliser un grid pour mieux organiser les boutons de fichiers
            files_frame.columnconfigure(0, weight=1)
            files_frame.columnconfigure(1, weight=1)
            files_frame.columnconfigure(2, weight=1)
            files_frame.columnconfigure(3, weight=1)

            ttk.Button(files_frame, text="Ouvrir Rapport Texte", command=lambda: self.open_file(self.results_txt_file)).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W+tk.E)
            ttk.Button(files_frame, text="Ouvrir Fichier Excel", command=lambda: self.open_file(self.excel_file)).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
            ttk.Button(files_frame, text="Ouvrir Graphiques (Analyse)", command=lambda: self.open_folder(Path(self.output_dir_var.get()) / "visualizations")).grid(row=0, column=2, padx=5, pady=5, sticky=tk.W+tk.E)
            ttk.Button(files_frame, text="Ouvrir Dossier Résultats", command=self.open_results_folder).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W+tk.E)

            ttk.Button(files_frame, text="Ouvrir Rapport Backtesting", command=lambda: self.open_file(self.backtesting_report_file)).grid(row=1, column=0, padx=5, pady=5, sticky=tk.W+tk.E)
            # NOTE (audit C2) : la fin de setup_results_tab a ete perdue (troncature de fichier).
            return
        except Exception:
            return
