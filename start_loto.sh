#!/bin/bash
# Script bash pour installer les dependances et demarrer l'application Loto
# Auteur: Systeme de prediction Loto
# Date: 2025

echo "========================================"
echo "   LOTO - Interface Web de Prediction"
echo "========================================"
echo ""

# Changer vers le repertoire du script
cd "$(dirname "$0")"

# Verifier si Python est installe
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "[ERREUR] Python n'est pas installe ou non accessible."
    echo "Veuillez installer Python 3.7 ou superieur."
    exit 1
fi

# Utiliser python3 si disponible, sinon python
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
    PIP_CMD=pip3
else
    PYTHON_CMD=python
    PIP_CMD=pip
fi

echo "[INFO] Python detecte:"
$PYTHON_CMD --version
echo ""

# Verifier si l'environnement virtuel existe
if [ -f "env/bin/activate" ]; then
    echo "[INFO] Activation de l'environnement virtuel existant..."
    source env/bin/activate
else
    echo "[INFO] Creation de l'environnement virtuel..."
    $PYTHON_CMD -m venv env
    if [ $? -ne 0 ]; then
        echo "[ERREUR] Impossible de creer l'environnement virtuel."
        exit 1
    fi
    echo "[INFO] Activation de l'environnement virtuel..."
    source env/bin/activate
fi
echo ""

# Mettre a jour pip
echo "[INFO] Mise a jour de pip..."
$PIP_CMD install --upgrade pip --quiet
echo ""

# Verifier et installer les dependances depuis requirements.txt
echo "[INFO] Verification des dependances depuis requirements.txt..."
if [ -f "requirements.txt" ]; then
    # Utiliser le script Python pour verifier les packages
    if [ -f "check_requirements.py" ]; then
        $PYTHON_CMD check_requirements.py
        CHECK_RESULT=$?
    else
        # Fallback: installer directement si le script n'existe pas
        echo "[AVERTISSEMENT] Script check_requirements.py non trouve, installation directe..."
        CHECK_RESULT=1
    fi
    
    # Si des packages manquent, les installer
    if [ $CHECK_RESULT -eq 1 ]; then
        echo ""
        echo "[INFO] Installation des packages manquants..."
        $PIP_CMD install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "[ERREUR] Erreur lors de l'installation des dependances."
            exit 1
        fi
        echo "[SUCCES] Packages manquants installes avec succes."
    else
        echo "[INFO] Passage direct au demarrage..."
    fi
else
    echo "[ERREUR] Le fichier requirements.txt est introuvable."
    exit 1
fi
echo ""

# Verifier et corriger les dependances PyArrow/scikit-learn
echo "[INFO] Verification des dependances critiques scikit-learn..."
$PYTHON_CMD -c "import sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    SKLEARN_ERROR=1
else
    $PYTHON_CMD -c "from sklearn.ensemble import RandomForestClassifier" 2>/dev/null
    if [ $? -ne 0 ]; then
        SKLEARN_ERROR=1
    else
        SKLEARN_ERROR=0
    fi
fi

if [ "$SKLEARN_ERROR" = "1" ]; then
    echo "[ATTENTION] Probleme detecte avec scikit-learn."
    echo "[INFO] Correction automatique en cours..."
    echo ""
    echo "[INFO] Desinstallation de PyArrow..."
    $PIP_CMD uninstall pyarrow -y >/dev/null 2>&1
    echo "[INFO] Reinstallation de scikit-learn..."
    $PIP_CMD uninstall scikit-learn -y >/dev/null 2>&1
    $PIP_CMD install scikit-learn --upgrade --no-deps
    echo "[INFO] Installation des dependances de scikit-learn..."
    $PIP_CMD install numpy scipy joblib threadpoolctl
    $PYTHON_CMD -c "from sklearn.ensemble import RandomForestClassifier" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "[ERREUR] Impossible de corriger les dependances automatiquement."
        echo "[INFO] Essayez d'executer fix_dependencies.sh manuellement."
        echo "[INFO] L'application demarrera en mode fallback sans ML."
    else
        echo "[SUCCES] Dependances corrigees avec succes."
        echo "[ATTENTION] Les modeles existants peuvent etre incompatibles avec la nouvelle version."
        echo "[INFO] Il est recommande de reentrainer les modeles pour eviter les warnings."
    fi
    echo ""
else
    echo "[SUCCES] Dependances critiques OK."
fi
echo ""

# Verifier que les modeles existent
echo "[INFO] Verification des modeles de prediction..."
if [ ! -f "resultats_loto/models/rf_number_model.joblib" ]; then
    echo "[ATTENTION] Le modele rf_number_model.joblib est introuvable."
    echo "[INFO] Les modeles seront crees automatiquement au demarrage si scikit-learn fonctionne."
    echo "[INFO] Sinon, l'application utilisera le mode fallback avec frequences."
fi
if [ ! -f "resultats_loto/models/rf_chance_model.joblib" ]; then
    echo "[ATTENTION] Le modele rf_chance_model.joblib est introuvable."
    echo "[INFO] Il sera cree automatiquement lors de l'entrainement."
fi
if [ ! -f "resultats_loto/models/scaler_numbers.joblib" ]; then
    echo "[ATTENTION] Le scaler scaler_numbers.joblib est introuvable."
    echo "[INFO] Il sera cree automatiquement lors de l'entrainement."
fi
if [ ! -f "resultats_loto/models/scaler_chance.joblib" ]; then
    echo "[ATTENTION] Le scaler scaler_chance.joblib est introuvable."
    echo "[INFO] Il sera cree automatiquement lors de l'entrainement."
fi
echo ""

# Verifier que le fichier CSV existe
echo "[INFO] Verification du fichier de donnees..."
if [ ! -f "tirages_loto.csv" ]; then
    echo "[ATTENTION] Le fichier tirages_loto.csv est introuvable."
    echo "[INFO] Les statistiques ne seront pas disponibles."
    echo "[INFO] Le scraping automatique creera ce fichier lors du premier tirage."
else
    echo "[SUCCES] Fichier tirages_loto.csv trouve."
fi
echo ""

# Verifier si le port 5000 est deja utilise
echo "[INFO] Verification du port 5000..."
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1 || netstat -tuln 2>/dev/null | grep -q ":5000 "; then
    echo "[ATTENTION] Le port 5000 est deja utilise."
    echo "[INFO] Si une autre instance du serveur tourne, arretez-la d'abord."
    echo "[INFO] Commande pour voir le processus: lsof -i :5000 ou netstat -tuln | grep 5000"
    echo ""
    read -p "Voulez-vous continuer quand meme? (O/N): " CONTINUE
    if [ "$CONTINUE" != "O" ] && [ "$CONTINUE" != "o" ]; then
        echo "[INFO] Demarrage annule."
        exit 0
    fi
    echo ""
else
    echo "[SUCCES] Port 5000 disponible."
fi
echo ""

# Demarrer l'application Flask
echo "========================================"
echo "   Demarrage du serveur Flask..."
echo "========================================"
echo ""
echo "[INFO] Le serveur va demarrer sur http://0.0.0.0:5000"
echo "[INFO] Port: 5000"
echo ""
echo "[INFO] Acces disponibles:"
echo "  - IP directe: http://107.189.17.46:5000"
echo "  - Domaine: https://kenopredictionia.fr (si configure via reverse proxy)"
echo "  - Local: http://localhost:5000"
echo ""
echo "[INFO] Endpoints API:"
echo "  - Interface: http://107.189.17.46:5000/"
echo "  - API Test: http://107.189.17.46:5000/api/test"
echo "  - API Predict: http://107.189.17.46:5000/api/predict"
echo "  - API Stats: http://107.189.17.46:5000/api/stats"
echo "  - API Health: http://107.189.17.46:5000/api/health"
echo ""
echo "[INFO] Appuyez sur Ctrl+C pour arreter le serveur."
echo "[INFO] Le serveur est accessible depuis l'exterieur."
echo ""

$PYTHON_CMD app.py

echo ""
echo "[INFO] Serveur arrete."

