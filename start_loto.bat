@echo off
setlocal enabledelayedexpansion
REM Script batch pour installer les dependances et demarrer l'application Loto
REM Auteur: Systeme de prediction Loto
REM Date: 2025

echo ========================================
echo    LOTO - Interface Web de Prediction
echo ========================================
echo.

REM Changer vers le repertoire du script
cd /d "%~dp0"

REM Verifier si Python est installe
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe ou non accessible.
    echo Veuillez installer Python 3.7 ou superieur.
    echo Telechargez Python depuis: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [INFO] Python detecte:
python --version
echo.

REM Verifier si l'environnement virtuel existe
if exist "env\Scripts\activate.bat" (
    echo [INFO] Activation de l'environnement virtuel existant...
    call env\Scripts\activate.bat
) else (
    echo [INFO] Creation de l'environnement virtuel...
    python -m venv env
    if errorlevel 1 (
        echo [ERREUR] Impossible de creer l'environnement virtuel.
        pause
        exit /b 1
    )
    echo [INFO] Activation de l'environnement virtuel...
    call env\Scripts\activate.bat
)
echo.

REM Mettre a jour pip
echo [INFO] Mise a jour de pip...
python -m pip install --upgrade pip --quiet >nul 2>&1
echo.

REM Verifier et installer les dependances depuis requirements.txt
echo [INFO] Verification des dependances depuis requirements.txt...
if exist requirements.txt (
    REM Utiliser le script Python pour verifier les packages
    if exist check_requirements.py (
        python check_requirements.py
        set CHECK_RESULT=!errorlevel!
    ) else (
        REM Fallback: installer directement si le script n'existe pas
        echo [AVERTISSEMENT] Script check_requirements.py non trouve, installation directe...
        set CHECK_RESULT=1
    )
    
    REM Si des packages manquent, les installer
    if !CHECK_RESULT! EQU 1 (
        echo.
        echo [INFO] Installation des packages manquants...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo [ERREUR] Erreur lors de l'installation des dependances.
            pause
            exit /b 1
        )
        echo [SUCCES] Packages manquants installes avec succes.
    ) else (
        echo [INFO] Passage direct au demarrage...
    )
) else (
    echo [ERREUR] Le fichier requirements.txt est introuvable.
    pause
    exit /b 1
)
echo.

REM Verifier et corriger les dependances PyArrow/scikit-learn
echo [INFO] Verification des dependances critiques scikit-learn...
python -c "import sklearn" >nul 2>&1
if errorlevel 1 (
    set SKLEARN_ERROR=1
) else (
    python -c "from sklearn.ensemble import RandomForestClassifier" >nul 2>&1
    if errorlevel 1 (
        set SKLEARN_ERROR=1
    ) else (
        set SKLEARN_ERROR=0
    )
)

if "%SKLEARN_ERROR%"=="1" (
    echo [ATTENTION] Probleme detecte avec scikit-learn.
    echo [INFO] Correction automatique en cours...
    echo.
    echo [INFO] Desinstallation de PyArrow...
    pip uninstall pyarrow -y >nul 2>&1
    echo [INFO] Reinstallation de scikit-learn...
    pip uninstall scikit-learn -y >nul 2>&1
    pip install scikit-learn --upgrade --no-deps
    echo [INFO] Installation des dependances de scikit-learn...
    pip install numpy scipy joblib threadpoolctl
    python -c "from sklearn.ensemble import RandomForestClassifier" >nul 2>&1
    if errorlevel 1 (
        echo [ERREUR] Impossible de corriger les dependances automatiquement.
        echo [INFO] Essayez d'executer fix_dependencies.bat manuellement.
        echo [INFO] L'application demarrera en mode fallback sans ML.
    ) else (
        echo [SUCCES] Dependances corrigees avec succes.
        echo [ATTENTION] Les modeles existants peuvent etre incompatibles avec la nouvelle version.
        echo [INFO] Il est recommande de reentrainer les modeles pour eviter les warnings.
    )
    echo.
) else (
    echo [SUCCES] Dependances critiques OK.
)
echo.

REM Verifier que les modeles existent
echo [INFO] Verification des modeles de prediction...
if not exist "resultats_loto\models\rf_number_model.joblib" (
    echo [ATTENTION] Le modele rf_number_model.joblib est introuvable.
    echo [INFO] Les modeles seront crees automatiquement au demarrage si scikit-learn fonctionne.
    echo [INFO] Sinon, l'application utilisera le mode fallback avec frequences.
)
if not exist "resultats_loto\models\rf_chance_model.joblib" (
    echo [ATTENTION] Le modele rf_chance_model.joblib est introuvable.
    echo [INFO] Il sera cree automatiquement lors de l'entrainement.
)
if not exist "resultats_loto\models\scaler_numbers.joblib" (
    echo [ATTENTION] Le scaler scaler_numbers.joblib est introuvable.
    echo [INFO] Il sera cree automatiquement lors de l'entrainement.
)
if not exist "resultats_loto\models\scaler_chance.joblib" (
    echo [ATTENTION] Le scaler scaler_chance.joblib est introuvable.
    echo [INFO] Il sera cree automatiquement lors de l'entrainement.
)
echo.

REM Verifier que le fichier CSV existe
echo [INFO] Verification du fichier de donnees...
if not exist "tirages_loto.csv" (
    echo [ATTENTION] Le fichier tirages_loto.csv est introuvable.
    echo [INFO] Les statistiques ne seront pas disponibles.
    echo [INFO] Le scraping automatique creera ce fichier lors du premier tirage.
) else (
    echo [SUCCES] Fichier tirages_loto.csv trouve.
)
echo.

REM Verifier si le port 5000 est deja utilise
echo [INFO] Verification du port 5000...
netstat -ano | findstr ":5000" >nul 2>&1
if not errorlevel 1 (
    echo [ATTENTION] Le port 5000 est deja utilise.
    echo [INFO] Si une autre instance du serveur tourne, arretez-la d'abord.
    echo [INFO] Commande pour voir le processus: netstat -ano ^| findstr ":5000"
    echo.
    set /p CONTINUE="Voulez-vous continuer quand meme? (O/N): "
    if /i "%CONTINUE%"=="O" (
        echo [INFO] Continuation du demarrage...
    ) else (
        echo [INFO] Demarrage annule.
        pause
        exit /b 0
    )
    echo.
) else (
    echo [SUCCES] Port 5000 disponible.
)
echo.

REM Demarrer l'application Flask
echo ========================================
echo    Demarrage du serveur Flask...
echo ========================================
echo.
echo [INFO] Le serveur va demarrer sur http://0.0.0.0:5000
echo [INFO] Port: 5000
echo.
echo [INFO] Acces disponibles:
echo   - IP directe: http://107.189.17.46:5000
echo   - Domaine: https://kenopredictionia.fr (si configure via reverse proxy)
echo   - Local: http://localhost:5000
echo.
echo [INFO] Endpoints API:
echo   - Interface: http://107.189.17.46:5000/
echo   - API Test: http://107.189.17.46:5000/api/test
echo   - API Predict: http://107.189.17.46:5000/api/predict
echo   - API Stats: http://107.189.17.46:5000/api/stats
echo   - API Health: http://107.189.17.46:5000/api/health
echo.
echo [INFO] Appuyez sur Ctrl+C pour arreter le serveur.
echo [INFO] Le serveur est accessible depuis l'exterieur.
echo.

python app.py

REM Si le script se termine, attendre une touche
echo.
echo [INFO] Serveur arrete.
pause
