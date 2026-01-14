@echo off
REM Script batch pour installer uniquement les dependances
REM Utilisez ce script si vous voulez juste installer les dependances sans demarrer le serveur

echo ========================================
echo    LOTO - Installation des Dependances
echo ========================================
echo.

REM Changer vers le repertoire du script
cd /d "%~dp0"

REM Verifier si Python est installe
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe ou non accessible.
    echo Veuillez installer Python 3.7 ou superieur.
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
python -m pip install --upgrade pip
echo.

REM Installer les dependances
echo [INFO] Installation des dependances depuis requirements.txt...
if exist requirements.txt (
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERREUR] Erreur lors de l'installation des dependances.
        pause
        exit /b 1
    )
    echo.
    echo [SUCCES] Installation terminee avec succes!
    echo.
    echo Pour demarrer l'application, utilisez:
    echo   start_loto.bat
    echo ou
    echo   python app.py
) else (
    echo [ERREUR] Le fichier requirements.txt est introuvable.
    pause
    exit /b 1
)

echo.
pause

