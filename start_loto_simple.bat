@echo off
REM Script batch simplifie pour demarrer rapidement l'application Loto
REM Version simple sans verification approfondie

echo ========================================
echo    LOTO - Demarrage Rapide
echo ========================================
echo.

REM Changer vers le repertoire du script
cd /d "%~dp0"

REM Activer l'environnement virtuel si il existe
if exist "env\Scripts\activate.bat" (
    call env\Scripts\activate.bat
)

REM Installer/Mettre a jour les dependances rapidement
echo [INFO] Installation des dependances...
pip install -r requirements.txt --quiet

echo.
echo [INFO] Demarrage du serveur sur http://localhost:5000
echo.

REM Demarrer l'application
python app.py

pause

