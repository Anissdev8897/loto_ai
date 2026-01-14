@echo off
REM Script pour corriger les problèmes de dépendances PyArrow et scikit-learn
REM Solution pour l'erreur: DLL load failed while importing lib

echo ========================================
echo    Correction des dependances Python
echo ========================================
echo.

REM Changer vers le repertoire du script
cd /d "%~dp0"

REM Activer l'environnement virtuel si il existe
if exist "env\Scripts\activate.bat" (
    echo [INFO] Activation de l'environnement virtuel...
    call env\Scripts\activate.bat
)

echo.
echo [INFO] Désinstallation de PyArrow...
pip uninstall pyarrow -y

echo.
echo [INFO] Désinstallation de scikit-learn...
pip uninstall scikit-learn -y

echo.
echo [INFO] Installation de PyArrow (version compatible)...
pip install pyarrow

echo.
echo [INFO] Installation de scikit-learn (dernière version)...
pip install --upgrade scikit-learn

echo.
echo [INFO] Vérification de l'installation...
python -c "import sklearn; print('scikit-learn OK')"
python -c "import pyarrow; print('pyarrow OK')"

echo.
echo [SUCCES] Dépendances corrigées!
echo.
pause

