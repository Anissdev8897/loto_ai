@echo off
REM Script pour corriger definitivement le probleme PyArrow/scikit-learn
REM Desinstalle PyArrow et reinstalle scikit-learn proprement

echo ========================================
echo    Correction du probleme PyArrow
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
echo [INFO] Desinstallation de PyArrow...
pip uninstall pyarrow -y

echo.
echo [INFO] Desinstallation de scikit-learn...
pip uninstall scikit-learn -y

echo.
echo [INFO] Installation de scikit-learn sans PyArrow...
pip install scikit-learn --no-deps

echo.
echo [INFO] Installation des dependances de scikit-learn...
pip install numpy scipy joblib threadpoolctl

echo.
echo [INFO] Verification de l'installation...
python -c "from sklearn.ensemble import RandomForestClassifier; print('scikit-learn OK')"
if errorlevel 1 (
    echo [ERREUR] scikit-learn ne fonctionne toujours pas.
    echo [INFO] Essayez de reinstaller Python ou utilisez conda.
) else (
    echo [SUCCES] scikit-learn fonctionne correctement sans PyArrow!
)

echo.
pause

