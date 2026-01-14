@echo off
chcp 65001 >nul
echo ========================================
echo ENTRAÎNEMENT LOCAL DES MODÈLES LOTO
echo ========================================
echo.

REM Vérifier si Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installé ou pas dans le PATH
    pause
    exit /b 1
)

REM Vérifier si l'environnement virtuel existe
if not exist "env\Scripts\activate.bat" (
    echo [ERREUR] Environnement virtuel introuvable
    echo Veuillez d'abord exécuter install.bat
    pause
    exit /b 1
)

REM Activer l'environnement virtuel
echo Activation de l'environnement virtuel...
call env\Scripts\activate.bat
if errorlevel 1 (
    echo [ERREUR] Impossible d'activer l'environnement virtuel
    pause
    exit /b 1
)

REM Vérifier si le fichier CSV existe
if not exist "tirages_loto.csv" (
    echo [ERREUR] Fichier tirages_loto.csv introuvable
    echo Veuillez d'abord télécharger les données
    pause
    exit /b 1
)

echo.
echo Configuration de l'entraînement:
echo - Fichier CSV: tirages_loto.csv
echo - Répertoire modèles: resultats_loto\models
echo - Estimateurs: 200
echo - Itérations: 50
echo - Validation croisée: 5 folds
echo.
echo [INFO] L'entraînement peut prendre plusieurs minutes...
echo [INFO] Les logs sont enregistrés dans train_local.log
echo.

REM Lancer l'entraînement
python train_local.py --csv tirages_loto.csv --n-estimators 200 --n-iter 50 --cv 5

if errorlevel 1 (
    echo.
    echo [ERREUR] L'entraînement a échoué
    echo Consultez train_local.log pour plus de détails
    pause
    exit /b 1
)

echo.
echo ========================================
echo ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS
echo ========================================
echo Les modèles sont sauvegardés dans: resultats_loto\models
echo Vous pouvez maintenant les transférer vers le VPS
echo.
pause

