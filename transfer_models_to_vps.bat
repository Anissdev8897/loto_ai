@echo off
chcp 65001 >nul
echo ========================================
echo TRANSFERT DES MODÈLES VERS LE VPS
echo ========================================
echo.

REM Configuration
set VPS_IP=107.189.17.46
set VPS_USER=root
set VPS_PATH=/inetpub/wwwroot/bottrading_website/loto/resultats_loto/models
set LOCAL_MODELS=resultats_loto\models

REM Vérifier si les modèles existent localement
if not exist "%LOCAL_MODELS%\rf_number_model.joblib" (
    echo [ERREUR] Modèles locaux introuvables
    echo Veuillez d'abord exécuter train_local.bat
    pause
    exit /b 1
)

echo Fichiers à transférer:
dir /b "%LOCAL_MODELS%\*.joblib" 2>nul
echo.

REM Demander confirmation
set /p confirm="Voulez-vous transférer les modèles vers le VPS? (O/N): "
if /i not "%confirm%"=="O" (
    echo Transfert annulé
    pause
    exit /b 0
)

echo.
echo [INFO] Transfert en cours vers %VPS_IP%...
echo [INFO] Chemin VPS: %VPS_PATH%
echo.

REM Utiliser scp si disponible (nécessite OpenSSH ou PuTTY)
where scp >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] SCP n'est pas disponible
    echo Options:
    echo 1. Installer OpenSSH pour Windows
    echo 2. Utiliser WinSCP ou FileZilla manuellement
    echo 3. Utiliser SFTP manuellement
    echo.
    echo Fichiers à transférer manuellement:
    echo - %LOCAL_MODELS%\rf_number_model.joblib
    echo - %LOCAL_MODELS%\scaler_numbers.joblib
    echo - %LOCAL_MODELS%\rf_chance_model.joblib (si existe)
    echo - %LOCAL_MODELS%\scaler_chance.joblib (si existe)
    echo.
    echo Vers: %VPS_USER%@%VPS_IP%:%VPS_PATH%
    pause
    exit /b 1
)

REM Transférer les fichiers
echo Transfert des modèles...
scp "%LOCAL_MODELS%\rf_number_model.joblib" "%VPS_USER%@%VPS_IP%:%VPS_PATH%/"
if errorlevel 1 (
    echo [ERREUR] Échec du transfert de rf_number_model.joblib
    pause
    exit /b 1
)

scp "%LOCAL_MODELS%\scaler_numbers.joblib" "%VPS_USER%@%VPS_IP%:%VPS_PATH%/"
if errorlevel 1 (
    echo [ERREUR] Échec du transfert de scaler_numbers.joblib
    pause
    exit /b 1
)

if exist "%LOCAL_MODELS%\rf_chance_model.joblib" (
    scp "%LOCAL_MODELS%\rf_chance_model.joblib" "%VPS_USER%@%VPS_IP%:%VPS_PATH%/"
)

if exist "%LOCAL_MODELS%\scaler_chance.joblib" (
    scp "%LOCAL_MODELS%\scaler_chance.joblib" "%VPS_USER%@%VPS_IP%:%VPS_PATH%/"
)

echo.
echo ========================================
echo TRANSFERT TERMINÉ AVEC SUCCÈS
echo ========================================
echo Les modèles ont été transférés vers le VPS
echo Vous pouvez maintenant redémarrer l'application sur le VPS
echo.
pause

