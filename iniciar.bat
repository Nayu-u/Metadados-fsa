@echo off
title Kingambit 

echo [INFO] Instalando dependencias a partir do requirements.txt...
python -m pip install -r requirements.txt

echo.
echo [INFO] Iniciando o servidor...
python server.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERRO] Ocorreu um problema ao executar o servidor.
    echo Verifique se o Python esta instalado corretamente e se o arquivo 'server.py' existe no diretorio.
    pause
)
