@echo off
chcp 1251 >nul
setlocal enabledelayedexpansion

:: Строго переходим в папку, где лежит сам .bat файл
cd /d "%~dp0"

:: ИЗОЛЯЦИЯ UV
set "UV_CACHE_DIR=%~dp0.uv_cache"
set "UV_PYTHON_INSTALL_DIR=%~dp0.uv_python"

:: ==============================================================
:: БЛОК 1: ПЕРВЫЙ ЗАПУСК (СКАЧИВАНИЕ И УСТАНОВКА)
:: ==============================================================
if not exist ".venv\Scripts\python.exe" (
    echo ==========================================
    echo ПЕРВЫЙ ЗАПУСК: Инициализация окружения...
    echo ==========================================

    if not exist ".bin\uv.exe" (
        echo [1/5] Скачивание портативного менеджера uv...
        mkdir .bin 2>nul
        powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile '.bin\uv.zip'"
        powershell -NoProfile -Command "Expand-Archive -Path '.bin\uv.zip' -DestinationPath '.bin' -Force"
        move /y ".bin\uv-x86_64-pc-windows-msvc\uv.exe" ".bin\uv.exe" >nul
        rmdir /s /q ".bin\uv-x86_64-pc-windows-msvc"
        del /q ".bin\uv.zip"
    )

    echo [2/5] Скачивание портативного Python 3.13...
    ".bin\uv.exe" venv --python 3.13 .venv

    echo [3/5] Скачивание и установка нейросетевых библиотек...
    ".bin\uv.exe" pip install -r requirements.txt --index-strategy unsafe-best-match
    copy /y requirements.txt ".venv\requirements.installed" >nul

    echo[4/5] Очистка временных файлов для экономии места...
    ".bin\uv.exe" cache clean

    echo[5/5] Подготовка конфигурации...
    if not exist ".env" (
        echo # Конфигурация AI Media Organizer Pro > .env
        echo # HF_TOKEN=hf_ВАШ_ТОКЕН_ЗДЕСЬ >> .env
        echo PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True >> .env
        echo. >> .env
        echo # Если SageAttention требует CUDA Toolkit 13, укажите путь к папке ^(без кавычек^) >> .env
        echo # Например: CUSTOM_CUDA_PATH=C:\cuda_13.0 >> .env
        echo # CUSTOM_CUDA_PATH= >> .env
        echo.
        echo [ВНИМАНИЕ] Создан файл .env.
        echo Пожалуйста, впишите в него свой HF_TOKEN и запустите скрипт заново!
        notepad .env
        exit /b 0
    )
    echo ==========================================
    echo Установка успешно завершена!
    echo ==========================================
)

:: ==============================================================
:: БЛОК 2: ОБЫЧНЫЙ ЗАПУСК ПРИЛОЖЕНИЯ
:: ==============================================================

:: Проверка изменений в requirements.txt
if exist "requirements.txt" (
    if exist ".venv\requirements.installed" (
        fc requirements.txt ".venv\requirements.installed" >nul 2>nul
        if errorlevel 1 (
            echo [INFO] Обнаружены изменения в requirements.txt. Обновление зависимостей...
            ".bin\uv.exe" pip install -r requirements.txt --index-strategy unsafe-best-match
            copy /y requirements.txt ".venv\requirements.installed" >nul
            ".bin\uv.exe" cache clean
        )
    ) else (
        echo [INFO] Восстановление списка зависимостей...
        ".bin\uv.exe" pip install -r requirements.txt --index-strategy unsafe-best-match
        copy /y requirements.txt ".venv\requirements.installed" >nul
    )
)

echo Запуск AI Media Organizer Pro...

:: Подгружаем переменные из .env файла
if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do (
        if not "%%b"=="" set "%%a=%%b"
    )
)

:: ПОДМЕНА ПУТЕЙ CUDA (Если пользователь указал кастомный путь)
if defined CUSTOM_CUDA_PATH (
    echo [INFO] Найден кастомный путь CUDA: !CUSTOM_CUDA_PATH!
    set "CUDA_HOME=!CUSTOM_CUDA_PATH!"
    set "CUDA_PATH=!CUSTOM_CUDA_PATH!"
    :: Добавляем bin в PATH, чтобы система могла найти nvcc.exe и .dll файлы (например cublas.dll)
    set "PATH=!CUSTOM_CUDA_PATH!\bin;!PATH!"
)

:: Запуск Python напрямую
".venv\Scripts\python.exe" media_mind_ai.py %*

pause