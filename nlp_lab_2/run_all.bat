@echo off
chcp 65001
title Vectorization Project - Complete Setup

echo ====================================================
echo    Полная установка и запуск анализатора векторных пространств
echo ====================================================
echo.

:check_python
echo Проверка установки Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python не установлен или не добавлен в PATH
    echo Установите Python с официального сайта: https://python.org
    pause
    exit /b 1
)
echo ✅ Python обнаружен

:install_dependencies
echo.
echo 📦 Установка зависимостей...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Ошибка установки зависимостей
    pause
    exit /b 1
)
echo ✅ Зависимости установлены

:check_data
echo.
echo 📁 Проверка наличия данных...
if not exist "data\rbc_articles_words.jsonl" (
    echo ❌ Файл с корпусом не найден: data\rbc_articles_words.jsonl
    echo Поместите ваш корпус в указанную папку
    pause
    exit /b 1
)
echo ✅ Данные обнаружены

:run_experiments
echo.
echo 🚀 Запуск экспериментов по векторизации...
echo Это может занять несколько минут...
python run_experiments.py
if %errorlevel% neq 0 (
    echo ❌ Ошибка выполнения экспериментов
    pause
    exit /b 1
)

:launch_web
echo.
echo 🌐 Запуск веб-интерфейса...
echo Веб-приложение будет доступно по адресу: http://localhost:8501
echo.
echo Для остановки нажмите Ctrl+C в окне браузера
echo.
timeout /t 3 /nobreak >nul
python -m streamlit run run_web_app.py

echo.
echo ✅ Все компоненты успешно запущены!
echo.
echo 📊 Результаты экспериментов сохранены в:
echo    - vectorization_metrics.csv
echo    - models_comparison.csv
echo    - папка models/ с обученными моделями
echo.
echo 🌐 Веб-интерфейс запущен в браузере
echo.
pause