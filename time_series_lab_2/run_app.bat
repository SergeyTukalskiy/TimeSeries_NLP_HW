@echo off
chcp 65001 >nul

echo 📈 Запуск анализа временных рядов MOEX...
echo.

echo 📥 Установка необходимых библиотек...
python -m pip install -r requirements.txt

echo.
echo 🌐 Запуск веб-интерфейса...
echo 📍 Откройте http://localhost:8501 в браузере
python -m streamlit run app/streamlit_app.py

pause