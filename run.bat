@echo off
echo Iniciando FastAPI...
start cmd /k uvicorn main.api:api --reload --port 8000

timeout /t 3

echo Iniciando Streamlit...
start cmd /k streamlit run main/app.py
