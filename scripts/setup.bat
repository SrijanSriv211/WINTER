@echo off

if not exist .venv (
    echo Creating virtual environment '.venv'
    python -m venv .venv
)

echo Activating virtual environment
call .venv\Scripts\activate

echo Install dependencies from requirements.txt
pip install -r requirements.txt
