import os
from dotenv import load_dotenv

# Charger les variables depuis le fichier .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

def set_environment():
    variable_dict = globals().items();
    for var_name, var_value in variable_dict:
        if var_name.isupper():
            os.environ[var_name] = str(var_value)