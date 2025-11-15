import google.generativeai as genai
from config import GEMINI_API_KEY

# Configuration de l'API Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Lister tous les modèles disponibles
print("Modèles disponibles :")
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"- {model.name}")
