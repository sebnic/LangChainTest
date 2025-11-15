import replicate
import os
import requests
from config import set_environment
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from typing import Dict

# Charger les variables d'environnement depuis config.py
set_environment()

def generate_image_with_replicate(prompt: str, output_path: str = "generated_image.png"):
    """
    Génère une image avec Replicate (Stable Diffusion) et gère les erreurs.
    
    Args:
        prompt: Description de l'image à générer
        output_path: Chemin de sauvegarde de l'image
    """
    try:
        print("=" * 60)
        print("GÉNÉRATION D'IMAGE AVEC REPLICATE (Flux 1.1 Pro)")
        print("=" * 60)
        print(f"\nPrompt: {prompt}\n")
        
        # Vérifier que le token API est configuré
        api_token = os.getenv("REPLICATE_API_TOKEN")
        if not api_token or api_token == "your_replicate_token_here":
            raise ValueError("REPLICATE_API_TOKEN n'est pas configuré dans le fichier .env")
        
        print("⏳ Génération en cours...")
        
        # Utilisation du modèle Flux 1.1 Pro via Replicate
        # Modèle: Black Forest Labs Flux 1.1 Pro - génération d'images de pointe
        output = replicate.run(
            "black-forest-labs/flux-1.1-pro",
            input={
                "prompt": prompt,
                "width": 1024,
                "height": 1024,
                "output_format": "png",
                "output_quality": 100,
                "safety_tolerance": 2,
                "prompt_upsampling": True
            }
        )
        
        # Le résultat peut être une URL ou un objet FileOutput
        if output:
            # Gestion du FileOutput ou d'une URL directe
            if hasattr(output, 'url'):
                image_url = output.url
            elif isinstance(output, str):
                image_url = output
            elif isinstance(output, list) and len(output) > 0:
                image_url = output[0] if isinstance(output[0], str) else output[0].url
            else:
                # Pour les objets FileOutput, on peut lire directement
                image_url = str(output)
            
            print(f"✅ Image générée : {image_url}")
            
            # Télécharger et sauvegarder l'image
            print(f"⏳ Téléchargement de l'image vers {output_path}...")
            response = requests.get(image_url)
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Image sauvegardée avec succès : {output_path}")
                print("=" * 60)
                return True
            else:
                print(f"❌ Erreur lors du téléchargement: {response.status_code}")
                return False
        else:
            print("❌ Aucune image n'a été générée.")
            return False
            
    except ValueError as e:
        print(f"❌ Erreur de configuration: {str(e)}")
        print("\n💡 Veuillez ajouter votre token Replicate dans le fichier .env:")
        print("   REPLICATE_API_TOKEN=r8_votre_token_ici")
        return False
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération: {type(e).__name__}")
        print(f"   Détails: {str(e)}")
        print("\n💡 Vérifiez:")
        print("   1. Que votre token Replicate est valide")
        print("   2. Que vous avez une connexion internet")
        print("   3. Que vous avez des crédits sur votre compte Replicate")
        return False


def call_replicate_api(input_dict: Dict) -> str:
    """
    Appelle l'API Replicate pour générer une image.
    Retourne l'URL de l'image générée.
    """
    prompt = input_dict["prompt"]
    
    # Vérifier que le token API est configuré
    api_token = os.getenv("REPLICATE_API_TOKEN")
    if not api_token or api_token == "your_replicate_token_here":
        raise ValueError("REPLICATE_API_TOKEN n'est pas configuré dans le fichier .env")
    
    print(f"⏳ Génération de l'image avec Flux 1.1 Pro...")
    
    # Appel à Replicate
    output = replicate.run(
        "black-forest-labs/flux-1.1-pro",
        input={
            "prompt": prompt,
            "width": 1024,
            "height": 1024,
            "output_format": "png",
            "output_quality": 100,
            "safety_tolerance": 2,
            "prompt_upsampling": True
        }
    )
    
    # Extraire l'URL
    if hasattr(output, 'url'):
        image_url = output.url
    elif isinstance(output, str):
        image_url = output
    elif isinstance(output, list) and len(output) > 0:
        image_url = output[0] if isinstance(output[0], str) else output[0].url
    else:
        image_url = str(output)
    
    return image_url


def download_image(input_dict: Dict) -> Dict:
    """
    Télécharge l'image depuis l'URL et la sauvegarde.
    """
    image_url = input_dict["image_url"]
    output_path = input_dict["output_path"]
    
    print(f"⏳ Téléchargement de l'image vers {output_path}...")
    
    response = requests.get(image_url)
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Image sauvegardée avec succès : {output_path}")
        return {**input_dict, "success": True}
    else:
        print(f"❌ Erreur lors du téléchargement: {response.status_code}")
        return {**input_dict, "success": False}


def create_image_generation_chain(output_path: str = "generated_image.png"):
    """
    Crée une chaîne LangChain pour la génération d'images avec LCEL.
    
    Args:
        output_path: Chemin de sauvegarde de l'image
    
    Returns:
        Une chaîne LangChain exécutable
    """
    # Étape 1: Appel à l'API Replicate
    generate_step = RunnableLambda(call_replicate_api)
    
    # Étape 2: Préparation pour le téléchargement
    prepare_download = RunnableLambda(
        lambda image_url: {
            "image_url": image_url,
            "output_path": output_path
        }
    )
    
    # Étape 3: Téléchargement de l'image
    download_step = RunnableLambda(download_image)
    
    # Création de la chaîne LCEL complète
    chain = generate_step | prepare_download | download_step
    
    return chain


if __name__ == "__main__":
    print("=" * 60)
    print("GÉNÉRATION D'IMAGE AVEC LANGCHAIN + REPLICATE")
    print("=" * 60)
    print()
    
    # Description de l'image souhaitée
    prompt_description = "A surreal street with upside-down houses, inverted architecture, houses flipped upside down, detailed artistic style, photorealistic"
    
    print(f"Prompt: {prompt_description}\n")
    
    try:
        # Création de la chaîne LangChain avec LCEL
        chain = create_image_generation_chain(output_path="rue_maisons_inversees.png")
        
        # Exécution de la chaîne
        result = chain.invoke({"prompt": prompt_description})
        
        if result.get("success"):
            print("\n" + "=" * 60)
            print("✅ Génération terminée avec succès !")
            print("=" * 60)
        else:
            print("\n⚠️  La génération d'image a échoué.")
            
    except Exception as e:
        print(f"\n❌ Erreur: {type(e).__name__}")
        print(f"   Détails: {str(e)}")
