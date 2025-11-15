from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Any, Dict
import sys


def invoke_llm_chain_with_error_handling(chain: Any, input_data: Dict[str, str], context: str = "LLM") -> str:
    """
    Méthode centralisée pour gérer les erreurs lors des appels à un LLM.
    
    Args:
        chain: La chaîne LangChain à invoquer
        input_data: Les données d'entrée pour la chaîne
        context: Contexte de l'appel (pour le message d'erreur)
    
    Returns:
        str: Le résultat de la chaîne ou un message d'erreur
    """
    try:
        resultat = chain.invoke(input_data)
        return resultat
    
    except ConnectionError as e:
        error_msg = f"❌ Erreur de connexion à {context}: Impossible de se connecter au serveur Ollama."
        print(error_msg)
        print(f"   Détails: {str(e)}")
        print("   💡 Vérifiez qu'Ollama est bien lancé avec: ollama serve")
        return None
    
    except TimeoutError as e:
        error_msg = f"❌ Timeout lors de l'appel à {context}: Le serveur met trop de temps à répondre."
        print(error_msg)
        print(f"   Détails: {str(e)}")
        return None
    
    except ValueError as e:
        error_msg = f"❌ Erreur de valeur dans {context}: Données d'entrée invalides."
        print(error_msg)
        print(f"   Détails: {str(e)}")
        return None
    
    except Exception as e:
        error_msg = f"❌ Erreur inattendue lors de l'appel à {context}."
        print(error_msg)
        print(f"   Type d'erreur: {type(e).__name__}")
        print(f"   Détails: {str(e)}")
        return None


# Création du modèle Ollama
llm = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0.7
)

# Parser pour extraire le texte de la réponse
output_parser = StrOutputParser()

# PromptTemplate avec la contrainte d'expliquer en termes simples
prompt_template = PromptTemplate(
    input_variables=["concept"],
    template="""Tu es un excellent vulgarisateur scientifique. 
Ta mission est d'expliquer des concepts complexes de manière SIMPLE et ACCESSIBLE à tous.

Utilise des analogies de la vie quotidienne, évite le jargon technique, et reste clair.

Concept à expliquer : {concept}

Explication simple :"""
)

# Création de la chaîne LCEL : prompt | llm | parser
chain = prompt_template | llm | output_parser

# Exécution de la chaîne avec gestion d'erreurs
print("=" * 60)
print("EXPLICATION EN TERMES SIMPLES")
print("=" * 60)
print()

resultat = invoke_llm_chain_with_error_handling(
    chain=chain,
    input_data={"concept": "l'ordinateur quantique"},
    context="Ollama (deepseek-r1:1.5b)"
)

if resultat:
    print(resultat)
    print()
    print("=" * 60)
else:
    print("\n⚠️  L'exécution a échoué. Veuillez corriger les erreurs ci-dessus.")
    sys.exit(1)
