from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from config import GEMINI_API_KEY

# Création du modèle LangChain avec Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=2.0
)

# Parser pour extraire le texte de la réponse
output_parser = StrOutputParser()

# ÉTAPE 1 : Chaîne pour générer une blague
print("=" * 60)
print("ÉTAPE 1 : GÉNÉRATION DE LA BLAGUE (avec LCEL)")
print("=" * 60)

blague_prompt = PromptTemplate(
    input_variables=["question"],
    template="Tu es un assistant utile. Réponds de manière CONCISE et DIRECTE.\n\nQuestion: {question}\n\nRéponse:"
)

# Création de la chaîne avec LCEL : prompt | llm | parser
chain_blague = blague_prompt | llm | output_parser

# Exécution de la chaîne
blague = chain_blague.invoke({"question": "Raconte-moi une blague drôle"})
print(f"Blague: {blague}")
print()

# ÉTAPE 2 : Chaîne pour analyser la blague
print("=" * 60)
print("ÉTAPE 2 : ANALYSE DE LA BLAGUE (avec LCEL)")
print("=" * 60)

analyse_prompt = PromptTemplate(
    input_variables=["blague"],
    template="Tu es un assistant utile. Réponds de manière CONCISE et DIRECTE.\n\nBlague: {blague}\n\nQuestion: Pourquoi cette blague est-elle marrante ? Explique-moi.\n\nRéponse:"
)

# Création de la chaîne d'analyse avec LCEL
chain_analyse = analyse_prompt | llm | output_parser

# Exécution de la chaîne
analyse = chain_analyse.invoke({"blague": blague})
print(f"Analyse: {analyse}")
print()

# BONUS 1 : Chaîne avec RunnablePassthrough
print("=" * 60)
print("BONUS 1 : AVEC RunnablePassthrough")
print("=" * 60)

# RunnablePassthrough.assign() permet d'ajouter de nouvelles clés au dictionnaire
# tout en gardant les clés existantes
chain_with_passthrough = (
    RunnablePassthrough.assign(
        blague=(blague_prompt | llm | output_parser)
    )
    | RunnablePassthrough.assign(
        analyse=(analyse_prompt | llm | output_parser)
    )
)

# Exécution - retourne un dict avec question, blague ET analyse
resultat_passthrough = chain_with_passthrough.invoke({"question": "Raconte-moi une blague drôle"})
print(f"Question: {resultat_passthrough['question']}")
print(f"Blague: {resultat_passthrough['blague']}")
print(f"Analyse: {resultat_passthrough['analyse']}")
print()

# BONUS 2 : Chaîne avec itemgetter
print("=" * 60)
print("BONUS 2 : AVEC itemgetter")
print("=" * 60)

# itemgetter extrait une valeur spécifique d'un dictionnaire
# Parfait pour passer d'une étape à l'autre dans une chaîne
chain_with_itemgetter = (
    {"question": itemgetter("question")}  # Prépare l'input
    | blague_prompt 
    | llm 
    | output_parser
    | {"blague": RunnablePassthrough()}  # Transforme la string en dict {"blague": ...}
    | analyse_prompt
    | llm
    | output_parser
)

# Exécution
resultat_itemgetter = chain_with_itemgetter.invoke({"question": "Raconte-moi une blague drôle"})
print(f"Analyse (via itemgetter): {resultat_itemgetter}")
print()

# BONUS 3 : Chaîne complète élégante avec itemgetter + RunnablePassthrough
print("=" * 60)
print("BONUS 3 : COMBINAISON ÉLÉGANTE")
print("=" * 60)

# Chaîne qui garde toutes les données intermédiaires
chain_elegante = (
    {
        "question": itemgetter("question"),
        "blague": blague_prompt | llm | output_parser
    }
    | RunnablePassthrough.assign(
        analyse=(itemgetter("blague") | analyse_prompt | llm | output_parser)
    )
)

# Exécution - retourne tout : question, blague ET analyse
resultat_elegant = chain_elegante.invoke({"question": "Raconte-moi une blague drôle"})
print(f"Question originale: {resultat_elegant['question']}")
print(f"Blague générée: {resultat_elegant['blague']}")
print(f"Analyse: {resultat_elegant['analyse']}")
print("=" * 60)
