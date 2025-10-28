"""
Configurações centrais do projeto
"""
import os
from dotenv import load_dotenv
 
# Carrega variáveis de ambiente
load_dotenv()
 
# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
 
# Verificar se a chave existe
if not GROQ_API_KEY:
    raise ValueError("Por favor, configure GROQ_API_KEY no ficheiro .env")
 
# Modelos disponíveis no Groq (atualizados)
MODELOS_GROQ = {
    # Modelos Llama (Meta)
    "llama-3.3-70b": "llama-3.3-70b-versatile",      # Mais poderoso e versátil
    "llama-3.1-8b": "llama-3.1-8b-instant",          # Rápido e eficiente
    # Modelos OpenAI Open Source
    "gpt-oss-120b": "openai/gpt-oss-120b",     # Mais poderoso e versátil
    "gpt-oss-20b": "openai/gpt-oss-20b",       # Rápido e eficiente
}
 
# Modelo padrão para análise de dados
MODELO_PADRAO = MODELOS_GROQ["llama-3.3-70b"]
 
# Configurações de geração
TEMPERATURA_PADRAO = 0.1  # Baixa para análise de dados (mais determinístico)
MAX_TOKENS_PADRAO = 2048

# Configurações para visualização Plotly usadas pelo dashboard
PLOTLY_CONFIG = {
    # Esconder a barra de ferramentas por defeito (clean UI)
    "displayModeBar": False,
    # Tornar responsivo dentro do container
    "responsive": True,
    # Remover dicas flutuantes
    "showTips": False
}