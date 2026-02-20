import streamlit as st
import os
from openai import OpenAI

# ======================================
# CONFIGURAÇÃO DA PÁGINA
# ======================================
st.set_page_config(
    page_title="Industrial AI Assistant",
    layout="wide"
)

st.title("🏭 Industrial AI Assistant")
st.markdown("Inteligência Estratégica Proprietária para Indústria e Gestão")

# ======================================
# CAPTURA SEGURA DA API KEY
# ======================================
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY não configurada. Configure nas Secrets do Streamlit.")
    st.stop()

client = OpenAI(api_key=api_key)

# ======================================
# CARREGAR BASE DE CONHECIMENTO FIXA
# ======================================
def carregar_conhecimento():
    base_texto = ""
    pasta = "knowledge"

    if os.path.exists(pasta):
        for arquivo in os.listdir(pasta):
            caminho = os.path.join(pasta, arquivo)

            if arquivo.endswith(".txt"):
                try:
                    with open(caminho, "r", encoding="utf-8") as f:
                        base_texto += f.read() + "\n\n"
                except Exception as e:
                    st.warning(f"Erro ao ler {arquivo}: {e}")
    else:
        st.warning("⚠️ Pasta 'knowledge' não encontrada no repositório.")

    return base_texto

BASE_CONHECIMENTO = carregar_conhecimento()

# ======================================
# HISTÓRICO DE CONVERSA
# ======================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ======================================
# INPUT DO USUÁRIO
# ======================================
user_input = st.chat_input("Faça sua pergunta estratégica...")

# Mostrar histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ======================================
# PROCESSAMENTO DA IA
# ======================================
if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # ======================================
    # PROMPT ESTRATÉGICO PROFISSIONAL
    # ======================================
    contexto = f"""
Você é a Inteligência Artificial proprietária de uma metodologia estratégica industrial.

Sua função é atuar como:

- Consultor Industrial
- Mentor Empresarial
- Especialista em Otimização Financeira
- Estrategista de Crescimento

Baseie suas respostas PRIORITARIAMENTE na base de conhecimento abaixo.
Caso o tema não esteja explicitamente descrito, complemente com boas práticas reais de mercado.

========================
BASE DE CONHECIMENTO:
========================

{BASE_CONHECIMENTO}

========================
PERGUNTA DO USUÁRIO:
========================

{user_input}

Responda de forma:
- Estratégica
- Clara
- Aplicável
- Direta para tomada de decisão
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Você é um consultor estratégico industrial de alto nível."
                },
                {
                    "role": "user",
                    "content": contexto
                }
            ],
            temperature=0.3
        )

        answer = response.choices[0].message.content

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error("Erro ao conectar com a OpenAI.")
        st.write(e)
