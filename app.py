import streamlit as st
import os
from openai import OpenAI

# ==============================
# CONFIGURAÇÃO DA PÁGINA
# ==============================
st.set_page_config(
    page_title="Industrial AI Assistant",
    layout="wide"
)

st.title("🏭 Industrial AI Assistant")
st.markdown("Gestão de Processos | Otimização Financeira | Estratégia Industrial")

# ==============================
# CAPTURA SEGURA DA API KEY
# ==============================
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY não configurada. Configure nas Secrets do Streamlit.")
    st.stop()

client = OpenAI(api_key=api_key)

# ==============================
# CONTROLE DE CONVERSA
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "Você é um especialista em indústria, gestão de processos e otimização financeira empresarial."
        }
    ]

# ==============================
# INPUT DO USUÁRIO
# ==============================
user_input = st.chat_input("Digite sua pergunta industrial...")

# Mostrar histórico
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================
# PROCESSAMENTO DA IA
# ==============================
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state.messages
        )

        answer = response.choices[0].message.content

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error("Erro ao conectar com a OpenAI.")
        st.write(e)
