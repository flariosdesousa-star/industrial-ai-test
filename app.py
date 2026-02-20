import streamlit as st
import os
from openai import OpenAI
from pypdf import PdfReader

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Industrial AI Assistant", layout="wide")
st.title("🏭 Industrial AI Assistant")
st.markdown("Gestão de Processos | Estratégia | Inteligência Empresarial")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("Configure OPENAI_API_KEY nas Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

# ==============================
# BASE DE CONHECIMENTO
# ==============================
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = ""

uploaded_file = st.file_uploader("📂 Upload de material (PDF ou TXT)", type=["pdf", "txt"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        pdf = PdfReader(uploaded_file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        st.session_state.knowledge_base += text
    else:
        text = uploaded_file.read().decode("utf-8")
        st.session_state.knowledge_base += text

    st.success("Material adicionado à base de conhecimento.")

# ==============================
# CHAT
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Pergunte algo sobre indústria ou estratégia...")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input:

    # Monta contexto com material enviado
    contexto = f"""
    Use o seguinte material como base para responder:

    {st.session_state.knowledge_base}

    Pergunta do usuário:
    {user_input}
    """

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é especialista em gestão industrial e mentoria empresarial."},
            {"role": "user", "content": contexto}
        ]
    )

    answer = response.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
