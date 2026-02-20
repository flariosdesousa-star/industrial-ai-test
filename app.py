import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Industrial AI Assistant", layout="wide")

st.title("🏭 Industrial AI Assistant")
st.markdown("Gestão de Processos | Otimização Financeira | Estratégia Industrial")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Digite sua pergunta industrial...")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state.messages
        )
        answer = response.choices[0].message.content
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
