import streamlit as st
import os
import numpy as np
from openai import OpenAI

# ==============================
# CONFIGURAÇÃO DA PÁGINA
# ==============================
st.set_page_config(page_title="Industrial Strategic AI", layout="wide")

st.title("🏭 Industrial Strategic AI")
st.markdown("Motor Proprietário de Inteligência Estratégica")

# ==============================
# API KEY
# ==============================
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY não configurada nas Secrets do Streamlit.")
    st.stop()

client = OpenAI(api_key=api_key)

# ==============================
# CARREGAR BASE DE CONHECIMENTO
# ==============================
def carregar_conhecimento():
    pasta = "knowledge"
    textos = []

    if os.path.exists(pasta):
        for arquivo in os.listdir(pasta):
            if arquivo.endswith(".txt"):
                with open(os.path.join(pasta, arquivo), "r", encoding="utf-8") as f:
                    conteudo = f.read()
                    blocos = conteudo.split("\n\n")
                    textos.extend(blocos)
    else:
        st.warning("Pasta 'knowledge' não encontrada.")

    return textos

documentos = carregar_conhecimento()

# ==============================
# CRIAR EMBEDDINGS
# ==============================
@st.cache_data
def criar_embeddings(textos):
    embeddings = []
    for texto in textos:
        if len(texto.strip()) > 20:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texto
            )
            embeddings.append(response.data[0].embedding)
        else:
            embeddings.append(None)
    return embeddings

embeddings = criar_embeddings(documentos)

# ==============================
# FUNÇÃO DE SIMILARIDADE
# ==============================
def similaridade(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# ==============================
# BUSCA SEMÂNTICA
# ==============================
def buscar_contexto(pergunta):
    pergunta_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=pergunta
    ).data[0].embedding

    similaridades = []

    for emb in embeddings:
        if emb is not None:
            sim = similaridade(pergunta_embedding, emb)
            similaridades.append(sim)
        else:
            similaridades.append(-1)

    top_indices = np.argsort(similaridades)[-3:]

    contexto_relevante = ""
    for i in top_indices:
        contexto_relevante += documentos[i] + "\n\n"

    return contexto_relevante

# ==============================
# INICIALIZAR HISTÓRICO
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================
# INPUT DO USUÁRIO
# ==============================
user_input = st.chat_input("Faça sua pergunta estratégica...")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================
# PROCESSAMENTO
# ==============================
if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    contexto = buscar_contexto(user_input)

    prompt_final = f'''
Você é uma Inteligência Artificial estratégica baseada exclusivamente na metodologia proprietária Industrial Alpha.

OBJETIVO:
Interpretar a necessidade do usuário e aplicar os conceitos existentes na metodologia para gerar direcionamento estratégico prático.

REGRAS OBRIGATÓRIAS:
1. Analise profundamente a intenção da pergunta.
2. Identifique quais conceitos da base melhor se conectam com a necessidade apresentada.
3. Utilize raciocínio estratégico para aplicar esses conceitos.
4. NÃO crie novos pilares, dimensões ou métodos que não estejam explicitamente na base.
5. NÃO utilize teorias externas.
6. Se a pergunta estiver totalmente fora do escopo da metodologia, responda exatamente:
Essa solicitação não está contemplada na metodologia proprietária.

CONTEXTO DISPONÍVEL:
{contexto}

PERGUNTA:
{user_input}

RESPOSTA:
Forneça um direcionamento estratégico aplicado, utilizando exclusivamente os conceitos existentes na metodologia.
'''

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Você é um consultor estratégico industrial de alto nível que aplica exclusivamente a metodologia proprietária Industrial Alpha."
                },
                {
                    "role": "user",
                    "content": prompt_final
                }
            ],
            temperature=0.2
        )

        answer = response.choices[0].message.content

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error("Erro ao conectar com a OpenAI.")
        st.write(e)
