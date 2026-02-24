import streamlit as st
import os
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# CONFIGURAÇÃO
# ==============================
st.set_page_config(page_title="Industrial Strategic AI", layout="wide")
st.title("🏭 Industrial Strategic AI")
st.markdown("Motor Proprietário de Inteligência Industrial")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY não configurada.")
    st.stop()

client = OpenAI(api_key=api_key)

# ==============================
# CARREGAR BASE
# ==============================
def carregar_conhecimento():
    pasta = "knowledge"
    textos = []

    if os.path.exists(pasta):
        for arquivo in os.listdir(pasta):
            if arquivo.endswith(".txt"):
                with open(os.path.join(pasta, arquivo), "r", encoding="utf-8") as f:
                    conteudo = f.read()

                    # Quebra em blocos menores
                    blocos = conteudo.split("\n\n")
                    textos.extend(blocos)

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
            sim = cosine_similarity(
                [pergunta_embedding],
                [emb]
            )[0][0]
            similaridades.append(sim)
        else:
            similaridades.append(-1)

    top_indices = np.argsort(similaridades)[-3:]

    contexto_relevante = ""
    for i in top_indices:
        contexto_relevante += documentos[i] + "\n\n"

    return contexto_relevante

# ==============================
# CHAT
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Faça sua pergunta estratégica...")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    contexto = buscar_contexto(user_input)

    prompt_final = f"""
Você é uma Inteligência Artificial estratégica baseada exclusivamente na metodologia proprietária Industrial Alpha.

OBJETIVO:
Interpretar a necessidade do usuário e aplicar os conceitos existentes na metodologia para gerar direcionamento estratégico prático.

REGRAS OBRIGATÓRIAS:

1. Analise a intenção real da pergunta, mesmo que o usuário não utilize os termos exatos da metodologia.
2. Identifique quais conceitos da base de conhecimento melhor se conectam com a necessidade apresentada.
3. Utilize raciocínio estratégico para aplicar esses conceitos.
4. NÃO crie novos pilares, dimensões, métodos ou estruturas que não estejam explicitamente presentes na base.
5. NÃO complemente com teorias externas.
6. Se a pergunta estiver totalmente fora do escopo da metodologia, responda exatamente:
   "Essa solicitação não está contemplada na metodologia proprietária."

CONTEXTO RELEVANTE DA BASE:
{contexto}

PERGUNTA DO USUÁRIO:
{user_input}

RESPOSTA:
Apresente um direcionamento estratégico aplicado, utilizando exclusivamente os conceitos existentes na metodologia.
"""
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Consultor industrial estratégico proprietário."},
            {"role": "user", "content": prompt_final}
        ],
        temperature=0.1
    )

    answer = response.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
