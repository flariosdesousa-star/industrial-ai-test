import streamlit as st
import os
import numpy as np
from openai import OpenAI
import requests

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
# BUSCA SEMÂNTICA (RAG)
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
# GERAR VÍDEO NO HEYGEN
# ==============================
def gerar_video_heygen(texto):

    heygen_api_key = os.getenv("HEYGEN_API_KEY")

    if not heygen_api_key:
        return None, "HEYGEN_API_KEY não configurada."

    url = "https://api.heygen.com/v2/video/generate"

    headers = {
        "X-Api-Key": heygen_api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "video_inputs": [
            {
                "character": {
                    "type": "avatar",
                    "avatar_id": "Bryan_public"
                },
                "voice": {
                    "type": "text",
                    "input_text": texto,
                    "voice_id": "en-US-GuyNeural"
                }
            }
        ],
        "test": False
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        return None, f"Erro HeyGen: {response.text}"

    data = response.json()

    if "video_url" in data.get("data", {}):
        return data["data"]["video_url"], None

    if "video_id" in data.get("data", {}):
        video_id = data["data"]["video_id"]
        return None, f"Vídeo em processamento. ID: {video_id}"

    return None, "Resposta inesperada da API HeyGen."


# ==============================
# HISTÓRICO
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================
# INPUT
# ==============================
gerar_video = st.toggle("🎬 Gerar vídeo de mentoria executiva")

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

    # ==============================
    # PROMPT COMPLETO ATUALIZADO
    # ==============================
    if gerar_video:
        prompt_final = f"""
Você é a Inteligência Estratégica Oficial da metodologia proprietária Industrial Alpha.

MODO ATIVO: ROTEIRO DE VÍDEO DE MENTORIA EXECUTIVA.

MISSÃO CENTRAL:
Interpretar profundamente a intenção estratégica implícita na pergunta.
Identificar o problema estrutural.
Conectar PROBLEMA → CONCEITO → APLICAÇÃO PRÁTICA.

REGRAS ABSOLUTAS:
1. Use exclusivamente o conteúdo do CONTEXTO.
2. Não crie novos métodos ou frameworks.
3. Não utilize teorias externas.
4. Se não houver aderência clara, responda exatamente:
Essa solicitação não está contemplada na metodologia proprietária.

ESTRUTURA OBRIGATÓRIA:

🎯 Título Estratégico  
🔥 Abertura Executiva  
🧠 Diagnóstico Estratégico  
🏭 Aplicação Empresarial  
📈 Plano de Ação  
🚀 Encerramento Executivo  

CONTEXTO:
{contexto}

PERGUNTA:
{user_input}
"""
    else:
        prompt_final = f"""
Você é a Inteligência Estratégica Oficial da metodologia proprietária Industrial Alpha.

MISSÃO:
Interpretar profundamente a intenção do usuário.
Conectar problema → conceito → aplicação prática.

REGRAS ABSOLUTAS:
1. Use exclusivamente o conteúdo do CONTEXTO.
2. Não invente métodos.
3. Não use teorias externas.
4. Se não houver aderência clara, responda exatamente:
Essa solicitação não está contemplada na metodologia proprietária.

FORMATO DA RESPOSTA:

🧠 Diagnóstico Estratégico  
🏭 Conexão com a Metodologia  
📈 Aplicação Prática  
🚀 Direcionamento Executivo  

CONTEXTO:
{contexto}

PERGUNTA:
{user_input}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
Você é um consultor estratégico industrial sênior.
Aplique exclusivamente a metodologia Industrial Alpha.
Nunca invente conceitos externos.
Sempre conecte problema → conceito → aplicação prática.
"""
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

        # ==============================
        # GERAÇÃO DE VÍDEO
        # ==============================
        if gerar_video:
            with st.spinner("🎬 Gerando avatar executivo..."):
                video_url, erro = gerar_video_heygen(answer)

            if erro:
                st.warning(erro)
            elif video_url:
                st.markdown("### 🎥 Vídeo Gerado")
                st.video(video_url)

    except Exception as e:
        st.error("Erro ao conectar com a OpenAI.")
        st.write(e)
