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
# HISTÓRICO
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================
# INPUT
# ==============================
gerar_video = st.toggle("🎬 Gerar roteiro de vídeo de mentoria")

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

    if gerar_video:
        prompt_final = f"""
Você é uma Inteligência Artificial estratégica baseada exclusivamente na metodologia proprietária Industrial Alpha.

MODO ATIVO: GERAÇÃO DE ROTEIRO DE VÍDEO DE MENTORIA.

MISSÃO:
Interpretar profundamente a necessidade real do usuário, mesmo que ele não utilize os termos exatos da metodologia.
Você deve identificar a intenção estratégica implícita e conectar com os conceitos mais aderentes da base de conhecimento.

REGRAS ABSOLUTAS:
1. Utilize exclusivamente o conteúdo presente na base fornecida.
2. Nunca invente novos métodos, pilares ou teorias.
3. Se não houver aderência clara ao conteúdo, responda exatamente:
Essa solicitação não está contemplada na metodologia proprietária.

ESTRUTURA DO VÍDEO:
- 🎯 Título estratégico
- 🔥 Abertura com gancho executivo
- 🧠 Diagnóstico estratégico
- 🏭 Aplicação prática empresarial
- 📈 Plano de ação estruturado
- 🚀 Encerramento com direcionamento executivo

CONTEXTO DA METODOLOGIA:
{contexto}

PERGUNTA DO USUÁRIO:
{user_input}
"""
    else:
        prompt_final = f"""
Você é uma Inteligência Artificial estratégica baseada exclusivamente na metodologia proprietária Industrial Alpha.

MISSÃO:
Interpretar profundamente a intenção do usuário, mesmo que ele não utilize os termos exatos da metodologia.
Você deve entender o problema real e conectar com os conceitos mais aderentes da base de conhecimento.

REGRAS ABSOLUTAS:
1. Use exclusivamente os conceitos presentes no CONTEXTO.
2. Não crie novos frameworks.
3. Não utilize teorias externas.
4. Se a pergunta não estiver contemplada na metodologia, responda exatamente:
Essa solicitação não está contemplada na metodologia proprietária.

CONTEXTO DA METODOLOGIA:
{contexto}

PERGUNTA DO USUÁRIO:
{user_input}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
Você é um consultor estratégico industrial de alto nível.
Aplica exclusivamente a metodologia Industrial Alpha.
Interprete intenção implícita.
Conecte problema → conceito → aplicação prática.
Nunca invente novos métodos.
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

    except Exception as e:
        st.error("Erro ao conectar com a OpenAI.")
        st.write(e)
