import streamlit as st
import requests
import time
import datetime


st.set_page_config(
    page_title="MiniBot IA - Assistente Local da UFV",
    page_icon="✨",
    layout="centered",
)

SUGGESTIONS = {
    ":blue[:material/psychology:] O que é RAG?": (
        "Explique de forma simples o que é RAG e como funciona."
    ),
    ":green[:material/bolt:] Como acelerar um modelo LLM local?": (
        "Quais técnicas posso usar para melhorar a velocidade de um modelo rodando no Ollama?"
    ),
    ":orange[:material/smart_toy:] Como integrar FastAPI com LLM?": (
        "Como faço uma API com FastAPI que responde usando um modelo local no Ollama?"
    ),
}


if "messages" not in st.session_state:
    st.session_state.messages = []

if "prev_timestamp" not in st.session_state:
    st.session_state.prev_timestamp = datetime.datetime.fromtimestamp(0)

use_rag = st.toggle("Usar RAG", value=True)


def call_backend(prompt: str):
    """
    Chama sua API FastAPI que usa Ollama local.
    Streaming REAL com write_stream.
    """
    
    if use_rag:
        url = "http://127.0.0.1:8000/ask_ollama3_with_rag_endpoint"                    
    else:
        url = "http://127.0.0.1:8000/ask_ollama3"

    with requests.post(url, json={"prompt": prompt}, stream=True) as r:
        r.raise_for_status() 
        for chunk in r.iter_lines(): 
            if chunk: 
                yield chunk.decode("utf-8")



with st.expander("Adicionar documentos ao RAG", expanded=False):

    uploaded_files = st.file_uploader(
        "Envie PDFs, TXT ou DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Enviar arquivos"):
            files = [
                ("files", (file.name, file.getbuffer()))
                for file in uploaded_files
            ]

            r = requests.post(
                "http://127.0.0.1:8000/upload_docs",
                files=files,
            )

            if r.status_code == 200:
                st.success("Arquivos enviados com sucesso!")

    if st.button("⚙️ Processar documentos no RAG"):
        with st.spinner("Processando base vetorial..."):
            r = requests.post("http://127.0.0.1:8000/process_docs")

            if r.status_code == 200:
                st.success("Base vetorial atualizada com sucesso!")
                st.json(r.json())


header = st.container()
with header:
    st.html("<div style='font-size: 4rem; line-height: 1;'>❉</div>")
    st.title("MiniBot IA – Assistente Local")

user_first_interaction = (
    ("initial_question" in st.session_state and st.session_state.initial_question)
    or ("selected_suggestion" in st.session_state and st.session_state.selected_suggestion)
)

has_history = len(st.session_state.messages) > 0

if not user_first_interaction and not has_history:

    st.session_state.messages = []

    with st.container():
        st.chat_input("Faça uma pergunta...", key="initial_question")

        selected = st.pills(
            label="Exemplos",
            label_visibility="collapsed",
            options=SUGGESTIONS.keys(),
            key="selected_suggestion",
        )

    st.stop()


user_message = st.chat_input("Pergunte algo...")

if not user_message:
    if "initial_question" in st.session_state and st.session_state.initial_question:
        user_message = st.session_state.initial_question
    if "selected_suggestion" in st.session_state and st.session_state.selected_suggestion:
        user_message = SUGGESTIONS[st.session_state.selected_suggestion]

with header:
    if st.button("Restart", icon=":material/refresh:"):
        st.session_state.messages = []
        st.session_state.initial_question = None
        st.session_state.selected_suggestion = None
        st.rerun()



for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if user_message:

    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response_gen = call_backend(user_message)
            full_response = st.write_stream(response_gen)

    st.session_state.messages.append({"role": "user", "content": user_message})
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    st.rerun()