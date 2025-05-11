import os
import time
import streamlit as st
from PIL import Image
from accelerate import init_empty_weights

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint



DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    start_time = time.time()
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    elapsed_time = time.time() - start_time
    print(f"[Timing] Vectorstore loading time: {elapsed_time:.4f} seconds")
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    start_time = time.time()
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        max_length=512,
        return_full_text=False,
        token=HF_TOKEN
    )
    elapsed_time = time.time() - start_time
    print(f"[Timing] LLM loading time: {elapsed_time:.4f} seconds")
    return llm


def main():
    # Load custom chatbot icon first
    image_path = os.path.join("assets", "images", "chatbot_icon.png")
    try:
        full_path = os.path.abspath(image_path)
        st.sidebar.text(f"Image path: {full_path}")
        if os.path.exists(full_path):
            avatar_img = Image.open(full_path)
            st.sidebar.success("Custom avatar loaded successfully!")
        else:
            raise FileNotFoundError(f"Image not found at {full_path}")
    except Exception as e:
        st.sidebar.warning(f"Using default avatar. Error: {str(e)}")
        avatar_img = "🩺"  

    # Custom title with large avatar
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(avatar_img, width=100)  # Larger avatar
    with col2:
        st.markdown("<h1 style='font-size: 2.5em;'>Hi! I am Thoth Your Godly Advisor</h1>", unsafe_allow_html=True)

    # Initialize messages if not present
    if 'messages' not in st.session_state:
        st.session_state.messages = [{
            'role': 'assistant',
            'content': "<div style='font-size: 1.2em;'>I am ready to answer your question, ask me anything</div>"
        }]

    # Display all messages
    for message in st.session_state.messages:
        with st.chat_message(
            message['role'],
            avatar=avatar_img if message['role'] == 'assistant' else None
        ):
            if isinstance(message['content'], str) and message['content'].startswith('<div'):
                st.markdown(message['content'], unsafe_allow_html=True)
            else:
                st.markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the following pieces of context to provide a detailed, comprehensive answer to the question at the end.
                Combine information from multiple context sources when relevant.
                If you don't know the answer based on the context, say you don't know - don't make up an answer.
                Include relevant details and explanations from the context when possible.

                Context: {context}

                Question: {question}

                Provide a thorough answer with supporting details from the context:
                """
        
        HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN=os.environ.get("HF_TOKEN")

        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain=RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':10}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            start_time = time.time()
            response=qa_chain.invoke({'query':prompt})
            elapsed_time = time.time() - start_time
            print(f"[Timing] Query response generation time: {elapsed_time:.4f} seconds")

            result=response["result"]
            #source_documents=response["source_documents"]
            result_to_show=f"{result}\n\n_Response generated in {elapsed_time:.2f} seconds_"
            #response="Hi, I am MediBot!"
            st.chat_message('assistant', avatar=avatar_img).markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    import asyncio
    
    # Create new event loop for Streamlit
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    main()
