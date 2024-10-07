from dotenv import load_dotenv
import streamlit as st
from user_utils import *
import os

def main():
    load_dotenv()
    st.set_page_config(page_title="PerformanceX")
    st.title("PerformanceX ")
    st.header("Your Personalized AI Coach", divider="blue")
    st.sidebar.image("https://dcassetcdn.com/design_img/4077411/1363205/33612364/n1fphb57h65khkmdxps3m2h6b3_image.png")

    embeddings = create_embeddings()
    pinecone_key=st.secrets['PINECONE_API_KEY']
    index = pull_from_pinecone(pinecone_key, "us-east-1", "performancex", embeddings)
    
    if "history" not in st.session_state:
        st.session_state["history"] = []

    user_input = st.chat_input("How can PerformanceX help you today ?")
    
    if user_input:
        st.session_state["history"].append({"role": "user", "content": user_input})
        
        with st.spinner("I am working on it"):
            relevant_docs = get_similar_docs(index, user_input)
            if relevant_docs and is_relevant(relevant_docs[0], user_input):
                response = get_answer(relevant_docs, user_input)
            else:
                response = get_llm_answer(user_input)
        
        st.session_state["history"].append({"role": "assistant", "content": response})
    
    for message in st.session_state["history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if __name__ == '__main__':
    main()