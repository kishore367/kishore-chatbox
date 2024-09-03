import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

# Set the page configuration at the top of the script
st.set_page_config(page_title="AI ChatBot", page_icon="🤖", layout="wide")

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=1)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    #print(docs)
    if not docs:
        return None

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    return response["output_text"]

def fallback_gemini(user_question):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=1)
    response = model(user_question)
    return response["output_text"]

def display_chat():
    for message in st.session_state.messages:
        avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

def main():
    st.header("AI ChatBot 💁")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        # with st.chat_message("user", avatar='👨‍💻'):
        #     st.markdown(user_question)

        response = user_input(user_question)

        if response is None:
            response = fallback_gemini(user_question)

        st.session_state.messages.append({"role": "assistant", "content": response})

    display_chat()

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
