from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate


# Dataset: Unstructured Data pdf
if not st.session_state.authentication_status:
    st.info('Please Login from the Home page and try again.')
    st.stop()

st.header("⚡ Analyze Unstructured Textual data")
st.write("")
st.write("🟣 Here you can analyze textual descriptions about your competitors products")
st.write("🟣 Try some of the prompts in the drop down to begin")

load_dotenv()

if "messages1" not in st.session_state:
    st.session_state.messages1 = []

for message in st.session_state.messages1:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def get_res(question):
    loader = PyPDFLoader("Unstructured Data.pdf")

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key = os.getenv("OPENAI_API_KEY")))

    retriever = vectorstore.as_retriever()
    template = """Answer the question based only on the following context:
    {context}. 
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(api_key = os.getenv("OPENAI_API_KEY"))

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)

option = st.selectbox(
    'Query Unstructured Data', (
        'What kind of data is stored here',
        'List all the watch names',
        'What are some cons of the casio A168',
        'What is the Tissot My Lady watch all about',
        'What do you know about the casio a700',
        'What are some issues in the Seiko 5 Automatic',
        'What are some features of the Casio G Shock',
        'A brief on the timex expedition'
    ), index=None, placeholder='Try some default prompts to start out')

if prompt := st.chat_input("How can I help"):
    option = None

if option or prompt:
    if option:
        question = option
    else:
        question = prompt
    option = None
    prompt = None
    with st.chat_message("Human"):
        st.markdown(question)
    st.session_state.messages1.append({"role": "Human", "content": question})

    # Display chat-bot message in chat container
    with st.chat_message("AI"):
        message_placeholder = st.empty()
        full_response = ""

        response = get_res(question)

        st.markdown(response)

        st.session_state.messages1.append({"role": "AI", "content": response})