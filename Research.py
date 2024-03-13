import streamlit as st
import os
import pickle
from pathlib import Path
from openai import OpenAI
from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
import sqlite3
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
#from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
import streamlit_authenticator as stauth

if not st.session_state.authentication_status:
    st.info('Please Login from the Home page and try again.')
    st.stop()

load_dotenv()

model = ChatOpenAI(api_key = os.getenv("OPENAI_API_KEY"))
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat-history
if "messages" not in st.session_state:
    st.session_state.messages = []

# if "chain" not in st.session_state:
#     st.session_state.chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#con = sqlite3.connect('Customer.db')
db = SQLDatabase.from_uri("sqlite:///Customer.db")

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(model, db)
#chain = write_query | execute_query

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

answer = answer_prompt | model | StrOutputParser()
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)
#
# def get_schema(_):
#     return db.get_table_info()
#
# def run_query(query):
#     return db.run(query)
#
# template = """Based on the table schema below, write a SQL query that would answer the user's question:
# {schema}
#
# Question: {question}
# SQL Query:"""
# prompt = ChatPromptTemplate.from_template(template)
#
#
# sql_chain = (
#     RunnablePassthrough.assign(schema=get_schema)
#     | prompt
#     | model.bind(stop=["\nSQLResult:"])
#     | StrOutputParser()
# )
#
#
# template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
# {schema}
#
# Question: {question}
# SQL Query: {query}
# SQL Response: {response}"""
# prompt_response = ChatPromptTemplate.from_template(template)
#
#
# full_chain = (
#     RunnablePassthrough.assign(query=sql_chain).assign(
#         schema=get_schema,
#         response=lambda vars: run_query(vars["query"]),
#     )
#     | prompt_response
#     | model
# )

# template = """Based on the table schema below, write a SQL query that would answer the user's question:
# {schema}
#
# Question: {question}
# SQL Query:"""
# prompt = ChatPromptTemplate.from_template(template)
#
# template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
# {schema}
#
# Question: {question}
# SQL Query: {query}
# SQL Response: {response}"""
# prompt_response = ChatPromptTemplate.from_template(template)
#
# sql_response = (
#     RunnablePassthrough.assign(schema=get_schema)
#     | prompt
#     | model.bind(stop=["\nSQLResult:"])
#     | StrOutputParser()
# )
#
# full_chain = (
#     RunnablePassthrough.assign(query=sql_response).assign(
#         schema=get_schema,
#         response=lambda x: db.run(x["query"]),
#     )
#     | prompt_response
#     | model
# )

#csv = st.file_uploader(label="Upload your CSV File here")



st.title("Do nessacary research here")
st.title("")
st.text("Analyse customer base and competitors in the market before making decisions")

embedding = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
vector_store = FAISS.load_local("faiss_index", embedding)

# Set OpenAi Model

toolkit = SQLDatabaseToolkit(db=db, llm= model)
toolkit.get_tools()

memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

agent_executor = create_sql_agent(model, db=db, agent_type="openai-tools", verbose=True, memory=memory)








# if csv is not None:
#     save_folder = 'content'
#     save_path = Path(save_folder, csv.name)
#     with open(save_path, mode='wb') as w:
#         w.write(csv.getvalue())
#
#     if save_path.exists():
#         st.success(f'File {csv.name} is successfully saved!')
#     loader = CSVLoader(file_path=os.path.join('content/', csv.name))
#     data = loader.load()
#
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
#     text_chunks = text_splitter.split_documents(data)
#
#     model_name = 'text-embedding-ada-002'
#     embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
#
#     vector_store1 = FAISS.from_documents(text_chunks, embedding)

# st.session_state.chain = ConversationalRetrievalChain.from_llm(
#         llm=model,
#         retriever=vector_store.as_retriever(),
#         memory=memory)



# Get chat input
if question := st.chat_input("How can I help"):
    # Display user message in chat container
    with st.chat_message("user"):
        st.markdown(question)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})


    # Display chat-bot message in chat container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        #response = st.session_state.chain({"question": prompt})
        response = agent_executor.run({"input":question})
        #response = chain.invoke({'question': question})
        #st.markdown(response)
        st.write(response)
        # write response[ans]
        # chat history
    #     full_response += (response.answer[0].delta.content or "")
    #     message_placeholder.markdown(full_response + "▌")
    #     message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": response})
