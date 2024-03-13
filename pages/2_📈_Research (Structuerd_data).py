import streamlit as st
import sqlite3
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import time

# Dataset: CustomerDB
st.header("✨Understand your Customer base and Competitors")

if not st.session_state.authentication_status:
    st.info('Please Login from the Home page and try again.')
    st.stop()

load_dotenv()

llm = ChatOpenAI(api_key = os.getenv("OPENAI_API_KEY"))

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat-history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

db = SQLDatabase.from_uri("sqlite:///CustomerWatches.db")


def get_schema(_):
    return db.get_table_info()

def run_query(query):
    return db.run(query)

def get_sql_query(question):
    template = """Based on the table schema below, write a SQL query that would answer the user's question:
    {schema}
    
    Example:
    Question: How many customers prefer a Steel strap type?
    Answer: SELECT COUNT(*) AS Steel_Strap_Customers FROM Customers WHERE Strap_type = 'Steel';
    
    Question: list the names of the watches with the watch type equal to formal?
    Answer: SELECT Watch_name FROM Watches WHERE Watch_type = 'Formal';

    Question: {question}
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    sql_response_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
    )

    return sql_response_chain.invoke({"question": question})

def get_sql_responce(query):
    conn = sqlite3.connect('Customer.db')
    c = conn.cursor()
    c.execute(query)
    sqlRes = c.fetchall()
    conn.close()
    return sqlRes

def get_NL_responce(sql_responce,question):
    template = """Based on the Question and Sql response below write a Natural language response:

        Question: {question}
    SQL Response: {response}"""
    prompt_response = ChatPromptTemplate.from_template(template)

    full_chain = (prompt_response | llm)

    return full_chain.invoke({"question": question, "response": f"{sql_responce}"})

if question := st.chat_input("How can I help"):
    with st.chat_message("Human"):
        st.markdown(question)
    st.session_state.messages.append({"role": "Human", "content": question})

    with st.chat_message("AI"):
        # try:
        query = get_sql_query(question)
        st.write(query)
        sql_responce = get_sql_responce(query)
        NL_responce = get_NL_responce(sql_responce,question)
        st.markdown(NL_responce)
        st.session_state.messages.append({"role": "AI", "content": NL_responce})
        # except:
        #     st.text("try again")
