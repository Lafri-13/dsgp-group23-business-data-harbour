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
import os.path


# Dataset: CustomerDB
st.header("✨ Understand your Customer base and Competitors")
st.write("")
st.write("🟣 Here you can analyze your customers and competitors products by interacting with the chat bot")
st.write("🟣 Try some of the prompts in the drop down to begin")

if not st.session_state.authentication_status:
    st.info('Please Login from the Home page and try again.')
    st.stop()

load_dotenv()

#os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

llm = ChatOpenAI(api_key = os.getenv("OPENAI_API_KEY"))

# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-3.5-turbo"

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
    {schema}. Do not use 'information_schema' in your query. Use Like instead of '=' wherever possible. If the 
    users question does not contain something like 'customers' then query the watches table
    
    Example:
    Question: How many customers prefer a Steel strap type?
    Answer: SELECT COUNT(*) AS Steel_Strap_Customers FROM Customers WHERE Strap_type LIKE '%Steel%';
    
    Question: what are the columns in the Customers table?
    Answer: SELECT * FROM PRAGMA_TABLE_INFO('Customers');
    
    Question: list the names of the watches with the watch type equal to formal?
    Answer: SELECT Watch_name FROM Watches WHERE Watch_type Like '%Formal%';
    
    Question: "How many customers prefer a diameter size suitable for small wrists"
    Answer: Select Count(*) From Customers Where Diameter_size Like '%small wrists%'
    
    Question: "List all the brands of the watches"
    Answer: SELECT Brands FROM Watches GROUP BY Brands

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
    conn = sqlite3.connect('CustomerWatches.db')
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



option = st.selectbox(
    "Query Database", (
        "What are the tables in the database",
        'What are the columns in the customer table',
        'How many male customer exist',
        'How many customers are between the age range 20-25',
        "How many customers prefer the watch type formal",
        "How many customers prefer analog watches",
        "How many customers prefer steel strap type",
        "How many customer prefer watches with Quartz mechanics",
        'What are the columns in the Watches table',
        'List all the brands of the watches',
        'List all the watch types that exist',
        'How many quartz watches exist',
        "List the watch names with steel strap type",
        "list the names of the watches with a water resistence of 20 bar",
        "list the watch names which have the chronograph feature as an extra feature",
        "What is the price of the casio a700"
    ), index=None, placeholder='Try some default prompts to start out')


if prompt := st.chat_input("How can I help"):
    option = None


if option or prompt:
    if option:
        question = option
    else:
        question = prompt
    with st.chat_message("Human"):
        st.markdown(question)
    st.session_state.messages.append({"role": "Human", "content": question})

    with st.chat_message("AI"):
        # try:
        query = get_sql_query(question)
        #st.write(query)
        sql_responce = get_sql_responce(query)
        NL_responce = get_NL_responce(sql_responce,question)
        st.markdown(NL_responce.content)
        st.session_state.messages.append({"role": "AI", "content": NL_responce.content})
        question = None
        # except:
        #     st.text("try again")
