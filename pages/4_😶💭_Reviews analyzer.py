import streamlit as st
import sqlite3
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
import joblib
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Dataset: CReviews 1 DB
# Dataset: vectorizer file
# Dataset: trained lr

st.header("🌟 Analyze Customer Reviews")

nltk.download('stopwords')


load_dotenv()
# vectorizer = joblib.load("vectorizer_file.pkl")
# model = joblib.load("trained_ensemble_model.sav")
db = SQLDatabase.from_uri("sqlite:///CReviews1.db")
if 'vectorizer' not in st.session_state:
    st.session_state['vectorizer'] = joblib.load("vectorizer_file.pkl")
if 'model' not in st.session_state:
    st.session_state['model'] = joblib.load("trained_lr.sav")
# if "db" not in st.session_state:
#     st.session_state["db"] = SQLDatabase.from_uri("sqlite:///CReviews.db")
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_schema(_):
    return db.get_table_info()

def get_query(question):
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    template = """Based on the table schema below, write a SQL query that would answer the user's question, ignore the sentiment part of the question and focus only on the watch type:
    {schema}

    Example:
    Question: Give me the positive watch reviews for a watch type equal to digital.
    Answer:SELECT review FROM Watch_reviews WHERE watch_type = 'Digital'

    Example:
    Question: Give me the positive watch reviews for a watch type equal to smart.
    Answer: SELECT review FROM Watch_reviews WHERE  watch_type = 'Smart'

    Question: {question}
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    sql_response = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )

    return sql_response.invoke({"question":question})

def get_reviews(query):
    con = sqlite3.connect('CReviews1.db')
    c = con.cursor()
    c.execute(query)
    reviews = []
    for tuple in c.fetchall():
        reviews.append(tuple)
    return reviews

def stemming(content):
    port_stem = PorterStemmer()
    if not isinstance(content, str):
        # Convert non-string content to string
        content = str(content)
    # Remove everything other than letters in a text
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    # Convert every letter to lowercase
    stemmed_content = stemmed_content.lower()
    # Split the words and add to a list
    stemmed_content = stemmed_content.split()
    # Stem the words, but do not stem stopwords because they have no meaning
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    # Join the stemmed words
    stemmed_content = ' '.join(stemmed_content)

    return stemmed_content

def get_sentiment(reviews):
    positive_list = []
    negative_list = []
    neutral_list = []

    for review in reviews:
        processed_txt = stemming(review[0])
        vectorized_txt = st.session_state['vectorizer'].transform([processed_txt])
        prediction = st.session_state['model'].predict(vectorized_txt)
        if prediction == 1:
            positive_list.append(review[0])
        elif prediction == 0:
            negative_list.append(review[0])
        else:
            neutral_list.append(review[0])

    return positive_list, negative_list, neutral_list

if question := st.chat_input("How can I help"):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Display chat-bot message in chat container
    with st.chat_message("assistant"):
        #message_placeholder = st.empty()

        sql_query = get_query(question)
        st.write(sql_query)
        reviews = get_reviews(sql_query)
        positive_list, negative_list, neutral_list = get_sentiment(reviews)

        if "positive" in question:
            st.header("Positive reviews")
            for r in positive_list:
                st.write(f"-{r}")
        elif "negative" in question:
            st.header("Negative reviews")
            for r in negative_list:
                st.write(f"-{r}")
        else:
            st.header("Neutral reviews")
            for r in neutral_list:
                st.write(f"-{r}")

    #st.session_state.messages.append({"role": "assistant", "content": response})