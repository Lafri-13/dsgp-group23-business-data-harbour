import streamlit as st
import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os


# Dataset: customer index
# Dataset: Customers csv

# if not st.session_state.authentication_status:
#     st.info('Please Login from the Home page and try again.')
#     st.stop()

st.header("🎇Analyze Custemer Segments🎆")

@st.cache_data
def read_csv(csv):
    return pd.read_csv(csv)

@st.cache_resource
def load_vecDb():
    load_dotenv()
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    return FAISS.load_local("customer_index", embeddings)

df = read_csv("Customers.csv")
db = load_vecDb()
#st.write(df)

gd = GridOptionsBuilder.from_dataframe(df)
gd.configure_pagination(enabled=True)
gd.configure_default_column(editable=True)
gd.configure_selection(selection_mode='single',use_checkbox=True)
grid_options = gd.build()
table = AgGrid(df, gridOptions=grid_options)

if table["selected_rows"]:
    row = table["selected_rows"]
    cus_des = row[0]["Customer_description"]
    sim_cus = db.similarity_search_with_score(cus_des,k=10)
    for doc, score in sim_cus:
        st.write(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")
    # st.write(row[0]["Customer_id"])
    # st.write(row[0]["Customer_description"])
