import streamlit as st
import streamlit_authenticator as stauth
import pickle
from pathlib import Path



names = ["Shahid", "Uzama", "Lafri", "Vinod"]
usernames = ["Shahid", "Uzama", "Lafri", "Vinod"]

# load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    "sales_dashboard", "abcdef", cookie_expiry_days=0)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:

    st.title('🚢 WELCOME TO BUSINESS DATA HARBOR 🚢')
    st.write("🌊" * 32)
    #st.write("➖" * 32)
    st.text("🎈 Below are the page descriptions.")
    st.text("🎈 Pages are in the side bar ▶ top left corner")
    st.write("")
    st.write("")
    #st.text("🎈 Pages are in the side bar ▶ top left corner")
    # st.header('⛵ Test the Waters ⛵')
    # #st.subheader('Understand your customers and competition before you make decisions')
    # st.write("🌊"*32)
    st.subheader("💧 Research Structured: ")
    st.text("> Chat with tabular data store in a SQL database")
    st.text("> Data contains Customers and Watches")
    st.write("")
    st.write("")
    st.subheader("⚡ Research Unstructured: ")
    st.text("> Chat with textual data stored in a Vector Database")
    st.text("> Contains textual data about some watches")
    st.write("")
    st.write("")
    #st.write("➖" * 32)
    # st.header('🌊 Take the plunge🌊 ')
    # st.write("🌊" * 32)
    #st.subheader('Generate tailor made marketing decisions unique to each customer')
    st.subheader("🌀 Reviews analyzer: ")
    st.text("> A Chat bot used to analyze customer sentiments on Watch types")
    st.write("")
    st.write("")
    st.subheader("🔥 Decisions:")
    st.text("> Analyse Customer plots ")
    st.text("> Analyze Customer segments using kmeans and Vector database")
    st.write("")
    st.write("")
    st.subheader("🖼️ Content Generator:")
    st.text("> Generate tailor made content ")
    #authenticator.logout("Logout", "sidebar")


