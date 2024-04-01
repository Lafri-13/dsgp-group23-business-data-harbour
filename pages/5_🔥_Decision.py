import streamlit as st
import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
# Dataset: customer index
# Dataset: Customers csv

if not st.session_state.authentication_status:
    st.info('Please Login from the Home page and try again.')
    st.stop()

st.header("🎇Analyze Customer Segments🎆")


df = pd.read_csv("Customer_Details_and_Preferences.csv")
st.write(df)
@st.cache_data
def fig1():
    st.subheader("Density graphs")
    fig1 = plt.figure(1,figsize=(15,6))
    n=0
    for x in ['Age', 'Bought(for $)', 'Income(Family in k$)', 'Spending Score(1-100)']:
        n += 1
        plt.subplot(1,4,n)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        sns.distplot(df[x], bins=20)
        plt.title("Distplot of {}".format(x))
    st.pyplot(fig1)
    #plt.show()
@st.cache_data
def fig2():
    st.subheader("Count Graphs for male and female")
    fig2 = plt.figure(figsize=(15,6))
    sns.countplot(y='Gender', data=df)
    st.pyplot(fig2)
@st.cache_data
def fig3():
    st.subheader("The peaks of the data")
    fig3 = plt.figure(1,figsize=(15,6))
    n=0
    for cols in ['Age', 'Bought(for $)', 'Income(Family in k$)', 'Spending Score(1-100)']:
        n += 1
        plt.subplot(1,4,n)
        sns.set(style="whitegrid")
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        sns.violinplot(x=cols, y='Gender', data=df)
        plt.ylabel('Gender' if n == 1 else ' ')
        plt.title("Violin Plot")
    st.pyplot(fig3)
@st.cache_data
def fig4():
    st.subheader("Bar graph for each group")
    # Assigning the range group
    age_10_17 = df.Age[(df.Age >= 10) & (df.Age <=17)]
    age_18_40 = df.Age[(df.Age >= 18) & (df.Age <=40)]
    age_41_65 = df.Age[(df.Age >= 41) & (df.Age <=65)]
    age_65above = df.Age[df.Age >= 66]

    # bar graph for each group
    age_x = ['10-17', '18-40', '41-65', '65+']
    age_y = [len(age_10_17.values), len(age_18_40.values), len(age_41_65.values), len(age_65above.values)]
    fig4 = plt.figure(figsize=(15,6))
    sns.barplot(x=age_x, y=age_y, palette='mako')
    plt.title("Number of customers and Ages")
    plt.xlabel("Age")
    plt.ylabel("Number of Customers")
    st.pyplot(fig4)
@st.cache_data
def fig5():
    st.subheader("Spending score for each age group")
    # Assigning the range group
    ss_1_20 = df["Spending Score(1-100)"][(df["Spending Score(1-100)"] >= 1) & (df["Spending Score(1-100)"] <= 20)]
    ss_21_40 = df["Spending Score(1-100)"][(df["Spending Score(1-100)"] >= 21) & (df["Spending Score(1-100)"] <= 40)]
    ss_41_60 = df["Spending Score(1-100)"][(df["Spending Score(1-100)"] >= 41) & (df["Spending Score(1-100)"] <= 60)]
    ss_61_80 = df["Spending Score(1-100)"][(df["Spending Score(1-100)"] >= 61) & (df["Spending Score(1-100)"] <= 80)]
    ss_81_100 = df["Spending Score(1-100)"][(df["Spending Score(1-100)"] >= 81) & (df["Spending Score(1-100)"] <= 100)]

    # bar graph for each group
    ss_x = ['1-20','21-40', '41-60', '61-80', '81-100',]
    ss_y = [len(ss_1_20.values), len(ss_21_40.values), len(ss_41_60.values), len(ss_61_80.values), len(ss_81_100.values)]
    fig5 = plt.figure(figsize=(15,6))
    sns.barplot(x=ss_x, y=ss_y, palette='rocket')
    plt.title("Spending Scores")
    plt.xlabel("Score")
    plt.ylabel("Number of Customers having the store")
    st.pyplot(fig5)
@st.cache_data
def fig6():
    st.subheader("Annual income of a family for each age group")
    # Assigning the range group
    in_1_20 = df["Income(Family in k$)"][(df["Income(Family in k$)"] >= 1) & (df["Income(Family in k$)"] <= 20)]
    in_21_40 = df["Income(Family in k$)"][(df["Income(Family in k$)"] >= 21) & (df["Income(Family in k$)"] <= 40)]
    in_41_60 = df["Income(Family in k$)"][(df["Income(Family in k$)"] >= 41) & (df["Income(Family in k$)"] <= 60)]
    in_61_80 = df["Income(Family in k$)"][(df["Income(Family in k$)"] >= 61) & (df["Income(Family in k$)"] <= 80)]
    in_81_100 = df["Income(Family in k$)"][(df["Income(Family in k$)"] >= 81) & (df["Income(Family in k$)"] <= 100)]

    # bar graph for each group
    in_x = ['1-20','21-40', '41-60', '61-80', '81-100',]
    in_y = [len(in_1_20.values), len(in_21_40.values), len(in_41_60.values), len(in_61_80.values), len(in_81_100.values)]
    fig6 = plt.figure(figsize=(15,6))
    sns.barplot(x=in_x, y=in_y, palette='Spectral')
    plt.title("Anual Income of a Family")
    plt.xlabel("Income")
    plt.ylabel("Number of Customers")
    st.pyplot(fig6)
@st.cache_data
def fig7_elbow():
    st.subheader("Elbow Method")
    # Storing the relevant column details in x3
    global x3
    x3=df.loc[:, ["Age","Income(Family in k$)", "Spending Score(1-100)"]].values
    # empty list to store within cluster sum of squares
    wcss=[]
    # Creating Kmeans model for no of clusters k
    for k in range(1,11):
        kmeans = KMeans(n_clusters = k, init='k-means++', n_init = 10)
        # Fitting the model to x3
        kmeans.fit(x3)
        # storing all wcss for all k values
        wcss.append(kmeans.inertia_)
    # Elbow Graph
    fig7 = plt.figure(figsize=(15,6))
    plt.plot(range(1,11), wcss)
    plt.title('Elbow Method for Optimal K value')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Sum of Squared Distance (WCCS)')
    st.pyplot(fig7)
@st.cache_data
def fig8_kmeans():
    st.subheader("K-means cluster")
    # model with 7 clusters, 10 initializations
    kmeans = KMeans(n_clusters=7, n_init=10)
    # assigning cluster labels to dataframe
    clusters = kmeans.fit_predict(x3)
    df["label"] = clusters

    fig8 = px.scatter_3d(df, x='Age', y='Spending Score(1-100)', z='Income(Family in k$)', color='label',
                        opacity=0.8, size_max=5, labels={'label': 'Cluster'})

    st.plotly_chart(fig8)



fig1()
fig2()
fig4()
fig5()
fig6()
fig7_elbow()
fig8_kmeans()

@st.cache_data
def read_csv(csv):
    return pd.read_csv(csv)

@st.cache_resource
def load_vecDb():
    load_dotenv()
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    return FAISS.load_local("customer_index", embeddings)

df1 = read_csv("Customers.csv")
db = load_vecDb()
#st.write(df)

st.subheader("🎈 Use a Vector Database to get 10 similar customers to the one you select!!!")
st.text("Your customers with similar description will be retrieved below 🎈")
gd = GridOptionsBuilder.from_dataframe(df1)
gd.configure_pagination(enabled=True)
gd.configure_default_column(editable=True)
gd.configure_selection(selection_mode='single',use_checkbox=True)
grid_options = gd.build()
table = AgGrid(df1, gridOptions=grid_options)

if table["selected_rows"]:
    row = table["selected_rows"]
    cus_des = row[0]["Customer_description"]
    sim_cus = db.similarity_search_with_score(cus_des,k=10)
    for doc, score in sim_cus:
        st.write(f"🌟Customer description:")
        st.write(doc.page_content)
        st.text(f"🎈Customer id: {doc.metadata['Customer_id'] - 1}")
        st.text(f"🎈Similarity score: {score}")
        st.write('➖'*32)
