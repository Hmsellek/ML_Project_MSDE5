import streamlit as st
import pandas as pd
import pickle

st.image("http://www.ehtp.ac.ma/images/lo.png")
st.write("""
# MSDE5 : ML Course
## Project Segmentation of customer App

This app predicts the **Cluster** of customer based on RFM classification
""")

st.sidebar.image("https://thumbor.sd-cdn.fr/ESWkHL0JlBaiwUhodp6pe0TbX7A=/fit-in/724x493/cdn.sd-cdn.fr/wp-content/uploads/2019/07/RFM.png",width=300)

st.sidebar.header('Insertion of parameters')

def user_input_features():
    R = st.number_input('Recency')
    F = st.number_input('Frequency')
    M = st.number_input('Monetary')
    data = {'R': R,
            'F': F,
            'M': M,}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

kmeans_model=pickle.load(open("kmeans_model.pkl", "rb"))
prediction = kmeans_model.predict(df)
prediction_proba = kmeans_model.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(pd.DataFrame(kmeans_model.Cluster_))

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

