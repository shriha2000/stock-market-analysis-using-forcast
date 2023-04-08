# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""

import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression

st.title('Model Deployment: Logistic Regression')

st.sidebar.header('User Input Parameters')

def user_input_features():
    Open = st.sidebar.selectbox('Open',('1','0'))
    High = st.sidebar.selectbox('High',('1','0'))
    Low = st.sidebar.selectbox('Low',('1','0'))
    Close = st.sidebar.number_input("Insert Close")

    data = {'Open':Open,
            'High':High,
            'Low':Low,
            'Close':Close}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

reliance = pd.read_csv("RELIANCE.csv")
reliance = reliance.dropna()

X = reliance.iloc[:,[1,2,3,4]]
Y = reliance.iloc[:,0]
clf = LogisticRegression()
clf.fit(X,Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')

st.subheader('Prediction Probability')
st.write(prediction_proba)