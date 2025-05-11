#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 11 16:02:06 2025

@author: haenainglay
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Page config
st.set_page_config(page_title="Customer Clustering", layout="wide")
st.title("Customer Segmentation App (K-Means Clustering)")

# Load saved model and scaler
@st.cache_resource
def load_model():
    with open("kmeans_model.pkl", "rb") as model_file:
        kmeans = pickle.load(model_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return kmeans, scaler

kmeans, scaler = load_model()

# Load dataset
@st.cache_data
def load_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
    df = pd.read_excel(url)
    df.dropna(inplace=True)
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    features = df.groupby('CustomerID').agg({
        'Quantity': 'sum',
        'UnitPrice': 'mean',
        'TotalPrice': 'sum'
    })
    return features

features = load_data()
features_scaled = scaler.transform(features)
features['Cluster'] = kmeans.predict(features_scaled)

# Show Data Sample
st.subheader("Segmented Customer Data (Sample)")
st.write(features.head())

# Plot Clusters
st.subheader("Cluster Visualization (Quantity vs TotalPrice)")
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(x=features['Quantity'], y=features['TotalPrice'], hue=features['Cluster'], palette="Set2", ax=ax)
ax.set_title("Customer Segments")
st.pyplot(fig)

# Predict New Customer Segment
st.subheader("Predict Cluster for New Customer Data")

with st.form("prediction_form"):
    qty = st.number_input("Total Quantity Purchased", min_value=1)
    price = st.number_input("Average Unit Price", min_value=0.01)
    total = st.number_input("Total Purchase Value", min_value=0.01)
    submitted = st.form_submit_button("Predict Segment")

    if submitted:
        input_data = pd.DataFrame([[qty, price, total]], columns=['Quantity', 'UnitPrice', 'TotalPrice'])
        input_scaled = scaler.transform(input_data)
        prediction = kmeans.predict(input_scaled)[0]
        st.success(f"Predicted Cluster: {prediction}")

