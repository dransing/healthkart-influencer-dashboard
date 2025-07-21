import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import base64

# --- Load All Data ---
influencers = pd.read_csv('influencers.csv')
posts = pd.read_csv('posts.csv')
tracking = pd.read_csv('tracking_data_with_brand.csv')
payouts = pd.read_csv('payouts.csv')
sentiment = pd.read_csv('posts_with_sentiment.csv')
profit = pd.read_csv('profit_estimation.csv')
anomaly = pd.read_csv('anomaly_detected.csv')

# --- Preprocess ---
tracking['date'] = pd.to_datetime(tracking['date'])  # Ensure datetime for comparison

data = pd.merge(tracking, payouts, on='influencer_id')
data['ROAS'] = data['revenue'] / data['total_payout']
data['brand'] = tracking['brand']  # Brand column added

# --- Sidebar Filters ---
st.sidebar.header("Filters")
platform_filter = st.sidebar.selectbox("Platform", ["All"] + influencers['platform'].unique().tolist())
gender_filter = st.sidebar.selectbox("Gender", ["All"] + influencers['gender'].unique().tolist())
category_filter = st.sidebar.selectbox("Category", ["All"] + influencers['category'].unique().tolist())
brand_filter = st.sidebar.selectbox("Brand", ["All"] + tracking['brand'].dropna().unique().tolist())

filtered_influencers = influencers.copy()
if platform_filter != "All":
    filtered_influencers = filtered_influencers[filtered_influencers['platform'] == platform_filter]
if gender_filter != "All":
    filtered_influencers = filtered_influencers[filtered_influencers['gender'] == gender_filter]
if category_filter != "All":
    filtered_influencers = filtered_influencers[filtered_influencers['category'] == category_filter]

filtered_data = data.copy()
if brand_filter != "All":
    filtered_data = filtered_data[filtered_data['brand'] == brand_filter]

# Ensure 'date' column is datetime
filtered_data['date'] = pd.to_datetime(filtered_data['date'])

# --- Dashboard Title ---
st.title("📊 HealthKart Influencer Campaign Dashboard (Agentic AI Edition)")

# --- KPI Metrics ---
if 'orders' in filtered_data.columns:
    total_orders = filtered_data['orders'].sum()
else:
    total_orders = 0

total_revenue = filtered_data['revenue'].sum() if 'revenue' in filtered_data.columns else 0
total_payouts = filtered_data['total_payout'].sum() if 'total_payout' in filtered_data.columns else 0
overall_roas = round(total_revenue / total_payouts, 2) if total_payouts else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Orders", total_orders)
col2.metric("Total Revenue", f"Rs. {total_revenue}")
col3.metric("Total Payouts", f"Rs. {total_payouts}")
col4.metric("Overall ROAS", overall_roas)

# (Remaining sections go here...)

# Footer
st.caption("Built by Disha Ransingh · Internship Project for HealthKart")



