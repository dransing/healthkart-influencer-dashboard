import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Load Data
influencers = pd.read_csv('influencers.csv')
posts = pd.read_csv('posts.csv')
tracking = pd.read_csv('tracking_data.csv')
payouts = pd.read_csv('payouts.csv')

# Merge for ROAS
data = pd.merge(tracking, payouts, on='influencer_id')
data['ROAS'] = data['revenue'] / data['total_payout']

# Sidebar Filters
st.sidebar.header("Filters")
platform_filter = st.sidebar.selectbox("Platform", ["All"] + influencers['platform'].unique().tolist())
gender_filter = st.sidebar.selectbox("Gender", ["All"] + influencers['gender'].unique().tolist())
category_filter = st.sidebar.selectbox("Category", ["All"] + influencers['category'].unique().tolist())

# Apply Filters
filtered_influencers = influencers.copy()
if platform_filter != "All":
    filtered_influencers = filtered_influencers[filtered_influencers["platform"] == platform_filter]
if gender_filter != "All":
    filtered_influencers = filtered_influencers[filtered_influencers["gender"] == gender_filter]
if category_filter != "All":
    filtered_influencers = filtered_influencers[filtered_influencers["category"] == category_filter]

# Dashboard Title
st.title("📊 HealthKart Influencer Campaign Dashboard (Enhanced)")

# Metric Columns
total_orders = tracking['orders'].sum()
total_revenue = tracking['revenue'].sum()
total_payouts = payouts['total_payout'].sum()
overall_roas = round(total_revenue / total_payouts, 2)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Orders", total_orders)
col2.metric("Total Revenue", f"Rs. {total_revenue}")
col3.metric("Total Payouts", f"Rs. {total_payouts}")
col4.metric("Overall ROAS", overall_roas)

# Filtered Influencers Table
st.subheader("Influencer Overview")
st.dataframe(filtered_influencers)

# ROAS Chart using Plotly
st.subheader("ROAS by Influencer")
roas_chart = data.groupby('influencer_id')['ROAS'].mean().reset_index()
roas_chart = pd.merge(roas_chart, influencers[['id', 'name']], left_on='influencer_id', right_on='id')
fig = px.bar(roas_chart, x='name', y='ROAS', color='ROAS', color_continuous_scale='RdYlGn', title="Influencer ROAS")
st.plotly_chart(fig)

# Post-Level Insights
st.subheader("Top Posts by Reach")
top_posts = posts.sort_values(by="reach", ascending=False).head(5)
st.dataframe(top_posts[['influencer_id', 'platform', 'date', 'reach', 'likes', 'comments']])

# Simulated ROAS Trend Over Time
st.subheader("ROAS Trend by Campaign Date (Simulated)")
data['date'] = pd.to_datetime(data['date'])
roas_by_date = data.groupby('date')['ROAS'].mean().reset_index()
fig2 = px.line(roas_by_date, x='date', y='ROAS', markers=True, title="ROAS Trend Over Time")
st.plotly_chart(fig2)

st.caption("Built by Disha Ransingh · Internship Project for HealthKart")
