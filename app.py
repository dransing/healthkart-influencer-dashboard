import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Load Data
influencers = pd.read_csv('influencers.csv')
posts = pd.read_csv('posts.csv')
tracking = pd.read_csv('tracking_data.csv')
payouts = pd.read_csv('payouts.csv')

# Merge for ROI Calculation
data = pd.merge(tracking, payouts, on='influencer_id')
data['ROAS'] = data['revenue'] / data['total_payout']

# Dashboard Title
st.title("📊 HealthKart Influencer Campaign Dashboard")

# Filters
platform = st.selectbox("Choose Platform", ["All"] + list(influencers['platform'].unique()))
if platform != "All":
    filtered_influencers = influencers[influencers['platform'] == platform]
else:
    filtered_influencers = influencers

st.write("### Influencers Overview")
st.dataframe(filtered_influencers)

# ROAS Chart
st.write("### ROAS per Influencer")
roas_chart = data.groupby('influencer_id')['ROAS'].mean().reset_index()
roas_chart = pd.merge(roas_chart, influencers[['id', 'name']], left_on='influencer_id', right_on='id')

fig, ax = plt.subplots()
ax.bar(roas_chart['name'], roas_chart['ROAS'], color='skyblue')
ax.set_ylabel("ROAS")
ax.set_title("Return on Ad Spend by Influencer")
st.pyplot(fig)

# Summary Metrics
st.write("### Summary Metrics")
total_orders = tracking['orders'].sum()
total_revenue = tracking['revenue'].sum()
total_payouts = payouts['total_payout'].sum()
overall_roas = round(total_revenue / total_payouts, 2)

st.metric("Total Orders", total_orders)
st.metric("Total Revenue", f"₹{total_revenue}")
st.metric("Total Payouts", f"₹{total_payouts}")
st.metric("Overall ROAS", overall_roas)

