import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import base64

# --- Load All Data ---
influencers = pd.read_csv('influencers.csv')
posts = pd.read_csv('posts.csv')
tracking = pd.read_csv('tracking_data.csv')
payouts = pd.read_csv('payouts.csv')
sentiment = pd.read_csv('posts_with_sentiment.csv')
profit = pd.read_csv('profit_estimation.csv')
anomaly = pd.read_csv('anomaly_detected.csv')

# --- Preprocess ---
data = pd.merge(tracking, payouts, on='influencer_id')
data['ROAS'] = data['revenue'] / data['total_payout']
data['date'] = pd.to_datetime(data['date'])

# --- Sidebar Filters ---
st.sidebar.header("Filters")
platform_filter = st.sidebar.selectbox("Platform", ["All"] + influencers['platform'].unique().tolist())
gender_filter = st.sidebar.selectbox("Gender", ["All"] + influencers['gender'].unique().tolist())
category_filter = st.sidebar.selectbox("Category", ["All"] + influencers['category'].unique().tolist())

filtered_influencers = influencers.copy()
if platform_filter != "All":
    filtered_influencers = filtered_influencers[filtered_influencers['platform'] == platform_filter]
if gender_filter != "All":
    filtered_influencers = filtered_influencers[filtered_influencers['gender'] == gender_filter]
if category_filter != "All":
    filtered_influencers = filtered_influencers[filtered_influencers['category'] == category_filter]

# --- Dashboard Title ---
st.title("📊 HealthKart Influencer Campaign Dashboard (Agentic AI Edition)")

# --- KPI Metrics ---
total_orders = tracking['orders'].sum()
total_revenue = tracking['revenue'].sum()
total_payouts = payouts['total_payout'].sum()
overall_roas = round(total_revenue / total_payouts, 2)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Orders", total_orders)
col2.metric("Total Revenue", f"Rs. {total_revenue}")
col3.metric("Total Payouts", f"Rs. {total_payouts}")
col4.metric("Overall ROAS", overall_roas)

# --- Influencer Overview ---
st.subheader("Influencer Overview")
st.dataframe(filtered_influencers)

# --- ROAS Bar Chart ---
st.subheader("ROAS by Influencer")
roas_chart = data.groupby('influencer_id')['ROAS'].mean().reset_index()
roas_chart = pd.merge(roas_chart, influencers[['id', 'name']], left_on='influencer_id', right_on='id')
fig = px.bar(roas_chart, x='name', y='ROAS', color='ROAS', color_continuous_scale='RdYlGn')
st.plotly_chart(fig)

# --- ROAS Over Time ---
st.subheader("ROAS Trend Over Time")
roas_by_date = data.groupby('date')['ROAS'].mean().reset_index()
fig2 = px.line(roas_by_date, x='date', y='ROAS', title="ROAS Trend")
st.plotly_chart(fig2)

# --- ROI Simulator ---
st.subheader("ROI Simulator")
multiplier = st.slider("Adjust Payout Multiplier", 0.5, 2.0, 1.0)
simulated_payouts = payouts.copy()
simulated_payouts['adjusted_payout'] = simulated_payouts['total_payout'] * multiplier
simulated_total_payout = simulated_payouts['adjusted_payout'].sum()
simulated_roas = round(total_revenue / simulated_total_payout, 2)
st.metric("Simulated ROAS", simulated_roas)

# --- Smart Recommendations ---
st.subheader("💡 Smart Influencer Recommendations")
top_recommend = roas_chart[roas_chart['ROAS'] > 8]['name'].tolist()
drop_recommend = roas_chart[roas_chart['ROAS'] < 4]['name'].tolist()
if top_recommend:
    st.success(f"📈 Invest More in: {', '.join(top_recommend)}")
if drop_recommend:
    st.warning(f"📉 Consider Revising: {', '.join(drop_recommend)}")

# --- Natural Language Insights ---
st.subheader("🧠 Auto Insights")
yt_avg = roas_chart.merge(influencers, left_on='influencer_id', right_on='id')
yt_roas = yt_avg[yt_avg['platform'] == 'YouTube']['ROAS'].mean()
insta_roas = yt_avg[yt_avg['platform'] == 'Instagram']['ROAS'].mean()
delta = round((yt_roas - insta_roas) / insta_roas * 100, 1) if insta_roas else 0

st.markdown(f"""
- 📺 YouTube ROAS beats Instagram by **{delta}%**
- 🏆 Top Influencer: **{roas_chart.sort_values('ROAS', ascending=False)['name'].iloc[0]}**
- 🚨 Lowest Performer: **{roas_chart.sort_values('ROAS')['name'].iloc[0]}**
""")

# --- Sentiment Analysis ---
st.subheader("💬 Caption Sentiment Analysis")
fig_sent = px.histogram(sentiment, x='sentiment', color='sentiment', title="Post Sentiment Distribution")
st.plotly_chart(fig_sent)

# --- Profit Estimation ---
st.subheader("💸 Profit Estimation")

# Load and fix column names
profit = pd.read_csv('profit_estimation.csv')
profit.rename(columns={'Net_Profit': 'net_profit', 'Profit_Margin_%': 'margin'}, inplace=True)

# Show column list for debugging
st.write("Columns in your profit dataset:", profit.columns.tolist())

required_cols = ['influencer_id', 'net_profit', 'margin']

if all(col in profit.columns for col in required_cols):
    st.dataframe(profit[required_cols])

    # Ensure influencer_id is treated as a string for coloring
    profit['influencer_id'] = profit['influencer_id'].astype(str)

    # Plot net profit bar chart with safe color palette
    fig_profit = px.bar(
        profit,
        x='influencer_id',
        y='net_profit',
        color='influencer_id',
        title="Net Profit by Influencer",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    st.plotly_chart(fig_profit)

else:
    st.error("❌ One or more required columns (influencer_id, net_profit, margin) are missing in profit_estimation.csv.")


# --- Anomaly Detection ---
st.subheader("🚨 Anomaly Detection: ROAS Spikes & Drops")
if all(col in anomaly.columns for col in ['date', 'ROAS', 'Flag']):
    anomaly['date'] = pd.to_datetime(anomaly['date'])
    fig_anom = px.line(anomaly, x='date', y='ROAS', color='Flag', title="Anomaly Marked ROAS")
    st.plotly_chart(fig_anom)
else:
    st.warning("Required columns (date, ROAS, Flag) not found in anomaly_detected.csv")


# --- Clustering (KMeans) ---
st.subheader("🧩 Influencer Clustering")
features = influencers[['follower_count']].copy()
scaler = StandardScaler()
scaled = scaler.fit_transform(features)
model = KMeans(n_clusters=3, random_state=42)
influencers['cluster'] = model.fit_predict(scaled)
fig_clust = px.scatter(influencers, x='follower_count', y='id', color='cluster', title="Influencer Clusters")
st.plotly_chart(fig_clust)

# --- File Uploader for Real-Time Dataset Test ---
st.subheader("📤 Upload Your Own Influencer Data")
uploaded_file = st.file_uploader("Upload influencers.csv for testing", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Your Uploaded Dataset:")
    st.dataframe(df.head())

# --- PDF Export ---
st.subheader("📄 Export Insights PDF")
pdf_path = "HealthKart_Insights_Report_Clean.pdf"
with open(pdf_path, "rb") as f:
    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
pdf_link = f'<a href="data:application/pdf;base64,{base64_pdf}" download="HealthKart_Insights_Report_Clean.pdf">📥 Download Final PDF Report</a>'
st.markdown(pdf_link, unsafe_allow_html=True)

# Footer
st.caption("Built by Disha Ransingh · Internship Project for HealthKart")


