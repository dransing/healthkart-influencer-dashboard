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
tracking['date'] = pd.to_datetime(tracking['date'])
data = pd.merge(tracking, payouts, on='influencer_id')
data['ROAS'] = data['revenue'] / data['total_payout']
data['brand'] = tracking['brand']

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

filtered_data['date'] = pd.to_datetime(filtered_data['date'])

# --- Dashboard Title ---
st.title(" HealthKart Influencer Campaign Dashboard ")

# --- KPI Metrics ---
total_orders = filtered_data['orders'].sum() if 'orders' in filtered_data.columns else 0
total_revenue = filtered_data['revenue'].sum() if 'revenue' in filtered_data.columns else 0
total_payouts = filtered_data['total_payout'].sum() if 'total_payout' in filtered_data.columns else 0
overall_roas = round(total_revenue / total_payouts, 2) if total_payouts else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Orders", total_orders)
col2.metric("Total Revenue", f"Rs. {total_revenue}")
col3.metric("Total Payouts", f"Rs. {total_payouts}")
col4.metric("Overall ROAS", overall_roas)

# --- Influencer Overview ---
st.subheader("Influencer Overview")
st.dataframe(filtered_influencers)

# --- ROAS by Influencer ---
st.subheader("ROAS by Influencer")
roas_chart = filtered_data.groupby('influencer_id')['ROAS'].mean().reset_index()
roas_chart = pd.merge(roas_chart, influencers[['id', 'name']], left_on='influencer_id', right_on='id')
fig = px.bar(roas_chart, x='name', y='ROAS', color='ROAS', color_continuous_scale='RdYlGn')
st.plotly_chart(fig)

# --- Brand-wise ROAS ---
st.subheader("🏷️ Brand-wise ROAS")
brand_roas = filtered_data.groupby('brand').agg({'revenue': 'sum', 'total_payout': 'sum'}).reset_index()
brand_roas['ROAS'] = brand_roas['revenue'] / brand_roas['total_payout']
fig_brand_roas = px.bar(brand_roas, x='brand', y='ROAS', color='ROAS', color_continuous_scale='Viridis', title="ROAS by Brand")
st.plotly_chart(fig_brand_roas)

# --- ROAS Over Time ---
st.subheader("📈 ROAS Trend Over Time")
roas_by_date = filtered_data.groupby('date')['ROAS'].mean().reset_index()
fig2 = px.line(roas_by_date, x='date', y='ROAS', title="ROAS Trend")
st.plotly_chart(fig2)

# --- ROI Simulator ---
st.subheader("🧮 ROI Simulator")
multiplier = st.slider("Adjust Payout Multiplier", 0.5, 2.0, 1.0)
simulated_payouts = payouts.copy()
simulated_payouts['adjusted_payout'] = simulated_payouts['total_payout'] * multiplier
simulated_total_payout = simulated_payouts['adjusted_payout'].sum()
simulated_roas = round(total_revenue / simulated_total_payout, 2) if simulated_total_payout else 0
st.metric("Simulated ROAS", simulated_roas)

# --- Incremental ROAS ---
st.subheader("📈 Incremental ROAS: Before vs After Campaign")
campaign_start = st.date_input("📅 Select Campaign Start Date", value=pd.to_datetime("2025-06-01"))
campaign_start = pd.to_datetime(campaign_start)
before_campaign = filtered_data[filtered_data['date'] < campaign_start]
after_campaign = filtered_data[filtered_data['date'] >= campaign_start]
before_roas = before_campaign['revenue'].sum() / before_campaign['total_payout'].sum() if before_campaign['total_payout'].sum() > 0 else 0
after_roas = after_campaign['revenue'].sum() / after_campaign['total_payout'].sum() if after_campaign['total_payout'].sum() > 0 else 0
roas_df = pd.DataFrame({'Period': ['Before Campaign', 'After Campaign'], 'ROAS': [before_roas, after_roas]})
fig_roas_compare = px.bar(roas_df, x='Period', y='ROAS', color='Period', title="ROAS Before vs After Campaign")
st.plotly_chart(fig_roas_compare)
if before_roas and after_roas:
    percent_change = round((after_roas - before_roas) / before_roas * 100, 1)
    direction = "increased 📈" if percent_change > 0 else "decreased 📉"
    st.markdown(f"**ROAS {direction} by {abs(percent_change)}% after the campaign started.**")

# --- Smart Recommendations ---
st.subheader("🧠 Smart Influencer Recommendations")
top_recommend = roas_chart[roas_chart['ROAS'] > 8]['name'].tolist()
drop_recommend = roas_chart[roas_chart['ROAS'] < 4]['name'].tolist()
if top_recommend:
    st.success(f"📈 Invest More in: {', '.join(top_recommend)}")
if drop_recommend:
    st.warning(f"📉 Consider Revising: {', '.join(drop_recommend)}")

# --- Sentiment Analysis ---
st.subheader("💬 Caption Sentiment Analysis")
if 'sentiment' in sentiment.columns:
    fig_sent = px.histogram(sentiment, x='sentiment', color='sentiment', title="Post Sentiment Distribution")
    st.plotly_chart(fig_sent)
else:
    st.warning("Sentiment column not found in posts_with_sentiment.csv")

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


# --- Product-Platform Matching ---
st.subheader("🎯 Product-Platform Recommendations")
conversion_df = pd.merge(tracking, influencers[['id', 'platform']], left_on='influencer_id', right_on='id')
posts_agg = posts.groupby('influencer_id')['reach'].sum().reset_index().rename(columns={'reach': 'total_reach'})
conversion_df = pd.merge(conversion_df, posts_agg, on='influencer_id', how='left')
summary = conversion_df.groupby(['product', 'platform']).agg({'orders': 'sum', 'total_reach': 'sum'}).reset_index()
summary['conversion_rate'] = summary['orders'] / summary['total_reach']
summary = summary.dropna().sort_values('conversion_rate', ascending=False)
st.dataframe(summary.head(10))
fig_reco = px.bar(summary.head(10), x='product', y='conversion_rate', color='platform', barmode='group', title="Top Product-Platform Conversion Combos")
st.plotly_chart(fig_reco)

# --- ROAS Forecasting ---
st.subheader("🔮 Predictive ROAS Forecasting")

# Merge all required info
model_df = pd.merge(tracking, payouts, on='influencer_id')
model_df = pd.merge(model_df, influencers[['id', 'platform', 'category', 'follower_count']], left_on='influencer_id', right_on='id')

# Check if required numeric columns exist
required_cols = ['orders', 'revenue', 'follower_count', 'total_payout']
missing_cols = [col for col in required_cols if col not in model_df.columns]

if missing_cols:
    st.warning(f"Missing required columns for forecasting: {', '.join(missing_cols)}")
else:
    # Proceed only if all required columns exist
    model_df['ROAS'] = model_df['revenue'] / model_df['total_payout']
    model_df.dropna(subset=['ROAS'], inplace=True)

    # Encode categorical variables
    model_df = pd.get_dummies(model_df, columns=['platform', 'category'], drop_first=True)

    # Define features and target
    feature_cols = ['orders', 'revenue', 'follower_count'] + [col for col in model_df.columns if col.startswith('platform_') or col.startswith('category_')]
    X = model_df[feature_cols]
    y = model_df['ROAS']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test set
    predictions = model.predict(X_test)

    # Show results
    results_df = pd.DataFrame({'Actual ROAS': y_test, 'Predicted ROAS': predictions}).reset_index(drop=True)
    st.dataframe(results_df.head(10))

    # Visualize
    fig_pred = px.scatter(results_df, x='Actual ROAS', y='Predicted ROAS', trendline="ols", title="Actual vs Predicted ROAS")
    st.plotly_chart(fig_pred)


# --- File Uploader ---
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

# --- Footer ---
st.caption("Built by Disha Ransingh · Internship Project for HealthKart")




