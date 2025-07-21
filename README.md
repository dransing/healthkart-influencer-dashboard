# HealthKart Influencer Campaign ROI Dashboard

**Built by Disha Ransingh | Streamlit Dashboard | Internship Project @ HealthKart**

This interactive dashboard helps visualize, optimize, and track influencer marketing campaigns across platforms like Instagram and YouTube. It integrates data-driven insights, anomaly detection, sentiment analysis, and predictive modeling—all in one place to make influencer marketing measurable and strategic.

---

## Features

### Campaign Performance Tracker
- Total Orders, Revenue, Payouts, and ROAS (Return on Ad Spend)
- Platform, Gender, and Category filters
- Brand-wise ROAS comparison
- Daily ROAS trend over time

### Influencer Overview
- Interactive table of all filtered influencers
- ROAS by influencer as a color-coded bar chart

### Incremental ROAS Analysis
- Compare ROAS *before vs after* a selected campaign date
- Automatically calculates % increase or drop

### Smart Influencer Recommendations
- Highlights:
  - High performers (ROAS > 8) to invest more in
  - Underperformers (ROAS < 4) to reconsider

### Sentiment Analysis
- Histogram of caption sentiment (positive, neutral, negative)
- Based on NLP-processed post captions from `posts_with_sentiment.csv`

### Profit Estimation
- Net Profit and Margin by influencer
- Visualized with a Safe color bar plot

### Anomaly Detection
- Detect ROAS spikes or drops using statistical thresholds
- Line plot with color-coded anomaly markers

### Product-Platform Recommendations
- Identifies best-performing product-platform pairs based on conversion rates
- Helps brands decide where to advertise

### Predictive ROAS Forecasting
- Linear regression model to predict ROAS using:
  - Orders, Revenue, Follower count, Platform, Category
- Actual vs Predicted ROAS bar chart output

### Real-time CSV Upload
- Upload your own `influencers.csv` to test the dashboard dynamically

### One-click PDF Export
- Download your insights report as a clean, pre-generated PDF

---

## Files Included
HealthKart-Influencer-Dashboard/
│
├── app.py # Main Streamlit dashboard script
├── README.md # Dashboard documentation
├── HealthKart_Insights_Report_Clean.pdf # Final PDF insights report
├── influencers.csv # Influencer metadata (id, name, category, gender, follower_count, platform)
├── posts.csv # Post reach and captions (influencer_id, reach, caption, date)
├── payouts.csv # Payout details (influencer_id, basis, rate, total_payout, orders)
├── tracking_data_with_brand_FIXED.csv # Orders, revenue, brand tracking (influencer_id, product, date, orders, revenue, brand)
├── posts_with_sentiment.csv # Caption sentiment labels (caption, sentiment)
├── profit_estimation.csv # Profit and margin estimates (influencer_id, Net_Profit, Profit_Margin_%)
├── anomaly_detected.csv # ROAS anomaly data (date, ROAS, anomaly_score, is_anomaly)

---

## Tech Stack

- **Streamlit** – Interactive frontend  
- **Pandas** – Data processing  
- **Plotly Express** – Interactive charts  
- **Scikit-learn** – ML modeling for predictions  
- **Matplotlib** – Additional visuals  
- **Base64** – PDF encoding for download  

---

## About the Creator

**Disha Ransingh**  
Biomedical Health Sciences + Data Science @ Purdue University  
✉️ dransing@purdue.edu  
🔬 Focus: AI in healthcare, startup innovation, computational biology  

---

## Notes

- ✅ Dates must be in format `YYYY-MM-DD` in all CSVs.  
- ✅ Sample data is included for demonstration.  
- ✅ Final PDF is static, but can be customized with auto-export in future builds.  

---


## How to Run This Locally

1. Clone this repository:
```bash
git clone https://github.com/dransing/healthkart-influencer-dashboard
```

2. Install requirements:
```bash
pip install streamlit pandas matplotlib plotly scikit-learn
```

3. Run the app:
```bash
streamlit run app.py
```
---

## License

This project is licensed under the MIT License.

Then visit the deployed version here (optional):
 https://healthkart-influencer-dashboard-xh5dxof3pfyvwktrtdg3qc.streamlit.app/ in your browser


