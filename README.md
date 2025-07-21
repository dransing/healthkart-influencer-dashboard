# HealthKart Influencer Campaign Dashboard

A Streamlit-based dashboard that analyzes the ROI of influencer campaigns run by HealthKart across Instagram and YouTube.

---

## Features
- View campaign performance by influencer and platform
- ROAS (Return on Ad Spend) calculation per influencer
- Total orders, revenue, and payout summaries
- Platform filtering (Instagram, YouTube)

---

## Datasets Used (Simulated)
- `influencers.csv` – Influencer metadata
- `posts.csv` – Post engagement data
- `tracking_data.csv` – User orders and revenue
- `payouts.csv` – Payout type and amounts

---

## Metrics Tracked
- **ROAS** = Revenue ÷ Total Payout
- Total Orders, Total Revenue, Total Payouts
- Platform-wise filtering and influencer-level performance

---

## How to Run This Locally

1. Clone this repository:
```bash
git clone https://github.com/dransing/healthkart-influencer-dashboard
```

2. Install requirements:
```bash
pip install streamlit pandas matplotlib
```

3. Run the app:
```bash
streamlit run app.py
```
---

## License

This project is licensed under the MIT License.

Then visit `http://localhost:8501` in your browser.


