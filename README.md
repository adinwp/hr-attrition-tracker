# HR Attrition & Burnout Predictor 🏢

## Project Overview
Employee turnover is a significant cost center for any organization. This project is a machine learning-based web application designed to help HR departments transition from reactive replacement to proactive retention. By analyzing demographic data, workplace environment, and commute factors, the model predicts the probability of an employee resigning and allows HR leaders to simulate the impact of retention strategies.

**Live Application:** [Insert your Streamlit App URL here]

## Business Value
* **Proactive Intervention:** Identifies high-risk employees before they submit their resignation.
* **Policy Simulation:** Allows HR to test how changes in commute policies (e.g., remote work) or overtime reduction can lower attrition risk.
* **Data-Driven HR:** Replaces intuition with an XGBoost predictive model tailored to highlight critical burnout factors.

## Tech Stack
* **Language:** Python
* **Machine Learning:** XGBoost, Scikit-Learn
* **Data Manipulation:** Pandas, NumPy
* **Web Framework & UI:** Streamlit
* **Deployment:** Streamlit Community Cloud

## Key Features
1. **Interactive Dashboard:** A clean, user-friendly sidebar for HR to input employee profiles (Age, Department, Role, Commute Distance, Overtime, etc.).
2. **Real-time Prediction:** Instantly calculates the risk probability of resignation based on the inputs.
3. **Actionable Insights:** Categorizes risk levels (Low, Medium, Critical) and provides tailored recommendations for HR intervention.
4. **Custom Feature Engineering:** Incorporates realistic variables such as extreme commute distances and simulated toxic management scores to better reflect real-world burnout triggers.

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/hr-attrition-tracker.git](https://github.com/YourUsername/hr-attrition-tracker.git)
   cd hr-attrition-tracker
