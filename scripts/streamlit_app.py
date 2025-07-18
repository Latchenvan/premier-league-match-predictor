
import streamlit as st
import pandas as pd
import pickle
import datetime

# Load model
model = pickle.load(open("/Users/tobe2.0/Portfolio/premier league match predictor/data/xgb_model.pkl", "rb"))
# Load team form data
form_df = pd.read_csv("/Users/tobe2.0/Portfolio/premier league match predictor/data/team_form_stats.csv")

st.title("⚽ EPL Match Result Predictor")

st.markdown("Predict the outcome of a Premier League match using pre-match team form features and XGBoost.")

# Team selection
teams = sorted(form_df['Team'].unique())
home_team = st.selectbox("🏠 Home Team", teams)
away_team = st.selectbox("🚗 Away Team", [t for t in teams if t != home_team])

# Prediction button
if st.button("🔮 Predict Match Result"):
    # Get latest form for selected teams
    home_row = form_df[form_df['Team'] == home_team].iloc[-1]
    away_row = form_df[form_df['Team'] == away_team].iloc[-1]

    # Build input feature
    features = pd.DataFrame([{
        'home_avg_goals_for': home_row['avg_goals_for'],
        'home_avg_goals_against': home_row['avg_goals_against'],
        'home_avg_points': home_row['avg_match_points'],
        'away_avg_goals_for': away_row['avg_goals_for'],
        'away_avg_goals_against': away_row['avg_goals_against'],
        'away_avg_points': away_row['avg_match_points'],
        'Month': datetime.datetime.today().month,
        'Weekday': datetime.datetime.today().weekday()
    }])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    label_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
    st.success(f"🏁 **Predicted Result:** {label_map[prediction]}")

    st.subheader("📊 Prediction Confidence")
    st.bar_chart(pd.Series(probabilities, index=["Away Win", "Draw", "Home Win"]))
