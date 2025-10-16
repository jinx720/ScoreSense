import streamlit as st
import pandas as pd
import joblib
import random
import plotly.graph_objects as go

# --- Load saved models ---
clf = joblib.load(r"C:\Users\KIIT\Documents\ScoreSense\notebooks\models\clf.pkl")
xgb_away = joblib.load(r"C:\Users\KIIT\Documents\ScoreSense\notebooks\models\xgb_away.pkl")
xgb_home = joblib.load(r"C:\Users\KIIT\Documents\ScoreSense\notebooks\models\xgb_home.pkl")

# --- Load final processed DataFrame ---
df = joblib.load("data/final_df.pkl")

# --- Page setup ---
st.set_page_config(page_title="ScoreSense", layout="centered")
st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stSelectbox, .stButton > button {
        border-radius: 8px;
    }
    .score-card {
        background: #1e293b;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        text-align: center;
        margin-top: 20px;
        border: 1px solid #334155;
    }
    .form-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.85rem;
        color: white;
        margin-right: 4px;
    }
    .W { background-color: #16a34a; }
    .D { background-color: #eab308; }
    .L { background-color: #dc2626; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---
st.markdown("<h1 style='text-align:center; font-weight:600;'>ScoreSense</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94a3b8; font-size:0.95rem;'>Premier League Match Outcome & Scoreline Prediction</p>", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("Match Configuration")
home_team = st.sidebar.selectbox("Home Team", df['HomeTeam'].unique())
away_team = st.sidebar.selectbox("Away Team", df['AwayTeam'].unique())

if st.sidebar.button("Random Match"):
    home_team, away_team = random.sample(list(df['HomeTeam'].unique()), 2)
    st.sidebar.write(f"Selected: {home_team} vs {away_team}")

# --- Prediction section ---
st.markdown(f"<h3 style='text-align:center; font-weight:500;'>{home_team} vs {away_team}</h3>", unsafe_allow_html=True)

if st.button("Generate Prediction"):
    home_id = df[df['HomeTeam'] == home_team]['HomeTeamID'].iloc[0]
    away_id = df[df['AwayTeam'] == away_team]['AwayTeamID'].iloc[0]

    home_last = df[df['HomeTeam'] == home_team].sort_values('Date').iloc[-1]
    away_last = df[df['AwayTeam'] == away_team].sort_values('Date').iloc[-1]

    new_match = pd.DataFrame([{
        'HomeTeamID': home_id,
        'AwayTeamID': away_id,
        'HomeGoalsLast3': home_last['HomeGoalsLast3'],
        'AwayGoalsLast3': away_last['AwayGoalsLast3'],
        'DaysSinceHomeLast': home_last['DaysSinceHomeLast'],
        'DaysSinceAwayLast': away_last['DaysSinceAwayLast'],
        'B365H_Prob': home_last['B365H_Prob'],
        'B365D_Prob': home_last['B365D_Prob'],
        'B365A_Prob': home_last['B365A_Prob']
    }])

    probs = clf.predict_proba(new_match)[0]
    pred_home = round(xgb_home.predict(new_match)[0])
    pred_away = round(xgb_away.predict(new_match)[0])

    # --- Styled Result Card ---
    st.markdown(
        f"""
        <div class="score-card">
            <h2 style="color:#38bdf8; font-weight:500; margin-bottom:12px;">Predicted Score</h2>
            <h3 style="font-weight:400;">{home_team} {pred_home} - {pred_away} {away_team}</h3>
            <p style="margin-top:12px; color:#94a3b8;">Model Confidence: <b>{max(probs)*100:.1f}%</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Probability Chart with Plotly ---
    st.subheader("Win Probability Distribution")
    
    prob_data = {
        'Home Win': probs[2] * 100,
        'Draw': probs[1] * 100,
        'Away Win': probs[0] * 100
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(prob_data.keys()),
            y=list(prob_data.values()),
            text=[f'{v:.1f}%' for v in prob_data.values()],
            textposition='outside',
            marker_color=['#3b82f6', '#eab308', '#10b981'],
            hovertemplate='%{x}<br>%{y:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f1f5f9', size=12),
        yaxis=dict(
            title='Probability (%)',
            gridcolor='#334155',
            range=[0, max(prob_data.values()) * 1.15]
        ),
        xaxis=dict(title=''),
        margin=dict(t=20, b=40, l=40, r=40),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- Recent Form ---
    st.subheader("Recent Form (Last 5 Matches)")

    def form_html(team):
        team_df = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].sort_values('Date').tail(5)
        form = []
        for _, row in team_df.iterrows():
            if row['HomeTeam'] == team:
                res = 'W' if row['FTR'] == 'H' else ('D' if row['FTR'] == 'D' else 'L')
            else:
                res = 'W' if row['FTR'] == 'A' else ('D' if row['FTR'] == 'D' else 'L')
            form.append(f"<span class='form-badge {res}'>{res}</span>")
        return ''.join(form)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{home_team}:** {form_html(home_team)}", unsafe_allow_html=True)
    with col2:
        st.markdown(f"**{away_team}:** {form_html(away_team)}", unsafe_allow_html=True)