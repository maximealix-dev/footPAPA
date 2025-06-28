import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

API_KEY = "081ae98f42824c268ad824ea76387f0b"
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}

@st.cache_data(ttl=3600)
def get_teams():
    url = f"{BASE_URL}/competitions/FL1/teams"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        st.error(f"Erreur API équipes : {r.status_code}")
        return []
    data = r.json()
    return [team['name'] for team in data.get('teams', [])]

@st.cache_data(ttl=3600)
def get_fixtures():
    url = f"{BASE_URL}/competitions/FL1/matches?status=FINISHED&season=2023"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        st.error(f"Erreur API matchs : {r.status_code}")
        return pd.DataFrame()
    data = r.json()
    matches = []
    for match in data.get('matches', []):
        home = match['homeTeam']['name']
        away = match['awayTeam']['name']
        goals_home = match['score']['fullTime']['home']
        goals_away = match['score']['fullTime']['away']
        date = match['utcDate'][:10]
        if goals_home is None or goals_away is None:
            continue
        matches.append({
            'Team1': home,
            'Team2': away,
            'Team1_goals': goals_home,
            'Team2_goals': goals_away,
            'Date': date
        })
    return pd.DataFrame(matches)

st.title("PapaFoot - Prédicteur Ligue 1 (Alix,Seb,Ricky,Maxime,)")

teams = get_teams()
fixtures = get_fixtures()

if len(teams) == 0 or fixtures.empty:
    st.stop()

fixtures['Result'] = fixtures.apply(
    lambda row: 'Team1' if row['Team1_goals'] > row['Team2_goals']
    else 'Team2' if row['Team1_goals'] < row['Team2_goals']
    else 'Draw', axis=1)

features = pd.get_dummies(fixtures[['Team1', 'Team2']])
target = fixtures['Result']

model = RandomForestClassifier(random_state=42)
model.fit(features, target)

team1 = st.selectbox("🏠 Équipe à domicile :", teams)
team2 = st.selectbox("🚗 Équipe à l'extérieur :", teams)

if team1 == team2:
    st.warning("Choisis deux équipes différentes (marseille les meilleurs !")
    st.stop()

if st.button("🔮 Prédire le résultat"):
    input_data = pd.DataFrame(columns=features.columns)
    input_data.loc[0] = 0
    col_team1 = f"Team1_{team1}"
    col_team2 = f"Team2_{team2}"

    if col_team1 in input_data.columns:
        input_data.at[0, col_team1] = 1
    if col_team2 in input_data.columns:
        input_data.at[0, col_team2] = 1

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    labels = model.classes_

    st.subheader("📊 Résultat prédit :")
    if prediction == 'Team1':
        st.success(f"🏆 {team1} a plus de chances de gagner.")
    elif prediction == 'Team2':
        st.success(f"🏆 {team2} a plus de chances de gagner.")
    else:
        st.info("⚖️ Match nul probable.")

    st.subheader("🔢 Probabilités :")
    for label, p in zip(labels, proba):
        name = team1 if label == 'Team1' else team2 if label == 'Team2' else "Match nul"
        st.write(f"👉 {name} : **{p*100:.2f}%**")

st.subheader("📈 Forme récente des équipes")

for t in [team1, team2]:
    st.markdown(f"### 🔍 {t}")
    recent = fixtures[
        (fixtures['Team1'] == t) | (fixtures['Team2'] == t)
    ].sort_values('Date').tail(5).copy()

    def resultat(row):
        if (row['Team1'] == t and row['Team1_goals'] > row['Team2_goals']) or \
           (row['Team2'] == t and row['Team2_goals'] > row['Team1_goals']):
            return 'Victoire'
        elif (row['Team1'] == t and row['Team1_goals'] < row['Team2_goals']) or \
             (row['Team2'] == t and row['Team2_goals'] < row['Team1_goals']):
            return 'Défaite'
        else:
            return 'Nul'

    recent['Résultat'] = recent.apply(resultat, axis=1)

    buts_marques = []
    buts_encaisses = []
    dates = list(recent['Date'])

    for _, row in recent.iterrows():
        if row['Team1'] == t:
            buts_marques.append(row['Team1_goals'])
            buts_encaisses.append(row['Team2_goals'])
        else:
            buts_marques.append(row['Team2_goals'])
            buts_encaisses.append(row['Team1_goals'])

    df_buts = pd.DataFrame({
        'Date': dates,
        'Buts marqués': buts_marques,
        'Buts encaissés': buts_encaisses
    })

    fig = px.bar(df_buts, x='Date', y=['Buts marqués', 'Buts encaissés'],
                 title=f"Buts marqués et encaissés par {t} (5 derniers matchs)",
                 labels={'value': 'Nombre de buts', 'variable': 'Type de buts'})

    st.plotly_chart(fig)
