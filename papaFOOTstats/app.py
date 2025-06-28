import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier

API_KEY = "f21f112db4167d2a492997e44821921b"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {
    "x-apisports-key": API_KEY
}

def get_leagues():
    url = f"{BASE_URL}/leagues"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        st.error(f"Erreur API ligues: {r.status_code}")
        return {}
    data = r.json()
    leagues = data.get('response', [])
    # On garde les ligues avec saisons
    return {l['league']['id']: l['league']['name'] for l in leagues}

def get_seasons(league_id):
    url = f"{BASE_URL}/leagues/seasons"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        return []
    data = r.json()
    # En général on récupère les saisons par ligue dans /leagues endpoint, mais ici on prend les saisons générales
    return data.get('response', [])

def get_teams(league_id, season):
    url = f"{BASE_URL}/teams?league={league_id}&season={season}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        st.error(f"Erreur API équipes: {r.status_code}")
        return []
    data = r.json()
    teams = data.get('response', [])
    return [team['team']['name'] for team in teams]

def get_fixtures(league_id, season):
    url = f"{BASE_URL}/fixtures?league={league_id}&season={season}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        st.error(f"Erreur API matchs: {r.status_code}")
        return pd.DataFrame()
    data = r.json()
    fixtures = []
    for fix in data.get('response', []):
        home = fix['teams']['home']['name']
        away = fix['teams']['away']['name']
        goals_home = fix['goals']['home']
        goals_away = fix['goals']['away']
        if goals_home is None or goals_away is None:
            continue
        fixtures.append({
            'Team1': home,
            'Team2': away,
            'Team1_goals': goals_home,
            'Team2_goals': goals_away,
        })
    return pd.DataFrame(fixtures)

st.title("⚽ MAXFOOT AI - Prédicteur Multi-Ligue Multi-Saison")

leagues = get_leagues()
if not leagues:
    st.stop()

league_id = st.selectbox("Choisis la ligue", options=list(leagues.keys()), format_func=lambda x: leagues[x])
seasons = list(range(2024, 2017, -1))  # Tu peux ajuster les années dispo

season = st.selectbox("Choisis la saison", options=seasons)

teams = get_teams(league_id, season)
if not teams:
    st.warning("Pas d’équipes pour cette ligue/saison.")
    st.stop()

fixtures = get_fixtures(league_id, season)
if fixtures.empty:
    st.warning("Pas de données de match pour cette ligue/saison.")
    st.stop()

fixtures['Result'] = fixtures.apply(
    lambda r: 'Team1' if r['Team1_goals'] > r['Team2_goals']
    else 'Team2' if r['Team1_goals'] < r['Team2_goals']
    else 'Draw', axis=1)

features = pd.get_dummies(fixtures[['Team1', 'Team2']])
target = fixtures['Result']

model = RandomForestClassifier(random_state=42)
model.fit(features, target)

team1 = st.selectbox("🏠 Équipe à domicile", teams)
team2 = st.selectbox("🚗 Équipe à l'extérieur", [t for t in teams if t != team1])

if st.button("🔮 Prédire"):
    input_data = pd.DataFrame(columns=features.columns)
    input_data.loc[0] = 0
    col1 = f"Team1_{team1}"
    col2 = f"Team2_{team2}"
    if col1 in input_data.columns:
        input_data.at[0, col1] = 1
    if col2 in input_data.columns:
        input_data.at[0, col2] = 1

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    labels = model.classes_

    st.subheader("Résultat prédit :")
    if prediction == 'Team1':
        st.success(f"🏆 {team1} a le plus de chances de gagner.")
    elif prediction == 'Team2':
        st.success(f"🏆 {team2} a le plus de chances de gagner.")
    else:
        st.info("⚖️ Match nul probable.")

    st.subheader("Probabilités :")
    for label, p in zip(labels, proba):
        name = team1 if label == 'Team1' else team2 if label == 'Team2' else "Match nul"
        st.write(f"{name}: {p*100:.2f}%")
