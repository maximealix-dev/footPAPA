import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier

API_KEY = "f21f112db4167d2a492997e44821921b"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

def get_teams(league_id, season):
    url = f"{BASE_URL}/teams?league={league_id}&season={season}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        return []
    data = r.json()
    return [team['team']['name'] for team in data.get('response', [])]

def get_fixtures(league_id, season):
    url = f"{BASE_URL}/fixtures?league={league_id}&season={season}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
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
        fixtures.append({'Team1': home, 'Team2': away, 'Team1_goals': goals_home, 'Team2_goals': goals_away})
    return pd.DataFrame(fixtures)

st.title("Prédicteur Match Foot basique")

league_id = st.number_input("ID Ligue (ex: 61 pour Ligue 1)", min_value=1, value=61)
season = st.number_input("Saison (ex: 2023)", min_value=2000, max_value=2100, value=2023)

teams = get_teams(league_id, season)
if not teams:
    st.warning("Pas d'équipes trouvées")
    st.stop()

fixtures = get_fixtures(league_id, season)
if fixtures.empty:
    st.warning("Pas de données matchs")
    st.stop()

fixtures['Result'] = fixtures.apply(lambda r:
    'Team1' if r['Team1_goals'] > r['Team2_goals']
    else 'Team2' if r['Team1_goals'] < r['Team2_goals']
    else 'Draw', axis=1)

features = pd.get_dummies(fixtures[['Team1', 'Team2']])
target = fixtures['Result']

model = RandomForestClassifier(random_state=42)
model.fit(features, target)

team1 = st.selectbox("Équipe domicile", teams)
team2 = st.selectbox("Équipe extérieur", [t for t in teams if t != team1])

if st.button("Prédire"):
    input_df = pd.DataFrame(columns=features.columns)
    input_df.loc[0] = 0
    c1 = f"Team1_{team1}"
    c2 = f"Team2_{team2}"
    if c1 in input_df.columns:
        input_df.at[0, c1] = 1
    if c2 in input_df.columns:
        input_df.at[0, c2] = 1

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    labels = model.classes_

    st.write(f"Prédiction: {pred}")
    for l, p in zip(labels, proba):
        st.write(f"{l}: {p*100:.1f}%")
