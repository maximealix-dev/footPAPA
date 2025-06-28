import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier

API_KEY = "f21f112db4167d2a492997e44821921b"
HEADERS = {"X-Auth-Token": API_KEY}
BASE_URL = "https://api-football.com/demo/api/v2"  # Demo, limité

def get_teams(league_id, season):
    url = f"{BASE_URL}/teams/league/{league_id}/season/{season}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        return []
    data = r.json()
    teams = data.get('api', {}).get('teams', {})
    return [team['name'] for team in teams.values()]

def get_fixtures(league_id, season):
    url = f"{BASE_URL}/fixtures/league/{league_id}/season/{season}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        return pd.DataFrame()
    data = r.json()
    fixtures = []
    for fix_id, fix in data.get('api', {}).get('fixtures', {}).items():
        home = fix.get('homeTeam', '')
        away = fix.get('awayTeam', '')
        goals_home = fix.get('goalsHomeTeam', None)
        goals_away = fix.get('goalsAwayTeam', None)
        if None in (goals_home, goals_away):
            continue
        fixtures.append({'Team1': home, 'Team2': away, 'Team1_goals': goals_home, 'Team2_goals': goals_away})
    return pd.DataFrame(fixtures)

st.title("⚽ Prédicteur Foot Simple")

league_id = st.number_input("ID Ligue (ex: 61 pour Ligue 1)", min_value=1, value=61)
season = st.number_input("Saison (ex: 2023)", min_value=2000, max_value=2100, value=2023)

teams = get_teams(league_id, season)
if not teams:
    st.warning("Pas d'équipes trouvées. Vérifie ID ligue et saison.")
    st.stop()

fixtures = get_fixtures(league_id, season)
if fixtures.empty:
    st.warning("Pas de matchs trouvés pour cette ligue et saison.")
    st.stop()

fixtures['Result'] = fixtures.apply(
    lambda r: 'Team1' if r['Team1_goals'] > r['Team2_goals']
    else 'Team2' if r['Team1_goals'] < r['Team2_goals']
    else 'Draw', axis=1)

features = pd.get_dummies(fixtures[['Team1', 'Team2']])
target = fixtures['Result']

model = RandomForestClassifier(random_state=42)
model.fit(features, target)

team1 = st.selectbox("Équipe à domicile", teams)
team2 = st.selectbox("Équipe à l'extérieur", [t for t in teams if t != team1])

if st.button("Prédire"):
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

    st.write(f"Prédiction: {prediction}")
    for lab, p in zip(labels, proba):
        st.write(f"{lab}: {p*100:.1f}%")
