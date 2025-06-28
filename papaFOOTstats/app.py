import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier

API_KEY = "f21f112db4167d2a492997e44821921b"
HEADERS = {"X-Auth-Token": API_KEY}
BASE_URL = "https://api-football.com/demo/api/v2"  # change en prod si possible

def get_leagues():
    url = f"{BASE_URL}/leagues"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        return {}
    data = r.json()
    leagues = data.get('api', {}).get('leagues', [])
    return {l['league_id']: l['name'] for l in leagues if l.get('seasons')}

def get_teams(league_id, season):
    url = f"{BASE_URL}/teams/league/{league_id}/season/{season}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        return []
    data = r.json()
    if data.get('api'):
        teams = data['api'].get('teams', {})
        return [team['name'] for team in teams.values()]
    return []

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
        date = fix.get('event_date', '')[:10]
        if None in (goals_home, goals_away):
            continue
        fixtures.append({
            'Team1': home,
            'Team2': away,
            'Team1_goals': goals_home,
            'Team2_goals': goals_away,
            'Date': date
        })
    return pd.DataFrame(fixtures)

st.title("⚽ MAXFOOT AI - Prédicteur Multi-Ligue et Multi-Saison")

leagues = get_leagues()
if not leagues:
    st.error("Erreur récupération des ligues. Vérifie ta clé API.")
    st.stop()

league_id = st.selectbox("Choisis la ligue", options=list(leagues.keys()), format_func=lambda x: leagues[x])
season = st.selectbox("Choisis la saison", options=[2024, 2023, 2022, 2021])

teams = get_teams(league_id, season)
if not teams:
    st.error("Aucune équipe trouvée pour cette ligue/saison.")
    st.stop()

fixtures = get_fixtures(league_id, season)
if fixtures.empty:
    st.error("Aucun match trouvé pour cette ligue/saison.")
    st.stop()

# Crée la colonne résultat
fixtures['Result'] = fixtures.apply(
    lambda row: 'Team1' if row['Team1_goals'] > row['Team2_goals']
    else 'Team2' if row['Team1_goals'] < row['Team2_goals']
    else 'Draw', axis=1)

# Prépare les features (one-hot)
features = pd.get_dummies(fixtures[['Team1', 'Team2']])
target = fixtures['Result']

model = RandomForestClassifier(random_state=42)
model.fit(features, target)

team1 = st.selectbox("🏠 Équipe à domicile :", teams)
team2 = st.selectbox("🚗 Équipe à l'extérieur :", teams)

if team1 == team2:
    st.warning("Choisis deux équipes différentes !")
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
