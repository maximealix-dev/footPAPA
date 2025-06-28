import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

API_KEY = "f21f112db4167d2a492997e44821921b"
HEADERS = {"X-Auth-Token": API_KEY}
BASE_URL = "https://api-football.com/demo/api/v2"  # Modifie si prod

# Récupérer les ligues disponibles
def get_leagues():
    url = f"{BASE_URL}/leagues"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        return {}
    data = r.json()
    leagues = data.get('api', {}).get('leagues', [])
    # Retourne dict id: nom ligue, uniquement celles avec saisons dispo
    return {l['league_id']: l['name'] for l in leagues if l.get('seasons')}

# Récupérer les saisons dispo pour une ligue donnée
def get_seasons(league_id):
    url = f"{BASE_URL}/leagues/id/{league_id}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        return []
    data = r.json()
    seasons = data.get('api', {}).get('leagues', [])
    if seasons:
        return seasons[0].get('seasons', [])
    return []

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

# Prototype blessures statique (à améliorer)
blessures = {
    "Paris Saint-Germain": ["Neymar", "Mbappé"],
    "Marseille": ["Dieng"],
}

def check_blessures(equipe):
    return blessures.get(equipe, [])

def conseil_pari(prediction, proba, team1, team2):
    bless_team1 = check_blessures(team1)
    bless_team2 = check_blessures(team2)

    message = ""
    if proba[prediction] > 0.6:
        message = f"Pari conseillé : {prediction} probable."
    elif prediction == "Draw" and proba[prediction] > 0.5:
        message = "Pari conseillé : match nul possible."
    else:
        message = "Pari risqué, prudence."

    if bless_team1 or bless_team2:
        message += "\n⚠️ Blessures détectées chez "
        if bless_team1:
            message += f"{team1} ({', '.join(bless_team1)}) "
        if bless_team2:
            message += f"{team2} ({', '.join(bless_team2)})"
        message += " → reconsidère ton pari !"

    return message

# Streamlit app
st.title("⚽Papa FOOT IA - Multi-Ligue Multi-Saison")

leagues = get_leagues()
if not leagues:
    st.error("Impossible de récupérer les ligues. Vérifie ta clé API.")
    st.stop()

league_id = st.selectbox("Choisis la ligue", options=list(leagues.keys()), format_func=lambda x: leagues[x])
seasons = get_seasons(league_id)
if not seasons:
    st.error("Impossible de récupérer les saisons.")
    st.stop()

season = st.selectbox("Choisis la saison", options=seasons, format_func=lambda x: str(x))

with st.spinner("Chargement des équipes..."):
    teams = get_teams(league_id, season)
if not teams:
    st.error("Aucune équipe trouvée pour cette saison/ligue.")
    st.stop()

with st.spinner("Chargement des matchs..."):
    fixtures = get_fixtures(league_id, season)
if fixtures.empty:
    st.error("Aucun match trouvé pour cette saison/ligue.")
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

    conseil = conseil_pari(prediction, proba, team1, team2)
    st.warning(conseil)
