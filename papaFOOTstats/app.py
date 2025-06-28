import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier

API_KEY = "f21f112db4167d2a492997e44821921b"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# Exemple : équipes populaires hardcodées (ou à étendre)
POPULAR_TEAMS = [
    "Paris Saint Germain", "Marseille", "Lyon", "Real Madrid", "FC Barcelona",
    "Manchester United", "Liverpool", "Bayern Munich", "Juventus", "Chelsea"
]

def get_team_stats(team_name):
    # Recherche l’équipe via endpoint /teams?search=
    url = f"{BASE_URL}/teams?search={team_name}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200 or not r.json().get("response"):
        return None
    team_info = r.json()["response"][0]["team"]
    # Retourne quelques stats basiques (tu peux étendre)
    return {
        "Nom": team_info["name"],
        "Code": team_info["code"],
        "Pays": team_info["country"],
        "Fondation": team_info["founded"],
        "Logo": team_info["logo"]
    }

# Pour la prédiction, on simule un dataset minimal
def build_training_data():
    data = {
        'Team1_Paris Saint Germain': [1,0,0,1,0,0],
        'Team2_Marseille': [1,0,1,0,0,1],
        'Result': ['Team1','Team2','Draw','Team1','Draw','Team2']
    }
    df = pd.DataFrame(data)
    features = df.drop("Result", axis=1)
    target = df["Result"]
    return features, target

st.title("Prédicteur simple avec stats équipes")

team1 = st.selectbox("Équipe à domicile", POPULAR_TEAMS)
team2 = st.selectbox("Équipe à l'extérieur", [t for t in POPULAR_TEAMS if t != team1])

if team1 and team2:
    st.markdown("### Stats des équipes sélectionnées")
    stats1 = get_team_stats(team1)
    stats2 = get_team_stats(team2)
    if stats1:
        st.image(stats1["Logo"], width=100)
        st.write(stats1)
    if stats2:
        st.image(stats2["Logo"], width=100)
        st.write(stats2)

    features, target = build_training_data()
    model = RandomForestClassifier(random_state=42)
    model.fit(features, target)

    if st.button("Prédire le résultat"):
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

        st.write(f"### Prédiction : {pred}")
        for l, p in zip(labels, proba):
            st.write(f"{l}: {p*100:.1f}%")
