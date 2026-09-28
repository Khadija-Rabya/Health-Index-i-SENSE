"""
Tableau de bord temps reel — prediction du health_index i-SENSE a t+n, pour chaque
horizon benchmarque (10 min / 1h / 3h / 24h / 3j / 1 semaine) et chaque machine, avec le
modele le plus precis reellement mesure pour ce cas (cf.
rapport_prediction_health_index_comparaison_finale.md et realtime/artifacts.py).

Deux methodes possibles par cellule, lues depuis manifest.json :
  - "random_forest" : un modele entraine bat reellement la reference a cet horizon ;
  - "persistence"   : AUCUN modele entraine ne bat la simple reconduction de la valeur
    courante. Ce n'est pas un modele concurrent, c'est le PLANCHER de reference — la
    valeur est maintenue telle quelle et l'interface l'annonce explicitement
    (cf. hi_forecast/REPORT.md : aux horizons longs, tous les modeles entraines ont un
    R2 negatif, donc font moins bien que de ne rien predire).

Lancement : streamlit run realtime/dashboard.py
Pre-requis : python realtime/artifacts.py (une fois, genere realtime/artifacts/).
"""

import os
import sys
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api_client import fetch_recent_wide  # noqa: E402
from feature_pipeline import ASSET_COLS, build_live_snapshot, load_artifacts  # noqa: E402
from predict import predict_all_horizons  # noqa: E402

REFRESH_MS = 120_000  # 2 minutes, cf. plan valide avec l'utilisateur

STATE_COLOR = {"Normal": "#38a169", "Surveillance": "#dd6b20", "Alarme": "#e53e3e"}
ASSET_COLOR = {"Motosoufflante A": "#2b6cb0", "Motosoufflante B": "#dd6b20"}
# "persistence" n'est pas un modele concurrent mais l'absence de prevision exploitable :
# on l'affiche comme telle plutot que comme un nom de modele.
MODEL_LABEL = {
    "random_forest": "Random Forest",
    "persistence": "Aucune prévision fiable — valeur maintenue",
}

st.set_page_config(page_title="Health Index i-SENSE — temps réel", layout="wide")


@st.cache_resource(show_spinner=False)
def get_artifacts():
    return load_artifacts()


@st.cache_data(ttl=REFRESH_MS // 1000 - 5, show_spinner="Récupération des données i-SENSE…")
def get_snapshot(_artifacts):
    wide, fetched_at = fetch_recent_wide()
    if wide.empty:
        raise RuntimeError("L'API i-SENSE n'a retourné aucune mesure sur la fenêtre récente.")
    snapshot = build_live_snapshot(wide, _artifacts)
    predictions = predict_all_horizons(snapshot, _artifacts)
    return snapshot, predictions, fetched_at


def state_badge(state):
    color = STATE_COLOR.get(state, "#718096")
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:10px;font-weight:600;font-size:0.85em">{state}</span>'


def render_machine(asset, snapshot, data):
    st.subheader(asset)

    current = data["current_health_index"]
    state = data["current_state"]
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Health Index actuel", f"{current:.3f}")
        st.markdown(state_badge(state), unsafe_allow_html=True)
        st.caption(f"À {data['current_time']:%Y-%m-%d %H:%M} · état calculé via PCA T²/SPE + Isolation Forest "
                   "(même méthode que le reste du projet)")

    with c2:
        rows = []
        for h in data["horizons"]:
            rows.append({
                "Horizon": h["horizon_label"],
                "À": h["target_time"].strftime("%Y-%m-%d %H:%M"),
                "Health Index prédit": f"{h['predicted_health_index']:.3f}",
                "État prédit": h["predicted_state"],
                "Méthode": MODEL_LABEL.get(h["model"], h["model"]),
                "R² historique": f"{h['historical_r2']:.2f}",
            })
        table = pd.DataFrame(rows)

        def highlight_state(val):
            return f"background-color: {STATE_COLOR.get(val, '#fff')}22"

        def highlight_r2(val):
            r2 = float(val)
            return "color:#e53e3e" if r2 <= 0 else ("color:#dd6b20" if r2 < 0.5 else "")

        styled = table.style.map(highlight_state, subset=["État prédit"]).map(highlight_r2, subset=["R² historique"])
        st.dataframe(styled, hide_index=True, width="stretch")
        st.caption(
            "**« Aucune prévision fiable — valeur maintenue »** : à cet horizon et sur cette machine, "
            "aucun modèle entraîné ne fait mieux que reconduire la valeur actuelle du health index. "
            "La ligne affiche donc le health index courant, inchangé — ce n'est pas une prévision, "
            "c'est l'aveu qu'il n'y en a pas d'exploitable. "
            "\n\nÉtat prédit estimé à partir de la seule valeur de health_index prédite (les capteurs futurs "
            "n'existent pas encore) — pas directement comparable à l'état actuel, calculé différemment. "
            "R² historique ≤ 0 : la méthode fait moins bien que de prédire une constante "
            "(cf. rapport_prediction_health_index_comparaison_finale.md et hi_forecast/REPORT.md).")

    # --- Trajectoire : historique recent + points predits ---
    col = ASSET_COLS[asset]
    hist = snapshot.loc[snapshot[col] == 1].sort_values("created_at")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["created_at"], y=hist["health_index"], mode="lines", name="Historique (fenêtre récente)",
        line=dict(color=ASSET_COLOR[asset], width=2),
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>health_index=%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[h["target_time"] for h in data["horizons"]],
        y=[h["predicted_health_index"] for h in data["horizons"]],
        mode="markers+text", name="Prédictions t+n",
        text=[h["horizon_label"] for h in data["horizons"]],
        textposition="top center",
        marker=dict(size=10, color=[STATE_COLOR[h["predicted_state"]] for h in data["horizons"]],
                    line=dict(width=2, color="white")),
        hovertemplate="%{text}<br>%{x|%Y-%m-%d %H:%M}<br>health_index=%{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color="#a0aec0", annotation_text="repère 0.5")
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title=None, yaxis_title="health_index (1 = sain)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
    )
    st.plotly_chart(fig, width="stretch")


def main():
    st.title("Health Index i-SENSE — prédiction temps réel à t+n")
    st.caption("Motosoufflante A & B · méthode la plus précise mesurée pour chaque horizon. "
               "Les horizons où aucun modèle ne bat la reconduction de la valeur courante sont "
               "signalés comme tels plutôt que présentés comme une prévision "
               "(cf. rapport_prediction_health_index_comparaison_finale.md).")

    st_autorefresh(interval=REFRESH_MS, key="autorefresh")

    with st.sidebar:
        st.header("Contrôles")
        if st.button("Rafraîchir maintenant", width="stretch"):
            get_snapshot.clear()
            st.rerun()
        st.caption(f"Rafraîchissement automatique toutes les {REFRESH_MS // 60000} minutes.")

    artifacts = get_artifacts()

    try:
        snapshot, predictions, fetched_at = get_snapshot(artifacts)
        st.session_state["last_good"] = (snapshot, predictions, fetched_at)
        st.sidebar.success(f"Dernière mise à jour : {fetched_at:%Y-%m-%d %H:%M:%S}")
    except Exception as exc:
        if "last_good" in st.session_state:
            snapshot, predictions, fetched_at = st.session_state["last_good"]
            st.error(f"Échec de la récupération/traitement des données live ({exc}). "
                     f"Affichage du dernier snapshot valide ({fetched_at:%Y-%m-%d %H:%M:%S}).")
        else:
            st.error(f"Échec de la récupération/traitement des données live ({exc}). "
                     "Aucun snapshot précédent disponible.")
            return

    col_a, col_b = st.columns(2)
    for column, asset in zip([col_a, col_b], ASSET_COLS):
        with column:
            if asset in predictions:
                render_machine(asset, snapshot, predictions[asset])
            else:
                st.warning(f"Aucune donnée récente pour {asset} sur la fenêtre récupérée.")


if __name__ == "__main__":
    main()
