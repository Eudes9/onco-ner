# streamlit/app.py
"""
Interface Streamlit pour onco-ner.
Extraction et normalisation d'entités oncologiques depuis des textes cliniques.
"""

import requests
import streamlit as st

# --- Configuration ---
API_URL = "http://localhost:8000"

LABEL_COLORS = {
    "morphologie": "#FF6B6B",      # rouge
    "topographie": "#4ECDC4",      # turquoise
    "differenciation": "#45B7D1",  # bleu
    "expression_CIM": "#96CEB4",   # vert
}

LABEL_NAMES = {
    "morphologie": "Morphologie",
    "topographie": "Topographie",
    "differenciation": "Différenciation",
    "expression_CIM": "Expression CIM",
}

# --- Page config ---
st.set_page_config(
    page_title="onco-ner — Extraction d'entités oncologiques",
    page_icon="🏥",
    layout="wide",
)

# --- Header ---
st.title("🏥 onco-ner")
st.markdown(
    "**Extraction et normalisation d'entités oncologiques** "
    "depuis des textes cliniques en français."
)
st.divider()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Paramètres")

    confidence_threshold = st.slider(
        "Seuil de confiance minimum",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.05,
        help="Les entités avec un score inférieur à ce seuil seront masquées.",
    )

    min_length = st.slider(
        "Longueur minimale de l'entité (caractères)",
        min_value=1,
        max_value=20,
        value=4,
        step=1,
        help="Filtre les entités trop courtes comme 'du', 'la', 'un'.",
    )

    fuzzy = st.checkbox(
        "Matching approximatif (fuzzy)",
        value=True,
        help="Activer la recherche approximative pour la normalisation ICD-O.",
    )

    fuzzy_threshold = st.slider(
        "Seuil fuzzy match",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.05,
        help="Seuil de similarité pour la normalisation ICD-O.",
    ) if fuzzy else 0.8

    st.divider()
    st.subheader("🎨 Légende")
    for label, color in LABEL_COLORS.items():
        st.markdown(
            f"<span style='background-color:{color}; padding: 2px 8px; "
            f"border-radius: 4px; color: white; font-weight: bold;'>"
            f"{LABEL_NAMES[label]}</span>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Health check
    try:
        resp = requests.get(f"{API_URL}/", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            st.success("✅ API connectée")
            st.caption(f"Modèle : `{data['model'].split('/')[-1]}`")
            st.caption(
                f"Normalizer : {'✅' if data['normalizer'] else '❌'}"
            )
        else:
            st.error("❌ API non disponible")
    except Exception:
        st.error("❌ API non disponible — lancez l'API d'abord")
        st.code("uvicorn api.main:app --port 8000")

# --- Zone de texte ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Texte clinique")
    text_input = st.text_area(
        "Entrez un texte clinique en français",
        height=200,
        placeholder=(
            "Ex: Patient présentant un carcinome canalaire infiltrant "
            "du sein gauche, stade T2N1M0, grade SBR II..."
        ),
        label_visibility="collapsed",
    )

    analyze_btn = st.button(
        "🔍 Analyser",
        type="primary",
        use_container_width=True,
        disabled=not text_input.strip(),
    )

# --- Analyse ---
if analyze_btn and text_input.strip():
    try:
        with st.spinner("Analyse en cours..."):
            response = requests.post(
                f"{API_URL}/predict",
                json={
                    "text": text_input,
                    "fuzzy": fuzzy,
                    "fuzzy_threshold": fuzzy_threshold,
                },
                timeout=120,
            )

        if response.status_code == 200:
            result = response.json()
            entities = result["entities"]

            # Filtrer par seuil de confiance ET longueur minimale
            filtered_entities = [
                e for e in entities
                if e["score"] >= confidence_threshold
                and len(e["text"].strip()) >= min_length
            ]
            filtered_out = len(entities) - len(filtered_entities)

            st.divider()

            # --- Métriques ---
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Entités détectées", result["n_entities"])
            with col_m2:
                st.metric("Après filtrage", len(filtered_entities))
            with col_m3:
                st.metric("Filtrées (bruit)", filtered_out)
            with col_m4:
                avg_score = (
                    round(
                        sum(e["score"] for e in filtered_entities)
                        / len(filtered_entities),
                        3,
                    )
                    if filtered_entities else 0
                )
                st.metric("Score moyen", avg_score)

            st.divider()

            # --- Texte surligné ---
            st.subheader("📄 Texte annoté")

            if filtered_entities:
                # Trier les entités par position de début
                sorted_entities = sorted(
                    filtered_entities, key=lambda e: e["start"]
                )

                # Construire le HTML avec surlignage
                html = ""
                last_end = 0
                text = result["text"]

                for entity in sorted_entities:
                    start = entity["start"]
                    end = entity["end"]
                    label = entity["label"]
                    score = entity["score"]
                    icdo = entity.get("icdo_code") or ""
                    color = LABEL_COLORS.get(label, "#CCCCCC")
                    label_name = LABEL_NAMES.get(label, label)

                    # Texte avant l'entité
                    html += text[last_end:start]

                    # Badge ICD-O si disponible
                    icdo_badge = (
                        f"<span style='font-size:0.7em; margin-left:4px; "
                        f"background-color: rgba(0,0,0,0.2); "
                        f"padding: 1px 4px; border-radius: 3px;'>"
                        f"{icdo}</span>"
                        if icdo else ""
                    )

                    # Entité surlignée
                    html += (
                        f"<mark style='background-color:{color}; "
                        f"padding: 2px 6px; border-radius: 4px; "
                        f"color: white; font-weight: bold;'>"
                        f"{text[start:end]}"
                        f"<sup style='font-size:0.65em; margin-left:3px; "
                        f"opacity:0.9;'>{label_name}</sup>"
                        f"{icdo_badge}"
                        f"</mark>"
                    )
                    last_end = end

                # Texte après la dernière entité
                html += text[last_end:]

                st.markdown(
                    f"<div style='line-height: 2.5; font-size: 1.05em; "
                    f"padding: 20px; background-color: #f8f9fa; "
                    f"border-radius: 8px; border: 1px solid #dee2e6;'>"
                    f"{html}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info(
                    "Aucune entité détectée au-dessus du seuil de confiance "
                    "et de longueur minimum. "
                    "Essayez de baisser les seuils dans la barre latérale."
                )

            st.divider()

            # --- Tableau récapitulatif ---
            if filtered_entities:
                st.subheader("📊 Tableau récapitulatif")

                table_data = []
                for e in filtered_entities:
                    table_data.append({
                        "Entité": e["text"],
                        "Type": LABEL_NAMES.get(e["label"], e["label"]),
                        "Code ICD-O": e.get("icdo_code") or "—",
                        "Score": e["score"],
                        "Longueur": len(e["text"]),
                        "Position": f"{e['start']}-{e['end']}",
                    })

                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Score": st.column_config.ProgressColumn(
                            "Score",
                            min_value=0,
                            max_value=1,
                            format="%.4f",
                        ),
                        "Code ICD-O": st.column_config.TextColumn(
                            "Code ICD-O",
                            help=(
                                "Code ICD-O normalisé. "
                                "Disponible si le CSV FRACCO est chargé "
                                "(variable CSV_PATH)."
                            ),
                        ),
                        "Longueur": st.column_config.NumberColumn(
                            "Longueur",
                            help="Nombre de caractères de l'entité",
                        ),
                    },
                )

                # --- Distribution par type ---
                st.subheader("📈 Distribution par type d'entité")
                label_counts = {}
                for e in filtered_entities:
                    label_name = LABEL_NAMES.get(e["label"], e["label"])
                    label_counts[label_name] = label_counts.get(label_name, 0) + 1

                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    st.bar_chart(label_counts)
                with col_chart2:
                    for label_name, count in sorted(
                        label_counts.items(), key=lambda x: x[1], reverse=True
                    ):
                        st.metric(label_name, count)

                # Export JSON
                st.download_button(
                    label="⬇️ Télécharger les résultats (JSON)",
                    data=response.text,
                    file_name="onco_ner_results.json",
                    mime="application/json",
                )

        else:
            st.error(
                f"Erreur API : {response.status_code} — {response.text}"
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Impossible de se connecter à l'API. "
            "Assurez-vous que l'API est lancée sur le port 8000."
        )
    except requests.exceptions.Timeout:
        st.error(
            "⏱️ Timeout — le texte est peut-être trop long. "
            "Essayez avec un texte plus court."
        )
    except Exception as e:
        st.error(f"Erreur inattendue : {e}")

# --- Footer ---
st.divider()
st.caption(
    "onco-ner v0.1.0 — Modèle : XLM-RoBERTa optimisé sur FRACCO | "
    "Développé dans le cadre d'un projet de NLP médical"
)