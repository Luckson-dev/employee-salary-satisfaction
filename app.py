import os
import sys
import pandas as pd
import streamlit as st
from src.prediction import RENTPrediction

# Page Configuration
st.set_page_config(
    page_title="Prédiction de Satisfaction",
    page_icon="✅",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS Style
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 100%);
        color: #F8FAFC;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    h1, h2, h3 {
        color: #E2E8F0 !important;
        font-weight: 700;
    }
    
    .result-card {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
        margin-top: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        animation: fadeIn 0.5s ease-in;
    }
    
    .result-label {
        font-size: 1rem;
        color: #C7D2FE;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 10px;
    }
    
    .result-value {
        font-size: 3.5rem;
        font-weight: 800;
        color: #FFFFFF;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .result-sub {
        font-size: 1.1rem;
        color: #E0E7FF;
        margin-top: 5px;
        font-weight: 500;
    }

    .stNumberInput > div > div > input, .stSelectbox > div > div > select {
        background-color: #334155 !important;
        color: #F8FAFC !important;
        border: 1px solid #475569 !important;
        border-radius: 8px;
    }
    
    label {
        color: #CBD5E1 !important;
        font-weight: 500;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #6366F1 0%, #8B5CF6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.3s;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.5);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

satisfaction_mapping = ['Très insatisfait', 'Insatisfait', 'Neutre', 'Satisfait', 'Très satisfait']
balance_mapping = ['Très mauvais', 'Mauvais', 'Moyen', 'Bon', 'Excellent']

# 1. Initialisation avec des valeurs textuelles valides
if 'form_data' not in st.session_state:
    st.session_state['form_data'] = {
        'Equilibre_Vie_Travail': "Moyen",
        'Satisfaction_Salaire': "Neutre",
        'Heures_Formation': 10,
        'Nombre_Absences': 2,
        'Heures_Supplementaires': 5,
        'Salaire_Mensuel_BIF': 500000,
        'Age': 30,
    }

if 'prediction_result' not in st.session_state:
    st.session_state['prediction_result'] = None

# Chargement du modèle
@st.cache_resource
def load_model():
    return RENTPrediction(file_path="models/final_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

st.title("Prédiction de Classification")
st.markdown("<p style='color: #94A3B8; text-align: center; margin-bottom: 30px;'>Entrez les détails pour obtenir une prédiction</p>", unsafe_allow_html=True)

with st.form("classification_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        eq_vie = st.selectbox(
            "Équilibre Vie/Travail ?",
            options=balance_mapping,
            index=balance_mapping.index(st.session_state['form_data']['Equilibre_Vie_Travail'])
        )
        h_form = st.number_input(
            "Heures Formation",
            min_value=0,
            value=int(st.session_state['form_data']['Heures_Formation'])
        )
        h_supp = st.number_input(
            "Heures Supplémentaires",
            min_value=0,
            value=int(st.session_state['form_data']['Heures_Supplementaires'])
        )
        
    with col2:
        sat_sal = st.selectbox(
            "Satisfaction Salaire",
            options=satisfaction_mapping,
            index=satisfaction_mapping.index(st.session_state['form_data']['Satisfaction_Salaire'])
        )
        absences = st.number_input(
            "Nombre d'Absences",
            min_value=0,
            value=int(st.session_state['form_data']['Nombre_Absences'])
        )
        salaire = st.number_input(
            "Salaire Mensuel (BIF)",
            min_value=0,
            value=int(st.session_state['form_data']['Salaire_Mensuel_BIF']), 
            step=50000
        )
        
    with col3:
        age = st.number_input(
            "Âge", 
            min_value=18, 
            max_value=100, 
            value=int(st.session_state['form_data']['Age'])
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Lancer la Prédiction", use_container_width=True)
    
    if submitted:
        st.session_state['form_data'] = {
            'Equilibre_Vie_Travail': eq_vie,
            'Satisfaction_Salaire': sat_sal,
            'Heures_Formation': h_form,
            'Nombre_Absences': absences,
            'Heures_Supplementaires': h_supp,
            'Salaire_Mensuel_BIF': salaire,
            'Age': age
        }
        
        input_df = pd.DataFrame([st.session_state['form_data']])
        
        try:
            pred_raw = model.predict(input_df)
            pred_value = int(pred_raw[0])
            
            if isinstance(pred_value, (int, float)):
                if 0 < pred_value < 1:
                    label = "Oui" if pred_value >= 0.5 else "Non"
                else:
                    label = "Oui" if pred_value == 1 else "Non"
            else:
                label = str(pred_value)

            st.session_state['prediction_result'] = label
            
        except Exception as e:
            st.error(f"Erreur de prédiction : {e}")
            st.session_state['prediction_result'] = "Erreur"

# Affichage du résultat
if st.session_state['prediction_result'] is not None:
    result = st.session_state['prediction_result']
    
    color_class = "#22c55e" if result == "Oui" else "#ef4444"
    icon = "✅" if result == "Oui" else "❌"
    
    st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Résultat de la Prédiction</div>
            <div class="result-value" style="color: {color_class}">{icon} {result}</div>
            <div class="result-sub">Basé sur les données fournies</div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Remplissez le formulaire et cliquez sur 'Lancer la Prédiction' pour voir le résultat.")