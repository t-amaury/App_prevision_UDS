import streamlit as st
import pandas as pd
import joblib
import os
import git
from pathlib import Path
import shutil
from fonctions_utiles.pretraitement_facteurs_externes import pretraitement2
from fonctions_utiles.update_predictions import update_predictions, update_predictions_UDS

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Prévision du trafic',
    page_icon=':airplane:',
)

# Custom CSS to expand the width of the columns
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1600px;
        padding-left: 8%;
        padding-right: 5%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Utility Functions

def load_model(region_A, region_B):
    """Load the prediction models for a given pair of regions."""
    prophet_model_path = f'model/prophet_model_{region_A}_{region_B}.joblib'
    exp_smoothing_model_path = f'model/exp_smoothing_model_{region_A}_{region_B}.joblib'
    
    model = {}
    if os.path.exists(prophet_model_path):
        model['prophet'] = joblib.load(prophet_model_path)
    if os.path.exists(exp_smoothing_model_path):
        model['exp_smoothing'] = joblib.load(exp_smoothing_model_path)
    
    return model

@st.cache_data
def load_data():
    """Load all necessary data."""
    data = {
        'prevision': pd.read_csv(Path(__file__).parent / 'data/prevision_data.csv'),
        'prevision_UDS' :  pd.read_csv(Path(__file__).parent / 'data/prevision_data_UDS.csv'),
        'eurocontrol': pd.read_csv(Path(__file__).parent / 'data/eurocontrol_data.csv'),
        'facteur_externe': pd.read_csv(Path(__file__).parent / 'data/facteur_externe_data.csv'),
        'facteur_externe_petrole': pd.read_csv(Path(__file__).parent / 'data/facteur_externe_prix_petrole.csv'),
        'facteur_externe_modifiable': pd.read_csv(Path(__file__).parent / 'data/facteur_externe_data_modifiable.csv'),
        'facteur_externe_petrole_modifiable': pd.read_csv(Path(__file__).parent / 'data/facteur_externe_prix_petrole_modifiable.csv'),
        'translation': pd.read_csv(Path(__file__).parent / 'data/translation_code_region.csv')
    }
    
    # Transformation des données de prévision
    MIN_YEAR_PREVISION = 2024
    MAX_YEAR_PREVISION = 2030
    data['prevision'] = data['prevision'].melt(
        ['pair'],
        [str(x) for x in range(MIN_YEAR_PREVISION, MAX_YEAR_PREVISION + 1)],
        'Year',
        'prevision',
    )
    data['prevision']['Year'] = pd.to_numeric(data['prevision']['Year'])

    # Transformation des données de prévision UDS
    data['prevision_UDS'] = data['prevision_UDS'].melt(
        ['pair'],
        [str(x) for x in range(MIN_YEAR_PREVISION, MAX_YEAR_PREVISION + 1)],
        'Year',
        'prevision',
    )
    data['prevision_UDS']['Year'] = pd.to_numeric(data['prevision_UDS']['Year'])
    
    # Transformation des données Eurocontrol
    MIN_YEAR_EUROCONTROL = 2019
    MAX_YEAR_EUROCONTROL = 2030
    data['eurocontrol'] = data['eurocontrol'].melt(
        ['pair'],
        [str(x) for x in range(MIN_YEAR_EUROCONTROL, MAX_YEAR_EUROCONTROL + 1)],
        'Year',
        'prevision',
    )
    data['eurocontrol']['Year'] = pd.to_numeric(data['eurocontrol']['Year'])
    
    return data

def save_data(data, type='facteur_externe'):
    """Save modified external factors data."""
    if type == 'facteur_externe':
        data.to_csv(Path(__file__).parent / 'data/facteur_externe_data_modifiable.csv', index=False)
    elif type == 'facteur_externe_petrole':
        data.to_csv(Path(__file__).parent / 'data/facteur_externe_prix_petrole_modifiable.csv', index=False)

def save_uploaded_files(uploaded_files):
    """Save and modify uploaded files to a temporary directory and return their paths."""
    saved_file_paths = []

    # Temporary directory for saving uploaded files
    save_dir = Path(__file__).parent / 'temp_uploaded_files'
    save_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    for uploaded_file in uploaded_files:
        file_path = save_dir / uploaded_file.name
        noms_colonnes = ['Origin of message','Flight date','Time of departure/entry','Departure aerodrome','Arrival aerodrome','Type of aircraft','User Number','User Nationality','Correction Code','Main Exemption Code','Weight Coefficient','Distance Coefficient','Service Units','Exemption Indicator','Flight category',]
        # Read the uploaded file into a pandas DataFrame
        df = pd.read_csv(uploaded_file, sep=';', usecols = noms_colonnes)

        # Example: Automatically drop unnecessary columns to reduce file size
        # columns_to_drop = ['Column1', 'Column2']  # Replace with actual column names
        # df = df.drop(columns=columns_to_drop, errors='ignore')

        # Save the modified DataFrame back to CSV
        df.to_csv(file_path, index=False)
        saved_file_paths.append(file_path)
    return saved_file_paths

def push_to_github(file_paths):
    """Commit and push changes to GitHub."""
    try:
        repo = git.Repo(Path(__file__).parent)
        for file_path in file_paths:
            repo.git.add(str(file_path))
        repo.index.commit("upload fichier abacus")
        origin = repo.remote(name='origin')

        # Use the PAT for authentication
        github_token = st.secrets["github"]["token"]
        origin_url = origin.url
        if origin_url.startswith("git@"):
            repo_path = origin_url.split(':')[1]
            origin_url = f"https://{github_token}@github.com/{repo_path}"
        else:
            origin_url = origin_url.replace("https://", f"https://{github_token}@")

        repo.git.pull(origin_url)
        repo.git.push(origin_url)
        st.sidebar.success("fichier(s) committé(s) et poussé(s) sur GitHub")
    except Exception as e:
        st.sidebar.error(f"Erreur lors du commit et push: {str(e)}")
        st.error(f"Erreur détaillée : {e}")
    # Clean up: remove the temporary directory and its contents after pushing
    shutil.rmtree(save_dir)

def translate_pair(pair, translation_df):
    """Translate a pair code into region names."""
    code1, code2 = pair.split('_')
    region1 = translation_df[translation_df['code'] == code1]['region'].iloc[0]
    region2 = translation_df[translation_df['code'] == code2]['region'].iloc[0]
    return f"{region1}--{region2}"

# -----------------------------------------------------------------------------
# Load and Prepare Data

data = load_data()
prevision_df = data['prevision']
prevision_UDS_df = data['prevision_UDS']
eurocontrol_df = data['eurocontrol']
facteur_externe_df = data['facteur_externe']
facteur_externe_petrole_df = data['facteur_externe_petrole']
region_translation_df = data['translation']

prevision_df['pair_nom'] = prevision_df['pair'].apply(lambda x: translate_pair(x, region_translation_df))
prevision_UDS_df['pair_nom'] = prevision_UDS_df['pair'].apply(lambda x: translate_pair(x, region_translation_df))

if 'facteur_externe_modifiable_df' not in st.session_state:
    st.session_state.facteur_externe_modifiable_df = data['facteur_externe_modifiable']
if 'facteur_externe_petrole_modifiable_df' not in st.session_state:
    st.session_state.facteur_externe_petrole_modifiable_df = data['facteur_externe_petrole_modifiable']

# -----------------------------------------------------------------------------
# Sidebar: Modify and Save Data

st.sidebar.header('Données de Facteur Externe')
st.sidebar.dataframe(facteur_externe_df)
st.sidebar.dataframe(facteur_externe_petrole_df)

st.sidebar.header('Modifier les Données')
edited_data = st.sidebar.data_editor(st.session_state.facteur_externe_modifiable_df, key="data_editor_1")

# Sélection de l'intervalle des années
min_year = int(st.session_state.facteur_externe_petrole_modifiable_df.columns[2])
max_year = int(st.session_state.facteur_externe_petrole_modifiable_df.columns[-2])

selected_years = st.sidebar.slider(
    'Sélectionnez l\'intervalle des années à ajuster',
    min_year, max_year, (min_year, max_year)
)

adjustment_percentage = st.sidebar.number_input(
    'Entrez le pourcentage d\'ajustement des prix du pétrole (%)',
    value=0.0
)

edited_data_petrole = st.sidebar.data_editor(st.session_state.facteur_externe_petrole_modifiable_df, key="data_editor_2")

if st.sidebar.button('Sauvegarder les modifications'):
    save_data(edited_data, type='facteur_externe')
    st.session_state.facteur_externe_modifiable_df = edited_data

    for year in range(selected_years[0], selected_years[1] + 1):
        year_str = str(year)
        if year_str in edited_data_petrole.columns:
            edited_data_petrole[year_str] *= (1 + (adjustment_percentage / 100))
    save_data(edited_data_petrole, type='facteur_externe_petrole')
    st.session_state.facteur_externe_petrole_modifiable_df = edited_data_petrole

    # Recalculate predictions with updated factors
    predictions = update_predictions(edited_data, edited_data_petrole, prevision_df, "2023-12-31", "2031-01-05")
    predictions_UDS = update_predictions_UDS(edited_data, edited_data_petrole, prevision_UDS_df, "2023-12-31", "2031-01-05")
    st.write(predictions)
    st.write(predictions.columns)
       # Transformation des données de prévision
    MIN_YEAR_PREVISION = 2024
    MAX_YEAR_PREVISION = 2030
    predictions = predictions.melt(
        ['pair'],
        [str(x) for x in range(MIN_YEAR_PREVISION, MAX_YEAR_PREVISION + 1)],
        'Year',
        'prevision',
    )
    predictions['Year'] = pd.to_numeric(predictions['Year'])

    # Transformation des données de prévision
    predictions_UDS = predictions_UDS.melt(
        ['pair'],
        [str(x) for x in range(MIN_YEAR_PREVISION, MAX_YEAR_PREVISION + 1)],
        'Year',
        'prevision',
    )
    predictions_UDS['Year'] = pd.to_numeric(predictions_UDS['Year'])
    predictions['pair_nom'] = predictions['pair'].apply(lambda x: translate_pair(x, region_translation_df))
    predictions_UDS['pair_nom'] = predictions_UDS['pair'].apply(lambda x: translate_pair(x, region_translation_df))

    st.session_state['predictions'] = predictions
    st.session_state['predictions_UDS'] = predictions_UDS
    st.sidebar.success('Prédictions mises à jour avec succès!')

if st.sidebar.button('Réinitialiser'):
    save_data(facteur_externe_df, type='facteur_externe')
    st.session_state.facteur_externe_modifiable_df = facteur_externe_df.copy()
    save_data(facteur_externe_petrole_df, type='facteur_externe_petrole')
    st.session_state.facteur_externe_petrole_modifiable_df = facteur_externe_petrole_df.copy()
    st.session_state.pop('predictions', None)
    st.rerun()

# -----------------------------------------------------------------------------
# Sidebar: Upload Files

st.sidebar.header("Uploader des fichiers Abacus")
uploaded_files = st.sidebar.file_uploader(
    "Choisissez des fichiers Abacus",
    accept_multiple_files=True
)

if uploaded_files:
    file_paths = save_uploaded_files(uploaded_files)
    
    if st.sidebar.button("Commit et Push vers GitHub"):
        push_to_github(file_paths)

##-----------------------------------------------------------------------------
#Main Page: Display Predictions
#Add some spacing
st.write('')
st.write('')
if 'predictions' in st.session_state:
    predictions = st.session_state['predictions']
else:
    predictions = prevision_df

if 'predictions_UDS' in st.session_state:
    predictions_UDS = st.session_state['predictions_UDS']
else:
    predictions_UDS = prevision_UDS_df

min_value = predictions['Year'].min()
max_value = predictions['Year'].max()

from_year, to_year = st.slider(
    'Which years are you interested in?',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value]
)

countries = predictions['pair_nom'].unique()

if not len(countries):
    st.warning("Select at least one country")

selected_countries = st.multiselect(
    'Which countries would you like to view?',
    countries,
    ['France--France']
)

st.write('')
st.write('')
st.write('')

filtered_prevision_df = predictions[
    (predictions['pair_nom'].isin(selected_countries))
    & (predictions['Year'] <= to_year)
    & (from_year <= predictions['Year'])
]

filtered_prevision_df['Year'] = filtered_prevision_df['Year'].astype(str)

all_pair = predictions[
    (predictions['Year'] <= to_year)
    & (from_year <= predictions['Year'])
]

all_pair['Year'] = all_pair['Year'].astype(str)

# Calculate TCMA for each pair_nom
def calculate_tcma(df, from_year, to_year):
    start_value = df[df['Year'] == from_year]['prevision'].values[0]
    end_value = df[df['Year'] == to_year]['prevision'].values[0]
    n_years = to_year - from_year
    tcma = ((end_value / start_value) ** (1 / n_years) - 1) * 100
    return tcma

#Pour UDS
filtered_prevision_UDS_df = predictions_UDS[
    (predictions_UDS['pair_nom'].isin(selected_countries))
    & (predictions_UDS['Year'] <= to_year)
    & (from_year <= predictions['Year'])
]
filtered_prevision_UDS_df['Year'] = filtered_prevision_UDS_df['Year'].astype(str)
all_pair_UDS = predictions_UDS[
    (predictions_UDS['Year'] <= to_year)
    & (from_year <= predictions_UDS['Year'])
]
all_pair_UDS['Year'] = all_pair_UDS['Year'].astype(str)
##

# Calculate total prevision sum for each pair_nom and TCMA
prevision_sum_df = predictions.groupby('pair_nom')['prevision'].sum().reset_index()
prevision_sum_UDS_df = predictions_UDS.groupby('pair_nom')['prevision'].sum().reset_index()
prevision_sum_df['TCMA'] = prevision_sum_df['pair_nom'].apply(lambda x: calculate_tcma(predictions[predictions['pair_nom'] == x], from_year, to_year))
# Renommer la colonne 'prevision' pour éviter les conflits lors de la fusion
prevision_sum_df.rename(columns={'prevision': 'prevision_standard'}, inplace=True)
prevision_sum_UDS_df.rename(columns={'prevision': 'prevision_uds'}, inplace=True)
# Fusionner les DataFrames basés sur la colonne 'pair'
combined_prevision_sum_df = pd.merge(prevision_sum_df, prevision_sum_UDS_df, on='pair_nom')
# Sélectionner les 20 paires avec la plus grande somme de prévisions (selon le modèle standard)
top_20_prevision_sum_df = combined_prevision_sum_df.nlargest(20, 'prevision_standard')

# Plot the first chart and show the table
st.header('Prévision over time by Pair', divider='gray')

col1, col2 = st.columns([2, 1])

with col1:
    st.line_chart(
        filtered_prevision_df,
        x='Year',
        y='prevision',
        color='pair_nom',
    )

    st.line_chart(
        filtered_prevision_UDS_df,
        x='Year',
        y='prevision',
        color='pair_nom',
    )

with col2:
    st.subheader('Top 20 paires par somme des prévisions')
    st.dataframe(top_20_prevision_sum_df)


include_eurocontrol = st.checkbox('Include Eurocontrol data')

if include_eurocontrol:
    filtered_eurocontrol_df = eurocontrol_df[
        (eurocontrol_df['Year'] <= to_year)
        & (from_year <= eurocontrol_df['Year'])
    ]

# Calculate sum of pairs
sum_prevision_df = all_pair.groupby('Year')['prevision'].sum().reset_index()
sum_prevision_df['Type'] = 'Prévision Totale'

# Calculate sum of pairs UDS
sum_prevision_UDS_df = all_pair_UDS.groupby('Year')['prevision'].sum().reset_index()
sum_prevision_UDS_df['Type'] = 'Prévision Totale'

if include_eurocontrol:
    sum_eurocontrol_df = filtered_eurocontrol_df.groupby('Year')['prevision'].sum().reset_index()
    sum_eurocontrol_df['Type'] = 'Eurocontrol'
    sum_prevision_df = pd.concat([sum_prevision_df, sum_eurocontrol_df])      

# Plot the second chart
st.header('Total Prévision over time', divider='gray')

st.line_chart(
    sum_prevision_df,
    x='Year',
    y='prevision',
    color='Type',
)

st.line_chart(
    sum_prevision_UDS_df,
    x='Year',
    y='prevision',
    color='Type',
)

# Exportation
# -----------------------------------------------------------------------------
# Exportation des données en CSV

# Exporter les facteurs externes modifiables
csv_facteurs_externes = st.session_state.facteur_externe_modifiable_df.to_csv(index=False)
st.download_button(
    label="Télécharger les facteurs externes modifiables en CSV",
    data=csv_facteurs_externes,
    file_name='facteurs_externes_modifiables.csv',
    mime='text/csv',
)

# Exporter les prix du pétrole modifiables
csv_petro = st.session_state.facteur_externe_petrole_modifiable_df.to_csv(index=False)
st.download_button(
    label="Télécharger les prix du pétrole modifiables en CSV",
    data=csv_petro,
    file_name='prix_petrole_modifiables.csv',
    mime='text/csv',
)

# Calculs pour les prévisions
# Assurez-vous que 'predictions' et 'predictions_UDS' sont définis et filtrés correctement avant
# Calcul de la somme des prévisions pour chaque paire et TCMA
prevision_sum_standard_df = predictions.groupby('pair_nom')['prevision'].sum().reset_index()
prevision_sum_UDS_df = predictions_UDS.groupby('pair_nom')['prevision'].sum().reset_index()

prevision_sum_standard_df['TCMA'] = prevision_sum_standard_df['pair_nom'].apply(
    lambda x: calculate_tcma(predictions[predictions['pair_nom'] == x], from_year, to_year)
)

prevision_sum_UDS_df['TCMA'] = prevision_sum_UDS_df['pair_nom'].apply(
    lambda x: calculate_tcma(predictions_UDS[predictions_UDS['pair_nom'] == x], from_year, to_year)
)

# Fusion des DataFrames de prévisions standard et UDS
combined_prevision_df = pd.merge(
    prevision_sum_standard_df, 
    prevision_sum_UDS_df, 
    on='pair_nom', 
    suffixes=('_standard', '_UDS')
)

# Calcul des prévisions totales
sum_prevision_standard_df = all_pair.groupby('Year')['prevision'].sum().reset_index()
sum_prevision_standard_df['Type'] = 'Prévision Totale Standard'

sum_prevision_UDS_df = all_pair_UDS.groupby('Year')['prevision'].sum().reset_index()
sum_prevision_UDS_df['Type'] = 'Prévision Totale UDS'

# Fusion des prévisions totales
total_prevision_df = pd.concat([sum_prevision_standard_df, sum_prevision_UDS_df])

# Exporter les prévisions combinées et totales
csv_combined_prevision = combined_prevision_df.to_csv(index=False)
st.download_button(
    label="Télécharger les prévisions combinées (Standard et UDS) en CSV",
    data=csv_combined_prevision,
    file_name='previsions_combined.csv',
    mime='text/csv',
)

csv_total_prevision = total_prevision_df.to_csv(index=False)
st.download_button(
    label="Télécharger les prévisions totales en CSV",
    data=csv_total_prevision,
    file_name='total_prevision_data.csv',
    mime='text/csv',
)
