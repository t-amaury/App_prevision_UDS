import streamlit as st
import pandas as pd

def reindex_exogenes(start_forecast, end_forecast, facteur_externe):
    date_rng2 = pd.date_range(start=start_forecast, end=end_forecast, freq='D')
    weekly_exog2 = facteur_externe.resample('D').ffill().reindex(date_rng2).fillna(method='ffill')
    return(weekly_exog2)

def transform_covid_exo(weekly_exog2):
    weekly_exog2['covid'] = 0
    return(weekly_exog2)

def filtrage_pays(weekly_exog2, pays_A, pays_B):
    weekly_exog2 = weekly_exog2.loc[:, [pays_A, pays_B, "covid"]]
    return weekly_exog2

def reformatage(weekly_exog2):
    # Reformater les données pour StatsForecast
    test_exog = weekly_exog2.reset_index()
    test_exog.columns = ['ds', 'GDP_A', 'GGE_A', 'population_A', 'GDP_B', 'GGE_B', 'population_B', 'covid']
    test_exog['unique_id'] = 1  # Ajouter une colonne 'unique_id' avec une valeur constante
    test_exog = test_exog[['unique_id', 'ds', 'GDP_A', 'GGE_A', 'population_A', 'GDP_B', 'GGE_B', 'population_B', 'covid']]  # Réordonner les colonnes
    return(test_exog)
    
def pretraitement2(start_forecast, end_forecast, facteur_externe, pays_A, pays_B):
    facteur_externe = facteur_externe.set_index(["ICAO_Code", "Subject Descriptor"]).transpose().loc['2017':'2029']
    facteur_externe = facteur_externe.transpose().groupby(level=[0,1]).sum().transpose()
    facteur_externe.index= pd.to_datetime(facteur_externe.index, format='%Y')
    weekly_exog2 = reindex_exogenes(start_forecast, end_forecast, facteur_externe)
    weekly_exog2 = transform_covid_exo(weekly_exog2)
    weekly_exog2 = filtrage_pays(weekly_exog2, pays_A, pays_B)
    test_exog = reformatage(weekly_exog2)
    return test_exog
