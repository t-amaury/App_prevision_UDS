import numpy as np
np.float_ = np.float64
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import os
import joblib
from fonctions_utiles.pretraitement_facteurs_externes import pretraitement2

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

def load_model_UDS(region_A, region_B):
    """Load the prediction models for a given pair of regions."""
    prophet_model_path = f'model_UDS/prophet_model_{region_A}_{region_B}.joblib'
    exp_smoothing_model_path = f'model_UDS/exp_smoothing_model_{region_A}_{region_B}.joblib'
    
    model = {}
    if os.path.exists(prophet_model_path):
        model['prophet'] = joblib.load(prophet_model_path)
    if os.path.exists(exp_smoothing_model_path):
        model['exp_smoothing'] = joblib.load(exp_smoothing_model_path)
    return model

def regroupement_annees2(forecasts_dict):
    rows = []
    for key, series in forecasts_dict.items():
        # Convertir la série en DataFrame
        df = series.to_frame(name='value')
        # Ajouter la colonne pour la paire
        df['pair'] = '_'.join(key)
        # Ajouter la colonne pour l'année
        df['year'] = df.index.year
        rows.append(df)

    combined_df = pd.concat(rows)
    grouped = combined_df.groupby(['pair', 'year'])['value'].sum().reset_index()
    pivot_df = grouped.pivot_table(index='pair', columns='year', values='value', fill_value=0)
    pivot_df.columns = pivot_df.columns.astype(str)
    pivot_df.reset_index(inplace=True)
    return pivot_df
    
def update_predictions(facteur_externe_df, facteur_externe_petrole_df, prevision_df, start_date, end_date):
    """Update predictions based on modified external factors."""
    # Placeholder for recalculating predictions
    forecasts_dict = {}

    # Get all unique region pairs
    region_pairs = prevision_df['pair'].unique()
    
    for pair in region_pairs:
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        region_A, region_B = pair.split('_')
        model_dict = load_model(region_A, region_B)
        test_exog = pretraitement2(start_date, end_date, facteur_externe_df, facteur_externe_petrole_df, region_A, region_B)

        if 'prophet' in model_dict:
            model = model_dict["prophet"]
            forecast_prophet = model.predict(test_exog)
            forecast = pd.Series(list(forecast_prophet["yhat"]), index=list(forecast_prophet["ds"]))
        elif "exp_smoothing" in model_dict:
            model = model_dict["exp_smoothing"]
            results = model.fit()
            params = model.params
            forecast = model.predict(start = start_date, end = end_date, params=params)
            forecast = pd.Series(forecast, index=date_range)
        else:
            forecast = [0.147] * (len(date_range))
            forecast = pd.Series(forecast, index=date_range)
        forecasts_dict[(region_A, region_B)] = forecast
    predictions = regroupement_annees2(forecasts_dict)
    return predictions

def update_predictions_UDS(facteur_externe_df, facteur_externe_petrole_df, prevision_df, start_date, end_date):
    """Update predictions based on modified external factors."""
    # Placeholder for recalculating predictions
    forecasts_dict = {}

    # Get all unique region pairs
    region_pairs = prevision_df['pair'].unique()
    
    for pair in region_pairs:
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        region_A, region_B = pair.split('_')
        model_dict = load_model_UDS(region_A, region_B)
        test_exog = pretraitement2(start_date, end_date, facteur_externe_df, facteur_externe_petrole_df, region_A, region_B)

        if 'prophet' in model_dict:
            model = model_dict["prophet"]
            forecast_prophet = model.predict(test_exog)
            forecast = pd.Series(list(forecast_prophet["yhat"]), index=list(forecast_prophet["ds"]))
        elif "exp_smoothing" in model_dict:
            model = model_dict["exp_smoothing"]
            results = model.fit()
            params = model.params
            forecast = model.predict(start = start_date, end = end_date, params=params)
            forecast = pd.Series(forecast, index=date_range)
        else:
            forecast = [0.147] * (len(date_range))
            forecast = pd.Series(forecast, index=date_range)
        forecasts_dict[(region_A, region_B)] = forecast
    predictions = regroupement_annees2(forecasts_dict)
    return predictions
