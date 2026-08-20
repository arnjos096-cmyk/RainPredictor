import os
import pickle
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from model import EnhancedRainfallLSTM, RainfallLSTM

app = FastAPI(
    title="RainPredictor AI",
    description="Time-series LSTM Rainfall Forecasting & Meteorological Inference Service",
    version="1.0.0"
)

# CORS middleware for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Feature specifications
FEATURES = [
    {"id": "temperature", "name": "Temperature", "unit": "°C", "min": -10.0, "max": 50.0, "default": 22.0, "desc": "Ambient air temperature"},
    {"id": "humidity", "name": "Relative Humidity", "unit": "%", "min": 0.0, "max": 100.0, "default": 75.0, "desc": "Relative humidity percentage"},
    {"id": "pressure", "name": "Barometric Pressure", "unit": "hPa", "min": 960.0, "max": 1050.0, "default": 1008.0, "desc": "Atmospheric air pressure"},
    {"id": "wind_speed", "name": "Wind Speed", "unit": "km/h", "min": 0.0, "max": 150.0, "default": 24.0, "desc": "Sustained wind velocity"},
    {"id": "wind_direction", "name": "Wind Direction", "unit": "°", "min": 0.0, "max": 360.0, "default": 210.0, "desc": "Compass angle of wind"},
    {"id": "soil_moisture", "name": "Soil Moisture", "unit": "%", "min": 0.0, "max": 100.0, "default": 55.0, "desc": "Ground soil water saturation"},
    {"id": "solar_radiation", "name": "Solar Radiation", "unit": "W/m²", "min": 0.0, "max": 1200.0, "default": 120.0, "desc": "Global horizontal irradiance"},
    {"id": "cloud_cover", "name": "Cloud Cover", "unit": "%", "min": 0.0, "max": 100.0, "default": 85.0, "desc": "Sky cloud cover percentage"},
    {"id": "dew_point", "name": "Dew Point", "unit": "°C", "min": -15.0, "max": 35.0, "default": 18.0, "desc": "Atmospheric moisture saturation temp"},
    {"id": "evapotranspiration", "name": "Evapotranspiration", "unit": "mm/h", "min": 0.0, "max": 2.5, "default": 0.15, "desc": "Evaporation and transpiration rate"},
    {"id": "rainfall_mm", "name": "Rainfall (Target)", "unit": "mm", "min": 0.0, "max": 100.0, "default": 0.0, "desc": "Current hour measured precipitation"}
]

FEATURE_COLUMNS = [f["id"] for f in FEATURES]

# Global model and scaler variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
scaler = None
expected_features_dim = 11

def load_resources():
    global model, scaler, expected_features_dim
    if os.path.exists('scaler.pkl'):
        try:
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
                expected_features_dim = getattr(scaler, 'n_features_in_', 11)
        except Exception as e:
            print(f"Error loading scaler: {e}")
            scaler = None
    else:
        scaler = None

    # Load weights
    if os.path.exists('model.pth'):
        checkpoint = torch.load('model.pth', map_location=device, weights_only=False)
        # Check if checkpoint has attention/feature_proj
        if 'attention.attention.weight' in checkpoint or 'feature_proj.weight' in checkpoint:
            model = EnhancedRainfallLSTM(actual_input_size=19, legacy_input_size=11, hidden_size=128, num_layers=2, dropout=0.3689).to(device)
        else:
            model = RainfallLSTM(input_size=11, hidden_size=64, num_layers=2, dropout=0.2).to(device)
            
        try:
            model.load_state_dict(checkpoint)
            model.eval()
            print(f"Loaded trained model.pth successfully. (Type: {type(model).__name__}, Scaler Dim: {expected_features_dim})")
        except Exception as e:
            print(f"State dict mismatch, fallback to new init: {e}")
            model = RainfallLSTM(input_size=11, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    else:
        model = RainfallLSTM(input_size=11, hidden_size=64, num_layers=2, dropout=0.2).to(device)
        print("Warning: model.pth not found. Initialized baseline RainfallLSTM.")

load_resources()

# Pydantic Schemas
class WeatherHour(BaseModel):
    temperature: float = 22.0
    humidity: float = 75.0
    pressure: float = 1008.0
    wind_speed: float = 24.0
    wind_direction: float = 210.0
    soil_moisture: float = 55.0
    solar_radiation: float = 120.0
    cloud_cover: float = 85.0
    dew_point: Optional[float] = None
    evapotranspiration: Optional[float] = None
    rainfall_mm: float = 0.0

class SequencePredictRequest(BaseModel):
    sequence: List[WeatherHour] # 24 timesteps
    hour_of_day: Optional[int] = 14
    day_of_year: Optional[int] = 180

class SingleHourPredictRequest(BaseModel):
    current: WeatherHour
    pressure_trend: Optional[str] = "falling" # "falling", "steady", "rising"
    hour_of_day: Optional[int] = 14
    day_of_year: Optional[int] = 180
    
class ForecastRequest(BaseModel):
    sequence: List[WeatherHour]
    steps: int = Field(default=6, ge=1, le=24)
    hour_of_day: Optional[int] = 14

def categorize_rainfall(mm: float) -> Dict[str, Any]:
    if mm < 0.05:
        return {
            "tier": "Dry / No Rain",
            "code": "DRY",
            "color": "#10b981",
            "risk_level": "None",
            "probability": max(2.0, min(15.0, mm * 100)),
            "summary": "Clear skies or dry conditions expected over the next hour."
        }
    elif mm < 2.5:
        return {
            "tier": "Light Rain / Drizzle",
            "code": "LIGHT",
            "color": "#38bdf8",
            "risk_level": "Low",
            "probability": min(85.0, 50.0 + mm * 15.0),
            "summary": "Scattered light showers or mist. Low impact on outdoor activities."
        }
    elif mm < 7.6:
        return {
            "tier": "Moderate Rain",
            "code": "MODERATE",
            "color": "#f59e0b",
            "risk_level": "Moderate",
            "probability": min(95.0, 80.0 + mm * 2.0),
            "summary": "Steady rainfall accumulation. Reduced road visibility and wet pavement."
        }
    else:
        return {
            "tier": "Heavy Rain / Storm",
            "code": "HEAVY",
            "color": "#ef4444",
            "risk_level": "High / Severe",
            "probability": min(99.9, 92.0 + mm * 0.8),
            "summary": "Torrential downpour with storm risks, flash runoff, and high precipitation."
        }

def compute_meteorological_insights(features_24h_11: np.ndarray, pred_mm: float) -> List[Dict[str, str]]:
    insights = []
    
    # Pressure drop over 24h
    pressures = features_24h_11[:, 2]
    p_delta = pressures[-1] - pressures[0]
    p_last = pressures[-1]
    
    if p_delta < -3.0 or p_last < 1005:
        insights.append({
            "type": "warning",
            "icon": "fa-gauge-simple-high",
            "title": "Barometric Depletion Detected",
            "detail": f"Atmospheric pressure stands at {p_last:.1f} hPa ({p_delta:+.1f} hPa 24h delta), facilitating convective cloud formation."
        })
    elif p_last > 1020:
        insights.append({
            "type": "positive",
            "icon": "fa-sun",
            "title": "High Pressure Ridge",
            "detail": f"Strong anticyclonic ridge ({p_last:.1f} hPa) suppressing cloud convection."
        })

    # Humidity and Cloud Cover
    last_humidity = features_24h_11[-1, 1]
    last_clouds = features_24h_11[-1, 7]
    if last_humidity > 85 and last_clouds > 80:
        insights.append({
            "type": "alert",
            "icon": "fa-cloud-showers-heavy",
            "title": "Atmospheric Moisture Saturation",
            "detail": f"Relative humidity at {last_humidity:.0f}% with {last_clouds:.0f}% cloud coverage creates near-critical precipitation potential."
        })

    # Wind gusts
    last_wind = features_24h_11[-1, 3]
    if last_wind > 35:
        insights.append({
            "type": "warning",
            "icon": "fa-wind",
            "title": "Elevated Wind Speeds",
            "detail": f"Sustained winds of {last_wind:.1f} km/h indicate incoming squall or active frontal boundary."
        })
        
    return insights

def format_hour_to_11(h: WeatherHour) -> List[float]:
    temp = h.temperature
    hum = h.humidity
    cloud = h.cloud_cover
    rad = h.solar_radiation
    
    dew = h.dew_point if h.dew_point is not None else (temp - ((100.0 - hum) / 5.0))
    evapo = h.evapotranspiration if h.evapotranspiration is not None else max(0.0, (rad * 0.001) + (max(0.0, temp) * 0.01) - (hum * 0.001))
    
    return [
        temp, hum, h.pressure, h.wind_speed, h.wind_direction,
        h.soil_moisture, rad, cloud, dew, evapo, h.rainfall_mm
    ]

def transform_to_features_matrix(raw_11: np.ndarray, base_hour: int = 14, day_of_year: int = 180) -> np.ndarray:
    """
    Transforms (24, 11) raw array into (24, 19) or (24, 11) based on expected scaler features.
    """
    global expected_features_dim
    if expected_features_dim == 11 or scaler is None:
        return raw_11
        
    # Build 19 feature DataFrame
    df = pd.DataFrame(raw_11, columns=[
        'temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction',
        'soil_moisture', 'solar_radiation', 'cloud_cover', 'dew_point',
        'evapotranspiration', 'rainfall_mm'
    ])
    
    hours = np.array([(base_hour - (23 - i)) % 24 for i in range(24)])
    df['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
    
    days_in_year = 365.25
    df['day_sin'] = np.sin(2 * np.pi * day_of_year / days_in_year)
    df['day_cos'] = np.cos(2 * np.pi * day_of_year / days_in_year)
    
    df['pressure_roll_3'] = df['pressure'].rolling(window=3).mean().bfill()
    df['pressure_roll_6'] = df['pressure'].rolling(window=6).mean().bfill()
    df['humidity_roll_3'] = df['humidity'].rolling(window=3).mean().bfill()
    df['humidity_roll_6'] = df['humidity'].rolling(window=6).mean().bfill()
    
    target_col = 'rainfall_mm'
    features_19 = [c for c in df.columns if c != target_col] + [target_col]
    df = df[features_19]
    return df.values

# API Endpoints
@app.get("/api/status")
def get_status():
    global model, scaler
    if model is None:
        load_resources()
        
    return {
        "status": "online",
        "device": str(device),
        "model_type": type(model).__name__ if model else "None",
        "model_loaded": model is not None and os.path.exists('model.pth'),
        "scaler_loaded": scaler is not None,
        "features_dim": expected_features_dim,
        "features": FEATURES,
        "sequence_length": 24
    }

@app.get("/api/scenarios")
def get_scenarios():
    return [
        {
            "id": "thunderstorm",
            "name": "Approaching Thunderstorm",
            "description": "Rapid barometric pressure plunge, dense dark cumulonimbus clouds, escalating gusty winds and near-saturation humidity.",
            "icon": "bolt",
            "badge": "Severe Storm Risk",
            "color": "#ef4444",
            "weather": {
                "temperature": 21.5,
                "humidity": 94.0,
                "pressure": 993.0,
                "wind_speed": 42.0,
                "wind_direction": 225.0,
                "soil_moisture": 65.0,
                "solar_radiation": 40.0,
                "cloud_cover": 98.0,
                "dew_point": 20.3,
                "evapotranspiration": 0.05,
                "rainfall_mm": 1.2
            },
            "pressure_trend": "falling"
        },
        {
            "id": "monsoon",
            "name": "Tropical Monsoon Influx",
            "description": "Continuous high-humidity tropical flow, saturated ground moisture, thick overcast skies and sustained showers.",
            "icon": "cloud-rain",
            "badge": "Heavy Continuous Rain",
            "color": "#3b82f6",
            "weather": {
                "temperature": 26.0,
                "humidity": 92.0,
                "pressure": 1002.0,
                "wind_speed": 28.0,
                "wind_direction": 190.0,
                "soil_moisture": 85.0,
                "solar_radiation": 110.0,
                "cloud_cover": 92.0,
                "dew_point": 24.5,
                "evapotranspiration": 0.12,
                "rainfall_mm": 3.8
            },
            "pressure_trend": "steady"
        },
        {
            "id": "drizzle",
            "name": "Morning Misty Drizzle",
            "description": "Cool morning air, dew point matched to ambient temperature, low wind, overcast stratiform clouds.",
            "icon": "cloud-sun-rain",
            "badge": "Light Precipitation",
            "color": "#0ea5e9",
            "weather": {
                "temperature": 14.0,
                "humidity": 89.0,
                "pressure": 1012.0,
                "wind_speed": 10.0,
                "wind_direction": 90.0,
                "soil_moisture": 45.0,
                "solar_radiation": 90.0,
                "cloud_cover": 75.0,
                "dew_point": 12.2,
                "evapotranspiration": 0.08,
                "rainfall_mm": 0.0
            },
            "pressure_trend": "steady"
        },
        {
            "id": "clear_sky",
            "name": "Sunny Anticyclone",
            "description": "High barometric pressure dome, crisp clear skies, low relative humidity, strong direct solar irradiance.",
            "icon": "sun",
            "badge": "Dry & Fair",
            "color": "#10b981",
            "weather": {
                "temperature": 28.0,
                "humidity": 32.0,
                "pressure": 1024.0,
                "wind_speed": 12.0,
                "wind_direction": 45.0,
                "soil_moisture": 22.0,
                "solar_radiation": 880.0,
                "cloud_cover": 5.0,
                "dew_point": 9.5,
                "evapotranspiration": 0.75,
                "rainfall_mm": 0.0
            },
            "pressure_trend": "rising"
        },
        {
            "id": "passing_squall",
            "name": "Passing Squall Front",
            "description": "Sudden sharp wind shifts, moderate temperature dip, temporary pressure trough with moderate precipitation.",
            "icon": "wind",
            "badge": "Moderate Shower",
            "color": "#f59e0b",
            "weather": {
                "temperature": 18.5,
                "humidity": 82.0,
                "pressure": 1006.0,
                "wind_speed": 34.0,
                "wind_direction": 280.0,
                "soil_moisture": 48.0,
                "solar_radiation": 220.0,
                "cloud_cover": 80.0,
                "dew_point": 15.0,
                "evapotranspiration": 0.18,
                "rainfall_mm": 0.4
            },
            "pressure_trend": "falling"
        }
    ]

@app.get("/api/historical")
def get_historical_slice(start_idx: int = 1000, length: int = 24):
    if not os.path.exists('synthetic_weather_data.csv'):
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    df = pd.read_csv('synthetic_weather_data.csv')
    total = len(df)
    start_idx = max(0, min(start_idx, total - length - 1))
    
    slice_df = df.iloc[start_idx : start_idx + length]
    records = slice_df.to_dict(orient="records")
    next_actual_rain = float(df.iloc[start_idx + length]['rainfall_mm'])
    
    return {
        "start_index": start_idx,
        "total_records": total,
        "date_start": str(slice_df.iloc[0]['date']),
        "date_end": str(slice_df.iloc[-1]['date']),
        "sequence": records,
        "next_actual_rainfall_mm": round(next_actual_rain, 2)
    }

@app.post("/api/predict")
def predict_sequence(req: SequencePredictRequest):
    global model, scaler, expected_features_dim
    if model is None or scaler is None:
        load_resources()
        if model is None or scaler is None:
            raise HTTPException(status_code=500, detail="Model or Scaler not loaded.")
            
    if len(req.sequence) != 24:
        raise HTTPException(status_code=400, detail=f"Expected 24 timesteps, got {len(req.sequence)}")
        
    raw_11 = np.array([format_hour_to_11(h) for h in req.sequence]) # shape (24, 11)
    
    # Transform to expected feature matrix (11 or 19)
    feature_matrix = transform_to_features_matrix(raw_11, base_hour=req.hour_of_day or 14, day_of_year=req.day_of_year or 180)
    
    # Scale input
    scaled_matrix = scaler.transform(feature_matrix)
    input_tensor = torch.tensor(scaled_matrix, dtype=torch.float32).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        scaled_pred = model(input_tensor).cpu().numpy()[0, 0]
        
    # Target col idx is the last column
    target_idx = feature_matrix.shape[1] - 1
    dummy_pred = np.zeros((1, feature_matrix.shape[1]))
    dummy_pred[0, target_idx] = scaled_pred
    unscaled_pred = scaler.inverse_transform(dummy_pred)[0, target_idx]
    pred_mm = max(0.0, round(float(unscaled_pred), 2))
    
    category = categorize_rainfall(pred_mm)
    insights = compute_meteorological_insights(raw_11, pred_mm)
    
    return {
        "predicted_rainfall_mm": pred_mm,
        "category": category,
        "insights": insights,
        "current_conditions": {
            "temperature": round(raw_11[-1, 0], 1),
            "humidity": round(raw_11[-1, 1], 1),
            "pressure": round(raw_11[-1, 2], 1),
            "cloud_cover": round(raw_11[-1, 7], 1),
            "wind_speed": round(raw_11[-1, 3], 1)
        }
    }

@app.post("/api/predict-single")
def predict_single(req: SingleHourPredictRequest):
    curr = req.current
    trend = req.pressure_trend or "falling"
    hour = req.hour_of_day or 14
    day = req.day_of_year or 180
    
    seq = []
    for step in range(24):
        h_offset = 23 - step
        past_hour = (hour - h_offset) % 24
        
        diurnal_temp = 3.0 * np.sin(2 * np.pi * (past_hour - 8) / 24)
        curr_diurnal = 3.0 * np.sin(2 * np.pi * (hour - 8) / 24)
        t_val = curr.temperature + (diurnal_temp - curr_diurnal)
        
        if trend == "falling":
            p_val = curr.pressure + (h_offset * 0.25)
        elif trend == "rising":
            p_val = curr.pressure - (h_offset * 0.25)
        else:
            p_val = curr.pressure + np.sin(step) * 0.3
            
        factor = (24 - h_offset) / 24.0
        if trend == "falling":
            c_val = max(10.0, curr.cloud_cover * (0.4 + 0.6 * factor))
            hum_val = min(100.0, curr.humidity * (0.6 + 0.4 * factor))
            w_val = max(5.0, curr.wind_speed * (0.5 + 0.5 * factor))
        else:
            c_val = max(5.0, curr.cloud_cover * (1.2 - 0.2 * factor))
            hum_val = max(20.0, curr.humidity * (1.1 - 0.1 * factor))
            w_val = max(5.0, curr.wind_speed * (1.1 - 0.1 * factor))
            
        daylight = (past_hour > 6) and (past_hour < 18)
        if daylight:
            rad_val = 800.0 * np.sin(np.pi * (past_hour - 6) / 12) * (1.0 - 0.7 * (c_val / 100.0))
        else:
            rad_val = 0.0
            
        dew_val = t_val - ((100.0 - hum_val) / 5.0)
        evapo_val = max(0.0, (rad_val * 0.001) + (max(0.0, t_val) * 0.01) - (hum_val * 0.001))
        
        rain_val = 0.0
        if step == 23:
            rain_val = curr.rainfall_mm
        elif trend == "falling" and c_val > 80 and step > 18:
            rain_val = np.random.uniform(0.0, 0.4)
            
        seq.append(WeatherHour(
            temperature=round(float(t_val), 1),
            humidity=round(float(hum_val), 1),
            pressure=round(float(p_val), 1),
            wind_speed=round(float(w_val), 1),
            wind_direction=curr.wind_direction,
            soil_moisture=round(float(curr.soil_moisture), 1),
            solar_radiation=round(float(rad_val), 1),
            cloud_cover=round(float(c_val), 1),
            dew_point=round(float(dew_val), 1),
            evapotranspiration=round(float(evapo_val), 2),
            rainfall_mm=round(float(rain_val), 2)
        ))
        
    return predict_sequence(SequencePredictRequest(sequence=seq, hour_of_day=hour, day_of_year=day))

@app.post("/api/predict-forecast")
def predict_multi_step_forecast(req: ForecastRequest):
    global model, scaler, expected_features_dim
    if model is None or scaler is None:
        load_resources()
        if model is None or scaler is None:
            raise HTTPException(status_code=500, detail="Model or Scaler not loaded.")
            
    if len(req.sequence) != 24:
        raise HTTPException(status_code=400, detail=f"Expected 24 timesteps, got {len(req.sequence)}")
        
    raw_11 = np.array([format_hour_to_11(h) for h in req.sequence])
    base_hour = req.hour_of_day or 14
    
    forecast = []
    working_raw_11 = np.copy(raw_11)
    
    for step in range(req.steps):
        current_hour = (base_hour + step) % 24
        feature_matrix = transform_to_features_matrix(working_raw_11, base_hour=current_hour)
        scaled_window = scaler.transform(feature_matrix)
        input_tensor = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            scaled_pred = model(input_tensor).cpu().numpy()[0, 0]
            
        target_idx = feature_matrix.shape[1] - 1
        dummy_pred = np.zeros((1, feature_matrix.shape[1]))
        dummy_pred[0, target_idx] = scaled_pred
        unscaled_pred = scaler.inverse_transform(dummy_pred)[0, target_idx]
        pred_mm = max(0.0, round(float(unscaled_pred), 2))
        
        cat = categorize_rainfall(pred_mm)
        forecast.append({
            "hour_ahead": step + 1,
            "predicted_rainfall_mm": pred_mm,
            "category": cat
        })
        
        # Roll forward
        next_row = np.copy(working_raw_11[-1])
        next_row[10] = pred_mm # rainfall_mm
        if pred_mm > 0.1:
            next_row[5] = min(100.0, next_row[5] + pred_mm * 1.5)
            next_row[2] = max(970.0, next_row[2] - 0.2)
            next_row[1] = min(100.0, next_row[1] + 1.0)
        else:
            next_row[2] = min(1030.0, next_row[2] + 0.3)
            
        working_raw_11 = np.vstack([working_raw_11[1:], next_row])
        
    return {
        "forecast": forecast
    }

# Mount static folder
os.makedirs("static", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
