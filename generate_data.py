import pandas as pd
import numpy as np

def generate_synthetic_weather_data(output_file='synthetic_weather_data.csv'):
    np.random.seed(42)
    
    # 5 years, hourly
    dates = pd.date_range(start='2020-01-01', periods=43800, freq='h')
    n = len(dates)
    
    # Time variables
    hour_of_day = dates.hour.values
    day_of_year = dates.dayofyear.values
    
    # 1. Temperature (°C)
    # Yearly seasonal variation + daily variation + noise
    # Base temp around 15C, yearly amplitude 10C, daily amplitude 5C
    yearly_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    daily_temp = 5 * np.sin(2 * np.pi * (hour_of_day - 8) / 24)
    temperature = yearly_temp + daily_temp + np.random.normal(0, 2, n)
    
    # 2. Barometric Pressure (hPa)
    # Base pressure with yearly seasonality
    pressure = 1013 + 10 * np.sin(2 * np.pi * day_of_year / 365.25) + np.random.normal(0, 3, n)
    
    # Create a smoothed random walk to represent broad weather fronts (storms)
    # A larger window (24 hours) makes weather systems last realistically long
    storm_front = np.convolve(np.random.normal(0, 1, n), np.ones(24)/24, mode='same')
    
    # Physical Correlation 1: Pressure drops significantly when a storm front moves in
    pressure -= storm_front * 8
    
    # 3. Cloud Cover (0-100%)
    # Clouds increase with the storm front
    cloud_cover = 30 + storm_front * 40 + np.random.normal(0, 10, n)
    cloud_cover = np.clip(cloud_cover, 0, 100)
    
    # 4. Relative Humidity (%)
    # Physical Correlation 2: Humidity correlates with cloud cover and inversely with temperature
    humidity = 60 - 0.5 * temperature + 0.4 * cloud_cover + np.random.normal(0, 5, n)
    humidity += storm_front * 15  # Humidity spikes during storms
    humidity = np.clip(humidity, 10, 100)
    
    # 5. Rain Generation (Fixing Zero-Inflation)
    # Rain happens when humidity is high, pressure is low, AND the storm front is active
    # By tuning these thresholds, we can target ~15% rainy hours instead of 1.7%
    is_raining = (humidity > 75) & (pressure < 1012) & (storm_front > 0.2)
    
    # Add random scattered showers (2% chance) to prevent perfect deterministic predictability
    scattered_showers = np.random.random(n) < 0.02
    is_raining = is_raining | scattered_showers
    
    # Generate Rainfall Volume
    rainfall_mm = np.zeros(n)
    
    # Physical Correlation 3: Rain intensity is driven by how low the pressure drops
    # The lower the pressure, the higher the exponential scale (heavier rain)
    intensity_scale = np.clip((1015 - pressure) * 0.5, 0.5, 10.0)
    
    # Apply exponential distribution so most rain is light, but extremes can happen
    rainfall_mm[is_raining] = np.random.exponential(scale=intensity_scale[is_raining])
    
    # 6. Wind Speed (km/h)
    # Higher during storms (low pressure)
    wind_speed = np.random.lognormal(mean=2, sigma=0.5, size=n)
    wind_speed[is_raining] += np.random.uniform(5, 20, np.sum(is_raining))
    wind_speed = np.clip(wind_speed, 0, 150)
    
    # 6. Wind Direction (degrees)
    # Random walk mostly
    wind_dir = (np.cumsum(np.random.normal(0, 10, n)) % 360)
    
    # 7. Soil Moisture (%)
    # Spikes after rain, decays exponentially
    soil_moisture = np.zeros(n)
    current_moisture = 20.0
    for i in range(n):
        if rainfall_mm[i] > 0:
            current_moisture = min(100.0, current_moisture + rainfall_mm[i] * 5)
        else:
            current_moisture = max(10.0, current_moisture * 0.995) # decay
        soil_moisture[i] = current_moisture
        
    # 8. Solar Radiation (W/m²)
    # Depends on time of day and cloud cover
    # Peaks around solar noon (12:00)
    solar_rad = np.zeros(n)
    daylight = (hour_of_day > 6) & (hour_of_day < 18)
    solar_rad[daylight] = 800 * np.sin(np.pi * (hour_of_day[daylight] - 6) / 12)
    # Reduce by cloud cover
    solar_rad = solar_rad * (1 - 0.7 * (cloud_cover / 100))
    solar_rad = np.clip(solar_rad + np.random.normal(0, 10, n), 0, 1200)
    solar_rad[~daylight] = 0
    
    # 9. Dew Point (°C)
    # Approximation formula based on temp and humidity
    dew_point = temperature - ((100 - humidity) / 5.0)
    
    # 10. Evapotranspiration Rate (mm/hr)
    # Depends on solar radiation, temperature, and inversely on humidity
    evapo = (solar_rad * 0.001) + (np.maximum(0, temperature) * 0.01) - (humidity * 0.001)
    evapo = np.clip(evapo + np.random.normal(0, 0.05, n), 0, 2)
    evapo[~daylight] = 0
    
    # Compile dataset
    df = pd.DataFrame({
        'date': dates,
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed,
        'wind_direction': wind_dir,
        'soil_moisture': soil_moisture,
        'solar_radiation': solar_rad,
        'cloud_cover': cloud_cover,
        'dew_point': dew_point,
        'evapotranspiration': evapo,
        'rainfall_mm': rainfall_mm
    })
    
    df.set_index('date', inplace=True)
    df.to_csv(output_file)
    print(f"Generated {n} rows of synthetic weather data and saved to {output_file}")

if __name__ == '__main__':
    generate_synthetic_weather_data()
