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
    # Mean around 1013 hPa, fluctuates around 990 to 1030
    # Pressure drops before and during rain
    pressure = 1013 + 10 * np.sin(2 * np.pi * day_of_year / 365.25) + np.random.normal(0, 3, n)
    
    # Let's define rain events first so we can correlate other variables.
    # Rain is more likely when pressure is dropping and in certain seasons, but let's make it simpler.
    # Probability of rain increases when pressure is low.
    rain_prob = np.where(pressure < 1010, 0.15, 0.02) 
    # Add a bit of seasonality to rain
    rain_prob += 0.05 * np.sin(2 * np.pi * day_of_year / 365.25)
    rain_prob = np.clip(rain_prob, 0, 1)
    
    # Generating rain events (Markov chain like would be better, let's use a smoothed random walk for storms)
    storm_index = np.convolve(np.random.normal(0, 1, n), np.ones(6)/6, mode='same')
    is_raining = (storm_index > 0.8) & (rain_prob > 0.05)
    
    rainfall_mm = np.zeros(n)
    rainfall_mm[is_raining] = np.random.exponential(2, np.sum(is_raining)) # mostly light rain, some heavy
    
    # 3. Cloud Cover (0-100%)
    # Highly correlated with rain. Near 100% when raining.
    cloud_cover = np.random.uniform(0, 50, n)
    cloud_cover[is_raining] = np.random.uniform(80, 100, np.sum(is_raining))
    cloud_cover = np.clip(cloud_cover + np.random.normal(0, 10, n), 0, 100)
    
    # Drop pressure specifically when it rains to enforce correlation
    pressure[is_raining] -= np.random.uniform(2, 5, np.sum(is_raining))
    
    # 4. Relative Humidity (%)
    # Correlated with cloud cover and rain, inversely with temperature
    humidity = 50 - 0.5 * temperature + 0.3 * cloud_cover + np.random.normal(0, 5, n)
    humidity[is_raining] = np.random.uniform(85, 100, np.sum(is_raining))
    humidity = np.clip(humidity, 10, 100)
    
    # 5. Wind Speed (km/h)
    # Higher during storms (low pressure, raining)
    wind_speed = np.random.lognormal(mean=2, sigma=0.5, size=n)
    wind_speed[is_raining] += np.random.uniform(5, 15, np.sum(is_raining))
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
