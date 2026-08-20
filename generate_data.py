import pandas as pd
import numpy as np

def generate_insat_weather_data(output_file='synthetic_weather_data.csv'):
    """
    Generates synthetic 5-year hourly dataset for INSAT-3D/3DR High Impact Rainfall Nowcasting.
    Features (11 Channels):
    1. tir1_temp: INSAT-3D Thermal IR 10.8µm Cloud Top Temperature (°C) [-85 to 40]
    2. wv_channel: INSAT-3DR 6.8µm Upper Tropospheric Water Vapor (%) [0 to 100]
    3. cloud_top_height: Convective Cloud Vertical Depth (km) [0 to 18]
    4. cape_index: Convective Available Potential Energy (J/kg) [0 to 5000]
    5. pressure: Surface Barometric Pressure (hPa) [960 to 1040]
    6. humidity: Boundary Layer Relative Humidity (%) [0 to 100]
    7. temperature: Surface Ambient Temperature (°C) [-10 to 50]
    8. moisture_conv: Horizontal Water Vapor Flux Convergence (g/kg/h) [0 to 15]
    9. wind_speed: Surface Wind Squall Speed (km/h) [0 to 160]
    10. wind_shear: 850-200 hPa Deep Layer Shear (m/s) [0 to 40]
    11. rainfall_mm: Satellite Hydro-Estimator Rain Rate (mm/h) [0 to 150]
    """
    np.random.seed(42)
    
    # 5 years, hourly = 43,800 records
    dates = pd.date_range(start='2020-01-01', periods=43800, freq='h')
    n = len(dates)
    
    hour_of_day = dates.hour.values
    day_of_year = dates.dayofyear.values
    
    # Monsoon Seasonality (June to October: Days 150 to 280)
    monsoon_factor = np.exp(-0.5 * ((day_of_year - 215) / 45.0) ** 2) # Peak in early August
    diurnal_heating = np.sin(np.pi * (hour_of_day - 6) / 12)
    diurnal_heating = np.where((hour_of_day >= 6) & (hour_of_day <= 20), diurnal_heating, -0.3)
    
    # Baseline Surface Temperature (°C)
    base_temp = 24.0 + 8.0 * np.sin(2 * np.pi * (day_of_year - 100) / 365.25) + 5.0 * diurnal_heating
    temperature = base_temp + np.random.normal(0, 1.5, n)
    temperature = np.clip(temperature, -5.0, 48.0)
    
    # Synoptic Pressure (hPa) - Low in monsoon & cyclonic depressions
    synoptic_pressure = 1012.0 - 10.0 * monsoon_factor + np.random.normal(0, 2.5, n)
    # Diurnal semi-diurnal atmospheric tidal wave
    pressure = synoptic_pressure + 1.2 * np.cos(4 * np.pi * hour_of_day / 24)
    
    # CAPE Instability (J/kg) - Spikes with afternoon heating and monsoon moisture
    base_cape = 400.0 + 2200.0 * monsoon_factor + 600.0 * np.maximum(0, diurnal_heating)
    cape_index = np.clip(base_cape + np.random.exponential(400, n), 50.0, 4800.0)
    
    # Boundary Layer Humidity (%)
    humidity = 45.0 + 45.0 * monsoon_factor - 0.4 * (temperature - 25.0) + np.random.normal(0, 5, n)
    humidity = np.clip(humidity, 15.0, 99.0)
    
    # Moisture Flux Convergence (g/kg/h)
    moisture_conv = np.maximum(0.1, 1.5 + 5.5 * monsoon_factor + np.random.normal(0, 1.0, n))
    
    # Vertical Wind Shear (m/s)
    wind_shear = np.clip(8.0 + 10.0 * monsoon_factor + np.random.normal(0, 3.0, n), 2.0, 38.0)
    
    # Wind Speed (km/h)
    wind_speed = np.clip(12.0 + 8.0 * monsoon_factor + np.random.exponential(10, n), 2.0, 140.0)
    
    # Convective Cell Activity Index (Smoothed cluster for storms)
    storm_noise = np.random.normal(0, 1, n)
    storm_wave = np.convolve(storm_noise, np.ones(10)/10.0, mode='same')
    
    convective_trigger = (monsoon_factor * 0.5) + (cape_index / 5000.0 * 0.4) + (moisture_conv / 15.0 * 0.3) + (storm_wave * 0.3)
    is_convective_storm = convective_trigger > 0.65
    is_extreme_cloudburst = (convective_trigger > 0.90) & (cape_index > 2800) & (moisture_conv > 7.0)
    is_moderate_rain = (convective_trigger > 0.45) & (~is_convective_storm)
    
    # Satellite Channels:
    # 1. TIR-1 Brightness Temperature (°C) - Very cold in convective storm cores
    tir1_temp = np.zeros(n)
    # Default fair weather cloud top temp (or ground surface temp when clear)
    tir1_temp[~is_convective_storm & ~is_moderate_rain] = 12.0 - 0.3 * temperature[~is_convective_storm & ~is_moderate_rain] + np.random.normal(0, 5, np.sum(~is_convective_storm & ~is_moderate_rain))
    # Moderate clouds
    tir1_temp[is_moderate_rain] = np.random.uniform(-35.0, -10.0, np.sum(is_moderate_rain))
    # Convective storm clouds
    tir1_temp[is_convective_storm] = np.random.uniform(-65.0, -40.0, np.sum(is_convective_storm))
    # Extreme Cloudburst deep convective towers
    tir1_temp[is_extreme_cloudburst] = np.random.uniform(-82.0, -66.0, np.sum(is_extreme_cloudburst))
    tir1_temp = np.clip(tir1_temp, -85.0, 38.0)
    
    # 2. Water Vapor 6.8µm Channel (%)
    wv_channel = np.clip(30.0 + 50.0 * monsoon_factor + (humidity * 0.2) + np.random.normal(0, 6, n), 10.0, 99.0)
    wv_channel[is_convective_storm] = np.random.uniform(85.0, 99.0, np.sum(is_convective_storm))
    
    # 3. Cloud Top Height (km)
    cloud_top_height = np.zeros(n)
    cloud_top_height[~is_convective_storm & ~is_moderate_rain] = np.random.uniform(0.5, 4.0, np.sum(~is_convective_storm & ~is_moderate_rain))
    cloud_top_height[is_moderate_rain] = np.random.uniform(5.0, 10.0, np.sum(is_moderate_rain))
    cloud_top_height[is_convective_storm] = np.random.uniform(11.0, 15.5, np.sum(is_convective_storm))
    cloud_top_height[is_extreme_cloudburst] = np.random.uniform(15.0, 17.8, np.sum(is_extreme_cloudburst))
    
    # Adjust pressure, moisture conv, wind speed during storms
    pressure[is_convective_storm] -= np.random.uniform(4.0, 12.0, np.sum(is_convective_storm))
    pressure[is_extreme_cloudburst] -= np.random.uniform(8.0, 20.0, np.sum(is_extreme_cloudburst))
    moisture_conv[is_convective_storm] += np.random.uniform(3.0, 7.0, np.sum(is_convective_storm))
    moisture_conv[is_extreme_cloudburst] += np.random.uniform(6.0, 10.0, np.sum(is_extreme_cloudburst))
    wind_speed[is_convective_storm] += np.random.uniform(15.0, 45.0, np.sum(is_convective_storm))
    humidity[is_convective_storm] = np.random.uniform(88.0, 99.0, np.sum(is_convective_storm))
    
    # Synthetic Rainfall Rate (mm/h)
    rainfall_mm = np.zeros(n)
    # Light/Moderate rain
    rainfall_mm[is_moderate_rain] = np.random.exponential(3.5, np.sum(is_moderate_rain))
    # Heavy convective rain (15-50 mm/h)
    rainfall_mm[is_convective_storm] = np.random.uniform(15.0, 48.0, np.sum(is_convective_storm)) + np.random.exponential(6.0, np.sum(is_convective_storm))
    # Extreme Cloudbursts (50-130 mm/h)
    rainfall_mm[is_extreme_cloudburst] = np.random.uniform(55.0, 115.0, np.sum(is_extreme_cloudburst)) + np.random.exponential(12.0, np.sum(is_extreme_cloudburst))
    rainfall_mm = np.clip(rainfall_mm, 0.0, 150.0)
    
    # Add False Alarm / Cirrus Anvil Overhang Case (cold clouds without rain)
    # In 2% of dry hours, set tir1_temp cold (-55°C) but low humidity and 0 moisture conv, 0 rain
    cirrus_mask = (~is_convective_storm & ~is_moderate_rain) & (np.random.uniform(0, 1, n) < 0.02)
    tir1_temp[cirrus_mask] = np.random.uniform(-65.0, -48.0, np.sum(cirrus_mask))
    cloud_top_height[cirrus_mask] = np.random.uniform(12.0, 14.5, np.sum(cirrus_mask))
    humidity[cirrus_mask] = np.random.uniform(25.0, 55.0, np.sum(cirrus_mask))
    moisture_conv[cirrus_mask] = np.random.uniform(0.1, 1.2, np.sum(cirrus_mask))
    rainfall_mm[cirrus_mask] = 0.0
    
    # Construct DataFrame with exactly the 11 INSAT features
    df = pd.DataFrame({
        'date': dates,
        'tir1_temp': np.round(tir1_temp, 2),
        'wv_channel': np.round(wv_channel, 2),
        'cloud_top_height': np.round(cloud_top_height, 2),
        'cape_index': np.round(cape_index, 1),
        'pressure': np.round(pressure, 2),
        'humidity': np.round(humidity, 2),
        'temperature': np.round(temperature, 2),
        'moisture_conv': np.round(moisture_conv, 2),
        'wind_speed': np.round(wind_speed, 2),
        'wind_shear': np.round(wind_shear, 2),
        'rainfall_mm': np.round(rainfall_mm, 2)
    })
    
    df.set_index('date', inplace=True)
    df.to_csv(output_file)
    print(f"Generated {n} records of INSAT-3D/3DR satellite & meteorological data -> {output_file}")
    print("Rainfall statistics:")
    print(f"  Dry hours: {np.sum(rainfall_mm < 0.1)} ({np.mean(rainfall_mm < 0.1)*100:.1f}%)")
    print(f"  Light (<7.6 mm): {np.sum((rainfall_mm >= 0.1) & (rainfall_mm < 7.6))}")
    print(f"  Moderate (7.6-35.6 mm): {np.sum((rainfall_mm >= 7.6) & (rainfall_mm < 35.6))}")
    print(f"  Heavy (35.6-64.5 mm): {np.sum((rainfall_mm >= 35.6) & (rainfall_mm < 64.5))}")
    print(f"  Extreme / Cloudburst (>64.5 mm): {np.sum(rainfall_mm >= 64.5)}")

if __name__ == '__main__':
    generate_insat_weather_data()

