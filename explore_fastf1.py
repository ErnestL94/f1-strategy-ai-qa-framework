import fastf1 as ff1
import pandas as pd


# Enable caching to speed up data retrieval
ff1.Cache.enable_cache('data/cache')   

# Load Silverstone 2023 Race
print("Loading Silverstone 2023 Race data...")
session = ff1.get_session(2023, 'Silverstone', 'R')
session.load()

print("/n" + "=" *60)
print("Silverstone 2023 - Data available")
print("=" *60)

#1. Race Results
print("\n Race Results:")
print(session.results[['Abbreviation', 'Position', 'GridPosition', 'Points']].head(10))

#2. Driver's laps e.g. Verstappen
ver_laps = session.laps.pick_drivers('VER')
print(f"\n Verstappen's Race:")
print(f"Total Laps: {len(ver_laps)}")
print(f"Fastest Lap: {ver_laps['LapTime'].min()}")

#3. Pit Stops
print("\n Pit Stops:")
for driver in ['VER', 'HAM', 'LEC']:
    driver_laps = session.laps.pick_drivers(driver)
    pit_laps = driver_laps[driver_laps['PitOutTime'].notna()]
    print(f"\n{driver}:")
    for _, lap in pit_laps.iterrows():
       print(f" Lap {lap['LapNumber']}: {lap['Compound']} -> (pit) -> Next Compound")

#4. Weather Data
print("\n Weather Data:")
print(session.weather_data.head())

#5. Interesting strategic moment
print("\n Strategic Moment Analysis - Lap 40-45:")
lap_42 = ver_laps[ver_laps['LapNumber'] == 42].iloc[0]
print(f"Verstappen Lap 42:")
print(f"Tire: {lap_42['Compound']}")
print(f"Tire Age: {lap_42['TyreLife']} laps")
print(f"Lap Time: {lap_42['LapTime']}")
print(f"Position: {lap_42['Position']}")

print("\nAnalysis complete for data needed.")