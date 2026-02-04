"""
FastF1 Race Data Explorer

Usage:
    python explore_fastf1.py --year 2023 --track Silverstone
    python explore_fastf1.py --year 2023 --track Spa-Francorchamps
    python explore_fastf1.py --year 2023 --track Singapore
"""

import argparse
from pathlib import Path

import fastf1


def load_race_session(year: int, track: str, session_type: str = "R"):
    """Load a race session from FastF1"""
    # Enable cache
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    print(f"Loading {track} {year} Race...")
    session = fastf1.get_session(year, track, session_type)
    session.load()

    return session


def main():
    parser = argparse.ArgumentParser(description="Explore F1 race data")
    parser.add_argument("--year", type=int, default=2023, help="Season year")
    parser.add_argument("--track", type=str, required=True, help="Track name")
    parser.add_argument("--drivers", nargs="+", help="Specific drivers (default: top 5)")

    args = parser.parse_args()

    # Load session
    session = load_race_session(args.year, args.track)

    print("\n" + "=" * 60)
    print(f"{args.track.upper()} {args.year} - DATA AVAILABLE")
    print("=" * 60)

    # 1. Race Results
    print("\n📊 RACE RESULTS:")
    print(session.results[["Abbreviation", "Position", "GridPosition", "Points"]].head(10))

    # 2. Pit stops
    drivers = args.drivers if args.drivers else session.results.head(5)["Abbreviation"].tolist()

    print("\n🔧 PIT STOPS:")
    for driver in drivers:
        driver_laps = session.laps.pick_drivers(driver)
        pit_laps = driver_laps[driver_laps["PitOutTime"].notna()]

        print(f"\n{driver}:")
        for _, lap in pit_laps.iterrows():
            print(f"  Lap {lap['LapNumber']}: {lap['Compound']}")

    # 3. Weather data
    print("\n🌦️ WEATHER DATA AVAILABLE:")
    print(session.weather_data.head())

    # 4. Example lap detail
    print(f"\n🎯 EXAMPLE - {drivers[0]} Lap 30:")
    driver_laps = session.laps.pick_drivers(drivers[0])
    try:
        lap_30 = driver_laps[driver_laps["LapNumber"] == 30].iloc[0]
        print(f"  Tire: {lap_30['Compound']}")
        print(f"  Tire Age: {lap_30['TyreLife']} laps")
        print(f"  Lap Time: {lap_30['LapTime']}")
        print(f"  Position: {lap_30['Position']}")
    except:
        print("  (Lap 30 data not available)")

    print("\n✅ Explore the data above and identify 3-5 strategic moments!")


if __name__ == "__main__":
    main()
