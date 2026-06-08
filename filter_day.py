#!/usr/bin/env python3
"""
Filter one day of raw US-wide AIS data to San Diego Harbor bounds.
Called by download_filter.sh for each day.

Usage: python3 filter_day.py <DD>  (e.g. python3 filter_day.py 01)
"""
import sys
import pandas as pd

# San Diego Harbor bounds (matching grid.py hard-coded filter)
LAT_MIN, LAT_MAX = 32.0, 33.0
LON_MIN, LON_MAX = -118.0, -117.0

day = sys.argv[1]
filename = f"AIS_2017_01_{day}.csv"

try:
    df = pd.read_csv(
        filename,
        usecols=['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'Heading'],
        low_memory=False
    )
    total = len(df)

    mask = (
        (df['LAT'] >= LAT_MIN) & (df['LAT'] <= LAT_MAX) &
        (df['LON'] >= LON_MIN) & (df['LON'] <= LON_MAX)
    )
    df_sd = df[mask]

    # Append to master CSV (header already written by shell script)
    df_sd.to_csv('san_diego_jan2017.csv', mode='a', header=False, index=False)
    print(f"Day {day}: {len(df_sd):>6,} San Diego records  (from {total:>9,} total US records)")

except FileNotFoundError:
    print(f"WARNING: {filename} not found — skipping day {day}")
except Exception as e:
    print(f"ERROR processing day {day}: {e}")
