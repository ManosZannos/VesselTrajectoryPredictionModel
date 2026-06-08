#!/usr/bin/env python3
"""
preprocess_sfdata.py
Adapted from S&F's preprocess_data.py for the day-based NOAA AIS format.

Reads:   raw_data/san_diego_jan2017.csv   (merged San Diego data)
Writes:  processed_data/11.csv            (compatible with grid.py)

What it does (same as original preprocess_data.py):
  - Per-vessel: ceil timestamps to nearest minute
  - Remove duplicate timestamps (keep first)
  - Resample to 1-minute intervals with linear interpolation (max 5-gap fill)
  - Fix Heading=511 (invalid AIS value) with forward/backward fill
  - Drop rows with missing LAT/LON
"""
from __future__ import print_function
import os
import sys
sys.dont_write_bytecode = True
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

IN_FILE  = 'raw_data/san_diego_jan2017.csv'
OUT_FILE = 'processed_data/11.csv'

if not os.path.isdir('processed_data/'):
    os.makedirs('processed_data/')

# ── Load ──────────────────────────────────────────────────────────────
print(f"Reading {IN_FILE} ...")
df = pd.read_csv(
    IN_FILE,
    header=0,
    parse_dates=['BaseDateTime'],
    usecols=['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'Heading']
)
df.sort_values(['BaseDateTime'], inplace=True)
vessels = df['MMSI'].unique()
print(f"Unique vessels : {len(vessels):,}")
print(f"Total records  : {len(df):,}")
print(f"Date range     : {df['BaseDateTime'].min()} → {df['BaseDateTime'].max()}")
print()

# ── Per-vessel resampling ─────────────────────────────────────────────
frames = []

for v, vessel in enumerate(vessels):
    print(f"\r  {v+1:>4}/{len(vessels)}  MMSI {vessel}", end='', flush=True)

    vd = df.loc[df['MMSI'] == vessel].copy()

    # Round up to nearest minute, drop duplicates
    vd['BaseDateTime'] = pd.to_datetime(
        vd['BaseDateTime'], format='%Y-%m-%dT%H:%M:%S', errors='coerce'
    )
    vd['BaseDateTime'] = vd['BaseDateTime'].dt.ceil('min')
    vd = vd.loc[~vd['BaseDateTime'].duplicated(keep='first')]

    # Resample to 1-min grid, interpolate gaps ≤5 min
    vd = vd.set_index('BaseDateTime').resample('1min').interpolate(limit=5)
    vd.reset_index('BaseDateTime', inplace=True)
    vd = vd.dropna(subset=['LAT', 'LON'])

    if vd.empty:
        continue

    try:
        vd = vd.set_index('BaseDateTime')

        # Fix Heading=511 (invalid AIS sentinel value)
        vd['Heading'] = vd['Heading'].round().astype('int32')
        if not len(vd['Heading'].unique()) == 1:
            if 511 in vd['Heading'].values:
                vd['Heading'].replace(511, method='ffill', inplace=True)
                vd['Heading'].replace(511, method='bfill', inplace=True)

        frames.append(vd)
    except (ValueError, TypeError):
        pass  # skip vessels with entirely invalid heading

# ── Merge & save ──────────────────────────────────────────────────────
print(f"\n\nSuccessfully processed {len(frames):,} vessels.")
print("Combining and sorting ...")

out_frame = pd.concat(frames)
out_frame.index.name = 'BaseDateTime'
out_frame.sort_values(['BaseDateTime'], inplace=True)

print(f"Saving to {OUT_FILE} ...")
out_frame.to_csv(OUT_FILE, index=True)

print()
print("=== preprocess_sfdata.py complete ===")
print(f"  Output shape : {out_frame.shape}")
print(f"  Date range   : {out_frame.index.min()} → {out_frame.index.max()}")
print(f"  Output file  : {OUT_FILE}")
