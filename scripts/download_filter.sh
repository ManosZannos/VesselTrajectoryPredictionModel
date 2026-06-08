#!/bin/bash
# ============================================================
# download_filter.sh
# Downloads January 2017 AIS data (day-by-day format) and
# filters immediately to San Diego Harbor to save disk space.
#
# Output: raw_data/san_diego_jan2017.csv
# Run from project root: bash scripts/download_filter.sh
# ============================================================
set -e

BASE_URL="https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2017"

echo "=== Setting up directories ==="
mkdir -p raw_data processed_data
cd raw_data

# Write CSV header once
echo "MMSI,BaseDateTime,LAT,LON,SOG,Heading" > san_diego_jan2017.csv

echo "=== Starting download of January 2017 AIS data ==="
echo "    Filtering to San Diego: LAT [32,33], LON [-118,-117]"
echo ""

for day in $(seq -w 1 31); do
    ZIPFILE="AIS_2017_01_${day}.zip"
    CSVFILE="AIS_2017_01_${day}.csv"

    echo "--- Day $day / 31 ---"

    # Download
    echo "  Downloading $ZIPFILE ..."
    wget -q --show-progress "${BASE_URL}/${ZIPFILE}"

    # Unzip (extracts CSVFILE in current directory)
    echo "  Extracting ..."
    unzip -q "$ZIPFILE"
    rm "$ZIPFILE"

    # Filter to San Diego and append to master CSV
    echo "  Filtering ..."
    python3 ../filter_day.py "$day"

    # Remove full US-wide CSV immediately to save space
    rm -f "$CSVFILE"

    echo "  Cumulative records: $(( $(wc -l < san_diego_jan2017.csv) - 1 ))"
    echo "  Working dir size:   $(du -sh . | cut -f1)"
    echo ""
done

echo "============================================"
echo "Download complete."
echo "Final record count: $(( $(wc -l < san_diego_jan2017.csv) - 1 ))"
echo "File size:          $(du -sh san_diego_jan2017.csv | cut -f1)"
echo "============================================"
