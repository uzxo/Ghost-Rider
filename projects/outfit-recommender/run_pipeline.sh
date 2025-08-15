#!/bin/bash
# Usage: ./run_pipeline.sh YOUR_ACCESS_KEY

YOUR_ACCESS_KEY=$1

if [ -z "$YOUR_ACCESS_KEY" ]; then
  echo "Usage: $0 YOUR_ACCESS_KEY"
  exit 1
fi

export YOUR_ACCESS_KEY=$1

Rscript /Users/uzairzaidi/Desktop/product_scraping.R
Rscript /Users/uzairzaidi/Desktop/weatherstack_api.R
Rscript /Users/uzairzaidi/Desktop/etl.R

Rscript /Users/uzairzaidi/Desktop/run_ootd_api.R &
sleep 5

curl "http://localhost:8000/ootd" --output ootd_plot.png
echo "Outfit of the Day plot saved as ootd_plot.png"
