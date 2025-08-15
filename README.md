---
title: "AM05 AUT24 Final Project: Outfit Recommendation System"
author: "Uzair Zaidi"
editor_options: 
  markdown: 
    wrap: 72
---

# Introduction

This project implements an automated "Outfit of the Day" (OOTD)
recommendation system based on the current weather conditions in London.
The system integrates the following components:

-   **Data Acquisition (Web Scraping):** Scrapes clothing items from
    ASOS, downloading product images and details.
-   **ETL and Database Management:** Cleans and stores scraped data into
    a SQLite database (`closet.db`).
-   **Weather Data Integration:** Fetches current weather data for
    London using the Weatherstack API.
-   **Recommendation Logic:** Applies simple rules to select outfits
    (tops, bottoms, shoes, coat, and accessories) based on weather
    conditions.
-   **API Endpoints:** Provides a Plumber API with `/ootd` and
    `/rawdata` endpoints.
-   **Automation with Bash Script:** A `run_pipeline.sh` script
    automates the entire workflow from data scraping to generating the
    final OOTD image.

By the end of the pipeline, the system produces an `ootd_plot.png` file
showing the recommended outfit annotated with the current date, weather,
and temperature.

# Prerequisites

## Software & Dependencies

-   **R (version ≥ 3.6)** and **Rscript**
-   **curl** installed on your system
-   **SQLite** (no additional server needed, just RSQLite)
-   **ImageMagick** installed for the `magick` R package.

## R Packages

Install the required packages:

``` r
install.packages(c(
  "rvest", "httr", "jsonlite", "DBI", "RSQLite", "plumber",
  "magick", "dplyr", "magrittr"
))
```

# Project Structure

-   **product_scraping.R**: Scrapes product data and images from the
    web.

-   **weatherstack_api.R**: Fetches current weather data using the
    Weatherstack API.

-   **etl.R**: Cleans data and populates the SQLite database.

-   **ootd_api.R**: Defines the API endpoints using Plumber.

-   **run_ootd_api.R**: Runs the API server.

-   **run_pipeline.sh**: Bash script that automates the entire pipeline.

-   **images/**: Directory containing product images.

-   **closet.db**: SQLite database file containing the closet data.

### Make Sure You Change These To Your Own Paths For The Scripts in The run_pipeline.sh Script:

Rscript /Users/uzairzaidi/Desktop/product_scraping.R

Rscript /Users/uzairzaidi/Desktop/weatherstack_api.R

Rscript /Users/uzairzaidi/Desktop/etl.R

Rscript /Users/uzairzaidi/Desktop/run_ootd_api.R &

# Instructions

### - Set Your Weatherstack API Key You must have a Weatherstack API key. Set it as an environment variable before running the pipeline:

### - Change all File Paths in run_ootd_api.R & run_pipeline.sh to your own File Paths on your Machine (where you saved the scripts)

### 

*bash*

export YOUR_ACCESS_KEY=<your_weatherstack_key>

Note: replace <your_weatherstack_key> with your actual WeatherstackAPI
key.

Running the Entire Pipeline Use the run_pipeline.sh script to run the
entire pipeline from start to finish:

*bash*

./run_pipeline.sh YOUR_ACCESS_KEY


In run_ootd_api.R:


\ # r \<- plumb("/Users/your_name/File Location/ootd_api.R")

Run the Bash script in the correct directory where you saved the file,
for example, /Users/your_name/File Location/run_pipeline.sh
YOUR_ACCESS_KEY

Note: replace YOUR_ACCESS_KEY with your actual WeatherstackAPI key.

### This Will:

-   Run product_scraping.R to scrape product data and images, producing
    products_raw.csv and populating images/.

-   Run weatherstack_api.R to fetch current weather data
    (weather_data.rds).

-   Run etl.R to clean and load data into closet.db.

-   Run run_ootd_api.R to start the Plumber API server on
    <http://localhost:8000>.

-   Call the /ootd endpoint via curl, producing ootd_plot.png.

-   After completion, you should find ootd_plot.png in the correct
    directory showing the recommended outfit for today's weather in
    London.

## Script Details

**product_scraping.R**

Scrapes at least 25 clothing items covering the categories: Shoes,
Bottoms, Tops, Coats, and Accessories. Downloads images and saves them
into images/. Creates products_raw.csv for the ETL step.

**weatherstack_api.R**

Uses your provided YOUR_ACCESS_KEY to call the Weatherstack API. Saves
the current weather data for London as weather_data.rds.

**etl.R**

Reads products_raw.csv, cleans the data, and inserts it into closet.db
under the closet table. Ensures the database has all required fields.

**ootd_api.R**

Defines two endpoints using Plumber: /ootd: Returns a JSON with
recommended items and generates ootd_plot.png. /rawdata: Returns all
products from the closet table as JSON.

#### The recommendation logic is based on temperature thresholds and weather conditions:

-   Hot/Warm (≥25°C): Light clothing, no coats

-   Mild (15°C to \<25°C): Comfortable clothing plus a light
    (non-waterproof) coat.

-   Cold (\<15°C): Warmer clothing and heavier coats.

-   Rain: Prefer waterproof or rain-specific coats, exclude sunglasses.

-   Sun: Include sunglasses if available.

#### **run_ootd_api.R**

Starts the Plumber API on <http://localhost:8000>.

#### **run_pipeline.sh**

Orchestrates the entire pipeline from start to finish: Runs all R
scripts in sequence. Starts the API. Calls /ootd to produce the final
ootd_plot.png.

#### **Weather Integration & Logic**

The weather data retrieved from Weatherstack is parsed, and the
recommend_outfit function in ootd_api.R uses it to select items from the
database that match the conditions.

### **Examples:**

Cold & Rainy: Warm items, waterproof coat, no sunglasses.

Mild & Sunny: Comfortable items, a light coat, and sunglasses if
available.

Hot & Clear: Light items, shorts, tees, no coats.

API Endpoints

/ootd Method: GET Parameters (optional): temperature (numeric)
description (string) If parameters are not provided, it uses
weather_data.rds for the current weather.

Response:

JSON object with selected items (top, coat, bottom, shoes, accessory).
The script also saves an ootd_plot.png image displaying the chosen
outfit. /rawdata Method: GET Returns all product data in closet.db as
JSON. Example Output After running:

bash

./run_pipeline.sh YOUR_ACCESS_KEY

Check:

ootd_plot.png: Should display an annotated image of the chosen outfit.

closet.db: Contains your product data. weather_data.rds: Current weather
info for London.

products_raw.csv: Raw scraped products.

images/: Contains downloaded product images.

## Troubleshooting & FAQ Port in Use Error:

If run_ootd_api.R fails to start the API due to port conflicts, either
kill the process using that port or change the port in run_ootd_api.R
and adjust the curl call in run_pipeline.sh.

lsof -i :8000   -\> checks port 8000

kill \# -\> where \# = ID-in-use from check\^

**No 'current' Weather Data:** Check your Weatherstack API key and
subscription limits. The script stops if current data is not available.

**Insufficient Items in Categories:** If the logic doesn’t find suitable
items, ensure that your scraped dataset contains enough variety. You may
need to scrape more items or adjust conditions.

**NULL Parameter Warning in Swagger Docs:** The code uses missing()
checks instead of NULL defaults for parameters, which should resolve
NULL-related warnings.

### If No outfit is being produced and no product_raw data shown,

Set your environment key

Run product_scraping.R

Run the weatherstack_api using:

"Rscript /Users/your_name/Directory/weatherstack_api.R"

then run etl.R

execute run_pipeline.sh, and it should work.

### Additional Notes:

Consider adding more items or logic enhancements for better
recommendations and customizable wardrobes. Test various scenarios by
calling /ootd with different temperature and description parameters.
