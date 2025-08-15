library(RSQLite)
library(DBI)
library(dplyr)
library(magrittr)

# Connect to SQL database
conn <- dbConnect(SQLite(), dbname = "closet.db")

dbExecute(conn, "
    CREATE TABLE IF NOT EXISTS closet (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        category TEXT,
        image_path TEXT
    )
")

# Read the raw product data
products <- read.csv("products_raw.csv", stringsAsFactors = FALSE)

# Clean data
products_clean <- products[complete.cases(products), ]

# Insert cleaned data (overwrites)
dbWriteTable(conn, "closet", products_clean, overwrite = TRUE, row.names = FALSE)

dbDisconnect(conn)
cat("Database created and populated successfully.\n")
