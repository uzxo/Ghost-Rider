library(plumber)

# Load the API
r <- plumb("/Users/uzairzaidi/Desktop/ootd_api.R")

# Run the API on port 8000
r$run(port = 8000)
