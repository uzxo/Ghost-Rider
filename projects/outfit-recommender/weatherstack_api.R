library(httr)
library(jsonlite)
library(dplyr)
library(magrittr)

# Retrieve API key from environment variable
api_key <- Sys.getenv("YOUR_ACCESS_KEY")
if (api_key == "") {
  stop("API key not provided. Please set YOUR_ACCESS_KEY as an environment variable.")
}

response <- tryCatch(
  GET(
    url = "http://api.weatherstack.com/current",
    query = list(
      access_key = api_key,
      query = "London"
    ),
    add_headers(`User-Agent` = "R-script")
  ),
  error = function(e) {
    stop("Failed to make API request: ", e$message)
  }
)

if (http_error(response)) {
  stop("HTTP error occurred: ", status_code(response))
}

weather_data <- tryCatch(
  content(response, as = "text", encoding = "UTF-8") %>% fromJSON(flatten = TRUE),
  error = function(e) {
    stop("Failed to parse API response: ", e$message)
  }
)

if (!"current" %in% names(weather_data) || is.null(weather_data$current)) {
  stop("API response does not contain 'current' weather data.")
}

# Save weather data
saveRDS(weather_data, "weather_data.rds")

current_temperature <- weather_data$current$temperature
weather_descriptions <- weather_data$current$weather_descriptions
cat("Current Temperature:", current_temperature, "°C\n")
cat("Weather Description:", paste(weather_descriptions, collapse = ", "), "\n")
