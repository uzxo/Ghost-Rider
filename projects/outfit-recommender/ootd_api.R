library(plumber)
library(DBI)
library(RSQLite)
library(jsonlite)
library(magick)

#* @apiTitle Outfit Recommendation API
#* @serializer json

db_connect <- function() {
  dbConnect(SQLite(), dbname = "closet.db")
}

load_weather_data <- function() {
  tryCatch({
    weather_data <- readRDS("weather_data.rds")
    list(
      temperature = weather_data$current$temperature,
      description = tolower(weather_data$current$weather_descriptions[[1]])
    )
  }, error = function(e) {
    message("Error loading weather data: ", e$message)
    stop("Weather data fetch failed.")
  })
}

# Query a category with conditions and fallback
get_item_with_fallback <- function(conn, category, condition = NULL) {
  # Try specific condition if provided
  if (!is.null(condition)) {
    query <- paste0("SELECT * FROM closet WHERE category = '", category, "' AND (", condition, ") ORDER BY RANDOM() LIMIT 1")
    result <- dbGetQuery(conn, query)
    if (nrow(result) > 0) {
      return(result)
    }
  }
  
  # If no condition provided or no result found, fallback to any item in the category
  fallback_query <- paste0("SELECT * FROM closet WHERE category = '", category, "' ORDER BY RANDOM() LIMIT 1")
  fallback_result <- dbGetQuery(conn, fallback_query)
  return(fallback_result)
}

recommend_outfit <- function(conn, temp, desc) {
  outfit <- list()
  
  if (temp >= 25) {
    # Hot/Warm weather: no coat
    outfit$top <- get_item_with_fallback(conn, "Tops", "(name LIKE '%short sleeve%' OR name LIKE '%tee%' OR name LIKE '%tank%' OR name LIKE '%henley%' OR name LIKE '%pique%') AND name NOT LIKE '%long sleeve%'")
    outfit$bottom <- get_item_with_fallback(conn, "Bottoms", "name NOT LIKE '%heavy%' AND (name LIKE '%shorts%' OR name LIKE '%chinos%' OR name LIKE '%light%' OR name LIKE '%denim%' OR name LIKE '%short%)")
    outfit$shoes <- get_item_with_fallback(conn, "Shoes", "name LIKE '%sneakers%' OR name LIKE '%sandals%' OR name LIKE '%flip flops%' OR name LIKE '%trainers%'")
    # No coat in hot weather
  } else if (temp >= 15 && temp < 25) {
    # Mild/Comfortable weather: Add a light coat
    outfit$top <- get_item_with_fallback(conn, "Tops", "name LIKE '%long sleeve%' OR name LIKE '%short sleeve%' OR name LIKE '%pique%' OR name LIKE '%tee%'")
    outfit$bottom <- get_item_with_fallback(conn, "Bottoms", "(name LIKE '%jeans%' OR name LIKE '%chinos%' OR name LIKE '%trousers%' OR name LIKE '%denim%')")
    outfit$shoes <- get_item_with_fallback(conn, "Shoes", "name LIKE '%sneakers%' OR name LIKE '%loafers%' OR name LIKE '%trainers%'")
    
    # Mild weather coat conditions:
    # Include something like a jacket or blazer, exclude heavy/waterproof/puffer styles.
    # Adjust these conditions based on what's in your closet database.
    outfit$coat <- get_item_with_fallback(
      conn, "Coats", 
      "(name LIKE '%jacket%' OR name LIKE '%blazer%' OR name LIKE '%light%') 
       AND name NOT LIKE '%heavy%' 
       AND name NOT LIKE '%waterproof%' 
       AND name NOT LIKE '%raincoat%' 
       AND name NOT LIKE '%puffer%' 
       AND name NOT LIKE '%windbreaker%' 
       AND name NOT LIKE '%anorak%' 
       AND name NOT LIKE '%parka%' 
       AND name NOT LIKE '%fleece%'"
    )
    
  } else {
    # Cold weather: (warm coats)
    outfit$top <- get_item_with_fallback(conn, "Tops", "name NOT LIKE '%light%' AND (name LIKE '%sweater%' OR name LIKE '%wool%' OR name LIKE '%long sleeve%' OR name LIKE '%pique%')")
    outfit$coat <- get_item_with_fallback(conn, "Coats", "(name LIKE '%jacket%' OR name LIKE '%puffer%' OR name LIKE '%parka%' OR name LIKE '%fleece%' OR name LIKE '%raincoat%' OR name LIKE '%windbreaker%' OR name LIKE '%anorak%')")
    outfit$bottom <- get_item_with_fallback(conn, "Bottoms", "name NOT LIKE '%shorts%' AND name NOT LIKE '%short%' AND (name LIKE '%warm%' OR name LIKE '%jogger%' OR name LIKE '%trousers%' OR name LIKE '%chino%')")
    outfit$shoes <- get_item_with_fallback(conn, "Shoes", "(name LIKE '%boots%' OR name LIKE '%warm%')")
  }
  
  # Polo top rule
  if (!is.null(outfit$top) && nrow(outfit$top) > 0 && grepl("polo", outfit$top$name, ignore.case = TRUE)) {
    outfit$bottom <- get_item_with_fallback(conn, "Bottoms", "(name LIKE '%trousers%' OR name LIKE '%chinos%' OR name LIKE '%denim%')")
  }
  
  # Weather-based accessories
  is_rain <- grepl("\\brain\\b", desc) && !grepl("\\bno\\b.*\\brain\\b", desc)
  is_sun <- grepl("\\bsun\\b", desc)
  
  # Waterproof Coats in Rainy Conditions
  if (is_rain) {
    wcoat <- get_item_with_fallback(conn, "Coats", "name LIKE '%waterproof%' OR name LIKE '%raincoat%' OR name LIKE '%windbreaker%' OR name LIKE '%puffer%' OR name LIKE '%mac%'")
    if (nrow(wcoat) > 0 && temp < 25) {
      outfit$coat <- wcoat
    }
    # Non-sunglasses accessory in rain
    outfit$accessory <- get_item_with_fallback(conn, "Accessories", "name NOT LIKE '%sunglasses%'")
  } else if (is_sun) {
    sunglasses <- dbGetQuery(conn, "SELECT * FROM closet WHERE category = 'Accessories' AND name LIKE '%sunglasses%' ORDER BY RANDOM() LIMIT 1")
    if (nrow(sunglasses) > 0) {
      outfit$accessory <- sunglasses
    } else {
      outfit$accessory <- get_item_with_fallback(conn, "Accessories", "name NOT LIKE '%sunglasses%' AND name NOT LIKE '%gloves%' AND name NOT LIKE '%scarf%' AND name NOT LIKE '%beanie%' AND name NOT LIKE '%earmuffs%'")
    }
  } else {
    nonsun_accessory <- get_item_with_fallback(conn, "Accessories", "name NOT LIKE '%sunglasses%' AND name NOT LIKE '%gloves%' AND name NOT LIKE '%scarf%' AND name NOT LIKE '%beanie%' AND name NOT LIKE '%earmuffs%'")
    if (nrow(nonsun_accessory) > 0) {
      outfit$accessory <- nonsun_accessory
    } else {
      outfit$accessory <- get_item_with_fallback(conn, "Accessories")
    }
  }
  
  return(outfit)
}

# Safely read an item's image
get_image <- function(item) {
  if (!is.null(item) && nrow(item) > 0 && !is.na(item$image_path) && file.exists(item$image_path)) {
    image_read(item$image_path)
  } else {
    NULL
  }
}

generate_ootd_image <- function(outfit, weather_desc, temperature) {
  # Common dimensions for item images
  item_width <- 300
  item_height <- 300
  
  
  prepare_item_image <- function(item, label) {
    # Create or load the item image
    if (!is.null(item) && nrow(item) > 0 && !is.na(item$image_path) && file.exists(item$image_path)) {
      img <- image_read(item$image_path)
    } else {
      # Placeholder if no item found
      img <- image_blank(width = item_width, height = item_height, color = "grey90")
    }
    
    # Resize and crop to uniform
    img <- image_scale(img, paste0(item_width, "x"))
    dims <- image_info(img)
    if (dims$height < item_height) {
      # Extend if not tall enough
      img <- image_extent(img, paste0(item_width, "x", item_height),
                          color = "white", gravity = "center")
    } else {
      # Crop if taller
      img <- image_crop(img, paste0(item_width, "x", item_height, "+0+0"))
    }
    
    # Annotate the label at bottom center
    # Using boxcolor for readability
    img <- image_annotate(
      img,
      label,
      gravity = "south",
      size = 20,
      color = "black",
      boxcolor = "rgba(255,255,255,0.7)",
      location = "+0+5"
    )
    
    return(img)
  }
  
  # Item image with labels
  top_img <- prepare_item_image(outfit$top, "Top")
  coat_img <- prepare_item_image(outfit$coat, "Coat")
  bottom_img <- prepare_item_image(outfit$bottom, "Bottom")
  shoes_img <- prepare_item_image(outfit$shoes, "Shoes")
  accessory_img <- prepare_item_image(outfit$accessory, "Accessory")
  
  # Rows
  row1 <- image_append(c(top_img, coat_img))
  row2 <- image_append(c(bottom_img, shoes_img))
  
  blank_item <- image_blank(width = item_width, height = item_height, color = "white")
  row3 <- image_append(c(blank_item, accessory_img, blank_item))
  
  # Combine rows into a single image
  combined_image <- image_append(c(row1, row2, row3), stack = TRUE)
  
  # Top banner for date and weather info
  banner_height <- 60
  banner_width <- image_info(combined_image)$width
  banner <- image_blank(width = banner_width, height = banner_height, color = "white")
  
  # Annotate Banner 
  banner <- image_annotate(
    banner,
    paste("Date:", Sys.Date(), "| Weather:", weather_desc, "| Temp:", temperature, "°C"),
    gravity = "center",
    size = 20,
    color = "black"
  )
  
  # Stack Banner
  final_image <- image_append(c(banner, combined_image), stack = TRUE)
  
  # Save Final Image
  output_path <- "ootd_plot.png"
  image_write(final_image, path = output_path, format = "png")
  
  return(output_path)
}

#* @get /ootd
#* @param temperature query optional numeric
#* @param description query optional string
function(temperature, description) {
  conn <- db_connect()
  on.exit(dbDisconnect(conn))
  
  # Check if missing arguments 
  if (missing(temperature) || missing(description)) {
    weather <- load_weather_data()
    temperature <- weather$temperature
    description <- weather$description
  }
  
  recommended_outfit <- recommend_outfit(conn, as.numeric(temperature), description)
  image_path <- generate_ootd_image(recommended_outfit, description, temperature)
  
  response <- list(
    top = if(!is.null(recommended_outfit$top)) recommended_outfit$top else NULL,
    coat = if(!is.null(recommended_outfit$coat)) recommended_outfit$coat else NULL,
    bottom = if(!is.null(recommended_outfit$bottom)) recommended_outfit$bottom else NULL,
    shoes = if(!is.null(recommended_outfit$shoes)) recommended_outfit$shoes else NULL,
    accessory = if(!is.null(recommended_outfit$accessory)) recommended_outfit$accessory else NULL,
    ootd_image_path = image_path
  )
  
  return(toJSON(response, pretty = TRUE, auto_unbox = TRUE))
}

#* @get /rawdata
function() {
  conn <- db_connect()
  on.exit(dbDisconnect(conn))
  
  data <- dbGetQuery(conn, "SELECT * FROM closet")
  return(toJSON(data, pretty = TRUE, auto_unbox = TRUE))
}
