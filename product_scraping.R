library(rvest)
library(httr)

# Function to scrape clothing data from single URL
scrape_clothing_data <- function(base_url) {
  response <- GET(
    base_url,
    user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
  )
  
  if (http_error(response)) {
    stop("HTTP error: ", status_code(response))
  }
  
  page_content <- content(response, as = "text", encoding = "UTF-8")
  webpage <- read_html(page_content)
  
# Extract product names
  product_names <- webpage %>% 
    html_nodes("p.productDescription_sryaw") %>% 
    html_text(trim = TRUE)
  
# Extract image URLs
  image_urls <- webpage %>% 
    html_nodes("article img") %>% 
    html_attr("src")
  
# Handle missing or alternative image sources
  missing_src <- is.na(image_urls) | image_urls == ""
  if (any(missing_src)) {
    alt_urls <- webpage %>% 
      html_nodes("article img") %>% 
      html_attr("data-src")
    image_urls[missing_src] <- alt_urls[missing_src]
  }
  
# Prepend 'https:' to protocol-relative URLs and remove query parameters
  image_urls <- ifelse(grepl("^//", image_urls), paste0("https:", image_urls), image_urls)
  image_urls <- gsub("\\?.*$", "", image_urls)
  
# Assign categories based on product names (original logic)
  assign_category <- function(product_name) {
    product_name_lower <- tolower(product_name) # convert once
    
    # Bottoms:
    if (grepl("belted trouser|belted pants", product_name_lower)) {
      return("Bottoms")
    } else if (grepl("joggers|pants|trousers|chinos|jogger|shorts|leggings|denim", product_name_lower)) {
      return("Bottoms")
      
      # Tops:
    } else if (grepl("t-shirt|tee|top|sweater|jumper|tank|jersey|shirt|polo|long sleeve|cardigan|hoodie|henley|blouse", product_name_lower)) {
      return("Tops")
      
      # Coats:
    } else if (grepl("jacket|coat|blazer|gilet|bomber|parka|puffer|mac|raincoat|windbreaker|anorak|fleece", product_name_lower)) {
      return("Coats")
      
      # Shoes:
    } else if (grepl("shoes|sneakers|boots|sandals|trainers|trainer|loafers|clogs|clog|moccasins|flip flops", product_name_lower)) {
      return("Shoes")
      
      # Accessories:
    } else if (grepl("hat|cap|sunglasses|scarf|umbrella|beanie|vault|backpack|gloves|bag|earmuffs|headband|necklace|bracelet|watch|tie|bowtie|wallet", product_name_lower)) {
      return("Accessories")
      
      # Belt as accessory if not trouser/pants
    } else if (grepl("belt", product_name_lower) && !grepl("trouser|pants|denim|chinos|shorts|leggings", product_name_lower)) {
      return("Accessories")
      
      # If no match, categorize as Other
    } else {
      return("Other")
    }
  }
  
  product_categories <- sapply(product_names, assign_category)
  
# Ensure lengths match
  min_length <- min(length(product_names), length(product_categories), length(image_urls))
  product_names <- product_names[1:min_length]
  product_categories <- product_categories[1:min_length]
  image_urls <- image_urls[1:min_length]
  
  clothing_data <- data.frame(
    Name = product_names,
    Category = product_categories,
    Image_URL = image_urls,
    stringsAsFactors = FALSE
  )
  
  return(clothing_data)
}

log_failed_url <- function(name, url, reason) {
  log_entry <- paste(Sys.time(), "- Failed:", name, "| URL:", url, "| Reason:", reason, "\n")
  cat(log_entry, file = "failed_downloads.log", append = TRUE)
}

download_images <- function(data) {
  if (!dir.exists("images")) {
    dir.create("images", showWarnings = FALSE)
    Sys.chmod("images", mode = "0777")
  }
  
  for (i in seq_along(data$Image_URL)) {
    filename <- paste0("images/", gsub("[^A-Za-z0-9]", "_", data$Name[i]), ".jpg")
    
    success <- FALSE
    for (attempt in 1:3) {
      response <- tryCatch(
        GET(data$Image_URL[i],
            user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64)"),
            write_disk(filename, overwrite = TRUE), timeout(15)),
        error = function(e) NULL
      )
      
      if (!is.null(response) && response$status_code == 200) {
        success <- TRUE
        break
      } else {
        Sys.sleep(2 ^ attempt)
        message("Retrying URL: ", data$Image_URL[i])
      }
    }
    
    if (!success) {
      log_failed_url(data$Name[i], data$Image_URL[i], "Failed after 3 attempts")
    } else {
      message("Successfully downloaded: ", data$Name[i])
    }
  }
}

scrape_multiple_categories <- function(url_list) {
  all_data <- list()
  
  for (url in url_list) {
    cat("Scraping URL:", url, "\n")
    
    category_data <- tryCatch(
      scrape_clothing_data(url),
      error = function(e) {
        message("Error scraping URL: ", url, ": ", e)
        return(NULL)
      }
    )
    
    if (!is.null(category_data)) {
      all_data[[length(all_data) + 1]] <- category_data
    }
  }
  
  combined_data <- do.call(rbind, all_data)
  row.names(combined_data) <- NULL
  
# Remove duplicates
  combined_data <- combined_data[!duplicated(combined_data$Name), ]
  
# Ensure at least 5 items per required category
  required_categories <- c("Shoes", "Bottoms", "Tops", "Coats", "Accessories")
  category_counts <- table(combined_data$Category)
  for (catg in required_categories) {
    if (!catg %in% names(category_counts) || category_counts[catg] < 5) {
      warning("Not enough items in category: ", catg)
    }
  }
  
# Download images
  download_images(combined_data)
  
# Prepare final data for ETL step
  combined_data$name <- combined_data$Name
  combined_data$category <- combined_data$Category
  combined_data$image_path <- paste0("images/", gsub("[^A-Za-z0-9]", "_", combined_data$Name), ".jpg")
  
# Write to CSV
  combined_data <- combined_data[, c("name", "category", "image_path")]
  write.csv(combined_data, "products_raw.csv", row.names = FALSE)
  
  return(combined_data)
}

# URL's
url_list <- c(
  "https://www.asos.com/men/t-shirts-vests/cat/?cid=7616",
  "https://www.asos.com/men/loungewear/joggers/cat/?cid=14274",
  "https://www.asos.com/men/jackets-coats/cat/?cid=3606",
  "https://www.asos.com/men/accessories/cat/?cid=4210",
  "https://www.asos.com/men/trousers-chinos/cat/?cid=4910",
  "https://www.asos.com/men/shoes-boots-trainers/cat/?cid=4209&currentpricerange=5-250",
  "https://www.asos.com/men/ctas/fashion-online-13/cat/?cid=13522&currentpricerange=5-450&refine=currentprice:90%3C450",
  "https://www.asos.com/men/polo-shirts/cat/?cid=4616&currentpricerange=5-220&refine=currentprice:95%3C220",
  "https://www.asos.com/men/a-to-z-of-brands/nike/cat/?cid=4766&refine=attribute_10992:61388",
  "https://www.asos.com/men/premium-brands/cat/?cid=27111&currentpricerange=5-620&refine=attribute_1047:8401",
  "https://www.asos.com/men/accessories/sunglasses/cat/?cid=6519&currentpricerange=5-245&refine=attribute_1046:8213",
  "https://www.asos.com/men/shoes-boots-trainers/boots/cat/?cid=5774",
  "https://www.asos.com/men/t-shirts-vests/long-sleeve-t-shirts/cat/?cid=13084#ctaref-cat_header",
  "https://www.asos.com/men/jackets-coats/raincoats/cat/?cid=51416",
  "https://www.asos.com/men/ctas/usa-fashion-online-16/cat/?cid=16694",
  "https://www.asos.com/men/shorts/cat/?cid=7078"
)

scraped_data <- scrape_multiple_categories(url_list)
print(head(scraped_data))