library(tidyverse)
library(ggplot2)
library(GGally)

# ---------- Functions -------------

plot_distributions <- function(data) {
  # Convert Class to factor for proper coloring
  data <- data %>% mutate(Class = factor(Class))
  
  # Reshape data to long format for facet plotting
  data_long <- data %>%
    pivot_longer(cols = -Class, names_to = "Feature",  values_to = "Value")
  
  # Create density plots for each feature
  ggplot(data_long, aes(x = Value, fill = Class)) +
    geom_histogram() +
    facet_wrap(~Feature, scales = "free") +
    labs(title = "Distribution of Each Feature",
         x = "Feature Value",
         y = "Density",
         fill = "Class") +
    scale_fill_manual(values = c("skyblue", "tomato")) 
}
plot_densities <- function(data) {
  # Convert Class to factor for proper coloring
  data <- data %>% mutate(Class = factor(Class))
  
  # Reshape data to long format for facet plotting
  data_long <- data %>%
    pivot_longer(cols = -Class, names_to = "Feature",  values_to = "Value")
  
  # Create density plots for each feature
  ggplot(data_long, aes(x = Value, fill = Class)) +
    geom_density(alpha = 0.6) +
    facet_wrap(~Feature, scales = "free") +
    labs(title = "Distribution of Each Feature",
         x = "Feature Value",
         y = "Density",
         fill = "Class") +
    scale_fill_manual(values = c("skyblue", "tomato")) 
}


scale_min_max <- function(data) {
  # Apply Min-Max scaling only to numeric columns
  data_scaled <- data
  numeric_columns <- sapply(data, is.numeric)  # Identify numeric columns
  
  # Apply scaling to numeric columns
  data_scaled[numeric_columns] <- lapply(data[numeric_columns], function(x) {
    (x - min(x)) / (max(x) - min(x))
  })
  
  return(data_scaled)
}


# -------------- Importing Dataset----------
dataset <- read.csv("bank_note_forgery.csv")

# --------- Shuffle Rows-------------
dataset <- dataset[sample(nrow(dataset)), ]



# --------- Check Missing and Duplicates -------------

missing_values <- sapply(dataset, function(x) sum(is.na(x)))
missing_summary <- data.frame(Feature = names(missing_values), MissingValues = missing_values)
missing_summary <- missing_summary[missing_summary$MissingValues > 0, ]  # Only show features with missing values

# Check for duplicates in the dataset
duplicate_rows <- duplicated(dataset)  # Logical vector indicating duplicated rows
duplicate_count <- sum(duplicate_rows)

# Output results
if (nrow(missing_summary) > 0) {
  cat("Missing Values Summary:\n")
  print(missing_summary)
} else {
  cat("No missing values found.\n")
}

if (duplicate_count > 0) {
  cat("\nDuplicates found: ", duplicate_count, "duplicate rows.\n")
} else {
  cat("\nNo duplicates found.\n")
}

# ---------------- Scaling Values------------
# plot_densities(dataset)

dataset <- scale_min_max(dataset)

# ---------------- Selecting Inputs and Outputs-----------

inputs <- dataset %>% 
            select(-Class)

outputs <- dataset %>% 
            select(Class)



ggplot(dataset) +
  geom_point(aes(x = Variance, y = Skewness, color = Class))
# ------------- Outliers Check --------------------
outlier_summary <- data.frame(feature = character(), 
                              outlier_count = numeric(), 
                              stringsAsFactors = FALSE)

for(col in names(dataset)[!names(dataset) %in% "class"]) {
  Q1 <- quantile(dataset[[col]], 0.25)
  Q3 <- quantile(dataset[[col]], 0.75)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  
  outlier_count <- sum(dataset[[col]] < lower_bound | dataset[[col]] > upper_bound)
  outlier_summary <- rbind(outlier_summary, 
                           data.frame(feature = col, 
                                      outlier_count = outlier_count))
}

print("Outlier summary:")
print(outlier_summary)

# Get summary statistics
summary_stats <- summary(dataset)
print("Summary statistics:")
print(summary_stats)




# Run the function
plot_distributions(dataset)
plot_densities(dataset)

write.csv(dataset, file = "preprocess_bank_note_forgery.csv", row.names = FALSE)



