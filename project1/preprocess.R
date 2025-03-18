library(caret)
library(tidyverse)
library(ggplot2)
library(corrplot)
library(ggcorrplot)
# -----------Importing Data-----------
raw_data <- read.csv("wisc_bc_data.csv")

View(raw_data)
dim(raw_data)
names <- sort(names(raw_data))
summary(raw_data)

# -----------Pre-processing-------------

detect_outliers <- function(x) {
  q1 <- quantile(x, 0.25)
  q3 <- quantile(x, 0.75)
  iqr <- q3 - q1
  lower_bound <- q1 - (1.5 * iqr)
  upper_bound <- q3 + (1.5 * iqr)
  
  outliers <- x[(x < lower_bound) | (x > upper_bound)]
  return(outliers)
}

# 1. Remove unnecessary Data


clean_data <- raw_data[-1] # Remove ID column

# 2.  Check for NULL/NA Values

contains_na <- any(is.na(raw_data))

print(paste("The given data contains missing values: ", contains_na ))

# 3. checking duplicates

duplicates <- clean_data %>% group_by_all() %>% filter(n() > 1)
print(paste("The dataset contains ", nrow(duplicates), " duplicate rows"))

# 4. Dividing inputs and outputs
X <- clean_data %>% select(-diagnosis)
Y <- clean_data %>% select(diagnosis)

# 5. Removing Outliers
# outliers <- detect_outliers(X)

# 6. Scaling inputs
X <- X %>%
  mutate_if(is.numeric, ~(. - min(.)) / (max(.) - min(.)))

# 7. Converting categories to numbers
Y$diagnosis <- ifelse(Y$diagnosis == "M", 1, 0) 



#-------------Data Visualization-------------
# cor_matrix <- cor(X)
# 
# corrplot(cor_matrix, method="color", type="upper", 
#          tl.col="black", tl.srt=45, 
#          title="Feature Correlation Matrix")

ggplot(raw_data, aes(x = radius_mean, y = perimeter_mean, color=diagnosis)) +
  geom_point()
ggplot(raw_data, aes(x = compactness_mean, y = perimeter_mean, color=diagnosis)) +
  geom_point()
ggplot(raw_data, aes(x = smoothness_mean, y = symmetry_mean, color = diagnosis)) +
  geom_point()
scaled_data <- clean_data %>% 
  mutate_if(is.numeric, ~(. - min(.)) / (max(.) - min(.)))


ggplot(raw_data, aes(x = diagnosis, fill = diagnosis)) +
  geom_bar() +
  geom_text(stat = 'count', aes(label = after_stat(count)), vjust = -0.5) +
  labs(title = "Distribution of Diagnosis Classes",
       x = "Diagnosis",
       y = "Count") +
  scale_fill_manual(values = c("M" = "skyblue", "B" = "tomato")) +
  theme_minimal()

mean_features <- raw_data %>%
  select(diagnosis, radius_mean, texture_mean, perimeter_mean, area_mean) %>%
  pivot_longer(cols = -diagnosis, names_to = "feature", values_to = "value")

 ggplot(mean_features, aes(x = diagnosis, y = value, fill = diagnosis)) +
  geom_boxplot() +
  facet_wrap(~feature, scales = "free_y") +
  labs(title = "Key Feature Distributions by Diagnosis",
       x = "Diagnosis",
       y = "Value") +
  scale_fill_manual(values = c("B" = "skyblue", "M" = "tomato")) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 4.3 Correlation Plot for Mean Features
mean_cols <- grep("_mean$", names(raw_data), value = TRUE)
mean_data <- raw_data[, c("diagnosis", mean_cols)]
mean_cors <- cor(mean_data[, -1])

ggcorrplot(mean_cors, hc.order = TRUE, type = "lower",
                 title = "Correlation of Mean Features",
                 lab = TRUE, lab_size = 2,
                 colors = c("#6D9EC1", "white", "#E46726"))
  
