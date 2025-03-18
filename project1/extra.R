scaled_data = data[-1] %>% mutate(across(where(is.numeric), scale))

scaled_M = scaled_data %>% filter(diagnosis == "M") %>% select(-diagnosis, -id)
scaled_B = scaled_data %>% filter(diagnosis == "B") %>% select(-diagnosis, -id)

calculate_metrics = function(data.frame) {
  max <- apply(data.frame, 2, max)
  min <- apply(data.frame, 2, min)
  mean <- (max + min) / 2
  
  dt_matrix = data.frame(name = colnames(data.frame),
                         min = as.numeric(as.character(min)),
                         max = as.numeric(as.character(max)),
                         mean = as.numeric(as.character(mean)))
  return(dt_matrix)
}

ci_B = calculate_metrics(scaled_B)
ci_B$diagnosis <- rep("B", ncol(scaled_B))

ci_M = calculate_metrics(scaled_M)
ci_M$diagnosis <- rep("M", ncol(scaled_M))

data_metrics = rbind(ci_M, ci_B)

ggplot(data_metrics, aes(name, mean, color = diagnosis)) +
  geom_point() +
  geom_errorbar(aes(ymin = min, ymax = max)) +
  coord_flip() +
  theme_minimal() +
  scale_color_manual(values = c('#33cc00', '#009AEF')) +
  labs(title = "Columns Means", color = "Class") +
  theme(axis.title.y = element_blank(),
        plot.title = element_text(size = 16, face = "bold"))

df_long = scaled_data %>% pivot_longer(cols = where(is.numeric), names_to = "variable")
df_long$variable = factor(df_long$variable, levels = levels(forcats::fct_rev((data_metrics$name))))


ggplot(df_long, aes(value, fill = diagnosis)) +
  geom_density(alpha = 0.4) +
  facet_wrap(~ variable, scales = "free") +
  labs(title = "Density", fill = "Class") +
  theme(axis.ticks.y = element_blank(),
        axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.x = element_blank(),
        axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        strip.text.x = element_text(size = 8, color = "black"),
        plot.title = element_text(size = 16, face = "bold")) +
  scale_fill_manual(values = c('#33cc00', '#009AEF'))

pca <- prcomp(x = scaled_data[,-c(1)], scale = TRUE, center = TRUE)
summary(pca)

pca_res <- as.data.frame(pca$x) %>%
  mutate(diagnosis = data$diagnosis)




density_pca <- pca_res %>% 
  pivot_longer(cols = where(is.numeric), names_to = "variable")
density_pca$variable = factor(density_pca$variable, levels = colnames(pca_res))


ggplot(density_pca, aes(value, fill = diagnosis)) +
  geom_density(alpha = 0.4) +
  facet_wrap(~ variable, scales = "free") +
  theme(axis.ticks.y = element_blank(),
        axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.x = element_blank(),
        axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        strip.text.x = element_text(size = 8, color = "black"),
        plot.title = element_text(size = 16, face = "bold.italic")) +
  scale_fill_manual(values = c('#33cc00', '#009AEF')) +
  ggtitle("Density")






# Load required packages
library(neuralnet)

# Read the data
data <- read.csv("wisc_bc_data.csv")

# Remove the first column if it's just row numbers
data <- data[, -1]

# Convert diagnosis to binary (1 for Malignant, 0 for Benign)
data$diagnosis <- ifelse(data$diagnosis == "M", 1, 0)

# Normalize/scale the numeric features
features <- data[, 2:31]  # All columns except id and diagnosis
scaled_features <- scale(features)

# Combine scaled features with the diagnosis column
data_scaled <- data.frame(diagnosis = data$diagnosis, scaled_features)

# Split into training and test sets (80% training, 20% test)
set.seed(123)
train_indices <- sample(1:nrow(data_scaled), size = 0.8 * nrow(data_scaled))
train_data <- data_scaled[train_indices, ]
test_data <- data_scaled[-train_indices, ]

# Create formula for the model
# This creates a formula like: diagnosis ~ radius_mean + texture_mean + ...
feature_names <- colnames(data_scaled)[-1]  # All column names except diagnosis
formula <- as.formula(paste("diagnosis ~", paste(feature_names, collapse = " + ")))

# Build the neural network
nn <- neuralnet(
  formula,
  data = train_data,
  hidden = c(5, 3),  # Two hidden layers with 5 and 3 neurons
  act.fct = "logistic",
  linear.output = FALSE,
  threshold = 0.01,
  stepmax = 1e5
)

# Plot the neural network
plot(nn, 
     rep = "best",
     col.hidden = "blue",
     col.hidden.synapse = "darkgreen",
     col.out = "red",
     col.out.synapse = "darkred",
     show.weights = TRUE,
     information = TRUE,
     dimension = 10)  # Adjust this for a larger plot due to many features

# Predict on test data
predictions <- compute(nn, test_data[, -1])
predicted_classes <- ifelse(predictions$net.result > 0.5, 1, 0)
actual_classes <- test_data$diagnosis

# Calculate accuracy
accuracy <- mean(predicted_classes == actual_classes)
cat("Accuracy:", round(accuracy * 100, 2), "%\n")

# Create confusion matrix
conf_matrix <- table(Actual = actual_classes, Predicted = predicted_classes)
print(conf_matrix)

