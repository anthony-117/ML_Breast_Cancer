source("preprocess.R")
source("NN.R")



NN <- NN(X = X, Y.d = Y,
         hidden_layers = c(10, 5, 3)
         )

result <- NN.train(NN,X, Y, 1000, 0.01, 0.9, verbose = TRUE)


NN.trained <- result$NN
training_data <- result$training_data



library(ggplot2)

# Create a data frame for cost history
df_cost <- data.frame(
  epoch = 1:length(training_data$cost_history),
  train_cost = training_data$cost_history,
  test_cost = training_data$test_cost_history
)

# Plot cost over epochs
ggplot(df_cost, aes(x = epoch)) +
  geom_line(aes(y = train_cost, color = "Train Cost")) +
  geom_line(aes(y = test_cost, color = "Test Cost")) +
  labs(title = "Training and Testing Cost Over Epochs", x = "Epoch", y = "Cost") +
  theme_minimal() +
  scale_color_manual(values = c("Train Cost" = "blue", "Test Cost" = "red"))



df_acc <- data.frame(
  epoch = 1:length(training_data$accuracy_history),
  accuracy = training_data$accuracy_history
)

ggplot(df_acc, aes(x = epoch, y = accuracy)) +
  geom_line(color = "green") +
  labs(title = "Accuracy Over Epochs", x = "Epoch", y = "Accuracy") +
  theme_minimal()



library(reshape2)

# Convert confusion matrix to a dataframe for ggplot
cm_df <- as.data.frame(as.table(training_data$final_confusion_matrix))

ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = 1.5, color = "white", size = 5) +
  labs(title = "Confusion Matrix Heatmap", x = "Predicted", y = "Actual") +
  scale_fill_gradient(low = "white", high = "blue") +
  theme_minimal()

library(plotly)

# Example: Tracking the evolution of one weight across epochs
weight_df <- data.frame(
  epoch = 1:length(NN$weights[[1]][,1]),
  weight_value = NN$weights[[1]][,1]
)

plot_ly(weight_df, x = ~epoch, y = ~weight_value, type = "scatter", mode = "lines") %>%
  layout(title = "Weight Changes Over Epochs", xaxis = list(title = "Epoch"), yaxis = list(title = "Weight Value"))
library(ggplot2)
library(reshape2)

# Assuming confusion matrices are stored in a list called `confusion_matrices`
cm <- training_data$confusion_matrices[[1]]  # Get the confusion matrix for a specific epoch

# Ensure the confusion matrix has row and column names
cm_melt <- melt(cm)

# Plot heatmap
ggplot(cm_melt, aes(x =1:1000, y = value, fill = value)) + 
  geom_tile() + 
  labs(title = "Confusion Matrix Heatmap", x = "Predicted", y = "Actual") +
  scale_fill_gradient(low = "white", high = "blue") +
  theme_minimal()


# 
# 
# 
# # Plot training progress (cost function)
# par(mfrow = c(1, 2))
# plot(1:length(NN.trained$cost_history), NN.trained$cost_history, 
#      type = "l", xlab = "Epoch", ylab = "Training Cost", 
#      main = "Neural Network Training Progress",
#      col = "blue")
# 
# # Plot test cost history
# lines(1:length(NN.trained$test_cost_history), NN.trained$test_cost_history, 
#       col = "red")
# legend("topright", legend = c("Training Cost", "Test Cost"), 
#        col = c("blue", "red"), lty = 1)
# 
# # Plot evaluation metrics
# plot_metrics(NN.trained)
# 
# # Reset plot parameters
# par(mfrow = c(1, 1))
# 
# # Make predictions on the entire dataset
# predictions <- NN.predict(NN.trained, X = X)
# 
# # Convert predictions to binary class (0 or 1)
# binary_predictions <- ifelse(predictions > 0.1, 1, 0)
# 
# # Create final confusion matrix for the entire dataset
# final_confusion_matrix <- table(
#   Predicted = factor(binary_predictions, levels = c(0, 1)),
#   Actual = factor(Y, levels = c(0, 1))
# )
# 
# # Calculate and print final metrics for the entire dataset
# cat("\nFinal Evaluation on Complete Dataset:\n")
# print(final_confusion_matrix)
# 
# # Calculate metrics from the confusion matrix
# final_metrics <- calculate_metrics(final_confusion_matrix)
# cat("\nFinal Metrics on Complete Dataset:\n")
# cat("Accuracy:", round(final_metrics$accuracy, 4), "\n")
# cat("Precision:", round(final_metrics$precision, 4), "\n")
# cat("Recall:", round(final_metrics$recall, 4), "\n")
# cat("F1 Score:", round(final_metrics$f1_score, 4), "\n")
# 
# # Save the trained model
# saveRDS(NN.trained, "trained_neural_network.rds")