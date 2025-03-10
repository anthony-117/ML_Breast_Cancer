source("preprocess.R")
source("NN.R")

NN <- NN(X = X, Y.d = Y,
         hidden_layers = c(6,3,2)
         )

result <- NN.train(NN,X, Y, 1000, 0.01, 0.9, verbose = TRUE)




NN.trained <- result$NN
training_data <- result$training_data


library(ggplot2)

# Define a range of learning rates
learning_rates <- c(0.001, 0.01, 0.1, 0.5)

# Train the network with different learning rates
training_results_lr <- NN.train_multiple.lr(NN, X, Y, epochs = 1000, learning_rates, momentum = 0.9, verbose = FALSE)

# Convert results into a dataframe for ggplot2
df_lr <- data.frame(Epoch = integer(), Cost = numeric(), Accuracy = numeric(), LearningRate = character())

for (lr in names(training_results_lr)) {
  df_temp <- data.frame(
    Epoch = 1:length(training_results_lr[[lr]]$cost_history),
    Cost = training_results_lr[[lr]]$cost_history,
    Accuracy = training_results_lr[[lr]]$accuracy_history,
    LearningRate = lr
  )
  df_lr <- rbind(df_lr, df_temp)
}

# Plot Cost History
ggplot(df_lr, aes(x = Epoch, y = Cost, color = LearningRate)) +
  geom_line(size = 1) +
  labs(title = "Effect of Learning Rate on Cost",
       x = "Epoch",
       y = "Cost") +
  theme_minimal()

# Plot Accuracy History
ggplot(df_lr, aes(x = Epoch, y = Accuracy, color = LearningRate)) +
  geom_smooth(size = 1) +
  labs(title = "Effect of Learning Rate on Accuracy",
       x = "Epoch",
       y = "Accuracy") +
  theme_minimal()


# Define a range of momentum values
momentums <- c(0.5, 0.7, 0.9, 0.99)

# Train the network with different momentum values
training_results_momentum <- NN.train_multiple.momentum(NN, X, Y, epochs = 1000, learning_rate = 0.01, momentums, verbose = FALSE)

# Convert results into a dataframe for ggplot2
df_momentum <- data.frame(Epoch = integer(), Cost = numeric(), Accuracy = numeric(), Momentum = character())

for (m in names(training_results_momentum)) {
  df_temp <- data.frame(
    Epoch = 1:length(training_results_momentum[[m]]$cost_history),
    Cost = training_results_momentum[[m]]$cost_history,
    Accuracy = training_results_momentum[[m]]$accuracy_history,
    Momentum = m
  )
  df_momentum <- rbind(df_momentum, df_temp)
}

# Plot Cost History
ggplot(df_momentum, aes(x = Epoch, y = Cost, color = Momentum)) +
  geom_line(size = 1) +
  labs(title = "Effect of Momentum on Cost",
       x = "Epoch",
       y = "Cost") +
  theme_minimal()

# Plot Accuracy History
ggplot(df_momentum, aes(x = Epoch, y = Accuracy, color = Momentum)) +
  geom_line(size = 1) +
  labs(title = "Effect of Momentum on Accuracy",
       x = "Epoch",
       y = "Accuracy") +
  theme_minimal()


# --------------------------



df_acc <- data.frame(
  epoch = 1:length(training_data$accuracy_history),
  accuracy = training_data$accuracy_history,
  cost = training_data$cost_history
)

# ggplot(df_acc, aes(x = epoch, y = accuracy)) +
#   geom_line(color = "green") +
#   geom_line(aes(y = cost), color = "yellow") +
#   labs(title = "Accuracy Over Epochs", x = "Epoch", y = "Accuracy") +
#   theme_minimal()
ggplot(df_acc, aes(x = epoch)) +
  geom_smooth(aes(y = accuracy), color="blue") 

# Create a smooth spline for the accuracy and cost data
accuracy_smooth <- smooth.spline(df_acc$epoch, df_acc$accuracy)
cost_smooth <- smooth.spline(df_acc$epoch, df_acc$cost)

# Plot the smoothed accuracy data
plot(accuracy_smooth, col = "skyblue", 
     xlab = "Epoch", ylab = "Accuracy",
     main = "Accuracy and Cost Over Epochs")

# Add a second y-axis on the right side
par(new = TRUE)

# Plot the smoothed cost data with the second y-axis
plot(cost_smooth, col = "tomato", axes = FALSE, xlab = "", ylab = "")

# Add the second y-axis
axis(side = 4)
mtext("Cost", side = 4, line = 3)

# Add a legend
legend("topright", legend = c("Accuracy", "Cost"), 
       col = c("skyblue", "tomato"), lty = 1)

  
# plot(df_acc$epoch, df_acc$accuracy, type = "l", col = "skyblue", 
#      xlab = "Epoch", ylab = "Accuracy",
#      main = "Accuracy and Cost Over Epochs")
# 
# # Add a second y-axis on the right side
# par(new = TRUE)
# 
# # Plot the cost with the second y-axis
# plot(df_acc$epoch, df_acc$cost, type = "l", col = "tomato",
#      axes = FALSE, xlab = "", ylab = "")
# 
# # Add the second y-axis
# axis(side = 4)
# mtext("Cost", side = 4, line = 3)
# 
# # Add a legend
# legend("topright", legend = c("Accuracy", "Cost"), 
#        col = c("skyblue", "tomato"), lty = 1)

# library(reshape2)
# 
# # Convert confusion matrix to a dataframe for ggplot
# cm_df <- as.data.frame(as.table(training_data$final_confusion_matrix))
# 
# ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Freq)) +
#   geom_tile() +
#   geom_text(aes(label = Freq), vjust = 1.5, color = "white", size = 5) +
#   labs(title = "Confusion Matrix Heatmap", x = "Predicted", y = "Actual") +
#   scale_fill_gradient(low = "white", high = "blue") +
#   theme_minimal()

# library(plotly)
# 
# # Example: Tracking the evolution of one weight across epochs
# weight_df <- data.frame(
#   epoch = 1:length(NN$weights[[1]][,1]),
#   weight_value = NN$weights[[1]][,1]
# )
# 
# plot_ly(weight_df, x = ~epoch, y = ~weight_value, type = "scatter", mode = "lines") %>%
#   layout(title = "Weight Changes Over Epochs", xaxis = list(title = "Epoch"), yaxis = list(title = "Weight Value"))
# library(ggplot2)
# library(reshape2)
# 
# # Assuming confusion matrices are stored in a list called `confusion_matrices`
# cm <- training_data$confusion_matrices[[1]]  # Get the confusion matrix for a specific epoch
# 
# # Ensure the confusion matrix has row and column names
# cm_melt <- melt(cm)
# 
# # Plot heatmap
# ggplot(cm_melt, aes(x =1:1000, y = value, fill = value)) + 
#   geom_tile() + 
#   labs(title = "Confusion Matrix Heatmap", x = "Predicted", y = "Actual") +
#   scale_fill_gradient(low = "white", high = "blue") +
#   theme_minimal()
# 

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