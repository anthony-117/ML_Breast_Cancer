source("preprocess.R")
source("NN.R")

NN <- NN(X = X, Y.d = Y,
         hidden_layers = c(10, 5, 3), 
         learning_rate = 0.01, 
         momentum = 0.9)

NN.trained <- NN.train(NN,1000, verbose = TRUE)

# Plot training progress (cost function)
par(mfrow = c(1, 2))
plot(1:length(NN.trained$cost_history), NN.trained$cost_history, 
     type = "l", xlab = "Epoch", ylab = "Training Cost", 
     main = "Neural Network Training Progress",
     col = "blue")

# Plot test cost history
lines(1:length(NN.trained$test_cost_history), NN.trained$test_cost_history, 
      col = "red")
legend("topright", legend = c("Training Cost", "Test Cost"), 
       col = c("blue", "red"), lty = 1)

# Plot evaluation metrics
plot_metrics(NN.trained)

# Reset plot parameters
par(mfrow = c(1, 1))

# Make predictions on the entire dataset
predictions <- NN.predict(NN.trained, X = X)

# Convert predictions to binary class (0 or 1)
binary_predictions <- ifelse(predictions > 0.1, 1, 0)

# Create final confusion matrix for the entire dataset
final_confusion_matrix <- table(
  Predicted = factor(binary_predictions, levels = c(0, 1)),
  Actual = factor(Y, levels = c(0, 1))
)

# Calculate and print final metrics for the entire dataset
cat("\nFinal Evaluation on Complete Dataset:\n")
print(final_confusion_matrix)

# Calculate metrics from the confusion matrix
final_metrics <- calculate_metrics(final_confusion_matrix)
cat("\nFinal Metrics on Complete Dataset:\n")
cat("Accuracy:", round(final_metrics$accuracy, 4), "\n")
cat("Precision:", round(final_metrics$precision, 4), "\n")
cat("Recall:", round(final_metrics$recall, 4), "\n")
cat("F1 Score:", round(final_metrics$f1_score, 4), "\n")

# Save the trained model
saveRDS(NN.trained, "trained_neural_network.rds")