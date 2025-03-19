source("dataset_creation.R")
source("NN.R")
source("visualize_training_results.R")

dataset <- read.csv("sine_wave_dataset_preprocess.csv")
raw_dataset <- read.csv("sine_wave_dataset.csv")

X <- as.matrix(dataset$X)
Y <- as.matrix(dataset$Y)

NN <- NN(X, Y, c(10,7,7))

epochs <- 10000

result <- NN.train(NN, X, Y, epochs = epochs, learning_rate = 0.001, momentum = 0.2)

NN.trained <- result$NN
# Plot the cost progression across all epochs
cost_progression_plot <- plot_cost_progression(result, 
                                               "Cost Progression")

`# Display the plot
print(cost_progression_plot)

# Plot predictions vs true values for a specific epoch (e.g., last epoch)
last_epoch <- epochs
plot(result$test_X[[last_epoch]], result$test_predictions[[last_epoch]],
     col = "blue", pch = 16, main = "Predictions vs True Values")
points(result$test_X[[last_epoch]], result$test_true_values[[last_epoch]],
       col = "red", pch = 16)
legend("topright", c("Predictions", "True Values"),
       col = c("blue", "red"), pch = 16)

plot_last_epoch(result, "My Neural Network Performance")

sequence <- seq(-1, 10, by = 0.01)
y.predicted <- NN.predict.scaled(NN.trained, sequence)
ggplot() +
  geom_line(aes(x = sequence, y = y.predicted))
