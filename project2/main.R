source("dataset_creation.R")
source("NN.R")
source("visualize_training_results.R")

dataset <- read.csv("sine_wave_dataset.csv")
X <- as.matrix(dataset$X)
Y <- as.matrix(dataset$Y)
NN <- NN(X, Y, c(20,5))
epochs <- 1000

result <- NN.train(NN, X, Y, epochs = 10000, learning_rate = 0.0005, momentum = 0)

# Plot predictions vs true values for a specific epoch (e.g., last epoch)
last_epoch <- epochs
plot(result$test_X[[last_epoch]], result$test_predictions[[last_epoch]], 
     col = "blue", pch = 16, main = "Predictions vs True Values")
points(result$test_X[[last_epoch]], result$test_true_values[[last_epoch]], 
       col = "red", pch = 16)
legend("topright", c("Predictions", "True Values"), 
       col = c("blue", "red"), pch = 16)

plot_last_epoch(result, "My Neural Network Performance")

