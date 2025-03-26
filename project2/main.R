source("dataset_creation.R")
source("NN_simple.R")
source("plots.R")

dataset <- read.csv("sine_wave_dataset_preprocess.csv")
raw_dataset <- read.csv("sine_wave_dataset.csv")
dataset <- dataset %>% 
              filter(X < 3 & X > -3)

X <- as.matrix(dataset$X)
Y <- as.matrix(dataset$Y)

nn <- NN.create(X, Y, c(30, 20, 10))

epochs <- 10000

# # result <- NN.train(nn, X, Y, epochs = epochs, learning_rate = 0.01, momentum = 0.9)
# result <- NN.train(nn, X, Y, epochs=epochs, learning_rate=0.01,  
#                        batch_type="mini", batch_size = 64, optimizer = "adagrad", verbose=TRUE)
result <- NN.train(nn, X, Y, epochs=10000, learning_rate=0.01, momentum=0.2,
                       batch_type="mini", batch_size=20, verbose=TRUE)
# result <- NN.train(nn, X, Y, epochs=1000, learning_rate=0.1, momentum=0.9, 
#                    batch_type="stochastic", batch_size=20, verbose=TRUE)
NN.trained <- result$NN
X.predict <- seq(5, 10, by = 0.1)
NN.predict(nn, X.predict, verbose = TRUE)
NN.predict.periodic(nn, X.predict)
# X <- seq(0,20, by = 0.1)
# predictions <- NN.predict(result$NN, X
# df <- data.frame(
#   x = X,
#   y = predictions,
#   y.true = calculate_y(X)
# )
# 
# ggplot(df) +
#   geom_line(aes(x = x, y = y)) +
#   geom_line(aes( x = x, y = y.true))



#-----------------Plots.R-----------------#
# # For validation set predictions vs true values
# validation_plot <- plot_last_epoch_predictions(result, "validation")
# print(validation_plot)
# 
# # For training set predictions vs true values
# plot_NN_results(result, epoch = 100)  # Plot predictions at epoch 100
# 
# library(ggplot2)
# library(tidyr)  # For pivot_longer()
# 
# # Ensure x.train.l is a vector (use first column if X is a matrix)
# if (is.matrix(result$X_train[[length(result$X_train)]])) {
#   x.train.l <- result$X_train[[length(result$X_train)]]  # Take first feature
# } else {
#   x.train.l <- result$X_train[[length(result$X_train)]]
# }
# 
# # Create data frame
# df <- data.frame(
#   X = x.train.l,
#   Y_Predicted = result$train_predictions[[length(result$train_predictions)]],
#   Y_True = result$train_true_values[[length(result$train_true_values)]]
# )
# 
# # Reshape data for ggplot
# df_long <- df %>%
#   pivot_longer(cols = c("Y_Predicted", "Y_True"), names_to = "Type", values_to = "Y")
# 
# # Plot
# ggplot(df_long, aes(x = X, y = Y, color = Type)) +
#   geom_line() +
#   labs(title = "Training Predictions vs True Values", x = "X", y = "Y") +
#   theme_minimal()
plot_cost_history(result, show_test = TRUE)

# Plot predictions vs true values for the test set
plot_predictions(result, dataset = "train")

# Visualize training progression every 100 epochs
plot_training_progression(result, interval = 100)
#-----------------Plots.R-----------------#

