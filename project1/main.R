source("preprocess.R")
source("NN.R")

NN <- NN(X = X, Y.d = Y,
         hidden_layers = c(5,3)
         )


result <- NN.train(NN,X, Y, 10000, 0.01, 0, verbose = TRUE)

NN.trained <- result$NN
training_data <- result$training_data


plot_confusion_matrix(NN.trained, X, Y)


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
  geom_line(se = FALSE) +
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
  geom_line() +
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

ggplot(df_acc, aes(x = epoch, y = accuracy)) +
  geom_line(color="blue") 
  

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

