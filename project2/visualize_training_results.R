# Function to plot true values vs predicted values using geom_line
plot_predictions_vs_true <- function(NN, X, Y, train_indices = NULL) {
  # If train_indices is not provided, create a random split
  if (is.null(train_indices)) {
    set.seed(123)
    train_indices <- sample(1:nrow(X), size = 0.8 * nrow(X))
  }
  
  # Split data into training and testing
  X_train <- X[train_indices, ]
  Y_train <- Y[train_indices, ]
  X_test <- X[-train_indices, ]
  Y_test <- Y[-train_indices, ]
  
  # Generate predictions
  train_preds <- NN.predict(NN, X_train)
  test_preds <- NN.predict(NN, X_test)
  
  # Create dataframes for plotting
  train_df <- data.frame(
    x = seq_along(Y_train),
    true = Y_train,
    predicted = train_preds,
    dataset = "Training"
  )
  
  test_df <- data.frame(
    x = seq_along(Y_test),
    true = Y_test,
    predicted = test_preds,
    dataset = "Testing"
  )
  
  # Check if ggplot2 is installed
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Please install ggplot2 package to use this function")
  }
  
  # Create plots
  library(ggplot2)
  
  # Training data plot
  train_plot <- ggplot(train_df, aes(x = x)) +
    geom_line(aes(y = true, color = "True Values"), size = 1) +
    geom_line(aes(y = predicted, color = "Predicted Values"), size = 1, linetype = "dashed") +
    scale_color_manual(values = c("True Values" = "steelblue", "Predicted Values" = "tomato")) +
    labs(title = "Training Set: True vs Predicted Values",
         x = "Data Point Index",
         y = "Value",
         color = "Legend") +
    xlim(-10,10)+
    theme_minimal()
  
  # Testing data plot
  test_plot <- ggplot(test_df, aes(x = x)) +
    geom_line(aes(y = true, color = "True Values"), size = 1) +
    geom_line(aes(y = predicted, color = "Predicted Values"), size = 1, linetype = "dashed") +
    scale_color_manual(values = c("True Values" = "steelblue", "Predicted Values" = "tomato")) +
    labs(title = "Testing Set: True vs Predicted Values",
         x = "Data Point Index",
         y = "Value",
         color = "Legend") +
    theme_minimal()
  
  # Calculate and display error metrics
  train_mse <- mean((Y_train - train_preds)^2)
  test_mse <- mean((Y_test - test_preds)^2)
  train_mae <- mean(abs(Y_train - train_preds))
  test_mae <- mean(abs(Y_test - test_preds))
  
  cat("Training MSE:", train_mse, "\n")
  cat("Testing MSE:", test_mse, "\n")
  cat("Training MAE:", train_mae, "\n")
  cat("Testing MAE:", test_mae, "\n")
  
  # Print plots
  print(train_plot)
  print(test_plot)
  
  # Return the plots for further customization if needed
  return(list(train_plot = train_plot, test_plot = test_plot))
}

# Alternative function using base R graphics instead of ggplot2
plot_predictions_vs_true_base <- function(NN, X, Y, train_indices = NULL) {
  # If train_indices is not provided, create a random split
  if (is.null(train_indices)) {
    set.seed(123)
    train_indices <- sample(1:nrow(X), size = 0.8 * nrow(X))
  }
  
  # Split data into training and testing
  X_train <- X[train_indices, ]
  Y_train <- Y[train_indices, ]
  X_test <- X[-train_indices, ]
  Y_test <- Y[-train_indices, ]
  
  # Generate predictions
  train_preds <- NN.predict(NN, X_train)
  test_preds <- NN.predict(NN, X_test)
  
  # Set up the plotting area
  par(mfrow = c(2, 1), mar = c(4, 4, 3, 1))
  
  # Training data plot
  plot(seq_along(Y_train), Y_train, type = "l", col = "steelblue", lwd = 2,
       main = "Training Set: True vs Predicted Values",
       xlab = "Data Point Index", ylab = "Value",
       ylim = range(c(Y_train, train_preds)))
  lines(seq_along(train_preds), train_preds, col = "tomato", lwd = 2, lty = 2)
  legend("topleft", legend = c("True Values", "Predicted Values"),
         col = c("steelblue", "tomato"), lwd = 2, lty = c(1, 2))
  
  # Testing data plot
  plot(seq_along(Y_test), Y_test, type = "l", col = "steelblue", lwd = 2,
       main = "Testing Set: True vs Predicted Values",
       xlab = "Data Point Index", ylab = "Value",
       ylim = range(c(Y_test, test_preds)))
  lines(seq_along(test_preds), test_preds, col = "tomato", lwd = 2, lty = 2)
  legend("topleft", legend = c("True Values", "Predicted Values"),
         col = c("steelblue", "tomato"), lwd = 2, lty = c(1, 2))
  
  # Reset plot parameters
  par(mfrow = c(1, 1))
  
  # Calculate and display error metrics
  train_mse <- mean((Y_train - train_preds)^2)
  test_mse <- mean((Y_test - test_preds)^2)
  train_mae <- mean(abs(Y_train - train_preds))
  test_mae <- mean(abs(Y_test - test_preds))
  
  cat("Training MSE:", train_mse, "\n")
  cat("Testing MSE:", test_mse, "\n")
  cat("Training MAE:", train_mae, "\n")
  cat("Testing MAE:", test_mae, "\n")
}

library(ggplot2)

plot_last_epoch <- function(nn_result, title = "Neural Network Last Epoch Results") {
  # Get the number of epochs from the cost history
  epochs <- length(nn_result$cost_history)
  
  # Extract the last epoch data
  last_test_X <- nn_result$test_X[[epochs]]
  last_test_predictions <- nn_result$test_predictions[[epochs]]
  last_test_true <- nn_result$test_true_values[[epochs]]
  
  # If X has multiple columns, we need to decide which one to plot
  # For simplicity, we'll use the first column if X is multi-dimensional
  if (is.matrix(last_test_X) || is.data.frame(last_test_X)) {
    plot_X <- last_test_X[, 1]
  } else {
    plot_X <- last_test_X
  }
  
  # Create a data frame for ggplot
  df <- data.frame(
    X = plot_X,
    Predictions = last_test_predictions,
    True_Values = last_test_true
  )
  
  # Plot using ggplot2
  p <- ggplot(df, aes(x = X)) +
    geom_line(aes(y = Predictions, color = "Predictions"), size = 1) +
    geom_line(aes(y = True_Values, color = "True Values"), size = 1) +
    labs(title = title, x = "X", y = "Y") +
    scale_color_manual(values = c("Predictions" = "blue", "True Values" = "red")) +
    theme_minimal() +
    theme(legend.title = element_blank()) +
    theme(legend.position = "top")
  
  # Display the plot
  print(p)
  
  # Calculate and display RMSE
  rmse <- sqrt(mean((last_test_predictions - last_test_true)^2, na.rm = TRUE))
  print(paste("RMSE:", round(rmse, 4)))
  
  # Return the plot data invisibly for further use if needed
  invisible(list(
    X = plot_X,
    predictions = last_test_predictions,
    true_values = last_test_true,
    rmse = rmse
  ))
}
plot_cost_progression <- function(training_results, title = "Cost Progression During Training") {
  # Extract cost history
  cost_history <- training_results$cost_history
  epochs <- 1:length(cost_history)
  
  # Create a dataframe for plotting
  plot_data <- data.frame(
    Epoch = epochs,
    Cost = cost_history
  )
  
  # Calculate initial and final costs for annotation
  initial_cost <- cost_history[1]
  final_cost <- cost_history[length(cost_history)]
  
  # Create the plot
  plot <- ggplot(plot_data, aes(x = Epoch, y = Cost)) +
    geom_line(color = "blue", linewidth = 1) +
    geom_point(data = plot_data[c(1, nrow(plot_data)), ], 
               color = c("darkred", "darkgreen"), size = 3) +
    geom_text(data = plot_data[c(1, nrow(plot_data)), ],
              aes(label = c(sprintf("Initial: %.6f", initial_cost), 
                            sprintf("Final: %.6f", final_cost))),
              vjust = c(-1, -1), hjust = c(0.5, 0.5), size = 3.5) +
    labs(
      title = title,
      subtitle = paste("From epoch 1 to", length(cost_history)),
      x = "Epoch",
      y = "Cost"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 12),
      axis.title = element_text(size = 12)
    )
  
  # If the cost range is very large, use log scale
  if (max(cost_history)/min(cost_history) > 100) {
    plot <- plot + scale_y_log10() +
      labs(subtitle = paste("From epoch 1 to", length(cost_history), "(log scale)"))
  }
  
  return(plot)
}
