library(tidyverse)
library(caret)

sigmoid <- function(x){
  return(1/(1 + exp(-x)))
}

sigmoid.derivative <- function(x){
  sig <- sigmoid(x)
  return(sig * (1 - sig))
}

activation.fn <- function(x){
  return(sigmoid(x))
}
activation.dfn <- function(x){
  return(sigmoid.derivative(x))
}

cost <- function(error){
  return(0.5 * colSums(error^2) )
}

dirac.hidden <- function(z.l, dirac.lp, weight.p){
  
  weight.p_nobias <- weight.p[-1, , drop = FALSE]
  
  # Matrix multiplication followed by element-wise multiplication
  dirac <- (dirac.lp %*% t(weight.p_nobias)) * activation.dfn(z.l)
  
  return(as.matrix(dirac))
}


initialize_weights <- function(layers) {
  weights = list()
  
  # Loop over layers to initialize weights
  for (i in 1:(length(layers) - 1)) {
    # Random weights between layers[i] and layers[i+1]
    weights[[i]] <- matrix(runif((layers[i] +1) * layers[i+1], min=-0.5, max=0.5), nrow=layers[i] +1, ncol=layers[i+1])
  }
  
  return(weights)
}



NN <-function(X, Y.d, hidden_layers, learning_rate, momentum, epochs){
  layers <- c(ncol(X), hidden_layers, ncol(Y.d))
  
  weights <- initialize_weights(layers)
  
  NN <- list(
    # X = X,
    # Y.d = Y.d,
    layers = layers,
    weights = weights,
    learning_rate = learning_rate,
    momentum = momentum,
    epochs = epochs,
    cost_history = numeric(epochs)
  )
  return(NN)
  
}

feed_forward <- function(NN, X.train){
  y.fw <- list()
  z.fw <- list()
  
  y <- X.train
  y.fw[[1]] <- y  # Store the input layer activation
  
  for(l in 1:length(NN$weights)){
    y.p <- as.matrix(cbind(-1, y))
    
    z.l <- y.p %*% NN$weights[[l]]
    z.fw[[l]] <- z.l
    
    y.l <- activation.fn(z.l)
    
    y.fw[[l+1]] <- y.l
    
    y <- y.l
  }
  
  return(list(
    y.fw = y.fw,  
    z.fw = z.fw   
  ))
}



back_propagation <- function(NN, z.fw,Y.train){
  dirac <- list()
  weight.nb <- length(NN$weights)
  
  z.L <- z.fw[[length(z.fw)]]
  error <- Y.train - activation.fn(z.L)
  
  # element wise multiplication
  dirac.L <- error * activation.dfn(z.L)
  
  dirac[[weight.nb]] <- as.matrix(dirac.L)
  
  # Dirac of hidden layers
  for(l in (weight.nb-1):1){
    dirac[[l]] <- dirac.hidden(z.fw[[l]], dirac[[l+1]], NN$weights[[l+1]] )
  } 
  return(dirac)
  
}

# Function to calculate evaluation metrics
calculate_metrics <- function(confusion_matrix) {
  # Extract values from confusion matrix
  TP <- confusion_matrix[2, 2]  # True Positives
  TN <- confusion_matrix[1, 1]  # True Negatives
  FP <- confusion_matrix[2, 1]  # False Positives
  FN <- confusion_matrix[1, 2]  # False Negatives
  
  # Handle potential division by zero
  accuracy <- (TP + TN) / sum(confusion_matrix)
  
  precision <- ifelse(TP + FP > 0, TP / (TP + FP), 0)
  
  recall <- ifelse(TP + FN > 0, TP / (TP + FN), 0)
  
  f1_score <- ifelse(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
  
  return(list(
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1_score = f1_score
  ))
}

NN.train <- function(NN, verbose = TRUE) {
  
  cost_history <- numeric(NN$epochs)
  
  old_weight <- NN$weights
  percent <- .8
  
  test_cost_history <- numeric(NN$epochs)
  confusion_matrices <- list()
  
  # For storing evaluation metrics
  accuracy_history <- numeric(NN$epochs)
  precision_history <- numeric(NN$epochs)
  recall_history <- numeric(NN$epochs)
  f1_history <- numeric(NN$epochs)
  
  for (epoch in 1:NN$epochs) {
    indices <- sample(1:nrow(X), size = percent * nrow(X))
    
    X.train <- X[indices,]
    Y.train <- Y[indices,]
    
    X.test <- X[-indices,]
    Y.test <- Y[-indices,]
    
    
    forward_result <- feed_forward(NN, X.train)
    y.fw <- forward_result$y.fw
    z.fw <- forward_result$z.fw
    
    output <- y.fw[[length(y.fw)]]
    
    error <- Y.train - output
    current_cost <- cost(error)
    cost_history[epoch] <- current_cost
    
    dirac <- back_propagation(NN, z.fw, Y.train)
    
    for (l in 1:length(NN$weights)) {
      
      if (l == 1) {
        a <- cbind(-1, X.train)
      } else {
        a <- cbind(-1, y.fw[[l]])
      }
      
      
      weight.delta <- NN$learning_rate * (t(a) %*% dirac[[l]]) + NN$momentum *(NN$weights[[l]] - old_weight[[l]])
      
      
      old_weight[[l]] <- NN$weights[[l]]
      
      NN$weights[[l]] <- NN$weights[[l]] + weight.delta
    }
    
    # Forward propagation on Testing Set
    test_result <- feed_forward(NN, X.test)
    test_output <- test_result$y.fw[[length(test_result$y.fw)]]
    
    # Compute Testing Cost
    error_test <- Y.test - test_output
    test_cost_history[epoch] <- cost(error_test)
    
    # Compute Confusion Matrix
    predicted_labels <- ifelse(test_output > 0.5, 1, 0)  # Using 0.5 as threshold for binary classification
    actual_labels <- Y.test
    
    # Create confusion matrix
    confusion_matrix <- table(Predicted = factor(predicted_labels, levels=c(0,1)), 
                              Actual = factor(actual_labels, levels=c(0,1)))
    
    # Handle case where some combinations might be missing in the confusion matrix
    if(nrow(confusion_matrix) < 2 || ncol(confusion_matrix) < 2) {
      # Create a complete 2x2 confusion matrix with zeros for missing combinations
      complete_matrix <- matrix(0, nrow=2, ncol=2)
      rownames(complete_matrix) <- c("0", "1")
      colnames(complete_matrix) <- c("0", "1")
      
      # Fill in the values that do exist
      for(i in rownames(confusion_matrix)) {
        for(j in colnames(confusion_matrix)) {
          complete_matrix[i, j] <- confusion_matrix[i, j]
        }
      }
      confusion_matrix <- complete_matrix
    }
    
    confusion_matrices[[epoch]] <- confusion_matrix
    
    # Calculate and store metrics
    metrics <- calculate_metrics(confusion_matrix)
    accuracy_history[epoch] <- metrics$accuracy
    precision_history[epoch] <- metrics$precision
    recall_history[epoch] <- metrics$recall
    f1_history[epoch] <- metrics$f1_score
    
    if (verbose && epoch %% 100 == 0) {
      cat("Epoch:", epoch, "Cost:", current_cost, 
          "Accuracy:", round(metrics$accuracy, 4),
          "Precision:", round(metrics$precision, 4), "\n")
    }
  }
  
  NN$cost_history <- cost_history
  NN$test_cost_history <- test_cost_history
  NN$final_output <- y.fw[[length(y.fw)]]
  NN$confusion_matrices <- confusion_matrices
  
  # Store the final evaluation metrics
  NN$final_confusion_matrix <- confusion_matrices[[NN$epochs]]
  NN$accuracy_history <- accuracy_history
  NN$precision_history <- precision_history
  NN$recall_history <- recall_history
  NN$f1_history <- f1_history
  
  # Store the final metrics
  final_metrics <- calculate_metrics(NN$final_confusion_matrix)
  NN$final_accuracy <- final_metrics$accuracy
  NN$final_precision <- final_metrics$precision
  NN$final_recall <- final_metrics$recall
  NN$final_f1 <- final_metrics$f1_score
  
  # Print final evaluation metrics
  if (verbose) {
    cat("\nFinal Evaluation Metrics:\n")
    cat("Accuracy:", round(NN$final_accuracy, 4), "\n")
    cat("Precision:", round(NN$final_precision, 4), "\n")
    cat("Recall:", round(NN$final_recall, 4), "\n")
    cat("F1 Score:", round(NN$final_f1, 4), "\n\n")
    
    # Print final confusion matrix
    cat("Final Confusion Matrix:\n")
    print(NN$final_confusion_matrix)
    cat("\n")
  }
  
  return(NN)
}

NN.predict <- function(NN, X) {
  
  y <- as.matrix(X)
  
  for (l in 1:length(NN$weights)) {
    y.p <- cbind(-1, y)
    z.l <- y.p %*% NN$weights[[l]]
    y <- activation.fn(z.l)
  }
  
  return(y)
}

# Function to plot evaluation metrics over epochs
plot_metrics <- function(NN) {
  epochs <- 1:NN$epochs
  
  # Create a data frame for plotting
  metrics_df <- data.frame(
    Epoch = rep(epochs, 4),
    Value = c(NN$accuracy_history, NN$precision_history, NN$recall_history, NN$f1_history),
    Metric = factor(rep(c("Accuracy", "Precision", "Recall", "F1 Score"), each = length(epochs)))
  )
  
  # Plot with ggplot2
  ggplot(metrics_df, aes(x = Epoch, y = Value, color = Metric)) +
    geom_line() +
    labs(title = "Evaluation Metrics Over Epochs",
         x = "Epoch",
         y = "Value") +
    theme_minimal() +
    scale_color_brewer(palette = "Set1")
}