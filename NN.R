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



NN <-function(X, Y.d, hidden_layers){
  
  layers <- c(ncol(X), hidden_layers, ncol(Y.d))
  
  weights <- initialize_weights(layers)
  
  NN <- list(
    layers = layers,
    weights = weights
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


output.classify <- function(output){
  predicted_labels <- ifelse(output > 0.1, 1, 0)  # Using 0.1 as threshold for binary classification
  return(predicted_labels)
}

NN.train <- function(NN, X, Y, epochs, learning_rate, momentum, verbose = TRUE) {
  
  cost_history <- numeric(epochs)
  test_cost_history <- numeric(epochs)
  confusion_matrices <- data.frame(Epoch = integer(), TP = integer(), FP = integer(), FN = integer(), TN = integer())
  
  old_weight <- NN$weights
  percent <- 0.8
  
  accuracy_history <- numeric(epochs)
  precision_history <- numeric(epochs)
  recall_history <- numeric(epochs)
  f1_history <- numeric(epochs)
  
  for (epoch in 1:epochs) {
    indices <- sample(1:nrow(X), size = percent * nrow(X))
    
    X.train <- X[indices, ]
    Y.train <- Y[indices, ]
    X.test <- X[-indices, ]
    Y.test <- Y[-indices, ]
    
    # Forward propagation
    forward_result <- feed_forward(NN, X.train)
    y.fw <- forward_result$y.fw
    z.fw <- forward_result$z.fw
    output <- y.fw[[length(y.fw)]]
    
    # Compute cost
    error <- Y.train - output
    current_cost <- cost(error)
    cost_history[epoch] <- current_cost
    
    # Backpropagation
    dirac <- back_propagation(NN, z.fw, Y.train)
    
    # Update weights
    for (l in 1:length(NN$weights)) {
      a <-  cbind(-1, y.fw[[l]])
      weight.delta <- learning_rate * (t(a) %*% dirac[[l]]) + momentum * (NN$weights[[l]] - old_weight[[l]])
      old_weight[[l]] <- NN$weights[[l]]
      NN$weights[[l]] <- NN$weights[[l]] + weight.delta
    }
    
    # Testing phase
    forward_result_test <- feed_forward(NN, X.test)
    test_output <- forward_result_test$y.fw[[length(forward_result_test$y.fw)]]
    
    # Compute test cost
    error_test <- Y.test - test_output
    test_cost_history[epoch] <- cost(error_test)
    
    # Compute confusion matrix
    predicted_labels <- output.classify(test_output)
    actual_labels <- Y.test
    cm <- table(Predicted = factor(predicted_labels, levels = c(1, 0)), 
                Actual = factor(actual_labels, levels = c(1, 0)))
    
    # Extract TP, FP, FN, TN
    TP <- if ("1" %in% rownames(cm) && "1" %in% colnames(cm)) cm["1", "1"] else 0
    FP <- if ("1" %in% rownames(cm) && "0" %in% colnames(cm)) cm["1", "0"] else 0
    FN <- if ("0" %in% rownames(cm) && "1" %in% colnames(cm)) cm["0", "1"] else 0
    TN <- if ("0" %in% rownames(cm) && "0" %in% colnames(cm)) cm["0", "0"] else 0
    
    confusion_matrices <- rbind(confusion_matrices, data.frame(Epoch = epoch, TP = TP, FP = FP, FN = FN, TN = TN))
    
    # Compute metrics
    metrics <- calculate_metrics(cm)
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
  
  # Store evaluation data separately
  training_data <- list(
    cost_history = cost_history,
    test_cost_history = test_cost_history,
    confusion_matrices = confusion_matrices,
    accuracy_history = accuracy_history,
    precision_history = precision_history,
    recall_history = recall_history,
    f1_history = f1_history
  )
  
  return(list(NN = NN, training_data = training_data))
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

NN.train_multiple.lr <- function(NN, X, Y, epochs, learning_rates, momentum, verbose = TRUE) {
  training_results <- list()
  
  for (lr in learning_rates) {
    cat("Training with learning rate:", lr, "\n")
    result <- NN.train(NN, X, Y, epochs, lr, momentum, verbose)
    training_results[[as.character(lr)]] <- result$training_data
  }
  
  return(training_results)
}


NN.train_multiple.momentum <- function(NN, X, Y, epochs, learning_rate, momentums, verbose = TRUE) {
  training_results <- list()
  
  for (m in momentums) {
    cat("Training with learning rate:", lr, "\n")
    result <- NN.train(NN, X, Y, epochs, learning_rate, m, verbose)
    training_results[[as.character(m)]] <- result$training_data
  }
  
  return(training_results)
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