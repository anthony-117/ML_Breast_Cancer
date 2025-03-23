library(tidyverse)
library(caret)

sigmoid <- function(z){
  return(1/(1 + exp(-z)))
}

sigmoid.derivative <- function(z){
  sig <- sigmoid(z)
  return(sig * (1 - sig))
}


activation.fn <- function(z){
  return(sigmoid(z))
}
activation.dfn <- function(z){
  return(sigmoid.derivative(z))
}


mse_cost <- function(y.predict, y.true){
  error <- y.true - y.predict
  return(mean(error^2) / 2)}
mse_derivative <- function(y.predict, y.true) {
  return(-(y.true - y.predict))
}


cost <- function(y.predict, y.true){
  return(mse_cost(y.predict, y.true))
}
cost.derivative <- function(y.predict, y.true){
  return(mse_derivative(y.predict, y.true))
}

# .p stands for plus 1 layer
dirac.hidden <- function(z.l, dirac.p, weight.p){
  
  weight.nobias <- weight.p[-1, , drop = FALSE]
  
  # Matrix multiplication followed by element-wise multiplication
  dirac <- (dirac.p %*% t(weight.nobias)) * activation.dfn(z.l)
  
  return(as.matrix(dirac))
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
initialize_weights <- function(layers) {
  weights = list()
  
  # Loop over layers to initialize weights
  for (i in 1:(length(layers) - 1)) {
    # Random weights between layers[i] and layers[i+1]
    weights[[i]] <- matrix(runif((layers[i] +1) * layers[i+1], min=-0.5, max=0.5), nrow=layers[i] +1, ncol=layers[i+1])
    # # Set first row to fixed value 0.5
    # weights[[i]][1,] <- 0.5
  }
  return(weights)
}



feed_forward <- function(NN, X.train){
  y.fw <- list()
  z.fw <- list()
  
  y <- X.train
  y.fw[[1]] <- y  # Store the input layer activation
  
  for(l in 1:length(NN$weights)){
    # add the -1 neuron 
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



back_propagation <- function(NN, z.fw, y.fw, Y.train){
  gradients <- list()
  m <- length(Y[,1])
  nb.weights <- length(NN$weights)
  output <- activation.fn(z.fw[[nb.weights]])
  # dy -> partial derivative of cost with respect
  # to y
  dy <- cost.derivative(y.predict = output, y.true = Y.train)
  for(l in nb.weights:1){
    weight <- NN$weights[[l]]
    dz <- dy * activation.dfn(z.fw[[l]])
    
    y.prev <- cbind(-1,y.fw[[l]])
    dW <- (t(y.prev) %*% dz)
    
    if (l > 1) {
      # Remove bias column for back propagation
      weight.nobias <- weight[-1, , drop = FALSE]
      dy <- dz %*% t(weight.nobias)
    }
    gradients[[l]] <- dW
    
  }
  return(gradients)
  
}


output.classify <- function(output){
  # Using 0.1 as threshold for binary classification
  predicted_labels <- ifelse(output > 0.1, 1, 0)  
  return(predicted_labels)
}

NN.train <- function(NN, X, Y, epochs, learning_rate, momentum, verbose = TRUE) {
  
  cost_history <- numeric(epochs)
  # test_cost_history <- numeric(epochs)
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
    # error <- Y.train - output
    current_cost <- cost(y.predict = output, y.true = Y.train)
    cost_history[epoch] <- current_cost
    
    # Backpropagation
    gradients <- back_propagation(NN, z.fw, y.fw, Y.train)
    
    # Update weights
    for (l in 1:length(NN$weights)) {
      weight.delta <- learning_rate * gradients[[l]] + 
        momentum * (NN$weights[[l]] - old_weight[[l]])
      
      # weight.delta[1,] <- 0
      
      old_weight[[l]] <- NN$weights[[l]]
      NN$weights[[l]] <- NN$weights[[l]] - weight.delta
    }
    
    
    # Compute confusion matrix
    predicted_labels <- NN.predict(NN, X.test)
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
    # test_cost_history = test_cost_history,
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
  y <- output.classify(y)
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
    cat("Training with learning rate:", m, "\n")
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

plot_confusion_matrix <- function(NN, X, Y) {
  # Get predictions from the neural network
  predictions <- NN.predict(NN, X)
  
  pred_vector <- as.vector(predictions)
  actual_vector <- as.vector(as.integer(Y[[1]]))
  
  # Create confusion matrix with factors having levels in the desired order (1 first, then 0)
  cm <- table(Predicted = factor(pred_vector, levels = c(1, 0)), 
              Actual = factor(actual_vector, levels = c(1, 0)))
  
  # Change the labels to 'M' and 'B'
  new_labels <- c("M", "B")
  colnames(cm) <- new_labels[match(colnames(cm), c("1", "0"))]
  rownames(cm) <- new_labels[match(rownames(cm), c("1", "0"))]
  
  # Extract metrics (now using updated labels)
  TP <- if ("M" %in% rownames(cm) && "M" %in% colnames(cm)) cm["M", "M"] else 0
  FP <- if ("M" %in% rownames(cm) && "B" %in% colnames(cm)) cm["M", "B"] else 0
  FN <- if ("B" %in% rownames(cm) && "M" %in% colnames(cm)) cm["B", "M"] else 0
  TN <- if ("B" %in% rownames(cm) && "B" %in% colnames(cm)) cm["B", "B"] else 0
  
  # Calculate performance metrics
  accuracy <- (TP + TN) / (TP + TN + FP + FN)
  precision <- ifelse(TP + FP > 0, TP / (TP + FP), 0)
  recall <- ifelse(TP + FN > 0, TP / (TP + FN), 0)
  f1_score <- ifelse(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
  
  # Create a data frame for ggplot2
  cm_df <- as.data.frame(as.table(cm))
  names(cm_df) <- c("Prediction", "Observed", "Freq")
  
  # Make sure the order is preserved in the plot
  cm_df$Prediction <- factor(cm_df$Prediction, levels = c("M", "B"))
  cm_df$Observed <- factor(cm_df$Observed, levels = c("M", "B"))
  
  # Create the heatmap using ggplot2
  library(ggplot2)
  
  p <- ggplot(cm_df, aes(x = Observed, y = Prediction, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "black", size = 8) +
    scale_fill_gradient(low = "#c1ff2e", high = "#25188c") +
    labs(title = "Confusion matrix", x = "Observed", y = "Prediction") +
    theme_minimal() +
    theme(
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 14),
      plot.title = element_text(size = 16, hjust = 0.5),
      legend.title = element_text(size = 12),
      legend.text = element_text(size = 10)
    )
  
  # Print metrics
  cat("Accuracy:", round(accuracy, 4), "\n")
  cat("Precision:", round(precision, 4), "\n")
  cat("Recall:", round(recall, 4), "\n")
  cat("F1 Score:", round(f1_score, 4), "\n")
  
  # Return both the confusion matrix, metrics, and the plot
  return(list(
    confusion_matrix = cm,
    metrics = list(
      accuracy = accuracy,
      precision = precision,
      recall = recall,
      f1_score = f1_score
    ),
    plot = p
  ))
}