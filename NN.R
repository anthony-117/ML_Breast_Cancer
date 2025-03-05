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
  return(-0.5 * colSums(error^2) )
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
    X = X,
    Y.d = Y.d,
    layers = layers,
    weights = weights,
    old_weights = weights,
    learning_rate = learning_rate,
    momentum = momentum,
    epochs = epochs,
    cost_history = numeric(epochs)
  )
  return(NN)
  
}
# 
# 
# feed_forward <- function(NN){
#   y.fw <- list()
#   z.fw <- list()
#   y <- as.matrix(NN$X)
#   for( l in 1:length(NN$weights)){
#     y.p <- cbind(-1,y)
#     z.l <- y.p %*% NN$weights[[l]]
#     z.fw[[l]] <- z.l
#     
#     y.l <- activation.fn(z.l)
#     
#     y.fw[[l]] <- y.l
#     y <- y.l
#   }
#   
#   return(list(
#     y.fw = y.fw,
#     z.fw = z.fw
#     ))
# }


feed_forward <- function(NN){
  y.fw <- list()
  z.fw <- list()
  
  # Start with input layer
  y <- as.matrix(NN$X)
  y.fw[[1]] <- y  # Store the input layer activation
  
  for(l in 1:length(NN$weights)){
    # Add bias term
    y.p <- cbind(-1, y)
    
    # Calculate weighted input
    z.l <- y.p %*% NN$weights[[l]]
    z.fw[[l]] <- z.l
    
    # Apply activation function
    y.l <- activation.fn(z.l)
    
    # Store activations (using l+1 since input is at position 1)
    y.fw[[l+1]] <- y.l
    
    # Set current layer output as input to next layer
    y <- y.l
  }
  
  return(list(
    y.fw = y.fw,  # List of all layer activations (including input)
    z.fw = z.fw   # List of all weighted inputs (pre-activation)
  ))
}



back_propagation <- function(NN, z.fw){
  dirac <- list()
  weight.nb <- length(NN$weights)
  
  z.L <- z.fw[[length(z.fw)]]
  error <- NN$Y.d - activation.fn(z.L)
  
  # element wise multiplication
  dirac.L <- error * activation.dfn(z.L)
  
  dirac[[weight.nb]] <- as.matrix(dirac.L)
  
  # Dirac of hidden layers
  for(l in (weight.nb-1):1){
    dirac[[l]] <- dirac.hidden(z.fw[[l]], dirac[[l+1]], NN$weights[[l+1]] )
  } 
  return(dirac)
  
}

NN.train <- function(NN, verbose = TRUE) {
  # Initialize cost history
  cost_history <- numeric(NN$epochs)
  
  # Store old weights for momentum calculations
  old_weight_updates <- lapply(NN$weights, function(w) matrix(0, nrow = nrow(w), ncol = ncol(w)))
  
  # Training loop
  for (epoch in 1:NN$epochs) {
    # Forward pass
    forward_result <- feed_forward(NN)
    y.fw <- forward_result$y.fw
    z.fw <- forward_result$z.fw
    
    # Get final output
    output <- y.fw[[length(y.fw)]]
    
    # Calculate error and cost
    error <- NN$Y.d - output
    current_cost <- sum(0.5 * colSums(error^2))
    cost_history[epoch] <- current_cost
    
    # Backward pass to get gradients
    dirac <- back_propagation(NN, z.fw)
    
    # Update weights using gradients, learning rate, and momentum
    for (l in 1:length(NN$weights)) {
      # Get activations from previous layer (with bias)
      if (l == 1) {
        a_prev <- cbind(-1, NN$X)
      } else {
        a_prev <- cbind(-1, y.fw[[l]])
      }
      
      # Calculate weight updates
      # Gradient is outer product of activations and delta
      gradient <- t(a_prev) %*% dirac[[l]]
      
      # Apply momentum
      weight_update <- NN$learning_rate * gradient + NN$momentum * old_weight_updates[[l]]
      
      # Store current update for next epoch's momentum
      old_weight_updates[[l]] <- weight_update
      
      # Update weights
      NN$weights[[l]] <- NN$weights[[l]] + weight_update
    }
    
    # Print progress if verbose
    if (verbose && epoch %% 100 == 0) {
      cat("Epoch:", epoch, "Cost:", current_cost, "\n")
    }
  }
  
  # Return updated NN with training results
  NN$cost_history <- cost_history
  NN$final_output <- y.fw[[length(y.fw)]]
  
  return(NN)
}

# Prediction function to use the trained network
predict_neural_network <- function(NN, new_data = NULL) {
  # If no new data is provided, use the training data
  if (is.null(new_data)) {
    X <- NN$X
  } else {
    X <- new_data
  }
  
  # Forward pass through the network
  y <- as.matrix(X)
  
  for (l in 1:length(NN$weights)) {
    y.p <- cbind(-1, y)
    z.l <- y.p %*% NN$weights[[l]]
    y <- activation.fn(z.l)
  }
  
  return(y)
}


