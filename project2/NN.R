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
  return(tanh(x))
}
activation.dfn <- function(x){
  return(1 - tanh(x)^2)
}

cost <- function(error){
  return(0.5 * colSums(error^2) /nrow(error) )
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
    # Set first row to fixed value 0.5
    weights[[i]][1,] <- 0.5
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



back_propagation <- function(NN, z.fw,Y.train){
  dirac <- list()
  weight.nb <- length(NN$weights)
  
  z.L <- z.fw[[length(z.fw)]]
  error <- Y.train - activation.fn(z.L)
  
  # element wise multiplication
  dirac.L <- error * activation.dfn(z.L)
  
  dirac[[weight.nb]] <- as.matrix(dirac.L)
  
  # Dirac of hidden layers
  for(l in seq(weight.nb - 1, 1, by = -1)){
    dirac[[l]] <- dirac.hidden(z.fw[[l]], dirac[[l+1]], NN$weights[[l+1]] )
  }
  return(dirac)
  
}


NN.train <- function(NN, X, Y, epochs, learning_rate, momentum, verbose = TRUE) {
  
  cost_history <- numeric(epochs)
  
  # Lists to store predictions, true values, and corresponding X values for each epoch
  train_predictions <- list()
  test_predictions <- list()
  train_true_values <- list()
  test_true_values <- list()
  train_X <- list()
  test_X <- list()
  
  best_weights <- NN$weights
  min_cost <- Inf  # Initialize minimum cost as infinity
  
  old_weight <- NN$weights
  percent <- 0.8  # Training split percentage
  
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
    
    # Store predictions, true values, and X values for training
    train_predictions[[epoch]] <- output
    train_true_values[[epoch]] <- Y.train
    train_X[[epoch]] <- X.train
    
    # Check if this is the best cost so far
    if (current_cost < min_cost) {
      min_cost <- current_cost
      best_weights <- NN$weights
    }
    
    # Backpropagation
    dirac <- back_propagation(NN, z.fw, Y.train)
    
    # Update weights
    for (l in 1:length(NN$weights)) {
      a <- cbind(-1, y.fw[[l]])
      weight.delta <- learning_rate * (t(a) %*% dirac[[l]]) + 
        momentum * (NN$weights[[l]] - old_weight[[l]])
      
      weight.delta[1,] <- 0
      
      old_weight[[l]] <- NN$weights[[l]]
      NN$weights[[l]] <- NN$weights[[l]] + weight.delta
    }
    
    # Test set prediction and true values
    test_output <- NN.predict(NN, X.test)
    test_predictions[[epoch]] <- test_output
    test_true_values[[epoch]] <- Y.test
    test_X[[epoch]] <- X.test
    
    if (verbose && epoch %% 100 == 0) {
      cat("Epoch:", epoch, "Cost:", current_cost, "\n")
    }
  }
  
  # Assign the best weights found
  NN$weights <- best_weights
  
  return(list(
    NN = NN,
    cost_history = cost_history,
    train_predictions = train_predictions,
    test_predictions = test_predictions,
    train_true_values = train_true_values,
    test_true_values = test_true_values,
    train_X = train_X,
    test_X = test_X
  ))
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
    cat("Training with learning rate:", m, "\n")
    result <- NN.train(NN, X, Y, epochs, learning_rate, m, verbose)
    training_results[[as.character(m)]] <- result$training_data
  }
  
  return(training_results)
}

