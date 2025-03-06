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

NN.train <- function(NN, verbose = TRUE) {
  
  cost_history <- numeric(NN$epochs)
  
  old_weight <- NN$weights
  percent <- .8
  
  test_cost_history <- numeric()
  confusion_matrices <- list()
  
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
      

      weight.delta <- NN$learning_rate * (t(a) %*% dirac[[l]]) + NN$momentum *(NN$weight[[l]] - old_weight[[l]])
      
      
      old_weight[[l]] <- NN$weights[[l]]
      
      NN$weights[[l]] <- NN$weights[[l]] + weight.delta
    }
    
    # --- 6. Forward propagation on Testing Set ---
    # NN$X <- X_test
    # NN$Y.d <- Y_test
    test_result <- feed_forward(NN, X.test)
    test_output <- test_result$y.fw[[length(test_result$y.fw)]]
    
    # --- 7. Compute Testing Cost ---
    error_test <- Y.test - test_output
    test_cost_history[epoch] <- cost(error_test)
    
    # --- 8. Compute Confusion Matrix ---
    predicted_labels <- ifelse(test_output > 0.1, 1, 0)  # Assuming binary classification
    actual_labels <- Y.test
    
    confusion_matrix <- table(Predicted = predicted_labels, Actual = actual_labels)
    confusion_matrices[[epoch]] <- confusion_matrix  # Store for analysis
    
    if (verbose && epoch %% 100 == 0) {
      cat("Epoch:", epoch, "Cost:", current_cost, "\n")
    }
  }
  
  NN$cost_history <- cost_history
  NN$final_output <- y.fw[[length(y.fw)]]
  
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


