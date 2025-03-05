library(tidyverse)
library(caret)

detect_outliers <- function(x) {
  q1 <- quantile(x, 0.25)
  q3 <- quantile(x, 0.75)
  iqr <- q3 - q1
  lower_bound <- q1 - (1.5 * iqr)
  upper_bound <- q3 + (1.5 * iqr)
  
  outliers <- x[(x < lower_bound) | (x > upper_bound)]
  return(outliers)
}

sigmoid <- function(x){
  return(1/(1 + exp(-x)))
}
sigmoid.derivative <- function(x){
  sig <- sigmoid(x)
  return(sig - sig ** 2)
}
error <- function(y.hat, y.d){
  return(-0.5 * sum(y.d - y.hat) ** 2)
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
    epochs = epochs
  )
  return(NN)
  
}

feed_forward <- function(NN, start_index , batch.size){
  y.fw <- list()
  y.hat <- vector()
  weights.nb <- length(NN$weights)
  for( i in 1:batch.size){
    y.fw[[1]] <- as.matrix(NN$X[i, , drop = FALSE])
    for( j in 1:length(NN$weights)){
      v <- c(-1, y.fw[[j]]) %*% NN$weights[[j]]
      y.fw[[j+1]] <- t(sigmoid(v))
    }
    y.hat[i] <- t(y.fw[[length(y.fw)]])
  }
  
  return(y.hat)
  
}

back_propagation <- function(NN, y.hat){
  dirac <- list()
  
  error <- error(y.fw, NN$Y)
  
  dirac.L <- error * y.hat (1 - y.hat)
  dirac <- append(dirac, dirac.L)
  
  weights.nb <- length(NN$weights)
  
  for (i in 2:length(layers_n)){
    dirac.j <- y.hat
    dirac <- append(dirac, dirac.j)
  }
  
  
}
