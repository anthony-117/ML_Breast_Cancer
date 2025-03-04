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
  return(1/(1-exp(-x)))
}
sigmoid.derivative <- function(x){
  sig <- sigmoid(x)
  return(sig - sig ** 2)
}
error <- function(x, y.d){
  return(-0.5 * sum(y.d - x) ** 2)
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
  layers <- c(ncol(X), hidden_layers, ncol(Y))
  
  weights <- initialize_weights(layers)
  
  NN <- list(
    X = X,
    Y.d = Y.d,
    layers = layers,
    weights = weights,
    learning_rate = learning_rate,
    momentum = momentum,
    error,
    dirac,
    epochs = epochs
  )
  return(NN)
  
}

feed_forward <- function(NN){
  y.fw <- list()
  for( i in 1:nrow(NN$X)){
    y.fw[[1]] <- NN$X[i,]
    for( j in 1:length(NN$layers)){
      v <- y.fw[[j]] %*% NN$weights[[j]]
      y.fw[[j+1]] <- sigmoid(v)
    }
  }
  
}
