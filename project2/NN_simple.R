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
  return (mean((y.predict - y.true)^2))
}
mse_derivative <- function(y.predict, y.true) {
  return(-(y.true - y.predict))

}
cost <- function(y.predict, y.true){
  return(mse_cost(y.predict, y.true))
}
cost.derivative <- function(y.predict, y.true){
  return(mse_derivative(y.predict, y.true))
}

weights.initialize <- function(layers) {
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

NN.create <- function(X, Y, hidden.layers){
  layers <- c(ncol(X), hidden.layers, ncol(Y))
  weights <- weights.initialize(layers)
  return(list(
    weights = weights,
    layers = layers
  ))
}

forward_propagation <- function(NN, X){
  y.fw <- list()
  z.fw <- list()
  
  y <- X
  y.fw[[1]] <- y  # Store the input layer
  
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

back_propagation <- function(NN, z.fw, y.fw, Y){
  Y <- as.matrix(Y)
  gradients <- list()
  m <- nrow(Y)
  nb.weights <- length(NN$weights)
  output <- y.fw[[length(y.fw)]]
  # dy -> partial derivative of cost with respect to y
  dy <- cost.derivative(y.predict = output, y.true = Y)
  for(l in nb.weights:1){
    weight <- NN$weights[[l]]
    # dz -> partial derivative of cost with respect to z
    dz <- dy * activation.dfn(z.fw[[l]])
    
    y.prev <- cbind(-1,y.fw[[l]])
    dW <- (t(y.prev) %*% dz) /m
    
    if (l > 1) {
      # Remove bias column for back propagation
      weight.nobias <- weight[-1, , drop = FALSE]
      dy <- dz %*% t(weight.nobias)
    }
    gradients[[l]] <- dW
    
  }
  return(gradients)
}
NN.train <- function(NN, X, Y, epochs, learning_rate, momentum, 
                     batch_type = "mini", batch_size = NULL, 
                     test_split = 0.2, verbose = TRUE) {
  cost_history <- numeric(epochs)
  test_cost_history <- numeric(epochs)
  
  # Lists to store predictions, true values, and corresponding X values
  train_predictions <- list()
  test_predictions <- list()
  train_true_values <- list()
  test_true_values <- list()
  X_train_stored <- list()     
  X_test_stored <- list() 
  
  best_weights <- NN$weights
  min_test_cost <- Inf
  
  old_weight <- NN$weights
  n_samples <- nrow(X)
  
  # Split the dataset between test and train sets
  test_indices <- sample(1:n_samples, size = round(test_split * n_samples))
  X.test <- as.matrix(X[test_indices, ])
  Y.test <- as.matrix(Y[test_indices, ])
  X.train_full <- as.matrix(X[-test_indices, ])
  Y.train_full <- as.matrix(Y[-test_indices, ])
  
  n_train <- nrow(X.train_full)
  
  # Determine batch size based on batch_type
  if (batch_type == "full") {
    batch_size <- n_train  # Use all training data
  } else if (batch_type == "stochastic") {
    batch_size <- 1  # Use single sample
  } else { # mini
    # If batch_size is NULL, use a default of approximately 10% of data
    if (is.null(batch_size)) {
      batch_size <- max(10, round(n_train * 0.1))
    }
  }
  
  for (epoch in 1:epochs) {
    # Shuffle training data for each epoch
    shuffle_indices <- sample(1:n_train)
    X.train_shuffled <- as.matrix(X.train_full[shuffle_indices, ])
    Y.train_shuffled <- as.matrix(Y.train_full[shuffle_indices, ])
    
    epoch_cost <- 0
    n_batches <- ceiling(n_train / batch_size)
    
    # Process each batch
    for (batch in 1:n_batches) {
      # Get batch data
      start_idx <- (batch - 1) * batch_size + 1
      end_idx <- min(batch * batch_size, n_train)
      
      X.batch <- as.matrix(X.train_shuffled[start_idx:end_idx, , drop = FALSE])
      Y.batch <- as.matrix(Y.train_shuffled[start_idx:end_idx, , drop = FALSE])
      
      # Forward propagation
      forward_result <- forward_propagation(NN, X.batch)
      y.fw <- forward_result$y.fw
      z.fw <- forward_result$z.fw
      output <- y.fw[[length(y.fw)]]
      
      batch_cost <- cost(y.predict = output, y.true = Y.batch)
      epoch_cost <- epoch_cost + (batch_cost * nrow(X.batch) / n_train)
      
      # Back-propagation
      gradients <- back_propagation(NN, z.fw, y.fw, Y.batch)
      
      # Update weights
      for (l in 1:length(NN$weights)) {
        weight.delta <- learning_rate * gradients[[l]] + 
          momentum * (NN$weights[[l]] - old_weight[[l]])
        
        old_weight[[l]] <- NN$weights[[l]]
        NN$weights[[l]] <- NN$weights[[l]] - weight.delta  
      }
    }
    
    # Store cost for this epoch
    cost_history[epoch] <- epoch_cost
    
    # Evaluate on test set
    test_output <- NN.predict(NN, X.test)
    test_cost <- cost(y.predict = test_output, y.true = Y.test)
    test_cost_history[epoch] <- test_cost
    
    # Save best model based on test performance
    if (test_cost < min_test_cost) {
      min_test_cost <- test_cost
      best_weights <- NN$weights
    }
    
    # Store every 10th epoch to save memory
    if (epoch %% 10 == 0) { 
      train_output <- NN.predict(NN, X.train_full)
      train_predictions[[length(train_predictions) + 1]] <- train_output
      train_true_values[[length(train_true_values) + 1]] <- Y.train_full
      X_train_stored[[length(X_train_stored) + 1]] <- X.train_full
      
      test_predictions[[length(test_predictions) + 1]] <- test_output
      test_true_values[[length(test_true_values) + 1]] <- Y.test
      X_test_stored[[length(X_test_stored) + 1]] <- X.test  
    }
    
    if (verbose && epoch %% 100 == 0) {
      
      cat(sprintf("Epoch %d | Train Cost: %.6f | Test Cost: %.6f\n",
                  epoch, epoch_cost, test_cost))
      
      df <- data.frame(
        X = as.numeric(X.test),
        Predictions = as.numeric(test_output),
        True_Values = as.numeric(Y.test)
      )
      df <- df[order(df$X), ]
      
      # Real-time plot
      p <- ggplot(df, aes(x = X)) +
        geom_line(aes(y = True_Values, color = "True Values"), size = 1) +
        # *** Solid blue line for predictions ***
        geom_line(aes(y = Predictions, color = "Predictions"), 
                  size = 1, linetype = "solid") +
        scale_color_manual(values = c("True Values" = "skyblue", "Predictions" = "tomato")) +
        labs(title = paste0("Epoch ", epoch, " | Cost: ", signif(epoch_cost, 4)),
             x = "X", y = "Y") +
        theme_minimal() +
        theme(legend.title = element_blank())
      
      print(p)
    }
  }
  
  # Restore best weights
  NN$weights <- best_weights
  
  return(list(
    NN = NN,
    cost_history = cost_history,
    test_cost_history = test_cost_history,
    train_predictions = train_predictions,
    test_predictions = test_predictions,
    train_true_values = train_true_values,
    test_true_values = test_true_values,
    X_train = X_train_stored,           
    X_test = X_test_stored, 
    best_test_cost = min_test_cost
  ))
}
NN.predict <- function(NN, X, verbose = FALSE) {
  
  y <- as.matrix(X)
  
  for (l in 1:length(NN$weights)) {
    y.p <- cbind(-1, y)
    z.l <- y.p %*% NN$weights[[l]]
    y <- activation.fn(z.l)
  }
  if(verbose){
    df <- data.frame(
      X = as.numeric(X),
      Predictions = as.numeric(y)
    )
    df <- df[order(df$X), ]
    
    p <- ggplot(df, aes(x = X)) +
      geom_line(aes(y = Predictions, color = "Predictions"), 
                size = 1, linetype = "solid") +
      scale_color_manual(values = c("Predictions" = "tomato")) +
      labs(title = "Prediction",
           x = "X", y = "Y") +
      theme_minimal() +
      theme(legend.title = element_blank())
    print(p)
  }

  
  return(y)
}


NN.predict.periodic <- function(NN, X.predict){
  X <- (X.predict %% 2)  
  Y <- NN.predict(NN, X)

  df <- data.frame(
    X = as.numeric(X.predict),
    Predictions = as.numeric(Y)
  )
  df <- df[order(df$X), ]
  
  # Real-time plot
  # Real-time plot
  p <- ggplot(df, aes(x = X)) +

    geom_line(aes(y = Predictions, color = "Predictions"), 
              size = 1, linetype = "solid") +
    scale_color_manual(values = c("Predictions" = "tomato")) +
    labs(title = "Prediction",
         x = "X", y = "Y") +
    theme_minimal() +
    theme(legend.title = element_blank())
  
  print(p)
  return(Y)
}