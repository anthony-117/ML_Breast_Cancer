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
se_cost <- function(y.predict, y.true){
  return(mean((y.predict - y.true)^2))
}

mse_derivative <- function(y.predict, y.true) {
  return(-(y.true - y.predict))
}

cross_entropy_cost <- function(y.predict, y.true) {
  # Add small epsilon to avoid log(0)
  epsilon <- 1e-15
  y.predict <- pmax(pmin(y.predict, 1 - epsilon), epsilon)
  return(-mean(y.true * log(y.predict) + (1 - y.true) * log(1 - y.predict)))
}

cross_entropy_derivative <- function(y.predict, y.true) {
  # Add small epsilon to avoid division by zero
  epsilon <- 1e-15
  y.predict <- pmax(pmin(y.predict, 1 - epsilon), epsilon)
  return(-(y.true / y.predict - (1 - y.true) / (1 - y.predict)))
}

mae_cost <- function(y.predict, y.true) {
  return(mean(abs(y.predict - y.true)))
}

mae_derivative <- function(y.predict, y.true) {
  return(sign(y.predict - y.true))
}

huber_cost <- function(y.predict, y.true, delta = 1.0) {
  error <- y.predict - y.true
  return(mean(ifelse(abs(error) <= delta, 
                     0.5 * error^2, 
                     delta * (abs(error) - 0.5 * delta))))
}

huber_derivative <- function(y.predict, y.true, delta = 1.0) {
  error <- y.predict - y.true
  return(ifelse(abs(error) <= delta, error, delta * sign(error)))
}


cost <- function(y.predict, y.true){
  return(mse_cost(y.predict, y.true))
}
cost.derivative <- function(y.predict, y.true){
  return(mse_derivative(y.predict, y.true))
}

weights.initialize <- function(layers, method = "xavier") {
  weights = list()
  
  for (l in 1:(length(layers) - 1)) {
    input_size <- layers[l]
    output_size <- layers[l+1]
    
    # Choose initialization method
    if (method == "xavier" || method == "glorot") {
      # Xavier/Glorot initialization - good for sigmoid/tanh
      limit <- sqrt(6 / (input_size + output_size))
      weights[[l]] <- matrix(
        runif((input_size + 1) * output_size, min = -limit, max = limit),
        nrow = input_size + 1,
        ncol = output_size
      )
    } else if (method == "he") {
      # He initialization - better for ReLU
      std <- sqrt(2 / input_size)
      weights[[l]] <- matrix(
        rnorm((input_size + 1) * output_size, mean = 0, sd = std),
        nrow = input_size + 1,
        ncol = output_size
      )
    } else if (method == "lecun") {
      # LeCun initialization
      std <- sqrt(1 / input_size)
      weights[[l]] <- matrix(
        rnorm((input_size + 1) * output_size, mean = 0, sd = std),
        nrow = input_size + 1,
        ncol = output_size
      )
    } else {
      # Default uniform initialization
      weights[[l]] <- matrix(
        runif((input_size + 1) * output_size, min = -0.5, max = 0.5),
        nrow = input_size + 1,
        ncol = output_size
      )
    }
    
    # Set bias values (can be initialized differently)
    weights[[l]][1,] <- 0  # Initialize biases to zero
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
NN.train <- function(NN, X, Y, epochs, learning_rate = 0.01, 
                     optimizer = "sgd", optimizer_params = list(),
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
  
  # Select the appropriate optimizer functions
  optimizer_functions <- get_optimizer_functions(optimizer)
  
  # Initialize optimizer state
  optimizer_state <- optimizer_functions$initialize(NN, optimizer_params)
  
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
      
      # Update weights using the selected optimizer's update function
      result <- optimizer_functions$update(NN, gradients, optimizer_state, learning_rate)
      NN <- result$NN
      optimizer_state <- result$state
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
        geom_line(aes(y = Predictions, color = "Predictions"), 
                  size = 1, linetype = "solid") +
        scale_color_manual(values = c("True Values" = "skyblue", "Predictions" = "tomato")) +
        labs(title = paste0("Epoch ", epoch, " | Cost: ", signif(epoch_cost, 4),
                            " | Optimizer: ", optimizer),
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
    best_test_cost = min_test_cost,
    optimizer = optimizer,
    optimizer_state = optimizer_state
  ))
}

# Function to select the appropriate optimizer functions
get_optimizer_functions <- function(optimizer) {
  optimizers <- list(
    sgd = list(
      initialize = initialize_sgd,
      update = update_sgd
    ),
    adam = list(
      initialize = initialize_adam,
      update = update_adam
    ),
    rmsprop = list(
      initialize = initialize_rmsprop,
      update = update_rmsprop
    ),
    adagrad = list(
      initialize = initialize_adagrad,
      update = update_adagrad
    )
  )
  
  if (!optimizer %in% names(optimizers)) {
    stop(paste("Unsupported optimizer:", optimizer, 
               "Available optimizers:", paste(names(optimizers), collapse=", ")))
  }
  
  return(optimizers[[optimizer]])
}

# SGD Optimizer Functions
initialize_sgd <- function(NN, params = list()) {
  state <- list()
  state$momentum <- if (!is.null(params$momentum)) params$momentum else 0.9
  state$velocity <- vector("list", length(NN$weights))
  
  for (l in 1:length(NN$weights)) {
    state$velocity[[l]] <- matrix(0, nrow = nrow(NN$weights[[l]]), ncol = ncol(NN$weights[[l]]))
  }
  
  return(state)
}

update_sgd <- function(NN, gradients, state, learning_rate) {
  momentum <- state$momentum
  
  for (l in 1:length(NN$weights)) {
    # Calculate velocity
    state$velocity[[l]] <- momentum * state$velocity[[l]] - learning_rate * gradients[[l]]
    # Update weights
    NN$weights[[l]] <- NN$weights[[l]] + state$velocity[[l]]
  }
  
  return(list(NN = NN, state = state))
}

# Adam Optimizer Functions
initialize_adam <- function(NN, params = list()) {
  state <- list()
  state$beta1 <- if (!is.null(params$beta1)) params$beta1 else 0.9
  state$beta2 <- if (!is.null(params$beta2)) params$beta2 else 0.999
  state$epsilon <- if (!is.null(params$epsilon)) params$epsilon else 1e-8
  state$m <- vector("list", length(NN$weights))
  state$v <- vector("list", length(NN$weights))
  state$t <- 0
  
  for (l in 1:length(NN$weights)) {
    state$m[[l]] <- matrix(0, nrow = nrow(NN$weights[[l]]), ncol = ncol(NN$weights[[l]]))
    state$v[[l]] <- matrix(0, nrow = nrow(NN$weights[[l]]), ncol = ncol(NN$weights[[l]]))
  }
  
  return(state)
}

update_adam <- function(NN, gradients, state, learning_rate) {
  state$t <- state$t + 1
  beta1 <- state$beta1
  beta2 <- state$beta2
  epsilon <- state$epsilon
  
  for (l in 1:length(NN$weights)) {
    # Update biased first moment estimate
    state$m[[l]] <- beta1 * state$m[[l]] + (1 - beta1) * gradients[[l]]
    # Update biased second raw moment estimate
    state$v[[l]] <- beta2 * state$v[[l]] + (1 - beta2) * (gradients[[l]]^2)
    
    # Compute bias-corrected first moment estimate
    m_corrected <- state$m[[l]] / (1 - beta1^state$t)
    # Compute bias-corrected second raw moment estimate
    v_corrected <- state$v[[l]] / (1 - beta2^state$t)
    
    # Update weights
    NN$weights[[l]] <- NN$weights[[l]] - learning_rate * m_corrected / (sqrt(v_corrected) + epsilon)
  }
  
  return(list(NN = NN, state = state))
}

# RMSprop Optimizer Functions
initialize_rmsprop <- function(NN, params = list()) {
  state <- list()
  state$decay_rate <- if (!is.null(params$decay_rate)) params$decay_rate else 0.9
  state$epsilon <- if (!is.null(params$epsilon)) params$epsilon else 1e-8
  state$cache <- vector("list", length(NN$weights))
  
  for (l in 1:length(NN$weights)) {
    state$cache[[l]] <- matrix(0, nrow = nrow(NN$weights[[l]]), ncol = ncol(NN$weights[[l]]))
  }
  
  return(state)
}

update_rmsprop <- function(NN, gradients, state, learning_rate) {
  decay_rate <- state$decay_rate
  epsilon <- state$epsilon
  
  for (l in 1:length(NN$weights)) {
    # Update cache
    state$cache[[l]] <- decay_rate * state$cache[[l]] + (1 - decay_rate) * (gradients[[l]]^2)
    # Update weights
    NN$weights[[l]] <- NN$weights[[l]] - learning_rate * gradients[[l]] / (sqrt(state$cache[[l]]) + epsilon)
  }
  
  return(list(NN = NN, state = state))
}

# Adagrad Optimizer Functions
initialize_adagrad <- function(NN, params = list()) {
  state <- list()
  state$epsilon <- if (!is.null(params$epsilon)) params$epsilon else 1e-8
  state$cache <- vector("list", length(NN$weights))
  
  for (l in 1:length(NN$weights)) {
    state$cache[[l]] <- matrix(0, nrow = nrow(NN$weights[[l]]), ncol = ncol(NN$weights[[l]]))
  }
  
  return(state)
}

update_adagrad <- function(NN, gradients, state, learning_rate) {
  epsilon <- state$epsilon
  
  for (l in 1:length(NN$weights)) {
    # Update cache (sum of squares of gradients)
    state$cache[[l]] <- state$cache[[l]] + gradients[[l]]^2
    # Update weights
    NN$weights[[l]] <- NN$weights[[l]] - learning_rate * gradients[[l]] / (sqrt(state$cache[[l]]) + epsilon)
  }
  
  return(list(NN = NN, state = state))
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
