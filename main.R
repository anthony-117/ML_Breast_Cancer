source("preprocess.R")
source("NN.R")

NN <- NN(X = X, Y.d = Y, 
            hidden_layers = c(10, 5, 3), 
            learning_rate = 0.01, 
            momentum = 0.9, 
            epochs = 1000)

NN.trained <- NN.train(my_nn, verbose = TRUE)

# Plot training progress
plot(1:length(trained_nn$cost_history), trained_nn$cost_history, 
     type = "l", xlab = "Epoch", ylab = "Cost", 
     main = "Neural Network Training Progress")

# Make predictions
predictions <- NN.predict(trained_nn, new_data = test_data)
