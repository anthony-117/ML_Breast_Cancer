source("preprocess.R")
source("NN.R")
source("plots.R")

library(tidyverse)

dataset <- read.csv("preprocess_bank_note_forgery.csv")

X <- dataset %>% 
  select(-Class)

Y <- dataset %>% 
  select(Class)
nn <- NN.create(X, Y, c(10))

epochs <- 2000

result <- NN.train(nn, X, Y, epochs=epochs, learning_rate=0.01, optimizer_params = list(momentum = 0.9),
                   batch_type="mini", batch_size = 64, optimizer = "rmsprop", verbose=TRUE)

NN.trained <- result$NN


#-----------------Plots.R-----------------#

plot_cost_history(result, show_test = TRUE)
# 
# # Plot predictions vs true values for the test set
# plot_predictions(result, dataset = "train")
# 
# # Visualize training progression every 100 epochs
# plot_training_progression(result, interval = 100)
#-----------------Plots.R-----------------#
