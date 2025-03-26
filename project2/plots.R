# Function 1: Plot cost history
plot_cost_history <- function(train_result, show_test = TRUE, log_scale = FALSE) {
  epochs <- length(train_result$cost_history)
  df <- data.frame(
    Epoch = 1:epochs,
    TrainCost = train_result$cost_history
  )
  
  p <- ggplot(df, aes(x = Epoch, y = TrainCost)) +
    geom_line(color = "tomato", size = 1) +
    labs(
      title = paste("Neural Network Training Cost History |", 
                    "Optimizer:", train_result$optimizer),
      x = "Epoch",
      y = "Cost"
    ) +
    theme_minimal()
  
  # Add test cost if requested
  if (show_test && !is.null(train_result$test_cost_history)) {
    df$TestCost <- train_result$test_cost_history
    p <- ggplot(df, aes(x = Epoch)) +
      geom_line(aes(y = TrainCost, color = "Training"), size = 1) +
      geom_line(aes(y = TestCost, color = "Testing"), size = 1) +
      scale_color_manual(values = c("Training" = "tomato", "Testing" = "skyblue")) +
      labs(
        title = paste("Neural Network Training Cost History |", 
                      "Optimizer:", train_result$optimizer),
        x = "Epoch",
        y = "Cost"
      ) +
      theme_minimal() +
      theme(legend.title = element_blank())
  }
  
  # Apply log scale if requested
  if (log_scale) {
    p <- p + scale_y_log10() +
      labs(y = "Cost (log scale)")
  }
  
  return(p)
}

# Function 2: Plot predictions vs true values for a specific epoch
plot_predictions <- function(train_result, dataset = "test", epoch_idx = NULL) {
  # Determine if we're plotting test or training data
  predictions <- if (dataset == "test") train_result$test_predictions else train_result$train_predictions
  true_values <- if (dataset == "test") train_result$test_true_values else train_result$train_true_values
  X_values <- if (dataset == "test") train_result$X_test else train_result$X_train
  
  # Check if data is available
  if (length(predictions) == 0) {
    stop("No prediction data available. Make sure that predictions were stored during training.")
  }
  
  # If epoch_idx not specified, use the last available epoch
  if (is.null(epoch_idx)) {
    epoch_idx <- length(predictions)
  } else {
    # Convert from actual epoch number to index in the stored data
    # Since we only store every 10th epoch
    epoch_idx <- min(ceiling(epoch_idx / 10), length(predictions))
  }
  
  # Extract data
  X <- X_values[[epoch_idx]]
  pred <- predictions[[epoch_idx]]
  true <- true_values[[epoch_idx]]
  
  # For 1D inputs, create a data frame for plotting
  if (ncol(X) == 1) {
    df <- data.frame(
      X = as.numeric(X),
      Predictions = as.numeric(pred),
      True_Values = as.numeric(true)
    )
    
    # Order by X for a cleaner line plot
    df <- df[order(df$X), ]
    
    # Create plot
    p <- ggplot(df, aes(x = X)) +
      geom_line(aes(y = True_Values, color = "True Values"), size = 1) +
      geom_line(aes(y = Predictions, color = "Predictions"), size = 1) +
      scale_color_manual(values = c("True Values" = "skyblue", "Predictions" = "tomato")) +
      labs(
        title = paste0("Neural Network Predictions | ", 
                       toupper(substr(dataset, 1, 1)), substr(dataset, 2, nchar(dataset)),
                       " Set | Epoch ", epoch_idx * 10),
        x = "X",
        y = "Y"
      ) +
      theme_minimal() +
      theme(legend.title = element_blank())
  } else {
    # For higher dimensional inputs, create a scatter plot of predicted vs actual values
    df <- data.frame(
      Predicted = as.numeric(pred),
      Actual = as.numeric(true)
    )
    
    # Add perfect prediction line
    min_val <- min(df$Predicted, df$Actual)
    max_val <- max(df$Predicted, df$Actual)
    
    p <- ggplot(df, aes(x = Actual, y = Predicted)) +
      geom_point(alpha = 0.5, color = "tomato") +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
      labs(
        title = paste0("Neural Network Predictions vs Actual Values | ", 
                       toupper(substr(dataset, 1, 1)), substr(dataset, 2, nchar(dataset)),
                       " Set | Epoch ", epoch_idx * 10),
        x = "Actual Values",
        y = "Predicted Values"
      ) +
      theme_minimal() +
      xlim(min_val, max_val) +
      ylim(min_val, max_val)
  }
  
  return(p)
}

# Function 3: Create an animation of predictions over epochs
plot_training_progression <- function(train_result, dataset = "test", interval = 100, animate = TRUE) {
  # Determine if we're plotting test or training data
  predictions <- if (dataset == "test") train_result$test_predictions else train_result$train_predictions
  true_values <- if (dataset == "test") train_result$test_true_values else train_result$train_true_values
  X_values <- if (dataset == "test") train_result$X_test else train_result$X_train
  
  # Check if data is available
  if (length(predictions) == 0) {
    stop("No prediction data available. Make sure that predictions were stored during training.")
  }
  
  # Create plots for each stored epoch
  plots <- list()
  
  for (i in 1:length(predictions)) {
    actual_epoch <- i * 10  # Since we store every 10th epoch
    
    # Skip if not at the specified interval
    if (actual_epoch %% interval != 0 && i != length(predictions)) {
      next
    }
    
    # Extract data
    X <- X_values[[i]]
    pred <- predictions[[i]]
    true <- true_values[[i]]
    
    # For 1D inputs, create a line plot
    if (ncol(X) == 1) {
      df <- data.frame(
        X = as.numeric(X),
        Predictions = as.numeric(pred),
        True_Values = as.numeric(true),
        Epoch = actual_epoch
      )
      
      # Order by X for a cleaner line plot
      df <- df[order(df$X), ]
      
      # Create plot
      p <- ggplot(df, aes(x = X)) +
        geom_line(aes(y = True_Values, color = "True Values"), size = 1) +
        geom_line(aes(y = Predictions, color = "Predictions"), size = 1) +
        scale_color_manual(values = c("True Values" = "skyblue", "Predictions" = "tomato")) +
        labs(
          title = paste0("Neural Network Training Progress | Epoch ", actual_epoch),
          x = "X",
          y = "Y"
        ) +
        theme_minimal() +
        theme(legend.title = element_blank())
    } else {
      # For higher dimensional inputs, create a scatter plot
      df <- data.frame(
        Predicted = as.numeric(pred),
        Actual = as.numeric(true),
        Epoch = actual_epoch
      )
      
      # Add perfect prediction line
      min_val <- min(df$Predicted, df$Actual)
      max_val <- max(df$Predicted, df$Actual)
      
      p <- ggplot(df, aes(x = Actual, y = Predicted)) +
        geom_point(alpha = 0.5, color = "tomato") +
        geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
        labs(
          title = paste0("Neural Network Predictions vs Actual Values | Epoch ", actual_epoch),
          x = "Actual Values",
          y = "Predicted Values"
        ) +
        theme_minimal() +
        xlim(min_val, max_val) +
        ylim(min_val, max_val)
    }
    
    plots[[length(plots) + 1]] <- p
  }
  
  # If animate is TRUE and gganimate is available, create an animation
  if (animate && requireNamespace("gganimate", quietly = TRUE)) {
    # For 1D inputs, animate progression
    if (ncol(X_values[[1]]) == 1) {
      # Combine all data frames
      all_epochs <- data.frame()
      
      for (i in 1:length(predictions)) {
        actual_epoch <- i * 10
        X <- X_values[[i]]
        pred <- predictions[[i]]
        true <- true_values[[i]]
        
        df <- data.frame(
          X = as.numeric(X),
          Predictions = as.numeric(pred),
          True_Values = as.numeric(true),
          Epoch = actual_epoch
        )
        df <- df[order(df$X), ]
        all_epochs <- rbind(all_epochs, df)
      }
      
      # Create animation
      p <- ggplot(all_epochs, aes(x = X)) +
        geom_line(aes(y = True_Values, color = "True Values"), size = 1) +
        geom_line(aes(y = Predictions, color = "Predictions"), size = 1) +
        scale_color_manual(values = c("True Values" = "skyblue", "Predictions" = "tomato")) +
        labs(
          title = "Neural Network Training Progress: Epoch {frame_time}",
          x = "X",
          y = "Y"
        ) +
        theme_minimal() +
        theme(legend.title = element_blank()) +
        gganimate::transition_time(Epoch) +
        gganimate::ease_aes('linear')
      
      return(gganimate::animate(p, nframes = length(predictions), fps = 2))
    } else {
      # For higher dimensional data
      all_epochs <- data.frame()
      
      for (i in 1:length(predictions)) {
        actual_epoch <- i * 10
        pred <- predictions[[i]]
        true <- true_values[[i]]
        
        df <- data.frame(
          Predicted = as.numeric(pred),
          Actual = as.numeric(true),
          Epoch = actual_epoch
        )
        all_epochs <- rbind(all_epochs, df)
      }
      
      min_val <- min(all_epochs$Predicted, all_epochs$Actual)
      max_val <- max(all_epochs$Predicted, all_epochs$Actual)
      
      p <- ggplot(all_epochs, aes(x = Actual, y = Predicted)) +
        geom_point(alpha = 0.5, color = "tomato") +
        geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
        labs(
          title = "Neural Network Progress: Epoch {frame_time}",
          x = "Actual Values",
          y = "Predicted Values"
        ) +
        theme_minimal() +
        xlim(min_val, max_val) +
        ylim(min_val, max_val) +
        gganimate::transition_time(Epoch) +
        gganimate::ease_aes('linear')
      
      return(gganimate::animate(p, nframes = length(predictions), fps = 2))
    }
  }
  
  # If not animating or gganimate is not available, return a list of plots
  return(plots)
}
plot_nn_costs <- function(files, 
                          title = "Training and Test Costs Across Neural Network Configurations",
                          output_file = "neural_network_costs.png",
                          width = 12, 
                          height = 7) {
  # Load required libraries
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  
  # Initialize an empty list to store cost histories
  cost_histories <- list()
  
  # Load cost histories from each file
  for (file in files) {
    # Extract the model name (without .Rdata extension)
    model_name <- sub("\\.Rdata$", "", file)
    
    # Load the file
    load(file)
    
    # Assuming the cost history is stored in a variable called 'result'
    if (exists("result")) {
      # Create a data frame with cost history and model name
      cost_df <- data.frame(
        Epoch = 1:length(result$cost_history),
        Train_Cost = result$cost_history,
        Test_Cost = result$test_cost_history,
        Model = model_name
      )
      
      cost_histories[[model_name]] <- cost_df
    }
  }
  
  # Combine all cost histories
  combined_costs <- do.call(rbind, cost_histories)
  
  # Find minimum points for each model and cost type
  min_points <- combined_costs %>%
    group_by(Model) %>%
    summarise(
      Train_Min_Cost = min(Train_Cost),
      Train_Min_Epoch = Epoch[which.min(Train_Cost)],
      Test_Min_Cost = min(Test_Cost),
      Test_Min_Epoch = Epoch[which.min(Test_Cost)]
    )
  
  # Plot using ggplot2
  p <- ggplot(combined_costs, aes(x = Epoch, y = Train_Cost, color = Model, linetype = "Train")) +
    geom_line(alpha = 0.7) +
    geom_line(aes(y = Test_Cost, color = Model, linetype = "Test"), alpha = 0.7) +
    
    # Add minimum points for Train Cost
    geom_point(
      data = min_points, 
      aes(x = Train_Min_Epoch, y = Train_Min_Cost, color = Model), 
      shape = 16, 
      size = 3
    ) +
    # Add minimum points for Test Cost
    geom_point(
      data = min_points, 
      aes(x = Test_Min_Epoch, y = Test_Min_Cost, color = Model), 
      shape = 16, 
      size = 3
    ) +
    
    # Add text labels for minimum points
    geom_text(
      data = min_points, 
      aes(x = Train_Min_Epoch, y = Train_Min_Cost, 
          label = sprintf("Train Min: %.4f (Ep. %d)", Train_Min_Cost, Train_Min_Epoch)),
      vjust = -0.5, 
      hjust = 0,
      size = 3
    ) +
    geom_text(
      data = min_points, 
      aes(x = Test_Min_Epoch, y = Test_Min_Cost, 
          label = sprintf("Test Min: %.4f (Ep. %d)", Test_Min_Cost, Test_Min_Epoch)),
      vjust = 1.5, 
      hjust = 0,
      size = 3
    ) +
    
    scale_linetype_manual(values = c("Train" = "solid", "Test" = "dashed")) +
    labs(
      title = title,
      x = "Epochs",
      y = "Cost",
      color = "Model",
      linetype = "Cost Type"
    ) +
    theme_minimal() +
    theme(legend.position = "right")
  
  # Print the plot
  print(p)
  
  # Save the plot
  ggsave(output_file, width = width, height = height, dpi = 300)
  
  # Return the minimum points for reference
  return(min_points)
}