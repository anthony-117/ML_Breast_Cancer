library(tidyverse)
# Set seed for reproducibility
set.seed(123)

# Generate 5000 random X values between -200 and 200 with 3 decimal places
X <- round(runif(6000, min = -100, max = 100), 3)

# Calculate Y values based on the given function
# y = 0.6*sin(πx) + 0.3*sin(3πx) + 0.1*sin(5πx) + 0.05*sin(7πx)
calculate_y <- function(x) {
  0.6 * sin(pi * x) + 
    0.3 * sin(3 * pi * x) + 
    0.1 * sin(5 * pi * x) + 
    0.05 * sin(7 * pi * x)
}

# Apply the function to all X values
Y <- sapply(X, calculate_y)

# Create the dataset
dataset <- data.frame(X = X, Y = Y)

# Display the first few rows
head(dataset)

# Summary statistics
summary(dataset)

# Generate the true Y values for the exact function over a specific X range
X_true <- seq(-10, 10, length.out = 1000)  # Using a smaller range for the true function
Y_true <- calculate_y(X_true)

# Plot only the dataset 
ggplot(dataset, aes(x = X, y = Y)) +
  geom_line(color = "skyblue", size = 1.2) +  # Thicker blue line
  geom_hline(yintercept = 0, color = "black", linetype = "solid", size = 0.8) +  # X-axis
  geom_vline(xintercept = 0, color = "black", linetype = "solid", size = 0.8) +  # Y-axis
  xlim(-10, 10) +
  ylim(-1, 1) +
  labs(title = "Improved Sine Wave Plot", x = "X-Axis", y = "Y-Axis") + 
  theme_minimal()  # Clean theme

# Plot the dataset and the complete function together
ggplot() +
  geom_line(data = dataset, aes(x = X, y = Y, color = "Random Sine Wave"), size = 1.2) +  # Thicker blue line
  geom_line(aes(x = X_true, y = Y_true, color = "True Function"), size = 1.2, linetype = "dashed") +  # True shape in red
  geom_hline(yintercept = 0, color = "black", linetype = "solid", size = 0.8) +  # X-axis
  geom_vline(xintercept = 0, color = "black", linetype = "solid", size = 0.8) +  # Y-axis
  xlim(-10, 10) +
  ylim(-1, 1) +
  labs(title = "Improved Sine Wave Plot with True Shape", x = "X-Axis", y = "Y-Axis") + 
  scale_color_manual(values = c("Random Sine Wave" = "skyblue", "True Function" = "tomato")) +  # Custom colors for legends
  theme_minimal() +  # Clean theme
  theme(legend.title = element_blank(), legend.position = "top")  # Position the legend at the top

# Save the dataset to a CSV file 
write.csv(dataset, "sine_wave_dataset.csv", row.names = FALSE)
