import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Simulated data for illustration purposes


data = pd.read_csv('data/Comparison 1 hour ahead.csv')

# Create some example data (adjust these as per your dataset)

# Create the plot
plt.figure(figsize=(10, 6))

# Plotting the lines
plt.plot(data['DateTime'], data['Actual'], label="Actual", color="blue", linewidth=2,)
plt.plot(data['DateTime'], data['Single WPD-LSTM'], label="Single WPD-LSTM", color="red", linewidth=2, linestyle="dashed")
plt.plot(data['DateTime'], data['Multi WPD-LSTM'], label="Multi WPD-LSTM", color="orange", linewidth=2, marker="o")
#plt.plot(data['DateTime'], data['Single EMD'], label="Single EMD-LSTM", color="orange", linewidth=1)
plt.plot(data['DateTime'], data['LSTM'], label="LSTM", color="green", linewidth=2, linestyle="dotted")
#plt.plot(data['DateTime'], data['Linear Regression'], label="Linear Regression", color="lightblue", linewidth=2, marker="^")
#plt.plot(data['DateTime'], data['Persistence'], label="Persistence", color="purple", linewidth=2)
#plt.plot(data['DateTime'], data['Multi EMD'], label="Multi EMD", color="orange", linewidth=1)

# Add labels and title
plt.xlabel('Time Index', fontsize=16)
plt.ylabel('PV Power (W)', fontsize=16)
#plt.title('Forecasting Results for Specific Period')

# Rotate x-axis labels to avoid overlap
plt.xticks(data['DateTime'][::8], rotation=45, fontsize=12)
plt.yticks(fontsize=12)
# Add grid
plt.grid(True, alpha=0.7)

# Add legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()


# Calculate errors
data['Error_Single_WPD-LSTM'] = data['Single WPD-LSTM'] - data['Actual']
data['Error_Multi_WPD-LSTM'] = data['Multi WPD-LSTM'] - data['Actual']
data['Error_Single_EMD-LSTM'] = data['Single EMD'] - data['Actual']
data['Error_LSTM'] = data['LSTM'] - data['Actual']
data['Error_lr'] = data['Linear Regression'] - data['Actual']

# Plotting
plt.figure(figsize=(10, 6))

# Scatter plot for each error
plt.scatter(data['DateTime'], data['Error_Single_WPD-LSTM'], label='Single WPD-LSTM', color='blue', s=10)
plt.scatter(data['DateTime'], data['Error_Multi_WPD-LSTM'], label='Multi WPD-LSTM', color='red', s=10)
plt.scatter(data['DateTime'], data['Error_Single_EMD-LSTM'], label='Single EMD-LSTM', color='orange', s=10)
plt.scatter(data['DateTime'], data['Error_LSTM'], label='LSTM', color='green', s=10)
plt.scatter(data['DateTime'], data['Error_lr'], label='Linear Regression', color='lightblue', s=10)

# Add title and labels
#plt.title('Forecasting Errors', fontsize=14)
plt.xlabel('Time Index', fontsize=16)
plt.ylabel('Error (W)', fontsize=16)

# Add grid
plt.grid(True, alpha=0.7)

# Add legend
plt.legend()

# Rotate x-axis labels
plt.xticks(data['DateTime'][::8], rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()




