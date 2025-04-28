import matplotlib.pyplot as plt
import numpy as np

# Updated methods list and nRMSE data
methods = ['Single WPD-LSTM', 'Multi WPD-LSTM', 'LSTM', 'Multi WPD-LSTM WWO', 'Persistence', 'Linear Regression']
nrmse_data = {
    '15 min ahead': [3.71, 2.67, 7.86, 3.12, 7.73, None],
    'hour ahead': [8.81, 7.31, 12.16, 11.31, 8.02, None],
    'day ahead': [12.93, 13.09, 13.98, 13.50, 16.00, 14.65]
}

# Function to sort and filter data
def sort_and_filter_data(data):
    return sorted([(method, value) for method, value in zip(methods, data) if value is not None], key=lambda x: x[1])

# Define a function to plot bar charts
def plot_bar_chart(data, title):
    fig, ax = plt.subplots(figsize=(12, 10))
    horizons = list(data.keys())
    colors = ['#1f77b4', '#d62728', '#2ca02c']  # Blue, Red, Green
    
    y_offset = 0
    
    for i, horizon in enumerate(horizons):
        sorted_data = sort_and_filter_data(data[horizon])
        methods_sorted, values_sorted = zip(*sorted_data)
        
        y_pos = np.arange(len(methods_sorted)) + y_offset
        bars = ax.barh(y_pos, values_sorted, align='center', height=0.7, label=horizon, color=colors[i])
        
        # Add text labels
        for bar, method in zip(bars, methods_sorted):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                    ha='left', va='center')
            ax.text(0, bar.get_y() + bar.get_height()/2, method, 
                    ha='right', va='center', fontsize=12, transform=ax.get_yaxis_transform())
        
        y_offset += len(methods_sorted) + 1
    
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_xlabel('nRMSE (%)')
    ax.set_title(title)
    ax.legend(loc='lower right', bbox_to_anchor=(1, -0.15), ncol=len(horizons))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.3)  # Adjust bottom and left margins
    plt.show()

# Plot nRMSE
plot_bar_chart(nrmse_data, 'nRMSE for Different Prediction Horizons (Lowest to Highest)')