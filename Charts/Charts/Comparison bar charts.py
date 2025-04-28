import matplotlib.pyplot as plt
import numpy as np

# Updated methods list and nRMSE data
methods = ['Single WPD-LSTM', 'Multi WPD-LSTM', 'LSTM', 'Multi WPD-LSTM without weight optimization',
           'Persistence']
nrmse_data = {
    '15 min': [3.71, 2.67, 8.51, 3.07, 9.7],
    'hour ahead': [8.88, 7.31, 11.44, 8.02, 15.52],
    'day ahead': [12.93, 13.09, 13.98, 13.5, 19.81]
}

# Define colors for each method
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


# Define a function to plot bar charts
def plot_bar_chart(data, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    horizons = list(data.keys())
    x = np.arange(len(horizons))
    width = 0.15  # Width of each bar

    for horizon in horizons:
        # Sort methods by nRMSE values for each horizon
        sorted_indices = sorted(range(len(methods)),
                                key=lambda k: data[horizon][k] if data[horizon][k] is not None else float('inf'))

        for i, idx in enumerate(sorted_indices):
            value = data[horizon][idx]
            if value is not None:
                bar = ax.bar(x[horizons.index(horizon)] + i * width, value, width,
                             label=methods[idx] if horizon == horizons[0] else "", color=colors[idx])

                # Add nRMSE value on top of each bar
                ax.text(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height(),
                        f'{value:.2f}', ha='center', va='bottom', rotation=0, fontsize=8)

    ax.set_ylabel('nRMSE (%)', fontsize=20)
    ax.set_xlabel('Prediction Horizon', fontsize=20)
    # ax.set_title(title)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(horizons)
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()


# Plot nRMSE
plot_bar_chart(nrmse_data, 'nRMSE for Different Prediction Horizons')
