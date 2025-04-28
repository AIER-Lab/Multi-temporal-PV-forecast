import pandas as pd

# Load the dataset
data = pd.read_csv('Dataset 2.csv')

# Convert the "x" column to datetime
data["Time"] = pd.to_datetime(data["x"])
data["Time"] = data["Time"].dt.floor('T')



if data["Time"].duplicated().any():
    print("Duplicate timestamps found. Removing duplicates...")
    data = data.drop_duplicates(subset="Time", keep="first")
print(data.columns)
data[" y"] = data[" y"].interpolate(method="linear")
# Define the desired frequency (e.g., 15 minutes)
desired_frequency = "1min"  # 'T' is deprecated, use 'min'
# Set the "Time" column as the index
data.set_index("Time", inplace=True)

# Resample the data to the desired frequency
df_resampled = data.resample(desired_frequency).asfreq()

# Interpolate missing values
df_resampled[" y"] = df_resampled[" y"].interpolate(method="linear")

desired_frequency_2 = "15min"
df_resampled_2 = data.resample(desired_frequency).asfreq()

# Interpolate missing values
df_resampled_2[" y"] = df_resampled_2[" y"].interpolate(method="linear")

# Print the resampled DataFrame
df_resampled_2.to_csv('resampled.csv')
