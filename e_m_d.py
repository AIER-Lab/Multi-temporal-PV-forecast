import matplotlib.pyplot as plt
import numpy as np
import emd
import pandas as pd

csv_file = "data\Dataset without WPD.csv"
df = pd.read_csv(csv_file, parse_dates=['DateTime'], index_col='DateTime')
p = df['P'].to_numpy()
ghi = df['GHI'].to_numpy()
# Visualise the time-series for analysis
plt.figure(figsize=(12, 4))
plt.plot(p)
plt.show()
plt.figure(figsize=(12, 4))
plt.plot(ghi)
plt.show()

p_imf = emd.sift.sift(p)
ghi_imf = emd.sift.sift(ghi)
print(p_imf.shape)
print(ghi_imf.shape)

emd.plotting.plot_imfs(p_imf)
plt.show()
emd.plotting.plot_imfs(ghi_imf)
plt.show()

p_imf_df = pd.DataFrame(p_imf, columns=['SS1', 'SS2', 'SS3', 'SS4', 'SS5', 'SS6', 'SS7', 'SS8', 'SS9', 'SS10', 'SS11'])
ghi_imf_df = pd.DataFrame(p_imf, columns=['GHI1', 'GHI2', 'GHI3', 'GHI4', 'GHI5', 'GHI6', 'GHI7', 'GHI8', 'GHI9',
                                          'GHI10', 'GHI11'])
df = df.reset_index()
df_combined = pd.concat([df, p_imf_df, ghi_imf_df], axis=1)
df_combined.to_csv('data/dataset_with_emd.csv', index=False)
col = df_combined.pop("GHI")
df_combined.insert(7, "GHI_", col)
col = df_combined.pop("P")
df_combined.insert(30, "P_input", col)
df_combined["P"] = df_combined["P_input"]
df_combined = df_combined.rename(columns={"GHI_": "GHI"})
df_combined = df_combined.set_index(df["DateTime"])
df_combined = df_combined.drop(columns="DateTime")
df_combined.to_csv("data/Dataset_with_EMD.csv")
print('done')
