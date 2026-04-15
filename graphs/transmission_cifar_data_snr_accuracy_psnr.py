import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('transmission_cifar_data_260415.csv')

# Group by model_snr and compute averages
grouped = df.groupby('model_snr').agg({
    'accuracy': 'mean',
    'psnr': 'mean'
}).reset_index()

print("Averages for each model_snr:")
print(grouped)

# Plot the graphs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy vs model_snr
ax1.plot(grouped['model_snr'], grouped['accuracy'], marker='o', linestyle='-')
ax1.set_xlabel('Model SNR')
ax1.set_ylabel('Average Accuracy')
ax1.set_title('Accuracy vs Model SNR')
ax1.grid(True)

# PSNR vs model_snr
ax2.plot(grouped['model_snr'], grouped['psnr'], marker='o', linestyle='-', color='orange')
ax2.set_xlabel('Model SNR')
ax2.set_ylabel('Average PSNR (dB)')
ax2.set_title('PSNR vs Model SNR')
ax2.grid(True)

plt.tight_layout()
plt.savefig('snr_accuracy_psnr_plot.png')
plt.show()