import matplotlib.pyplot as plt
import numpy as np

models = ['DWN', 'DT', 'RF', 'XGBoost', 'LightGBM', 'LSTM', 'GST', 'LST']
accuracies = [88.55, 83.22, 85.58, 80.11, 85.32, 87.58, 74.17, 78.33]

bar_width = 0.5
positions = np.arange(len(models))

plt.figure(figsize=(12, 6))
plt.bar(positions, accuracies, color='#587558', width=bar_width, alpha=0.7)

plt.ylabel('Accuracy (%)', fontsize=22, fontname='Times New Roman')
plt.ylim(70, 90)
plt.yticks([70, 75, 80, 85, 90], fontsize=18, fontname='Times New Roman')
plt.xticks(positions, models, fontname='Times New Roman')
plt.tick_params(axis='both', which='major', labelsize=18, labelcolor='black')

plt.tight_layout()
plt.savefig('fig9_Extra.png')
plt.show()
