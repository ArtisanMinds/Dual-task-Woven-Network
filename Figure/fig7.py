import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1.5, 1]})

models_bar = ['DT', 'RF', 'XGBoost', 'LightGBM', 'LSTM', 'GST', 'LST']
accuracy_few_samples = [88.13, 90.13, 90.44, 89.67, 85.19, 83.92, 84.73]
accuracy_normal = [89.57, 91.31, 91.42, 90.73, 89.68, 87.10, 88.33]
baseline = 91.75
bar_width = 0.306
r1 = np.arange(len(models_bar))
r2 = [x + bar_width for x in r1]

ax1.barh(r2, accuracy_few_samples, color='#587558', height=bar_width, alpha=0.7, label='Sub-Sampling')
ax1.barh(r1, accuracy_normal, color='#fdd000', height=bar_width, alpha=0.7, label='Normal')

ax1.axvline(x=baseline, color='b', linestyle='--', label=f'Baseline')
ax1.set_xlim(83, 92)

ax1.set_xlabel('Accuracy (%)', fontsize=22, fontname='Times New Roman')
ax1.set_yticks([r + bar_width / 2 for r in range(len(models_bar))])
ax1.set_yticklabels(models_bar, fontname='Times New Roman')
ax1.legend(loc='upper right', bbox_to_anchor=(0.95, 0.98), prop={'family': 'Times New Roman', 'size': 18})
ax1.tick_params(axis='both', which='major', labelsize=18, labelcolor='black')
ax1.text(-1.10, -0.125, '(a)', transform=plt.gca().transAxes, fontsize=20, fontname='Times New Roman', ha='center')


models_radar = ['DWN', 'DT', 'LightGBM', 'XGBoost', 'RF', 'LSTM', 'GST', 'LST']

accuracy = [91.75, 89.57, 90.73, 91.42, 91.31, 89.68, 87.10, 88.33]
noise_0001std = [91.71, 89.34, 90.54, 91.17, 91.04, 89.51, 87.00, 88.15]
noise_00025std = [91.47, 88.49, 89.51, 90.23, 90.12, 88.68, 86.56, 87.80]
noise_0005std = [90.60, 86.26, 88.12, 87.81, 87.69, 86.62, 85.27, 86.15]

num_vars = len(models_radar)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]
accuracy += accuracy[:1]
noise_0001std += noise_0001std[:1]
noise_00025std += noise_00025std[:1]
noise_0005std += noise_0005std[:1]

if 'ax2' in locals():
    ax2.remove()

ax2 = plt.subplot(122, polar=True)
ax2.plot(angles, accuracy, color='#a5d4d9', linewidth=3, label='Standard')
ax2.plot(angles, noise_0001std, color='#fdd000', linewidth=3, label='1% Std Noise')
ax2.plot(angles, noise_00025std, color='#ee9e7d', linewidth=3, label='2.5% Std Noise')
ax2.plot(angles, noise_0005std, color='#94c36c', linewidth=3, label='5% Std Noise')
ax2.scatter(angles, accuracy, color='#a5d4d9', s=40, zorder=5)
ax2.scatter(angles, noise_0001std, color='#fdd000', s=40, zorder=5)
ax2.scatter(angles, noise_00025std, color='#ee9e7d', s=40, zorder=5)
ax2.scatter(angles, noise_0005std, color='#94c36c', s=40, zorder=5)

ax2.set_ylim(85, 92)

ax2.set_yticks([85, 87, 89, 91, 92])
ax2.set_yticklabels(['85', '87', '89', '91', '92'], fontname='Times New Roman', fontsize=18, color='black')

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(models_radar, fontname='Times New Roman', fontsize=18, color='black', ha='center')

ax2.legend(loc='upper left', bbox_to_anchor=(0.7, 0.05), prop={'family': 'Times New Roman', 'size': 17})

ax2.tick_params(axis='both', which='major', labelsize=18, labelcolor='black', pad=15)

ax2.text(0.5, -0.23, '(b)', transform=plt.gca().transAxes, fontsize=20, fontname='Times New Roman', ha='center')

ax2.set_facecolor('#e5ecf6')

ax2.spines['polar'].set_color('white')
ax2.spines['polar'].set_linewidth(2)

ax2.grid(color='white', linestyle='-', linewidth=2)


plt.tight_layout()
plt.savefig('fig7_Perf.png')
plt.show()