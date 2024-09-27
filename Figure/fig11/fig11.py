import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('result.csv')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

label_abbr = ['  AB', 'CCS', '  CC', '  CD', '  TC', '  TD', '  SC', '  SD', 'CCS', '  CC', '  CD', '  TC', '  TD',
              '  SC', '  SD']
colors = ['black', '#2F4858', '#1F6570', '#258175', '#589B6A', '#9CAF5A', '#EABA5C', '#A38858']

selected_ids = [312, 337, 338]
filtered_data = data[data['id'].isin(selected_ids)]

fig, axs = plt.subplots(nrows=1, ncols=len(selected_ids), figsize=(20, 10))

def plot_intervals(ax, depths, values, color_map, num_classes=8, offset=0):
    for i in range(len(depths) - 1):
        for j in range(num_classes):
            ax.fill_betweenx(
                [depths[i], depths[i + 1]],
                offset + j, offset + j + 1,
                color=color_map[values[i]] if values[i] == j else 'white',
                edgecolor='none'
            )

y_limits = [(8, 0.5), (20, 0.5), (30, 0.5)]
y_ticks = [
    [0.5, 2, 4, 6, 8],
    [0.5, 5, 10, 15, 20],
    [0.5, 7.5, 15, 22.5, 30]
]
text_heights = [0.478, 0.451, 0.445]
subtitles = ['(a)', '(b)', '(c)']

for i, id_value in enumerate(selected_ids):
    id_data = filtered_data[filtered_data['id'] == id_value].sort_values(by='depth')
    depths = id_data['depth'].values
    labels = id_data['label'].values
    predictions = id_data['pred_label'].values

    axs[i].invert_yaxis()

    plot_intervals(axs[i], depths, predictions, colors, offset=7)
    plot_intervals(axs[i], depths, labels, colors)
    axs[i].axvline(x=8, color='black')
    axs[i].set_xlim(0, 15)
    axs[i].set_xticks(range(15))
    axs[i].set_xticklabels(label_abbr, ha='left')
    axs[i].set_ylim(y_limits[i])
    axs[i].set_yticks(y_ticks[i])
    axs[i].grid(True, linestyle='--')
    axs[i].tick_params(axis='y', labelsize=18)
    axs[i].set_xlabel(subtitles[i], fontsize=24)
    axs[i].text(3.5, text_heights[i], 'True Labels', ha='center', fontsize=16, color='black')
    axs[i].text(11.5, text_heights[i], 'Predictions', ha='center', fontsize=16, color='black')

fig.text(0.01, 0.55, 'Depth', va='center', rotation='vertical', fontsize=24)

legend_labels = [
    'Abnormal (AB)', 'Clay-like - Contractive - Sensitive (CCS)', 'Clay-like - Contractive (CC)',
    'Clay-like - Dilative (CD)', 'Transitional - Contractive (TC)', 'Transitional - Dilative (TD)',
    'Sand-like - Contractive (SC)', 'Sand-like - Dilative (SD)'
]
legend_patches = [plt.Line2D([0], [0], color=color, lw=8) for color in colors]

fig.legend(legend_patches, legend_labels, loc='lower center', ncol=4, frameon=False, fontsize=18)

plt.tight_layout(rect=[0.015, 0.085, 1, 1])

plt.savefig('fig11.png', bbox_inches='tight')
plt.show()
