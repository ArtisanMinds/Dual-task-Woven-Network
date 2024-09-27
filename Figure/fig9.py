import matplotlib.pyplot as plt

models = ['LST', 'GST', 'LSTM', 'LightGBM', 'XGBoost', 'RF', 'DT', 'DWN']
accuracy_valley_train = [81.3, 78.67, 77.14, 87.84, 88.31, 87.21, 84.32, 88.59]
accuracy_basin_train = [87.07, 85.63, 89.1, 87.89, 90.26, 89.52, 87.47, 90.64]

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

ax[0].barh(models, accuracy_valley_train, color='#587558', alpha=0.7)
ax[0].set_xlim(75, 92)
ax[0].invert_xaxis()

ax[1].barh(models, accuracy_basin_train, color='#fdd000', alpha=0.7)
ax[1].set_xlim(75, 92)

for axis in ax:
    axis.set_xlabel('Accuracy (%)', fontsize=22, fontname='Times New Roman')
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['bottom'].set_visible(True)
    axis.tick_params(left=False, labelleft=False, bottom=False)

ax[1].tick_params(left=True, labelleft=False, labelsize=20)
ax[0].tick_params(right=True)
custom_spacing = [-15.95, -14.60, -18.28, -18, -20.2, -18.08, -16.08, -19.7]

for i, model in enumerate(models):
    ax[1].text(accuracy_basin_train[i] + custom_spacing[i], i, model, va='center')

ax[0].text(0.5, -0.22, '(a)', transform=ax[0].transAxes, fontsize=24, fontname='Times New Roman', ha='center')
ax[0].spines['right'].set_visible(True)
ax[1].spines['left'].set_visible(True)
ax[1].text(0.5, -0.22, '(b)', transform=ax[1].transAxes, fontsize=24, fontname='Times New Roman', ha='center')

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig('fig8_Cross.png')
plt.show()
