import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from scipy.ndimage import gaussian_filter1d


def y_axis_formatter(x, pos):
    return '{:1.1f}'.format(x / 1e6)


# load data
df1 = pd.read_csv('6.21.csv')
df2 = pd.read_csv('6.13.csv')

# choose data
df1 = df1[(df1['qc'] <= 100) & (df1['qc'] >= 0) & (df1['fs'] <= 1000) & (df1['fs'] >= 0)]
df2 = df2[(df2['qc'] <= 5) & (df2['qc'] >= 0) & (df2['fs'] <= 10) & (df2['fs'] >= 0)]

fig, axes = plt.subplots(2, 2, figsize=(18, 12))


# plot frequency distribution curve and fill
def plot_freq_distribution(ax, data, bins, color):
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    smoothed_counts = gaussian_filter1d(counts, sigma=1.2)

    ax.plot(bin_centers, smoothed_counts, color=color, alpha=0)
    ax.fill_between(bin_centers, smoothed_counts, color=color, alpha=0.8)


plot_freq_distribution(axes[0, 0], df1['qc'], bins=500, color='#587558')
plot_freq_distribution(axes[0, 1], df1['fs'], bins=220, color='#fdd000')
plot_freq_distribution(axes[1, 0], df2['qc'], bins=115, color='#587558')
plot_freq_distribution(axes[1, 1], df2['fs'], bins=85, color='#fdd000')

# add text annotation
annotations = ['(a)', '(b)', '(c)', '(d)']
for i, ax in enumerate(axes.flat):
    ax.text(0.5, -0.195, annotations[i], transform=ax.transAxes, fontsize=22, fontname='Times New Roman', ha='center')

# set axis range
axes[0, 0].set_xlim([0, 60])
axes[0, 1].set_xlim([0, 600])
axes[1, 0].set_xlim([0.05, 5])
axes[1, 1].set_xlim([0.05, 8])
axes[0, 0].set_ylim([0, 2e6])
axes[0, 1].set_ylim([0, 3e6])
axes[1, 0].set_ylim([0, 2e6])
axes[1, 1].set_ylim([0, 3e6])

# set scientific notation
for ax in axes.flat:
    ax.set_xlabel('Value', fontsize=22, fontname='Times New Roman')
    ax.set_ylabel('Frequency', fontsize=22, fontname='Times New Roman')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(y_axis_formatter))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5e6))
    ax.tick_params(axis='both', which='major', labelsize=18, labelcolor='black')
    ax.annotate(r'$\times10^6$', xy=(0.01, 1.05), xycoords='axes fraction', fontsize=20,
                fontname='Times New Roman', ha='left', va='top')

plt.tight_layout()
plt.savefig('fig4.png')
plt.show()
