import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

# Load the data
df_qc = pd.read_csv('qc_var.csv')
df_fs = pd.read_csv('fs_var.csv')

# Create a figure and four subplots
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Customize the first scatter plot
axes[0, 0].scatter(df_qc['COV']/100, df_qc['SOF'], s=0.5, c='#587558', alpha=0.6)
axes[0, 0].set_xlim(0, 4)
axes[0, 0].set_ylim(0, 8)
axes[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(1))
axes[0, 0].yaxis.set_major_locator(ticker.MultipleLocator(1))
axes[0, 0].set_xlabel("Coefficient of Variation", fontsize=22, fontname='Times New Roman')
axes[0, 0].set_ylabel("Scale of Fluctuation", fontsize=22, fontname='Times New Roman')
axes[0, 0].tick_params(axis='both', which='major', labelsize=18, labelcolor='black')
axes[0, 0].text(0.5, -0.195, '(a)', transform=axes[0, 0].transAxes, fontsize=22, fontname='Times New Roman', ha='center')

# Customize the second scatter plot
axes[0, 1].scatter(df_fs['COV']/100, df_fs['SOF'], s=0.5, c='#fdd000', alpha=0.6)
axes[0, 1].set_xlim(0, 4)
axes[0, 1].set_ylim(0, 8)
axes[0, 1].xaxis.set_major_locator(ticker.MultipleLocator(1))
axes[0, 1].yaxis.set_major_locator(ticker.MultipleLocator(1))
axes[0, 1].set_xlabel("Coefficient of Variation", fontsize=22, fontname='Times New Roman')
axes[0, 1].set_ylabel("Scale of Fluctuation", fontsize=22, fontname='Times New Roman')
axes[0, 1].tick_params(axis='both', which='major', labelsize=18, labelcolor='black')
axes[0, 1].text(0.5, -0.195, '(b)', transform=axes[0, 1].transAxes, fontsize=22, fontname='Times New Roman', ha='center')

# Customize the third scatter plot
axes[1, 0].scatter(df_qc['Smoothness_First_Derivative']/100, df_qc['Smoothness_Second_Derivative']/10000, s=0.5, c='#587558', alpha=0.6)
axes[1, 0].set_xlim(0, 4)
axes[1, 0].set_ylim(0, 4)
axes[1, 0].xaxis.set_major_locator(ticker.MultipleLocator(1))
axes[1, 0].yaxis.set_major_locator(ticker.MultipleLocator(1))
axes[1, 0].set_xlabel("Standard Deviation of First Derivative", fontsize=22, fontname='Times New Roman')
axes[1, 0].set_ylabel("Standard Deviation of Second Derivative", fontsize=22, fontname='Times New Roman')
axes[1, 0].tick_params(axis='both', which='major', labelsize=18, labelcolor='black')
axes[1, 0].annotate(r'$\times10^2$', xy=(0.925, -0.08), xycoords='axes fraction', fontsize=20, fontname='Times New Roman', ha='left', va='top')
axes[1, 0].annotate(r'$\times10^4$', xy=(0.01, 1.05), xycoords='axes fraction', fontsize=20, fontname='Times New Roman', ha='left', va='top')
axes[1, 0].text(0.5, -0.195, '(c)', transform=axes[1, 0].transAxes, fontsize=22, fontname='Times New Roman', ha='center')

# Customize the fourth scatter plot
axes[1, 1].scatter(df_fs['Smoothness_First_Derivative']/1000, df_fs['Smoothness_Second_Derivative']/100000, s=0.5, c='#fdd000', alpha=0.6)
axes[1, 1].set_xlim(0, 4)
axes[1, 1].set_ylim(0, 4)
axes[1, 1].xaxis.set_major_locator(ticker.MultipleLocator(1))
axes[1, 1].yaxis.set_major_locator(ticker.MultipleLocator(1))
axes[1, 1].set_xlabel("Standard Deviation of First Derivative", fontsize=22, fontname='Times New Roman')
axes[1, 1].set_ylabel("Standard Deviation of Second Derivative", fontsize=22, fontname='Times New Roman')
axes[1, 1].tick_params(axis='both', which='major', labelsize=18, labelcolor='black')
axes[1, 1].annotate(r'$\times10^3$', xy=(0.925, -0.08), xycoords='axes fraction', fontsize=20, fontname='Times New Roman', ha='left', va='top')
axes[1, 1].annotate(r'$\times10^5$', xy=(0.01, 1.05), xycoords='axes fraction', fontsize=20, fontname='Times New Roman', ha='left', va='top')
axes[1, 1].text(0.5, -0.195, '(d)', transform=axes[1, 1].transAxes, fontsize=22, fontname='Times New Roman', ha='center')

# Adjust layout
plt.tight_layout()
plt.savefig('fig3.png')
# Display the plot
plt.show()
