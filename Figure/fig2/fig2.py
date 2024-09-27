import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df_result = pd.read_csv('qc_cont.csv')
df_fs = pd.read_csv('fs_cont.csv')

# 创建图像和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# 第一个散点图
ax1.scatter(df_result['Morans_I'], df_result['Gearys_C'], s=0.5, c='#587558', alpha=0.6)
ax1.plot([0, 1], [1, 0], '--', c='#231815')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel("Moran's I", fontsize=26, fontname='Times New Roman')
ax1.set_ylabel("Geary's C", fontsize=26, fontname='Times New Roman')
ax1.tick_params(axis='both', which='major', labelsize=22, labelcolor='black')
ax1.text(0.5, -0.215, '(a)', transform=ax1.transAxes, fontsize=26, fontname='Times New Roman', ha='center')

# 第二个散点图
ax2.scatter(df_fs['Morans_I'], df_fs['Gearys_C'], s=0.5, c='#fdd000', alpha=0.6)
ax2.plot([0, 1], [1, 0], '--', c='#231815')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xlabel("Moran's I", fontsize=26, fontname='Times New Roman')
ax2.set_ylabel("Geary's C", fontsize=26, fontname='Times New Roman')
ax2.tick_params(axis='both', which='major', labelsize=22, labelcolor='black')
ax2.text(0.5, -0.215, '(b)', transform=ax2.transAxes, fontsize=26, fontname='Times New Roman', ha='center')

# 调整布局
plt.tight_layout()
# 保存pdf
plt.savefig('fig2.png')
# 显示图像
plt.show()
