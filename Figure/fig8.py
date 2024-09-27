import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

confusion_matrices = {
    'a': np.array([[45712, 3941, 16, 772, 0, 16, 0],
    [3882, 115602, 3403, 781, 192, 0, 0],
    [6, 2649, 22547, 1, 836, 1, 1],
    [920, 1586, 16, 27824, 1349, 1181, 627],
    [0, 213, 1112, 855, 17604, 2, 1142],
    [0, 0, 0, 2296, 6, 66323, 5512],
    [0, 0, 5, 192, 1112, 2519, 117364]]),

    'b': np.array([ [15662, 2201, 7, 420, 0, 0, 0],
    [1544, 51437, 768, 598, 189, 0, 0],
    [32, 1808, 8922, 4, 386, 0, 1],
    [211, 329, 0, 10874, 502, 892, 180],
    [0, 53, 582, 377, 7067, 2, 524],
    [0, 0, 0, 584, 3, 30802, 2086],
    [0, 1, 2, 109, 521, 2178, 51822]]),

    'c': np.array([[12865, 4952, 5, 461, 0, 6, 1],
    [926, 52464, 686, 352, 108, 0, 0],
    [0, 2608, 8164, 1, 376, 1, 3],
    [296, 687, 4, 10396, 528, 917, 160],
    [0, 132, 713, 338, 6756, 9, 657],
    [17, 5, 0, 476, 3, 30507, 2467],
    [0, 1, 4, 57, 417, 1607, 52547]]),

    'd': np.array([[44404, 5217, 19, 801, 0, 16, 0],
    [4685, 114394, 3344, 1205, 232, 0, 0],
    [14, 2856, 22056, 17, 1096, 1, 1],
    [919, 1932, 43, 27233, 1470, 1324, 582],
    [0, 235, 1492, 892, 16990, 7, 1312],
    [0, 0, 0, 2704, 24, 65782, 5627],
    [0, 0, 4, 177, 1468, 2616, 116927]]),

    'e': np.array([ [14653, 3183, 19, 435, 0, 0, 0],
    [3351, 49165, 921, 923, 176, 0, 0],
    [74, 1887, 8393, 23, 775, 0, 1],
    [244, 522, 15, 10419, 534, 1094, 160],
    [0, 79, 850, 435, 6516, 15, 710],
    [1, 0, 0, 965, 19, 30211, 2279],
    [0, 1, 2, 134, 852, 2336, 51308]]),

    'f': np.array([ [12170, 5581, 19, 506, 0, 14, 0],
    [2581, 49892, 793, 1043, 226, 0, 1],
    [0, 2752, 7298, 19, 1074, 1, 9],
    [274, 1064, 12, 9652, 470, 1301, 215],
    [1, 233, 868, 323, 6334, 11, 835],
    [37, 5, 0, 789, 6, 30050, 2588],
    [0, 2, 7, 76, 548, 1637, 52363]])
}

def plot_confusion_matrices(cm_dict):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24, 16))
    axes = axes.flatten()

    labels = ['(a) DWN', '(b) XGBoost', '(c) LSTM', '(d) DWN', '(e) XGBoost', '(f) LSTM']

    for ax, (label, cm) in zip(axes, zip(labels, cm_dict.values())):
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Greens', ax=ax, cbar=False,
                    xticklabels=np.arange(1, cm.shape[1] + 1), yticklabels=np.arange(1, cm.shape[0] + 1),
                    annot_kws={"size": 18})
        ax.tick_params(axis='both', which='major', labelsize=18)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontname('Times New Roman')
            tick_label.set_fontsize(18)
        ax.text(0.5, -0.12, label, transform=ax.transAxes, fontsize=24, fontname='Times New Roman', ha='center')

    plt.tight_layout()
    plt.savefig('fig8_conf.png')
    plt.show()

plot_confusion_matrices(confusion_matrices)
