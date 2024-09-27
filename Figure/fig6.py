import matplotlib.pyplot as plt
import math
from matplotlib.ticker import ScalarFormatter

epochs_loss = range(5, 89)
train_loss = [
    1.5808675137925354, 1.3566605007639132, 1.174211972004376, 1.0305996984826837, 0.9254196266597757,
    0.8590632427364191, 0.7954054848701578, 0.7389466162295479, 0.7033498507042648, 0.6580826684470837,
    0.6258817253723309, 0.6008082348430352, 0.5741271774518648, 0.5578740828655246, 0.53720524762418,
    0.5166670812837949, 0.49860754679121255, 0.4920362160134894, 0.47551083828475016, 0.4647431021429359,
    0.45390594154946395, 0.44205041970698833, 0.43361881536343366, 0.41866268449774363, 0.4153066724422114,
    0.4015802414088671, 0.39513277417061937, 0.38999154087894117, 0.38105651306844635, 0.3747763072221468,
    0.36571638109913635, 0.3605735319708135, 0.3574546547074195, 0.3492311606112176, 0.34630631999103184,
    0.34024111596326295, 0.33443823029683173, 0.3309246090911008, 0.3247077152448048, 0.3213726658907922,
    0.3197862709418067, 0.31353719720358764, 0.31187849107345006, 0.30685703253363295, 0.3018978599841089,
    0.298880085640805, 0.29844321841454263, 0.2951696303876565, 0.29228670260312506, 0.2889627304362803,
    0.2873167247630323, 0.28570654480692553, 0.28156730869876845, 0.2803197364813663, 0.27836735362220577,
    0.27669473217629603, 0.27472134386083535, 0.2753197263135701, 0.2720182373384619, 0.2702087371397877,
    0.26863523344556706, 0.2677905174816784, 0.265321367183232, 0.26408529315455964, 0.2614391233856639,
    0.26431756136981227, 0.2628839277693697, 0.25992289612605035, 0.25931658503032795, 0.2587660370436887,
    0.25793803068419036, 0.25558286424530796, 0.25555246711337387, 0.25568623718285616, 0.2534308962016341,
    0.25422147454324884, 0.2528211595658315, 0.2533989911818747, 0.25216491138832403, 0.25247930669420476,
    0.2526207311296239, 0.25184940624349084, 0.2515454478416017, 0.2516454478416017
]

def learning_rate(epoch, warmup_start_lr, warmup_end_lr, warmup_epoch, cos_epoch, total_epochs):
    if epoch <= warmup_epoch:
        lr = warmup_start_lr + (warmup_end_lr - warmup_start_lr) * (epoch / warmup_epoch) * 1e4
    elif cos_epoch <= epoch <= total_epochs:
        t = (epoch - cos_epoch) / (total_epochs - cos_epoch)
        lr = max(warmup_end_lr * 0.5 * (1 + math.cos(math.pi * t)), 5e-5) * 1e4
    else:
        lr = 3e-4 * 1e4
    return lr

epochs_lr = range(101)
lr_values = [learning_rate(epoch, 1e-7, 3e-4, 5, 15, 100) for epoch in epochs_lr]
display_epochs_lr = [epoch + 1 for epoch in epochs_lr]

fig, ax = plt.subplots(1, 2, figsize=(18, 6))

ax[0].plot(display_epochs_lr, lr_values, linewidth=3, color='#587558')
ax[0].set_xlabel('Epoch', fontsize=24, fontname='Times New Roman')
ax[0].set_ylabel('Learning Rate', fontsize=24, fontname='Times New Roman')
ax[0].set_xlim(1, 100)
ax[0].set_ylim(0, 3.02)
ax[0].tick_params(axis='both', which='major', labelsize=20, labelcolor='black')
ax[0].annotate(r'$\times10^{-4}$', xy=(0.01, 1.05), xycoords='axes fraction', fontsize=20, fontname='Times New Roman', ha='left', va='top')
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax[0].yaxis.set_major_formatter(formatter)
ax[0].set_xticks([1] + list(range(20, 101, 20)))

ax[1].plot(epochs_loss, train_loss, linewidth=3, color='#fdd000')
ax[1].set_xlabel('Epoch', fontsize=24, fontname='Times New Roman')
ax[1].set_ylabel('Loss', fontsize=24, fontname='Times New Roman')
ax[1].tick_params(axis='both', which='major', labelsize=20, labelcolor='black')
ax[1].set_ylim(0.2, 1.5)
ax[1].set_xlim(5, 88)
ax[1].set_yticks([0.2, 0.5, 0.8, 1.2, 1.5])
ax[1].set_xticks([5] + list(range(15, 89, 10)))

ax[0].text(0.5, -0.22, '(a)', transform=ax[0].transAxes, fontsize=24, fontname='Times New Roman', ha='center')
ax[1].text(0.5, -0.22, '(b)', transform=ax[1].transAxes, fontsize=24, fontname='Times New Roman', ha='center')

plt.tight_layout()
plt.savefig('fig6_Train.png')
plt.show()
