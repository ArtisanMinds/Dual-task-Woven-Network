import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt

random_seed = 72
random.seed(random_seed)
np.random.seed(random_seed)

data = pd.read_csv('../../../Model/Data/cptu_data.csv')
data = data[['depth', 'qc', 'fs', 'u2', 'label']]
features = data[['qc', 'fs', 'depth', 'u2']]
labels = data['label']

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=random_seed)
print(X_train.shape, X_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = XGBClassifier(random_state=random_seed, eval_metric='mlogloss', n_jobs=-1)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
prediction_probs = model.predict_proba(X_test)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

if isinstance(shap_values, list):
    shap_values = shap_values[0]

mean_shap_values = np.mean(shap_values, axis=2)

feature_names = [r'$q_c$', r'$f_s$', 'depth', r'$u_2$']

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

norm = plt.Normalize(vmin=-1, vmax=1)
cmap = plt.cm.colors.LinearSegmentedColormap.from_list("", ['yellow', 'green'])

for i, feature_name in enumerate(feature_names):
    feature_shap_values = mean_shap_values[:, i]
    feature_values = X_train[:, i]
    colors = cmap(norm(feature_values))
    y_positions = np.where(
        np.random.rand(mean_shap_values.shape[0]) < 0.65,
        np.full(mean_shap_values.shape[0], i) + np.random.normal(0, 0.01, mean_shap_values.shape[0]),
        np.full(mean_shap_values.shape[0], i) + np.random.normal(0, 0.05, mean_shap_values.shape[0])
    )
    axs[1].scatter(feature_shap_values, y_positions, s=0.01, c=colors, alpha=0.5)

axs[1].set_yticks(range(len(feature_names)))
axs[1].set_xlabel("SHAP Value for Each Sample")
axs[1].invert_yaxis()

cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[1], label='Normalized Feature Value')
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

mean_abs_shap_values = np.mean(np.abs(mean_shap_values), axis=0)
axs[0].barh(range(len(mean_abs_shap_values)), mean_abs_shap_values, color='#587558', alpha=0.7)
axs[0].set_yticks(range(len(mean_abs_shap_values)))
axs[0].set_xlabel("Mean Absolute SHAP Value")
axs[0].invert_yaxis()

axs[0].text(0.5, -0.2, '(a)', transform=axs[0].transAxes, fontsize=18, va='top', ha='center')
axs[1].text(0.5, -0.2, '(b)', transform=axs[1].transAxes, fontsize=18, va='top', ha='center')

axs[0].set_yticklabels(feature_names)
axs[1].set_yticklabels(feature_names)

plt.tight_layout()
plt.savefig('fig1_Shap.png')
plt.show()
