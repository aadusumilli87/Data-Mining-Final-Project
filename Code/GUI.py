# Import Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load Datasets for each component
# Bankrupt Dataset - Using the Clean Version That the Models are Trained On
# NOTE: This file is initially exported in line 244 of the EDA script
# This is used for the correlation heatmap
bankrupt_clean = pd.read_csv('bankrupt_clean.csv', index_col=0)

# Load LPM Results
# NOTE: File is exported in line 144 of Inference script
LPM_Results = pd.read_csv('LPM_results.csv', index_col=0)

# Load Probit Results
# NOTE: File is exported in line 167 of Inference script
Probit_Results = pd.read_csv('Probit_Results.csv', index_col=0)

# Load PCA Results
# NOTE: File is exported in line 220 of Inference script
PCA_Var_Weight = pd.read_csv('PCA_VariableWeight.csv')
PCA_Var_Weight.set_index('Variables', inplace=True)

# Plots---------------------------------------------------------------------------------------------
# Correlation Matrix:
corr_mat = bankrupt_clean.corr('spearman')
f_heatmap, ax_heatmap = plt.subplots(figsize=(28, 25))
color_scheme = sns.diverging_palette(240, 10, as_cmap=True)
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, mask=mask, cmap=color_scheme, vmax=1, center=0, square=True, linewidths=0.5)
plt.title('Bankruptcy Data: Correlation Heatmap', fontsize=40)
plt.show()


# PCA Feature Composition
f_pca, ax_pca = plt.subplots(figsize=(12, 12))
sns.heatmap(PCA_Var_Weight, cmap='bwr')
plt.title('Visualizing Component Structure: \n What Factors Contribute to Bankruptcy?')
plt.show()

# EDA Plots
# Class Imbalance
f_imbalance, ax_imbalance = plt.subplots(figsize=(7, 7))
sns.countplot(bankrupt_clean['Bankrupt'])
plt.title('Bankruptcy Counts \n 0 = Not Bankrupt || 1 = Bankrupt')
plt.show()

# EDA Scatter Plots
# Debt Ratio vs Bankruptcy:
f_debt, ax_debt = plt.subplots(figsize=(7, 7))
sns.scatterplot(x='Operatingprofit/Paid-incapital', y='Totaldebt/Totalnetworth', hue='Bankrupt', data=bankrupt_clean)
plt.show()

#  Debt Ratio vs Asset Turnover
f_turn, ax_turn = plt.subplots(figsize=(7, 7))
sns.scatterplot(x='Totaldebt/Totalnetworth', y='TotalAssetTurnover', hue='Bankrupt', data=bankrupt_clean)
plt.show()

# Grouped Means of Important Variables:
bankrupt_means = bankrupt_clean.groupby('Bankrupt').mean()
# Filter on significant variables
means_index = LPM_Results.head(5).index
means_index= [i for i in means_index if i != 'Constant']
bankrupt_means = bankrupt_means.loc[:, means_index]
# Reshape
bankrupt_means.reset_index(inplace=True)
bankrupt_means = pd.melt(bankrupt_means, id_vars='Bankrupt', value_vars=means_index)
# Plot
f_bar, ax_bar = plt.subplots(figsize=(10, 10))
sns.barplot(x='variable', y='value', hue='Bankrupt', data=bankrupt_means)
plt.show()

# Tables------------------------------------------------------------------------------------------
# LPM Results
LPM_Results.head(5)

# Probit Results
Probit_Results.head(5)