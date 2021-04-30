import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy.builtins import Q
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Preparation:--------------------------------------------------------------------------------------
# NOTE: PCA Preparation: Using Original Data for PCA - We want the correlated variables in this instance
# Load Data - Unaltered (for PCA), and Cleaned (for regression)
bankrupt = pd.read_csv('C:\\Users\\Amar\\OneDrive\\Documents\\GW\\Spring 2021\\Data Mining\\Project\\Data-Mining-Final-Project\\data.csv', index_col=None)
# Remove Spaces in Column Names
bankrupt.columns = bankrupt.columns.str.replace(' ', '')
# Rename the Dependent Variable - the Question Mark may cause problems
bankrupt.rename(columns={'Bankrupt?': 'Bankrupt'}, inplace=True)
# Drop the dependent variable:
bankrupt.drop('Bankrupt', axis=1, inplace=True)
# Outlier Handling
# NOTE: Greyed out for PCA - data cleaning steps cause PCA to perform worse
#
# def outlier_detection(dataset):
#     outliers = dataset.quantile(1) - (dataset.quantile(0.75) * 1.5)  # This rule can be modified to be proper IQR, or whatever decision rule we want - TG: fine with keeping this definition or IQR
#     outliers = outliers[outliers > 1]
#     return outliers.index
#
#
# # Collect Outlier Columns
# bankrupt_outliers = bankrupt.loc[:, outlier_detection(bankrupt)]
#
# # Remove columns with severe outlier issues
# outliers_prop = bankrupt_outliers[bankrupt_outliers <= 1].count() / bankrupt_outliers.shape[0]
# # Many columns have > 99%, these can probably be imputed. For those under (TG: should be removed):
# outliers_le_99 = bankrupt_outliers.loc[:, outliers_prop[outliers_prop < .95].index]
# # Drop
# bankrupt.drop(outliers_le_99.columns, axis=1, inplace=True)
#
# # Apply Winsorization
# Winsor_set = bankrupt.max()
# Winsor_set = Winsor_set[Winsor_set > 1]
# # 17 Variables with outlier max values
# # Looking at this, we have 2 variables remaining with values > 1
#
#
# def winsorize(data, first_quant, second_quant):
#     first_thresh = data.quantile(first_quant)
#     # Isolate Columns with no outliers in 99th percentile values
#     first_thresh_cols = first_thresh[first_thresh < 1].index
#     for col in first_thresh_cols:
#         data[col] = np.where(data[col] > data[col].quantile(first_quant), data[col].quantile(first_quant), data[col])
#     # Isolate Columns Not Captured in the First Thresh
#     second_thresh = data.loc[:, ~data.columns.isin(first_thresh_cols)].columns
#     for col in second_thresh:
#         data[col] = np.where(data[col] > data[col].quantile(second_quant), data[col].quantile(second_quant), data[col])
#     return data
#
#
# # Apply to Data
# bankrupt.loc[:, Winsor_set.index] = winsorize(data=bankrupt.loc[:, Winsor_set.index], first_quant=0.99, second_quant=0.98)
#
# bankrupt['Interest-bearingdebtinterestrate'] = np.where(
#     bankrupt['Interest-bearingdebtinterestrate'] > 1, 1, bankrupt['Interest-bearingdebtinterestrate'])

# NOTE: Inferential Preparation
# Load Data - Uses dataset cleaned in the EDA.py file
bankrupt_inf = pd.read_csv('C:\\Users\\Amar\\OneDrive\\Documents\\GW\\Spring 2021\\Data Mining\\Project\\bankrupt_clean.csv')
# Remove the index column
bankrupt_inf.drop('Unnamed: 0', axis=1, inplace=True)

# Isolate Dependent
bankrupt_label = bankrupt_inf['Bankrupt']
bankrupt_inf.drop('Bankrupt', axis=1, inplace=True)
# Most Data Quality Issues are Taken Care of. However, There are still issues with dimensionality through redundent variables
corr_mat = bankrupt_inf.corr('spearman')
# Isolate variables which are >= .90 correlated


# def mutli_coll(corr_matrix):
#     corr_matrix = corr_matrix[(corr_matrix > 0.95) & (corr_matrix != 1)]
#     variables_rm = []
#     for col in corr_matrix:
#         current_series = corr_matrix[col].dropna()
#         current_vars = list(current_series.index)
#         current_vars.append(current_series.name)
#         coefficients = []
#         for i in current_vars:
#             r = np.abs(np.corrcoef(bankrupt_inf[i], bankrupt_label)[0, 1])
#             coefficients.append(r)
#             col_index = coefficients.index(min(coefficients))
#             variables_rm.append(current_vars[col_index])
#     return variables_rm


# collinear_vars = mutli_coll(corr_mat)  # Large Numbers of Collinear Variables
# Remove Duplicates
# collinear_vars = list(set(collinear_vars))

# Alternate Approach - Take the Upper Trianglur Part of the Correlation Matrix
upper_tri = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
# Find columns to drop
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.90)]
# Make a list of potential columns to drop
# Drop columns
bankrupt_trimmed = bankrupt_inf.drop(to_drop, axis=1)


# Econometric Modeling #-----------------------------------------------------------------------------
# Prepared the Trimmed Data
standard_scaler = StandardScaler()
bankrupt_trimmed = pd.DataFrame(standard_scaler.fit_transform(bankrupt_trimmed), columns=bankrupt_trimmed.columns)
# Note: Standard Scaling the Data changes coefficient interpretation:
# A one SD increase in X_i has an Beta_i increase on Y
# Add Intercept
bankrupt_trimmed['Constant'] = 1

# LPM
LPM = sm.OLS(exog=bankrupt_trimmed, endog=bankrupt_label, hasconst=True).fit(cov_type='HC1')
# Strong Multicollinearity Problems Persist. Using VIF, we can eliminate more variables:
VIF = pd.Series([variance_inflation_factor(bankrupt_trimmed.values, i) for i in range(bankrupt_trimmed.shape[1])],
          index=bankrupt_trimmed.columns)

# 3 variables have a VIF > 15. Removing those:
inf_vars = VIF[VIF > 15].index
bankrupt_trimmed.drop(inf_vars, axis=1, inplace=True)

# Rerun Regression
LPM_2 = sm.OLS(exog=bankrupt_trimmed, endog=bankrupt_label, hasconst=True).fit(cov_type='HC1')
print(LPM_2.summary())
# Export
LPM_results = pd.DataFrame(
    {'Parameters': LPM_2.params.values,
     'Standard Error': LPM_2.HC1_se.values,
     'T-Stat': LPM_2.tvalues.values,
     'P-values': LPM_2.pvalues.values},
    index=LPM_2.params.index
)
# Remove Insignificant Variables
LPM_results = LPM_results[LPM_results['P-values'] <= 0.05]

# Order on magnitude
LPM_results = LPM_results.sort_values(by='Parameters', ascending=False)
# Show 5 most important variables
print(LPM_results.head(5))
# Note: A one standard deviation increase in the Debt/Net Worth ratio corresponds with a 5.6% increase in P(Bankruptcy)
# Other variables have the same interpretation


# Probit
Probit = sm.Probit(exog=bankrupt_trimmed, endog=bankrupt_label).fit()
# Model Results
print(Probit)
# Export Results
Probit_Results = pd.DataFrame(
    {'Parameters': Probit.params.values,
     'T-Stat': Probit.tvalues.values,
     'P-values': Probit.pvalues.values},
    index=Probit.params.index
)
# Filter on significant variables
Probit_Results = Probit_Results[Probit_Results['P-values'] <= 0.05]
# Sort Values
Probit_Results = Probit_Results.sort_values('Parameters', ascending=False)
# Show most important determinants of Bankruptcy
print(Probit_Results.head(5))
# Note: A one SD increase in Operatingprofit/Paid-In Capital increases Z in Phi(z) by 0.467
# For reference, in a standard normal distribution, Phi(0) = 0.5
# Phi(0.467) = 0.68, so a one SD increase change in the profit/capital ratio is associated with an 18% increase in P(bankruptcy)
# As Probit is nonlinear, this will change for different values of Z
# Since data is standardized, if all Z-scores are 0 for an observation, we only have the constant P(-2.79) = 0.002
# In this case, P(const + 0.467) = 0.01, so the change is effectively insignificant

# PCA:------------------------------------------------------------------------------------------------
# Identify the number of components needed to explain the majority of the variance
var_explained = []
# Potential component numbers
n_components = np.arange(1, 10)
for n in n_components:
    pca = PCA(n_components=n)
    pca.fit_transform(bankrupt)
    variance = pca.explained_variance_ratio_.sum()
    var_explained.append(variance)
print(var_explained[7])
# 8 components explains some 97% of the variation

# Apply the PCA to the data
pca = PCA(n_components=8, random_state=505)
bankrupt_pca = pca.fit_transform(bankrupt)
# Convert Back to DataFrame
PC_cols = ['PC' + str(i) for i in range(1, 9)]
bankrupt_pca = pd.DataFrame(bankrupt_pca, columns=PC_cols)

# Isolate Components - Viewing Structure May Allow Us to See the Biggest Bankruptcy Determinants
components = pd.DataFrame(pca.components_.reshape(-1, 8), columns=PC_cols)
# Associate with the Variable Names
variables = []
variable_weight = []
for PC in PC_cols:
    primary_vars = np.abs(components[PC]).sort_values(ascending=False)[:5]
    for i in primary_vars.index:
        variable_weight.append(primary_vars[i])
        variables.append(bankrupt.columns[i])

# Create a DataFrame to store PC components
pc_dict = {'Variables': variables, 'Weight': variable_weight}
var_weight_df = pd.DataFrame(pc_dict)
# Since the top 5 variables are chosen from each component, I can demarcate with a running index
var_weight_df['Component'] = [i + 1 for i in range(8) for j in range(5)]
# Set index as variable name
# var_weight_df.set_index('Variables')
# Pivot Long to Wide
var_weight_df = var_weight_df.pivot(index='Variables', columns='Component', values='Weight')
# Sort Values
var_weight_df = var_weight_df.sort_values(var_weight_df.columns.to_list(), ascending=False)

# Plot the Components
plt.figure(figsize=(12, 12))
sns.heatmap(var_weight_df, cmap='bwr')
plt.title('Visualizing Component Structure: \n What Factors Contribute to Bankruptcy?')
plt.show()



