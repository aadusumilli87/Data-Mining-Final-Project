# NOTE: EDA Code Below --------------------------------------------------------------------------------------------------------------------------------------
# Import
# TODO: Add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from patsy.builtins import Q
import warnings
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn import utils
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy.builtins import Q
from sklearn.decomposition import PCA
# os.chdir('...') # Commenting out
# Initialization -----------------------------------------------------------------------
# Ignore Warnings
warnings.filterwarnings('ignore')
# Seaborn Plot Area
sns.set_style('whitegrid')
# Matplotlib Plot Area
plt.rc('font', size=20)
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('legend', fontsize=20)
plt.rc('figure', titlesize=20)
# Random Seed
seed = 101
# The StandardScaler
ss = StandardScaler()
# os Path
path = os.path.abspath(os.getcwd())
# Setup --------------------------------------------------------------------------------
# Load Data (Note: Raw Link keeps changing. Unsure how to get around this besides updating the link each session)
# url = 'https://raw.githubusercontent.com/aadusumilli87/Data-Mining-Final-Project/main/data.csv?token=AMPV3CTJRCPPMVIX5GQYKSTARG6AS'
# bankrupt = pd.read_csv(url, index_col=None)
# Hard encoding data.csv
bankrupt = pd.read_csv("data.csv", sep=',',header=0, index_col=None)
# Basic Cleaning:
# Establishing that there are no missing values
# Some columns have leading white space and almost all have white space in them - remove
# TODO: Note - Statsmodels doesn't like columns with spaces, but this makes everything look worse on plots - TG: Could replace the space with '_' or '.'
bankrupt.columns = bankrupt.columns.str.replace(' ', '')
# Remove
# Rename the Dependent Variable - the Question Mark may cause problems
bankrupt.rename(columns={'Bankrupt?': 'Bankrupt'}, inplace=True)
# EDA
# TODO: Clear that there are many redundant variables. 3 Net value per Share variables, 3 ROA variables, 3 Net Profit Growth Rate, etc
# TODO: EDA Gameplan - hone in on variables with v similar names and eliminate those less correlated with the dependent (keep original to test against)
# TODO: Variables to Remove: Net Income Flag (no variation); Interest-bearingdebtinterestrate; (all variables in the outliers_le_95 object)
# TODO: Other Outliers should be imputed (for predictive model datasets). For inferential models, those outliers should probably be removed
# TODO: For predictive models - most variables are bounded by 1, with outliers above. We can hone in on the outliers, replace them with NaN's,
# TODO: Then groupby transform to fill with group mean or median.
# Summary Overview - Get a Sense of What the Columns are.
bankrupt.info()  # No missing values at all, all data is numeric (no need for dummy encoding)
# Dimensions: 6819 * 96
# Correlation Matrix:
corr_mat = bankrupt.corr('spearman')
corr_mat.to_csv("SpearmanCorrMat.csv")
# Findings: Remove 'Net Income Flag' - No change in values makes no diff;
# The following have a Spearman Rank in the range [0.05, -0.05] (against Bankrupt?) - lowest impact:
#  Current Asset Turnover Rate*
#  Fixed Assets Turnover Frequency*
#  Quick Asset Turnover Rate*
#  Contingent liabilities/Net worth
#  Average Collection Days*
#  Cash Turnover Rate*
#  Revenue per person*
#  Inventory Turnover Rate (times)*
#  Operating Expense Rate*
#  Net Worth Turnover Rate (times)
#  Accounts Receivable Turnover*
#  Current Liabilities/Liability
#  Current Liability to Liability
#  Research and development expense rate*
#  Current Assets/Total Assets
# * indicates that these variables are also a part of bankrupt_outliers below
# Any Duplicate Values:
bankrupt.duplicated().sum()  # No identical rows

# Encoding Data and Handling Non-Numerical Data
# - No identifiers to remove, no date-time vars to standardize
# - Dropping 'NetIncomeFlag' since it has no values
bankrupt['NetIncomeFlag'].value_counts()
bankrupt.drop(['NetIncomeFlag'],axis=1,inplace=True)
# Summary Stats
describe_output = bankrupt.describe()
# TODO: Note - Clear that some columns have very large outliers that appear to be data quality problems - where 99% of observations are <= 1, and a few are
# TODO: (cont.) very very large. Will write a function to help identify these
# Outliers Identification
def outlier_detection(dataset):
    outliers = dataset.quantile(1) - (dataset.quantile(0.75) * 1.5)  # This rule can be modified to be proper IQR, or whatever decision rule we want - TG: fine with keeping this definition or IQR
    outliers = outliers[outliers > 1]
    return outliers.index
# Collect Outlier Columns
bankrupt_outliers = bankrupt.loc[:, outlier_detection(bankrupt)]
# Distribution
bankrupt_outliers.hist(figsize=(40, 45))
plt.show()
# Most of these seem to follow the same pattern - vast majority of observations are < 1, with a few that are extremely large
# Can either remove the variables or imputes - TG: happy to remove since most are insignificant in their correlation to 'Bankrupt?'
# How many values are below 1?
outliers_prop = bankrupt_outliers[bankrupt_outliers <= 1].count() / bankrupt_outliers.shape[0]
# Many columns have > 99%, these can probably be imputed. For those under (TG: should be removed):
outliers_le_99 = bankrupt_outliers.loc[:, outliers_prop[outliers_prop < .95].index]
# Are these variables strongly correlated (IE, should we try and find a way to preserve them?)
corr_mat.loc[:, outliers_le_99.columns].iloc[0]
# None are strongly correlated, highest at about 5%. These should be removed - TG: agree to remove
bankrupt.drop(outliers_le_99.columns, axis=1, inplace=True)

# How might we impute the outliers in the others?
outliers_99 = bankrupt_outliers.loc[:, outliers_prop > 0.95]
# Do the large values coincide with bankrupt companies disproportionately?
outliers_99 = pd.merge(outliers_99, bankrupt[['Bankrupt']], left_index=True, right_index=True, how='inner')
# We need to see how different these are for bankrupt vs non-bankrupt
outliers_99_grouped = outliers_99.groupby('Bankrupt').mean()
# Doesn't appear to be too much systemic, besides Revenue per Share, which may be ok
# We need to identify which of these are in close to bankruptcy cases
# 96th percentile observation shows all decimal numbers, above that we get very large numbers for many of the columns - using that as cutoff point:
def outlier_value_detection(dataset):
    column_list = []
    for col in dataset:
        current_series = dataset[str(col)]
        current_series = current_series[current_series > 1]
        current_series = pd.merge(current_series, bankrupt['Bankrupt'], left_index=True, right_index=True, how='inner')
        column_list.append(current_series)
    return column_list
outlier_values = outlier_value_detection(outliers_99)
# Counts of Bankruptcy cases in these variables:
count_list = []
for i in range(len(outlier_values) - 1):
    current_counts = outlier_values[i].groupby('Bankrupt').count()
    count_list.append(current_counts)
# Looking through the list, counts don't seem systemically biased.
# Looking at means:
mean_list = []
for i in range(len(outlier_values) - 1):
    current_counts = outlier_values[i].groupby('Bankrupt').mean()
    mean_list.append(current_counts)
# Winsorize the Data - In the columns that remain, the outlier problem is less significant than initially suspected
Winsor_set = bankrupt.max()
Winsor_set = Winsor_set[Winsor_set > 1]
# 17 Variables with outlier max values
# Set cutoff point - where do the values start to get very large?
x = bankrupt.loc[:, Winsor_set.index].quantile(0.98)
# Looking at this, we have 2 variables remaining with values > 1
def winsorize(data, first_quant, second_quant):
    first_thresh = data.quantile(first_quant)
    # Isolate Columns with no outliers in 99th percentile values
    first_thresh_cols = first_thresh[first_thresh < 1].index
    for col in first_thresh_cols:
        data[col] = np.where(data[col] > data[col].quantile(first_quant), data[col].quantile(first_quant), data[col])
    # Isolate Columns Not Captured in the First Thresh
    second_thresh = data.loc[:, ~data.columns.isin(first_thresh_cols)].columns
    for col in second_thresh:
        data[col] = np.where(data[col] > data[col].quantile(second_quant), data[col].quantile(second_quant), data[col])
    return data

# We can extend this as needed, however two sweeps covers all but two columns
# Apply the function
bankrupt.loc[:, Winsor_set.index] = winsorize(data=bankrupt.loc[:, Winsor_set.index], first_quant=0.99, second_quant=0.98)

# Todo: 2 Columns Not Captured - Interest Bearing Debt Rate, and Total Asset Groth Rate
# For the former, I will just do manually
bankrupt['Interest-bearingdebtinterestrate'] = np.where(
    bankrupt['Interest-bearingdebtinterestrate'] > 1, 1, bankrupt['Interest-bearingdebtinterestrate'])
# Export Outliers Cleaned Dataset
# bankrupt.to_csv('bankrupt_clean.csv')
# EDA Plots:
# Class Imbalance:
sns.countplot(bankrupt['Bankrupt'])
plt.title('Bankruptcy Counts \n 0 = Not Bankrupt || 1 = Bankrupt')
plt.show()

# General Distributions
bankrupt.hist(figsize=(40, 45), bins = 40)
plt.show()

# Boxpolots
plt.figure(figsize=(25, 25))
sns.boxplot(data=bankrupt, orient='h')
plt.show()

# Correlation Heatmap - All Variables
f, ax = plt.subplots(figsize=(31, 27))
color_scheme = sns.diverging_palette(240, 10, as_cmap=True)
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, mask=mask, cmap=color_scheme, vmax=1, center=0, square=True, linewidths=0.5)
plt.title('Bankruptcy Data: Correlation Heatmap', fontsize=40)
plt.show()
# Most variables are only weakly correlated with Bankruptcy - debt measures are positively correlated, and profit measures are negatively correlated
# Will hone in on the most correlated variables:

# Positive Correlations
corr_mat.iloc[:, 0].sort_values(ascending=False).iloc[0:26]
# Isolate Positively Correlated Variables - 26 with positive (spearman) correlation over 1% (TG: of which, 10 are under 5%)
# Debt Ratio is the most positively correlated - TG: I see Totaldebt/Totalnetworth as highest, then debtratio
# All positive correlations appear to be reflective of expenses and debt, which is as expected
# Isolate a new data frame with these variables
bankrupt_corrPos = bankrupt[corr_mat.iloc[:, 0].sort_values(ascending=False).iloc[0:26].index]
bankrupt_corrPos.to_csv("PosCorrMat.csv")
# Groupby - What differences are evident?
corrPos_grouped = bankrupt_corrPos.groupby('Bankrupt').mean()
# Some quality problems:
# TODO: Totaldebt/Totalnetworth has 8 values that are > 1, all are in the millions (probably data entry error) - should remove - TG: 6 of 8 values that are > 1, are associated with bankruptcy, I think we should keep
# TODO: Interest-bearingdebtinterestrate has 221 values way too high - can't trust these. Should remove this column
# Allocationrateperperson: 12 values > 1, all in the millions (none coincide with high values of Totaldebt/totalnetworth). Variable should be removed
# TG - suggestion to take logs of non-fractional values or outliers above Amar's 96 pctile rule

# Negative Correlations:
corr_mat.iloc[:, 0].sort_values(ascending=True).iloc[0:10]
# All negative correlations appear to be profitability metrics
# All negative correlations appear to be profitability and capital-at-hand metrics
# Groupby: What differences do we see for the dependent variable:
grouped_mean = bankrupt.groupby('Bankrupt').mean()
# NOTE: GUI Plots below-------------------------------------------------------------------------------------------------------------------------------------------------
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
# NOTE: Inference Code Below-------------------------------------------------------------------------------------------------------------------------------------------------
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
# Collect Results
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
# Export Results
# LPM_results.to_csv('LPM_results.csv')
# Show 5 most important variables
print(LPM_results.head(5))
# Note: A one standard deviation increase in the Debt/Net Worth ratio corresponds with a 5.6% increase in P(Bankruptcy)
# Other variables have the same interpretation

# Probit
Probit = sm.Probit(exog=bankrupt_trimmed, endog=bankrupt_label).fit()
# Model Results
print(Probit)
# Collect Results
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
# Export Results
# Probit_Results.to_csv('Probit_Results.csv')
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
# Export
# var_weight_df.to_csv('PCA_VariableWeight.csv')
# Plot the Components
plt.figure(figsize=(12, 12))
sns.heatmap(var_weight_df, cmap='bwr')
plt.title('Visualizing Component Structure: \n What Factors Contribute to Bankruptcy?')
plt.show()



