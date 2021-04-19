# Import
# TODO: Add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from patsy.builtins import Q
sns.set_style('whitegrid')

# Setup --------------------------------------------------------------------------------
# Load Data (Note: Raw Link keeps changing. Unsure how to get around this besides updating the link each session)
url = 'https://raw.githubusercontent.com/aadusumilli87/Data-Mining-Final-Project/main/data.csv?token=AMPV3CVXYF4RD2GKSZNK3RDAPXQQC'
bankrupt = pd.read_csv(url, index_col=None)

# Basic Cleaning:
# Established that there are no missing values
# Some columns have leading white space and almost all have white space in them - remove
# TODO: Note - Statsmodels doesn't like columns with spaces, but this makes everything look worse on plots
bankrupt.columns = bankrupt.columns.str.replace(' ', '')
# Remove
# Rename the Dependent Variable - the Question Mark may cause problems
bankrupt.rename(columns={'Bankrupt?': 'Bankrupt'}, inplace=True)


# EDA ---------------------------------------------------------------------------------
# TODO: Clear that there are many redundant variables. 3 Net value per Share variables, 3 ROA variables, 3 Net Profit Growth Rate, etc
# TODO: EDA Gameplan - hone in on variables with v similar names and eliminate those less correlated with the dependent (keep original to test against)
# TODO: Variables to Remove: Net Income Flag (no variation); Interest-bearingdebtinterestrate; (all variables in the outliers_le_95 object)
# TODO: Other Outliers should be imputed (for predictive model datasets). For inferential models, those outliers should probably be removed
# Summary Overview - Get a Sense of What the Columns are.
bankrupt.info()  # No missing values at all, all data is numeric (no need for dummy encoding)
# Dimensions: 6819 * 96

# Correlation Matrix:
corr_mat = bankrupt.corr('spearman')

# Any Duplicate Values:
bankrupt.duplicated().sum()  # No identical rows

# Summary Stats
describe_output = bankrupt.describe()
# TODO: Note - Clear that some columns have very large outliers that appear to be data quality problems - where 99% of observations are <= 1, and a few are
# TODO: (cont.) very very large. Will write a function to help identify these


def outlier_detection(dataset):
    outliers = dataset.quantile(1) - (dataset.quantile(0.75) * 1.5)  # This rule can be modified to be proper IQR, or whatever decision rule we want
    outliers = outliers[outliers > 1]
    return outliers.index


# Collect Outlier Columns
bankrupt_outliers = bankrupt.loc[:, outlier_detection(bankrupt)]
# Distribution
bankrupt_outliers.hist(figsize=(40, 45))
plt.show()
# Most of these seem to follow the same pattern - vast majority of observations are < 1, with a few that are extremely large
# Can either remove the variables or imputes
# How many values are below 1?
outliers_prop = bankrupt_outliers[bankrupt_outliers <= 1].count() / bankrupt_outliers.shape[0]
# Many columns have > 99%, these can probably be imputed. For those under:
outliers_le_99 = bankrupt_outliers.loc[:, outliers_prop[outliers_prop < .95].index]
# Are these variables strongly correlated (IE, should we try and find a way to preserve them?)
corr_mat.loc[:, outliers_le_99.columns].iloc[0]
# None are strongly correlated, highest at about 5%. These should be removed

# How might we impute the outliers in the others?
outliers_99 = bankrupt_outliers.loc[:, outliers_prop > 0.95]
# Do the large values coincide with bankrupt companies disproportionately?
outliers_99 = pd.merge(outliers_99, bankrupt[['Bankrupt']], left_index=True, right_index=True, how='inner')
# We need to see how different these are for bankrupt vs non-bankrupt
outliers_99_grouped = outliers_99.groupby('Bankrupt').mean()
# Doesn't appear to be too much systemic, besides Revenue per Share, which may be ok

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
# Isolate Positively Correlated Variables - 26 with positve (spearman) correlation over 1%
# Debt Ratio is the most positively correlated
# All positive correlations appear to be reflective of expenses and debt, which is as expected
# Isolate a new data frame with these variables
bankrupt_corrPos = bankrupt[corr_mat.iloc[:, 0].sort_values(ascending=False).iloc[0:26].index]
# Groupby - What differences are evident?
corrPos_grouped = bankrupt_corrPos.groupby('Bankrupt').mean()
# Some quality problems:
# TODO: Totaldebt/Totalnetworth has 8 values that are > 1, all are in the millions (probably data entry error) - should remove
# TODO: Interest-bearingdebtinterestrate has 221 values way too high - can't trust these. Should remove this column
# Allocationrateperperson: 12 values > 1, all in the millions (none coincide with high values of Totaldebt/totalnetworth). Variable should be removed

# Negative Correlations:
corr_mat.iloc[:, 0].sort_values(ascending=True).iloc[0:10]
# All negative correlations appear to be profitability metrics







# Groupby: What differences do we see for the dependent variable:
grouped_mean = bankrupt.groupby('Bankrupt?').mean()
# Some data quality problems are evident from this: 9 observations with Quick Ratio values that are clearly incorrect (in the thousands)
# Summary of Grouped Means - We can use this to look for data quality issues:


