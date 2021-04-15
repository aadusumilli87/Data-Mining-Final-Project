# Import
# TODO: Add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup --------------------------------------------------------------------------------
# Load Data
url = 'https://raw.githubusercontent.com/aadusumilli87/Data-Mining-Final-Project/main/data.csv?token=AMPV3CVJDA3ANQ7UU37XP5DAO5XGG'
bankrupt = pd.read_csv(url, index_col=None)


# EDA ---------------------------------------------------------------------------------

# Summary Overview - Get a Sense of What the Columns are.
bankrupt.info()  # No missing values at all, all data is numeric (no need for dummy encoding)
# Dimensions: 6819 * 96

# Summary Stats
describe_output = bankrupt.describe()

# Basic Cleaning:
# Established that there are no missing values
# Some columns have leading white space - will remove:
bankrupt.columns = bankrupt.columns.map(lambda x: x.strip())

# Any Duplicate Values:
bankrupt.duplicated().sum()  # No identical rows

# Correlation Matrix:
corr_mat = bankrupt.corr()
# Plot
sns.heatmap(corr_mat)
plt.show()

# This is a bit too convoluted to make any sense of (too many variables). Let's hone in on the dependent variable:
# Positive Correlations
corr_mat.iloc[:, 0].sort_values(ascending=False).iloc[0:10]
# Debt Ratio is the most positively correlated
# All positive correlations appear to be reflective of expenses and debt, which is as expected

# Negative Correlations:
corr_mat.iloc[:, 0].sort_values(ascending=True).iloc[0:10]
# All negative correlations appear to be profitability metrics

# TODO: Clear that there are many redundant variables. 3 Net value per Share variables, 3 ROA variables, 3 Net Profit Growth Rate, etc
# TODO: Variables to Remove: Net Income Flag (no variation)


# Groupby: What differences do we see for the dependent variable:
grouped_mean = bankrupt.groupby('Bankrupt?').mean()
# Some data quality problems are evident from this: 9 observations with Quick Ratio values that are clearly incorrect (in the thousands)
# Lower Return on Assets,