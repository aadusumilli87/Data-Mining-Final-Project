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
# url = 'https://raw.githubusercontent.com/aadusumilli87/Data-Mining-Final-Project/main/data.csv?token=AMPV3CVXYF4RD2GKSZNK3RDAPXQQC'
# bankrupt = pd.read_csv(url, index_col=None)

# Hard encoding data.csv
bankrupt = pd.read_csv("data.csv", sep=',',header=0, index_col=None)

# Basic Cleaning:
# Establishing that there are no missing values


# nan checker
def nan_checker(df):

    # Get the dataframe of variables with NaN, their proportion of NaN and data type
    df_nan = pd.DataFrame([[var, df[var].isna().sum() / df.shape[0], df[var].dtype]
                           for var in df.columns if df[var].isna().sum() > 0],
                          columns=['var', 'proportion', 'dtype'])

    # Sort df_nan in ascending order of the proportion of NaN
    df_nan = df_nan.sort_values(by='proportion', ascending=False).reset_index(drop=True)

    return df_nan
df_nan = nan_checker(bankrupt)
print("No variables with missing values:")
print(df_nan)     # Shows there are no columns with any missing values


# Some columns have leading white space and almost all have white space in them - remove
# TODO: Note - Statsmodels doesn't like columns with spaces, but this makes everything look worse on plots - TG: Could replace the space with '_' or '.'
bankrupt.columns = bankrupt.columns.str.replace(' ', '')
# Remove

# Rename the Dependent Variable - the Question Mark may cause problems
bankrupt.rename(columns={'Bankrupt?': 'Bankrupt'}, inplace=True)


# EDA ---------------------------------------------------------------------------------
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

# TG - multicollinearity suggestion - remove 17 cols that are >=|95%| collinearly related - NOT SPEARMAN
multicoll_mat = []
for i in range(len(corr_mat.columns)):
    for j in range(i):
        if(corr_mat.iloc[i,j] >= 0.95 or corr_mat.iloc[i,j] <= -0.95):
            if corr_mat.columns[j] not in multicoll_mat:
                multicoll_mat.append(corr_mat.columns[j])
print("Spearman Correlation Matrix Multicollinearity Columns:",len(multicoll_mat))
print(multicoll_mat)

corr_mtx = bankrupt.corr()
corr_mtx = corr_mtx.iloc[1:,1:]
multicoll_mtx = []
for i in range(len(corr_mtx.columns)):
    for j in range(i):
        if (corr_mtx.iloc[i, j] >= 0.95 or corr_mtx.iloc[i, j] <= -0.95):
            if corr_mtx.columns[j] not in multicoll_mtx:
                multicoll_mtx.append(corr_mtx.columns[j])
print("Pearson Correlation Matrix Multicollinearity Columns:",len(multicoll_mtx))
print(multicoll_mtx)
mcm_diff = [ele for ele in multicoll_mat if ele not in multicoll_mtx]
print("Columns identified by Spearman but not Pearson: ",mcm_diff) # Some are useful, removing only

#Removing multicollinearity columns identified by Pearson Corr
bankrupt = bankrupt.drop(multicoll_mtx,axis=1)

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

#Income-to-Bankruptcy Graph
itob = px.histogram(x=bankrupt['TotalAssetGrowthRate'],
                    color=bankrupt['Bankrupt'],
                   log_y=True,
                   template='ggplot2',
                  title='Income VS Bankrupcy',
                  width=700)
itob.show()

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
# Some data quality problems are evident from this: 9 observations with Quick Ratio values that are clearly incorrect (in the thousands)
# Summary of Grouped Means - We can use this to look for data quality issues:

# Redundent Variables - Eliminate variables that are near identical with others




# Demo Run of Predictive Model Data Preparation
# 1: Remove Variables with large proportions of outliers:
bankrupt_trimmed = bankrupt.drop(outliers_le_99.columns, axis=1)

# 2: Impute Outliers with Conditional Means



# Splitting the data
# Splits: Training (60%), Validation (20%), Testing (20%)
# Dividing dataset into training (60%) and test (40%)
df_train, df_test = train_test_split(bankrupt,
                                     train_size=0.6,
                                     random_state=seed,
                                     stratify=bankrupt['Bankrupt'])
# Dividing testing data into validation (50%) and testing (50%)
df_val, df_test = train_test_split(df_test,
                                   train_size=0.5,
                                   random_state=seed,
                                   stratify=df_test['Bankrupt'])

# Resetting the index
df_train, df_val, df_test = df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)
# Shapes of new splits
## Train
pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=['# rows', '# columns'])
## Val
pd.DataFrame([[df_val.shape[0], df_val.shape[1]]], columns=['# rows', '# columns'])
## Test
pd.DataFrame([[df_test.shape[0], df_test.shape[1]]], columns=['# rows', '# columns'])

#Splitting feature and target matrices
# Get the feature matrix
X_train = df_train[np.setdiff1d(df_train.columns, ['Bankrupt'])].values
X_val = df_val[np.setdiff1d(df_val.columns, ['Bankrupt'])].values
X_test = df_test[np.setdiff1d(df_test.columns, ['Bankrupt'])].values

# Get the target vector
y_train = df_train['Bankrupt'].values
y_val = df_val['Bankrupt'].values
y_test = df_test['Bankrupt'].values

# Standardizing the features
# Standardize the training data
X_train = ss.fit_transform(X_train)
# Standardize the validation data
X_val = ss.transform(X_val)
# Standardize the test data
X_test = ss.transform(X_test)

# Handling class imbalance
# Imbalance class distribution
pd.Series(y_train).value_counts()

# TSNE Scatter Plot
def plot_scatter_tsne(X, y, classes, labels, colors, markers, loc, dir_name, fig_name, random_seed):

    # Make directory
    directory = os.path.dirname(dir_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the tsne transformed training feature matrix
    X_embedded = TSNE(n_components=2, random_state=random_seed).fit_transform(X)

    # Get the tsne dataframe
    tsne_df = pd.DataFrame(np.column_stack((X_embedded, y)), columns=['x1', 'x2', 'y'])

    # Get the data
    data = {}
    for class_ in classes:
        data_x1 = [tsne_df['x1'][i] for i in range(len(tsne_df['y'])) if tsne_df['y'][i] == class_]
        data_x2 = [tsne_df['x2'][i] for i in range(len(tsne_df['y'])) if tsne_df['y'][i] == class_]
        data[class_] = [data_x1, data_x2]

    # The scatter plot
    fig = plt.figure(figsize=(8, 6))

    for class_, label, color, marker in zip(classes, labels, colors, markers):
        data_x1, data_x2 = data[class_]
        plt.scatter(data_x1, data_x2, c=color, marker=marker, s=120, label=label)

    # Set x-axis
    plt.xlabel('x1')

    # Set y-axis
    plt.ylabel('x2')

    # Set legend
    plt.legend(loc='best')

    # Save and show the figure
    plt.tight_layout()
    plt.savefig(dir_name + fig_name)
    plt.show()
# Plot
plot_scatter_tsne(X_train,
                  y_train,
                  [0, 1],
                  ['0', '1'],
                  ['blue', 'green'],
                  ['o', '^'],
                  'bottom-left',
                  path,
                  'scatter_plot_baseline.pdf',
                  seed)
# Handling class imbalance in the target using SMOTE -
smote = SMOTE(random_state=seed)
# Augment the training data
X_smote_train, y_smote_train = smote.fit_resample(X_train, y_train)
# See new distributions
pd.Series(y_smote_train).value_counts()
# Separate generated class from original class
def separate_generate_original(X_aug_train, y_aug_train, X_train, y_train, minor_class):
    # Make a copy of y_aug_train
    y_aug_gen_ori_train = np.array(y_aug_train)

    # For each sample in the augmented data
    for i in range(X_aug_train.shape[0]):
        # If the sample has the minor class
        if y_aug_gen_ori_train[i] == minor_class:
            # Flag variable, indicating whether a sample in the augmented data is the same as a sample in the original data
            same = False

            # For each sample in the original data
            for j in range(X_train.shape[0]):
                # If the sample has the minor class
                if y_train[j] == minor_class:
                    if len(np.setdiff1d(X_aug_train[i, :], X_train[j, :])) == 0:
                        # The two samples are the same
                        same = True
                        break

            # If the two samples are different
            if same is False:
                y_aug_gen_ori_train[i] = 2

    return y_aug_gen_ori_train
y_smote_gen_ori_train = separate_generate_original(X_smote_train, y_smote_train, X_train, y_train, 1)
# Plot of new SMOTE classes
plot_scatter_tsne(X_smote_train,
                  y_smote_gen_ori_train,
                  [0, 1, 2],
                  ['0', '1', '+1'],
                  ['blue', 'green', 'red'],
                  ['o', '^', 's'],
                  'bottom-right',
                  path,
                  'scatter_plot_smote.pdf',
                  seed)

# PCA
# Standardize the training data
X_train = ss.fit_transform(X_train)
y_train = ss.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
# Standardize the validation data
X_val = ss.fit_transform(X_val)
y_val = ss.fit_transform(y_val.reshape(-1, 1)).reshape(-1)
# Standardize the test data
X_test = ss.fit_transform(X_test)
y_test = ss.fit_transform(y_test.reshape(-1, 1)).reshape(-1)

# Encoding Training data for PCA
X_t = X_train.reshape(-1)
l_e = preprocessing.LabelEncoder()
enc = l_e.fit_transform(X_t)

# Applying PCA on training
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print("Explained variation per Principal Component: {}".format(pca.explained_variance_ratio_))
# PC1 holds 13.92% of info; PC2 holds 7.6% of info

#PCA Plot
#PC Values
bank = pd.DataFrame(data=X_train,columns=['PC-1','PC-2'])
plt.figure()
plt.xlabel("PC-1")
plt.ylabel("PC-2")
plt.title("PCA")
targets = [0,1,2]
colors = ['r','b','g']
for target, color in zip(targets, colors):
    ind = bankrupt['Bankrupt']==target
    plt.scatter(bank.loc[ind,'PC-1'],bank.loc[ind,'PC-2'],c=color,s=50)
plt.legend(targets,prop={'size':20})
plt.show()