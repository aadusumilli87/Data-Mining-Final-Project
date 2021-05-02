# Import Relevant Libraries
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QMessageBox
)
from PyQt5.uic import loadUi
from GUI_BR import Ui_MainWindow
import webbrowser

# Initialization
raw = pd.read_csv("data.csv", index_col=0)
clean = pd.read_csv("bankrupt_clean.csv", index_col=0)
corr_mat = clean.corr('spearman')
LPM_Results = pd.read_csv('LPM_results.csv', index_col=0)
PCA_Var_Weight = pd.read_csv('PCA_VariableWeight.csv')
PCA_Var_Weight.set_index('Variables', inplace=True)

# Operate GUI Window
class MyBR_GUI(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(MyBR_GUI,self).__init__()
        self.setupUi(self)
        #self.showMaximized()
        self.pushButton_click()
        self.aktions()
        self.klose()
        self.data_r()
        self.data_c()

# Implement pushbutton_click() function
    def pushButton_click(self):
        self.pb_PCA.clicked.connect(dpca)
        self.pB_LPM.clicked.connect(dlpm)
        self.pB_Probit.clicked.connect(dpro)
        self.pB_CHMap.clicked.connect(dchm)
        self.pb_PredModelResults.clicked.connect(dpre)

# Implement actions
    def aktions(self):
        self.actionCM_Complete.triggered.connect(dchm)
        self.actionVariables.triggered.connect(dvar)
        self.actionLPM_Results.triggered.connect(dlpm)
        self.actionProbit_Results.triggered.connect(dpro)
        self.actionVar_List.triggered.connect(dvar)
        self.actionVar_SS.triggered.connect(dvin)
        self.actionData_Shape.triggered.connect(dshp)
        self.actionData_Size.triggered.connect(dsiz)
        self.actionCM_Negative.triggered.connect(acmn)
        self.actionCM_Positive.triggered.connect(acmp)
        self.actionIP_ClassImbal.triggered.connect(ipci)
        self.actionIP_Grouped_Means.triggered.connect(ipgm)
        self.actionIP_PCA_FComp.triggered.connect(ippc)
        self.actionIP_Debt_Ratio_to_Bankruptcy.triggered.connect(ipdb)
        self.actionIP_Debt_Ratio_vs_Asset_Turnover.triggered.connect(ipda)
        self.actionView_Git.triggered.connect(vgit)
        self.actionView_Report.triggered.connect(vrep)
        self.actionView_Kaggle.triggered.connect(vkag)

# Implement Window CLosing
    def klose(self):
        self.actionClose.triggered.connect(self.close)

# Implement Dialog Window - Raw
    def data_r(self):
        self.actionRaw_Data.triggered.connect(draw)
        self.actionRaw_Data.triggered.connect(raw_dial)

# Implement Dialog Window - Clean
    def data_c(self):
        self.actionClean_Data.triggered.connect(dcle)
        self.actionClean_Data.triggered.connect(clean_dial)

# PCA Results Viewer
def dpca():
    PCA_Var_Weight = pd.read_csv('PCA_VariableWeight.csv')
    PCA_Var_Weight.set_index('Variables', inplace=True)
    print("\n \n")
    print("PCA Variables Weight:")
    print(PCA_Var_Weight)
    print("*"*75)

# LPM Results Viewer
def dlpm():
    LPM_Results = pd.read_csv('LPM_results.csv', index_col=0)
    print("\n \n")
    print("LPM Results:")
    print(LPM_Results)
    print("*"*75)

# Probit Results Viewer
def dpro():
    Probit_Results = pd.read_csv('Probit_Results.csv', index_col=0)
    print("\n \n")
    print("Probit Results:")
    print(Probit_Results)
    print("*" * 75)

# Correlation Heatmap Viewer
def dchm():
    corr_mat = clean.corr('spearman')
    f_heatmap, ax_heatmap = plt.subplots(figsize=(28, 25))
    color_scheme = sns.diverging_palette(240, 10, as_cmap=True)
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(corr_mat, mask=mask, cmap=color_scheme, vmax=1, center=0, square=True, linewidths=0.5)
    plt.title('Bankruptcy Data: Correlation Heatmap', fontsize=40)
    plt.show()

# Predictive Models Results Viewer
def dpre():
    modres = pd.read_csv("models results.csv",index_col=0)
    print("\n \n")
    print("Shallow Machine Learning Models' Results:")
    print(modres)
    print("*" * 75)

# Raw Data Head Viewer
def draw():
    print("\n \n")
    print("First 5 Rows of Raw Data:")
    print(raw.head())
    print("*" * 75)

# Raw Data MessageBox Dialog
def raw_dial():
    qm = QMessageBox()
    qm.setText("Data Imported: Raw Data")
    qm.setWindowTitle("Import Data")
    retval = qm.exec_()

# Clean Data Head Viewer
def dcle():
    print("\n \n")
    print("First 5 Rows of Clean Data:")
    print(clean.head())
    print("*" * 75)

# Clean Data MessageBox Dialog
def clean_dial():
    qn = QMessageBox()
    qn.setText("Data Imported: Clean Data")
    qn.setWindowTitle("Import Data")
    retvall = qn.exec_()

# Var List Viewer
def dvar():
    print("\n \n")
    print("Data Variables:")
    print(raw.info())
    print("*" * 75)

# Var Info Viewer
def dvin():
    print("\n \n")
    print("Data Variables' Summary Statistics:")
    print(raw.describe())
    print("*" * 75)

# Data Shape Viewer
def dshp():
    print("\n \n")
    print("Dataset Shape:")
    print(raw.shape)
    print("*" * 75)

# Data Size Viewer
def dsiz():
    print("\n \n")
    print("Dataset Size:")
    print(raw.size)
    print("*" * 75)

# Negative Corr Map Viewer
def acmn():
    corrNeg = corr_mat.iloc[:, 0].sort_values(ascending=True).iloc[0:53]
    print("\n \n")
    print("Negative Correlations with Bankruptcy:")
    print(corrNeg)
    print("*" * 75)

# Positive Corr Map Viewer
def acmp():
    corrPos = corr_mat.iloc[:, 0].sort_values(ascending=False).iloc[0:18]
    print("\n \n")
    print("Positive Correlations with Bankruptcy:")
    print(corrPos)
    print("*" * 75)

# Imbalance Plot Viewer
def ipci():
    f_imbalance, ax_imbalance = plt.subplots(figsize=(7, 7))
    sns.countplot(clean['Bankrupt'])
    plt.title('Bankruptcy Counts \n 0 = Not Bankrupt || 1 = Bankrupt')
    plt.show()

# Grouped Means Viewer
def ipgm():
    bankrupt_means = clean.groupby('Bankrupt').mean()
    # Filter on significant variables
    means_index = LPM_Results.head(5).index
    means_index = [i for i in means_index if i != 'Constant']
    bankrupt_means = bankrupt_means.loc[:, means_index]
    # Reshape
    bankrupt_means.reset_index(inplace=True)
    bankrupt_means = pd.melt(bankrupt_means, id_vars='Bankrupt', value_vars=means_index)
    # Plot
    f_bar, ax_bar = plt.subplots(figsize=(10, 10))
    sns.barplot(x='variable', y='value', hue='Bankrupt', data=bankrupt_means)
    plt.show()

# PCA Feature Component Viewer
def ippc():
    f_pca, ax_pca = plt.subplots(figsize=(12, 12))
    sns.heatmap(PCA_Var_Weight, cmap='bwr')
    plt.title('Visualizing Component Structure: \n What Factors Contribute to Bankruptcy?')
    plt.show()

# Debt-Ratio-to-Bankruptcy Viewer
def ipdb():
    f_debt, ax_debt = plt.subplots(figsize=(7, 7))
    sns.scatterplot(x='Operatingprofit/Paid-incapital', y='Totaldebt/Totalnetworth', hue='Bankrupt',
                    data=clean)
    plt.show()

# Debt-Ratio-to-Asset-Turnover Viewer
def ipda():
    f_turn, ax_turn = plt.subplots(figsize=(7, 7))
    sns.scatterplot(x='Totaldebt/Totalnetworth', y='TotalAssetTurnover', hue='Bankrupt', data=clean)
    plt.show()

# Redirect to GitRepo
def vgit():
    webbrowser.open("https://github.com/aadusumilli87/Data-Mining-Final-Project")

# Redirect to Git Report
def vrep():
    webbrowser.open("https://github.com/aadusumilli87/Data-Mining-Final-Project/blob/main/Report.pdf")

# Redirect to Kaggle Competition
def vkag():
    webbrowser.open("https://www.kaggle.com/fedesoriano/company-bankruptcy-prediction/code")

# GUI RUNNER
if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_pyqt_form = MyBR_GUI()
    my_pyqt_form.show()
    sys.exit(app.exec_())