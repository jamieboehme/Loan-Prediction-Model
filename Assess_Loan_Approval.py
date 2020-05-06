import Credentials as c
import pyodbc                       # This is needed to connect to MSSQL database
import pandas as pd                 # For creating and manipulating data in dataframes
import numpy as np                  # For mathematical calculation
import matplotlib
import matplotlib.pyplot as plt     # For plotting graphs
import seaborn as sns               # For plotting graphs
import warnings                     # To ignore any warnings
warnings.filterwarnings('ignore')

#Global Characteristics
colors = ['mediumturquoise', 'blueviolet', 'mediumvioletred', 'palegreen']
SMALL_SIZE = 6
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)

# region Methods to be used to graph/manipulate loan data sets for analysis
def graph_cat(variable_cat_or_ord):
    values = train[variable_cat_or_ord].value_counts(normalize=True)
    percent = values * 100
    return percent.plot.bar(title = variable_cat_or_ord, color = colors, ylim=(0,100))

def graph_num(variable_num):
    if variable_num != 'LoanAmount':
        return sns.distplot(train[variable_num])
    else:
        train_notnull = train.dropna()
        return sns.distplot(train_notnull['LoanAmount'])

def graph_boxplot(variable_num):
    if variable_num != 'LoanAmount':
        return train[variable_num].plot.box()
    else:
        train_notnull = train.dropna()
        return train_notnull['LoanAmount'].plot.box()

def cross_graph(tabular_num):
    var = pd.crosstab(train[tabular_num], train['Loan_Status'])
    graph = var.div(var.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, title = '{} vs Loan Status'.format(tabular_num), color=colors)
    return graph

def replace_mode(data,cat_variable):
    return data[cat_variable].fillna(data[cat_variable].mode()[0], inplace = True)

def replace_median(data,num_variable):
    return data[num_variable].fillna(data[num_variable].median(), inplace=True)
# endregion

# region Connect to MS SQL Server where Data Sets are stored
print('Connecting to database...')
cnxn = pyodbc.connect(server=c.server,
                    database='mischief_managed',
                    user=c.uid,
                    tds_version='7.4',
                    password=c.pwd,
                    port=1433,
                    driver='FreeTDS',
                    ansi = True)
# endregion

# region Verify PyCharm can see the tables wanted to be used
print('Verify PyCharm can see the tables wanted to be used')

cursor = cnxn.cursor()
for table_name in cursor.tables(tableType='TABLE'):
    print(table_name)
print('\n')
print('All tables are showing as expected!')
wait = input("PRESS ENTER TO CONTINUE.")
# endregion

# region Query and store data sets from MS SQL into a pandas dataframe for use
print('Storing train and test loan data sets in pandas dataframes...')
# Query and store train data in a pandas dataframe
train_sql = "Select * from mischief_managed.dbo.train_loan_data"
cursor.execute(train_sql)
train_loan = cursor.fetchall()
train = pd.read_sql(train_sql, cnxn).astype({'Credit_History': 'float64'})

# Query and store test data in a pandas dataframe
test_sql = "Select * from mischief_managed.dbo.test_loan_data"
cursor.execute(test_sql)
test_loan = cursor.fetchall()
test = pd.read_sql(test_sql, cnxn)

# Create copies of datasets (just in case)
train_original = train.copy()
test_original = test.copy()

# endregion

# region Review training and testing data sets for Loan Data
print('Review training and testing data sets for Loan Data')

print('\nReview all column names in train data set \n', train.columns)
print('\nReview all column names in test data set \n', test.columns)
print('We can see the two data sets have all the same columns except the test data set does not have Loan Status. This makes sense, because we want to use the test data set to predict loan status and then compare it to what was actual loan given.')
wait = input("PRESS ENTER TO CONTINUE.")

print('\nRecord and column count in train data set \n', train.shape)
print('\nRecord and column count in test data set \n', test.shape)
wait = input("PRESS ENTER TO CONTINUE.")

print('\nData types in train data set\n', train.dtypes)
print('\nData types in test data set\n', test.dtypes)
print('It is helpful to know the data types of this data set in order to visualize the distribution of values within each variable.')
wait = input("PRESS ENTER TO CONTINUE.")
# endregion

# region Review 'Loan Status' for training data set
print('Review Loan Status for training data set')
print('\nCount approved/disapproved loans\n', train['Loan_Status'].value_counts(), sep='')
print('\nPercent approved/disapproved loans\n', train['Loan_Status'].value_counts(normalize=True), sep='')
wait = input("PRESS ENTER TO CONTINUE.")
# endregion

# region Review categorical and ordinal variables graphically
print('Review categorical and ordinal variables graphically')
graph_cat_variables = ['Loan_Status', 'Gender', 'Married', 'Self_Employed', 'Credit_History', 'Dependents', 'Education', 'Property_Area']

fig1 = plt.figure(figsize=(14,12))
plt.suptitle('Distribution of Categorical or Ordinal Variables')

plotnum=1
for cat_var in graph_cat_variables:
    plt.subplot(1,8,plotnum)
    graph_cat(cat_var)
    plotnum +=1
plt.show()
print('In the graphs I can begin to understand the types of applicants requesting for a loan. This will be helpful in determining if any of these variables may influence whether a loan is approved or not.')
wait = input("PRESS ENTER TO CONTINUE.")
# endregion

# region Review numerical variables graphically
print('Review numeric variables graphically')
graph_numeric_var = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']

fig = plt.figure(figsize=(14,12))
plt.suptitle('Distribution of Numerical Variables')

plotnum=1
for num_var in graph_numeric_var:
    plt.subplot(1,6,plotnum)
    graph_num(num_var)
    plotnum +=1
    plt.subplot(1,6,plotnum)
    graph_boxplot(num_var)
    plotnum +=1
plt.show()

print('In these graphs I can now better understand outliers I may want to account for in my loan predictions. This is important so I do not skew my predictions.')
wait = input("PRESS ENTER TO CONTINUE.")
# endregion

# region Review loan status against categorical variables graphically
print('Review loan status against categorical variables graphically')

graph_cat_vs_target = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
for cat_and_target in graph_cat_vs_target:
    cross_graph(cat_and_target)
plt.show()

print('By graphing categorical variables against loan status, now I can better see what may influence a loan being approved or not.\n'
      '\t - Applicants who live in semisuburban areas have a greater chance of loan approval.\n'
      '\t - Applicants with good credit history have greater chance of loan approval.\n'
      '\t - Applicant employment does not appear to greatly affect loan status.\n'
      '\t - Graduated applicants have a slightly greater chance of loan approval.\n'
      '\t - Applicant dependent count appears to have a slight affect on loan status, but I would maybe want more data to validate that.\n'
      '\t - Applicants marriage status does not appear to greatly affect loan status.\n'
      '\t - Applicant gender does not appear to greatly affect loan status.\n'
      'All in all, it appears the variables that deem to have the most influence on loan status are:\n'
      '\t - Where your property resides, and\n\t - If you have good Credit History')
wait = input("PRESS ENTER TO CONTINUE.")
# endregion

# region Evaluate mean income of loans that are approved vs. disapproved
print('Evaluate mean income of loans that are approved vs. disapproved')

train_loan_group1 = train.groupby(['Loan_Status'])['ApplicantIncome'].mean()
income_bins = [0,2500,4000,6000,8100]                      # create income bins
income_group = ['Low', 'Average', 'High', 'Very High']     # develop bin labels
train['Income_Group'] = pd.cut(train['ApplicantIncome'], income_bins, labels=income_group)
cross_graph('Income_Group')
plt.show()

print('This graph is really interesting. It appears that those with average income have the highest chance of obtaining loan approval.'
      'Where as those with very high income are the least likely of obtaining loan approval. If we think back to our distribution of loan amounts and incomes, I am wondering'
      'if some of those outliers are appearing here.')
wait = input("PRESS ENTER TO CONTINUE.")
# endregion

# region Evaluate mean of total income of loans that are approved vs. disapproved
print('Evaluate mean of co-applicant income of loans that are approved vs. disapproved')

train['Total_Income']= train['ApplicantIncome'] + train['CoapplicantIncome']
combined_bins = [0, 2500, 4000, 6000, 8100]
combined_group = ['Low', 'Averge', 'High', 'Very High']
train['Total_Income_Group'] = pd.cut(train['Total_Income'], combined_bins, labels = combined_group)
cross_graph('Total_Income_Group')

print('Here we can see those with a combined of applicant and coapplicant income of low are likely to not get approved for a loan.'
      'Where as those in the average, high, and very high total income groups are more likely.')
plt.show()
wait = input("PRESS ENTER TO CONTINUE.")
# endregion

# region Evaluate the loan amount vs approved and disapproved loans
print('Evaluate the loan amount vs approved and disapproved loans.')

loan_bins = [0, 100, 200, 700]
loan_groups = ['Low', 'Average', 'High']
train['LoanAmount_Group'] = pd.cut(train['LoanAmount'], loan_bins, labels = loan_groups)
cross_graph('LoanAmount_Group')

plt.show()
print('As expected, those loans with a greater loan amount being requested are less likely to be approved.')
wait = input("PRESS ENTER TO CONTINUE.")
# endregion

# region Visualize the correlation of variables within the train loan data set
print('Visualize the correlation of variables within the train loan data set.')

train = train.drop(['Income_Group', 'LoanAmount_Group', 'Total_Income_Group', 'Total_Income'], axis = 1)
matrix = train.corr()
f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(matrix, vmax=.8, square = True, cmap = 'BuPu')

print('By seeing what variables are most correlated, we can see what may impact loan status the most. As expected credit'
      'history appears to greatly impact loan status. As we hypothesized earlier too, we can see that applicant income'
      'greatly affects the loan amount being requested, and could be the reason why those in very high income brackets may '
      'not be getting approved as often as we would think.')
plt.show()
wait = input("PRESS ENTER TO CONTINUE.")
# endregion

# region Impute missing values in data sets
print('Impute missing values in train data set; for missing values, numerical values will be replaced with the mean, and categorical values will be replaced with the mode.')
print('\nTrain Data Set w/ Nulls \n', train.isnull().sum(), sep='')     # evaluate count of null values
cat_variables_w_nulls = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term']
for remove_null_in_var in cat_variables_w_nulls:
    replace_mode(train,remove_null_in_var)
replace_median(train, 'LoanAmount')
print('Train Data Set - w/o Nulls \n', train.isnull().sum(), sep='')

wait = input("PRESS ENTER TO CONTINUE.")

print('Test Data Set w/ Nulls \n', test.isnull().sum(), sep='')     # evaluate count of null values
for remove_null_in_var in cat_variables_w_nulls:
    replace_mode(test,remove_null_in_var)
replace_median(test, 'LoanAmount')
print('Test Data Set w/o Nulls \n', test.isnull().sum(), sep='')

wait = input("PRESS ENTER TO CONTINUE.")
# endregion

# region Remove outliers from Loan Amount by taking the log of Loan Amount
print('Removing outliers... graphing histogram of Loan Amount with Outliers removed.')
train['LoanAmount_log'] = np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)

plt.show()
print('Now that the outliers have been removed, we will continue to prep the data to begin modeling.')
wait = input("PRESS ENTER TO CONTINUE.")
# endregion

# region Prep data set to begin model development to predict loan status
print('Prepping data set to begin model development to predict loan status.\n'
      '\t - drop Loan ID from both data sets as it is not useful for prediction purposes\n'
      '\t - drop Loan Amount log and Loan Status from the training data set as it is not useful for prediction purposes\n'
      '\t - make dummy variables for categorical variables such that we can use them like numerical variables\n'
      '\t - divide data set into training and validation data sets')

train = train.drop('Loan_ID', axis = 1)
train = train.drop('LoanAmount_log', axis = 1)
test = test.drop('Loan_ID', axis =1)

# put target variable Loan Status in separate data set in order to use Sklearn
X = train.drop('Loan_Status', 1)
y = train.Loan_Status

# make dummy variables for categorical variables by changing them to numerical variables
X = pd.get_dummies(X)
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# divide data set into training and validation
from sklearn.model_selection import train_test_split
x_train, x_ds, y_train, y_ds = train_test_split(X,y, test_size = .3)

print('For my project I will be testing if logistic regression, decision tree, and/or random forest modeling is acceptable to predict loan status.')
# endregion

# region Create prediction model of loan status using logistic regression using the training data set
print('Test prediction of loan status using logistic regression modeling.')

from sklearn.linear_model import LogisticRegression # needed for modeling
from sklearn.metrics import accuracy_score # needed to measure accuracy
model = LogisticRegression()
model.fit(x_train, y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100,
                   multi_class='over', n_jobs=1, penalty='12', random_state=1, solver='liblinear',tol=.0001,
                   verbose=0, warm_start=False)

# 16f) Predict the loan_status for validation set and calculate accuracy
pred_ds = model.predict(x_ds)
initial_logistic_model_score = accuracy_score(y_ds, pred_ds)
print('Initial logistic model score = ', initial_logistic_model_score)

wait = input("PRESS ENTER TO CONTINUE.")
# endregion

# region Validate logistic regression model using stratified k-folds - 1 method to validate models
print('Test validation of logistic regression model using stratified k-fold cross validation. Stratification is the process of rearranging the data so as to ensure that each fold is a good representative of the whole.')
from sklearn.model_selection import StratifiedKFold

i = 1
running_sum = 0
kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
for train_index, test_index in kf.split(X, y):
    print('{} of kfold {}'.format(i, kf.n_splits), end='')
    xtr, xv1 = X.loc[train_index], X.loc[test_index]
    ytr, yv1 = y[train_index], y[test_index]
    model = LogisticRegression(random_state=1)
    model.fit(xtr, ytr)
    pred_test = model.predict(xv1)
    score = accuracy_score(yv1, pred_test)
    print('accuracy_score: ', score)
    i += 1
    running_sum = running_sum + score
mean_accuracy = running_sum / 5
pred_test = model.predict(test)
pred = model.predict_proba(xv1)[:, 1]
print('Initial stratified k-fold model score = ', mean_accuracy, '...we want this number to be close to 1.0')
wait = input("PRESS ENTER TO CONTINUE.")
# endregion

# region Generate AUC value and plot the ROC Curve. We would hope to get a curve
print('Visualization of AUC value on an ROC Curve.')

from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(yv1, pred)
auc = metrics.roc_auc_score(yv1, pred)
print('AUC value = ', auc)
plt.figure(figsize=(12,8))
plt.plot(fpr, tpr, label = 'validation, auc =' + str(auc))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 4)

plt.show()
wait = input("PRESS ENTER TO CONTINUE.")
# endregion

# region Engineer new variables to see if we can find a better model than our initial logistic regression model
print('Engineering new variables including Total Income, Total Income log, EMI, and Balance Income.')

train['Total_Income']= train['ApplicantIncome'] + train['CoapplicantIncome']
test['Total_Income']= test['ApplicantIncome'] + test['CoapplicantIncome']
train['Total_Income_log'] = np.log(train['Total_Income'])
test['Total_Income_log'] = np.log(test['Total_Income'])
train['EMI']= train['LoanAmount'] / train['Loan_Amount_Term']
test['EMI'] = test['LoanAmount'] / test['Loan_Amount_Term']
train['Balance_Income']= (train['Total_Income'] - train['EMI']) * 1000
test['Balance_Income']= (test['Total_Income'] - test['EMI']) * 1000
# endregion

# region Review engineered variables graphically
print('Review engineered variables graphically.')

feature_variables = ['Total_Income', 'Total_Income_log', 'EMI', 'Balance_Income']
fig = plt.figure(figsize=(16,16))
plt.suptitle('Distribution of Engineered Variables') #figure title
plotnum=1
for engineered_variables in feature_variables:
    plt.subplot(1,4,plotnum)
    graph_num(engineered_variables)
    plotnum +=1

plt.show()
print('These look similar to what we would expect given our initial variables and distributions.')
wait = input("PRESS ENTER TO CONTINUE.")
# endregion

# region Drop variables used to create engineered variables and review columns in the loan training and test data sets
print('Drop variables used to create engineered variables and review columns in the loan training and test data sets.')
train = train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis = 1)
test = test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis = 1)

print('Columns in loan training data set', train.columns)
print('Columns in loan testing data set', test.columns)

wait = input("PRESS ENTER TO CONTINUE.")
# endregion

# region Re-test logistic regression model using engineered variables to predict loan status
print('Re-test logistic regression model using engineered variables.')
X = train.drop('Loan_Status', 1)
y = train.Loan_Status

j=1
running_sum = 0
kf = StratifiedKFold(n_splits = 5, random_state=1, shuffle = True)
for train_index, test_index in kf.split(X,y):
    print('{} of kfold {} '.format(j,kf.n_splits), end ='')
    xtr,xv1 = X.loc[train_index], X.loc[test_index]
    ytr,yv1 = y[train_index], y[test_index]
    model = LogisticRegression(random_state=1)
    model.fit(xtr, ytr)
    pred_test = model.predict(xv1)
    score = accuracy_score(yv1, pred_test)
    print('accuracy_score: ',score)
    j+=1
    running_sum = running_sum + score
print('Accuracy of logistic regression with engineered values', running_sum/5)
pred_test = model.predict(test)
pred = model.predict_proba(xv1)[:,1]
print('This is actually lower, so my engineered variables were not helpful for modeling our prediction using logistic regression')

wait = input("PRESS ENTER TO CONTINUE.")
# endregion

# region Test prediction of loan status using decision tree modeling
print('Test prediction of loan status using decision tree modeling.')
from sklearn import tree

i=1
running_sum = 0
kf = StratifiedKFold(n_splits = 5, random_state=1, shuffle = True)
for train_index, test_index in kf.split(X,y):
    print('{} of kfold {}'.format(i,kf.n_splits), end ='')
    xtr,xv1 = X.loc[train_index], X.loc[test_index]
    ytr,yv1 = y[train_index], y[test_index]
    model = tree.DecisionTreeClassifier(random_state=1)
    model.fit(xtr, ytr)
    pred_test = model.predict(xv1)
    score = accuracy_score(yv1, pred_test)
    print(' accuracy_score: ',score)
    i+=1
    running_sum = running_sum + score
print('Accuracy using decision tree model =', running_sum /5)
pred_test = model.predict(test)
pred = model.predict_proba(xv1)[:,1]
print('Decision tree modeling appears to be the worst model to predict loan status thus far.')

wait = input("PRESS ENTER TO CONTINUE.")
# endregion

# region Test prediction of loan status using random forest modeling
print('Test prediction of loan status using random forest modeling.')
from sklearn.ensemble import RandomForestClassifier

i=1
running_sum = 0
kf = StratifiedKFold(n_splits = 5, random_state=1, shuffle = True)
for train_index, test_index in kf.split(X,y):
    print('{} of kfold {}'.format(i,kf.n_splits), end ='')
    xtr,xv1 = X.loc[train_index], X.loc[test_index]
    ytr,yv1 = y[train_index], y[test_index]
    model = RandomForestClassifier(random_state=1, max_depth=10)
    model.fit(xtr, ytr)
    pred_test = model.predict(xv1)
    score = accuracy_score(yv1, pred_test)
    print(' accuracy_score: ',score)
    i+=1
    running_sum = running_sum + score
print('Accuracy using random forest model', running_sum / 5)
pred_test = model.predict(test)
pred = model.predict_proba(xv1)[:,1]
print('Interestingly, random forest modeling has given us the second best model to predict if a loan will be approved or not. We can further dimension random forest modeling by using a second method to determine the depth and estimates to be used in the model. We will do that next.')

wait = input("PRESS ENTER TO CONTINUE.")
# endregion

# region Determine if we can modify the random forest model by adjusting the depth and estimators of the model
print('Determining if we can modify the random forest model by adjusting the depth and estimators of the model.')
print('This may take awhile to compute...')
from sklearn.model_selection import GridSearchCV
paramgrid = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}
grid_search = GridSearchCV(RandomForestClassifier(random_state=1), paramgrid)

from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size= .3, random_state=1)
grid_search.fit(x_train, y_train)
GridSearchCV(cv = None, error_score= 'raise', estimator=RandomForestClassifier(bootstrap=True,
             class_weight= None, criterion='gini', max_depth=None, max_features='auto',
             max_leaf_nodes= None, min_impurity_decrease=0.0, min_impurity_split= None,
             min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=10, n_jobs=1, oob_score=False, random_state=1, verbose=0, warm_start=False),
             fit_params=None, iid = True, n_jobs=1, param_grid = {'max_depth': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
             'n_estimators': [1, 21, 41, 61, 81, 101, 121, 141, 161, 181]},pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
             scoring=None, verbose=0)

print(grid_search.best_estimator_) #max_depth = 3, n_estimators =81
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=3, max_features='auto',
                       max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,
                       n_estimators=81, n_jobs=1, oob_score=False, random_state=1, verbose=0, warm_start=False)

print('Best estimate can be found using max_depth = 3, and n_estimators = 81 given grid search.')
i = 1
running_sum = 0
kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
for train_index,test_index in kf.split(X,y):
    print('{} of kfold {}'.format(i, kf.n_splits), end = ' ')
    xtr,xv1 = X.loc[train_index],X.loc[test_index]
    ytr, yv1 = y[train_index], y[test_index]
    model = RandomForestClassifier(random_state=1, max_depth=3, n_estimators=81)
    model.fit(xtr,ytr)
    pred_test = model.predict(xv1)
    score = accuracy_score(yv1, pred_test)
    print('accuracy_score', score)
    i +=1
    running_sum = running_sum + score
print('Accuracy using more informed random forest model =', running_sum / 5)
print('The random forest modeling using max_depth = 3 and n_estimators = 81 is the second best model to predict whether a loan will be approved or not.')
pred_test = model.predict(test)
pred2 = model.predict_proba(test)[:, 1]
wait = input("PRESS ENTER TO CONTINUE.")
# endregion

# region Plot variables that deem to influence the model the most when doing random forest modeling using sklearn
print('Lastly, I want to plot the variables that deem to influence the random forest model the most using sklearn.')
importances=pd.Series(model.feature_importances_,index=X.columns)
importances.plot(kind='barh', figsize=(8,8))
plt.show()

print('As seen in the correlation matrix and proved here, credit history has significant importance on our models.'
      'However, it is interesting to see that some of the engineered variables like Total Income, Balance Income, and EMI'
      'way into the model. Lastly, Property Area is also showing to have an impact on the decision of loan status in this'
      'model as we saw in our original graphs.')

print('Following all the modeling tests, I would use logistic regression followed by random forest modeling to predict '
      'whether a loan would be approved or not.')
# endregion
