"""
LOAN CHALLENGE

ASSUMPTION #1: People of the same gender, who share a similar educational background, and live in a similar conditions lead a similar life style. Therefore, missing data is filled with grouped medians.
"""

#LIBRARIES
import pandas as pd 
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np

#FUNCTIONS

def status(feature):
	print 'PROCESSING', feature, ' : COMPLETED'
	
def combine_data():
	train = pd.read_csv('train.csv')
	test = pd.read_csv('test.csv')

	L = train.Loan_Status
	train.drop('Loan_Status', 1, inplace = True)
	
	labels = L.map({'Y':1,'N':0})
	
	combined = train.append(test)
	combined.reset_index(inplace = True)
	combined.drop('index', 1, inplace = True)
	status('dataset')
	return combined, labels
	
def process_gender():
	
	global combined
	
	combined['Gender'] = combined['Gender'].map({'Male':1,'Female':0})
	combined['Gender'].fillna(1, inplace = True)
	
	status('Gender')
	
def process_marital_status():
	
	global combined
	
	combined.Married.fillna('No', inplace = True)
	
	combined['Married'] = combined['Married'].map({'Yes':1,'No':0})
		
	status('Marital Status')
	
def process_dependants():
	
	global combined
	
	dep_dummies = pd.get_dummies(combined['Dependents'], prefix = 'Dep')
	
	combined = pd.concat([combined,dep_dummies], axis = 1)
	
	combined.drop('Dependents', axis = 1, inplace = True)
	
	status('Dependents')
	
def process_education():
	
	global combined
	
	combined['Education'] = combined['Education'].map({'Graduate':1,'Not Graduate':0})
		
	status('Education')
		
def	process_Self_Employed():
	
	global combined
	
	combined['Self_Employed'].fillna('No', inplace = True)
	
	combined['Self_Employed'] = combined['Self_Employed'].map({'Yes':1,'No':0})
		
	status('Self_Employed')
		
def process_Property_Area():

	global combined
	
	area_dummies = pd.get_dummies(combined['Property_Area'], prefix = 'Area')
	
	combined = pd.concat([combined, area_dummies], axis = 1)
	
	combined.drop('Property_Area', axis = 1, inplace = True)
	
	status('Property_Area')
	
def process_loan_amount():
	
	global combined
	
	table = combined.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
	
	def fage(x):
		return table.loc[x['Self_Employed'],x['Education']]
		
	combined['LoanAmount'].fillna(combined[combined['LoanAmount'].isnull()].apply(fage, axis = 1), inplace = True)
	
	status('Loan Amount')
	
def process_applicant_income():

	global combined
	
	combined['total_income'] = combined['ApplicantIncome'] + combined['CoapplicantIncome']
	
def add_tolerance():

	global combined

	combined['tolerance'] = combined['total_income']/combined['LoanAmount']
	 
def process_credits():
	
	global combined 
	
	combined.Credit_History.fillna(0 , inplace = True)
		
def process_loan_amount_term():
	combined.Loan_Amount_Term.fillna(360, inplace = True)
		

# PROCESSING DATA

combined, train_labels = combine_data()

process_gender()
process_marital_status()
process_dependants()
process_education()
process_Self_Employed()
process_Property_Area()
process_loan_amount()
process_applicant_income()
add_tolerance()
process_credits()
process_loan_amount_term()

Loan_ID = combined.Loan_ID.iloc[614:] 
combined.drop(['Loan_ID'], 1, inplace = True)
train_features = combined.head(614)


#IMPORTING LIBRARIES FOR SPOT TESTING
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectFromModel

'''
# SPOT TESTING

models=[]
models.append(('LR',LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('Random', RandomForestClassifier(n_estimators = 50, max_features = 'sqrt') ))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []

test_size = 0.20
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_features, train_labels, test_size=test_size,random_state=seed)

for name, model in models:
	kfold = model_selection.KFold(n_splits=10,random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
	print name, " = ", cv_results.mean()
'''	

#DETERMINING FEATURE IMPORTANCE
clf = RandomForestClassifier(n_estimators = 50, max_features = 'sqrt')
clf.fit(train_features, train_labels)
features = pd.DataFrame()
features['feature'] = train_features.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'],ascending = True, inplace = True)
features.set_index('feature', inplace = True)

model = SelectFromModel(clf, prefit=True) 
train_reduced = model.transform(train_features)

test_features = combined.iloc[614:]
test_reduced = model.transform(test_features)
	
##FINAL MODEL
clf = LogisticRegression()
clf.fit(train_features, train_labels)
pred = clf.predict(test_features)

#END OF CODE