import sys
from sklearn.svm import SVC
from sklearn.model_selection import KFold,cross_validate,GridSearchCV
from data_prepr import data_preprocessing
import numpy as np


def main():
	#if dataset is not provided on call terminate
	if len(sys.argv)<2:
		print("usage: python svm_parameter_tuning.py <input_file> ")
		sys.exit()

	#pass dataset and get the matrix containing the data vectors and data targets
	ret_value=data_preprocessing(sys.argv[1])
	data_matrix=ret_value[0]
	category_labels=ret_value[1]

	#create k_fold iterator to calculate metrics
	k_fold = KFold(n_splits=10)

	#perform grid search to determine parameter tuning
	c_range = [np.power(2.0,i) for i in range(-5, 10)]
	gamma_range = [np.power(2.0,i) for i in range(-10, -5)]
	param_grid = [{'kernel': ['rbf'], 'gamma': gamma_range,'C':c_range},{'kernel': ['linear'], 'C': c_range}]
	clf = GridSearchCV(SVC(),param_grid,cv=k_fold,scoring='accuracy',n_jobs=-1)
	clf.fit(data_matrix,category_labels)

	#print chosen hyperparameters
	print "Best accuracy achieved:"+ str(clf.best_score_) + " with below settings."
	for key,value in clf.best_params_.iteritems():
		print key + ":" + str(value)
	#save best hyperparameter values on a dictionary in file hyperparameter_values.py
	output=open('./hyperparameter_values.py','w')
	output.write('HYPERPARAMETER_VALUES={')
	first=True
	for key,value in clf.best_params_.iteritems():
		if first==True:
			output.write("\'"+key+"\':")
			first=False
		else:
			output.write(",\'"+key+"\':")

		if isinstance(value,str):
			output.write("\'"+value+"\'")
		else:
			output.write(str(value))
	output.write('}')



if __name__ == '__main__':
	main()