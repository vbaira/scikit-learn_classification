import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import RepeatedStratifiedKFold,StratifiedKFold,cross_validate
from sklearn import preprocessing
from my_knn import MyKNN
from data_prepr import data_preprocessing,btb_data_preprocessing
from hyperparameter_values import HYPERPARAMETER_VALUES 


def main():
	#if dataset is not provided on call terminate
	if len(sys.argv)<3:
		print("usage: python classifier_metrics.py <train_data_file> <test_data_file> ")
		sys.exit()

	#pass dataset and get the matrix containing the data vectors and data targets
	ret_value=data_preprocessing(sys.argv[1])
	data_matrix=ret_value[0]
	category_labels=ret_value[1]

	#create stratified k_fold iterator to calculate metrics
	k_fold = RepeatedStratifiedKFold(n_splits=10,n_repeats=3)
	sk_fold = StratifiedKFold(n_splits=10)
	metrics=['accuracy','precision_weighted','recall_weighted','f1_weighted']

	#create RandomForest classifier and calculate metrics
	rf_clf = RandomForestClassifier(n_jobs=-1)
	rf_result=cross_validate(rf_clf,data_matrix,category_labels, cv=k_fold,scoring=metrics,return_train_score=False, n_jobs=-1) 
	print "RANDOM FOREST:"
	for key,value in rf_result.iteritems():
		print key + " : " + str(np.round_(np.mean(value),decimals=5))
	print("\n")

	#create MNB classifier and calculate metrics
	#scale data matrix to positive values because MNB does not accept negative values
	#increasing scaling range increases accuracy up until scale is around 10
	scaler = preprocessing.MinMaxScaler(feature_range=(0,10),copy=True)
	scaled_data_matrix = scaler.fit_transform(data_matrix)
	mnb_clf = MultinomialNB()
	mnb_result=cross_validate(mnb_clf,scaled_data_matrix,category_labels, cv=k_fold,scoring=metrics,return_train_score=False, n_jobs=-1) 
	print "MULTINOMIAL NAIVE BAYES:"
	for key,value in mnb_result.iteritems():
		print key + " : " + str(np.round_(np.mean(value),decimals=5))
	print("\n")
	
	#load hyperparameters for svc classifier from file hyperparameter_values.py
	kernel_hp = HYPERPARAMETER_VALUES['kernel']
	c_hp = HYPERPARAMETER_VALUES['C']
	gamma_hp = 'auto'
	if kernel_hp == 'rbf':
		gamma_hp = HYPERPARAMETER_VALUES['gamma']
	#create svc classifier and calculate metrics
	svc_clf = SVC(kernel=kernel_hp, C=c_hp, gamma=gamma_hp) 
	svc_result=cross_validate(svc_clf,data_matrix,category_labels, cv=k_fold,scoring=metrics,return_train_score=False, n_jobs=-1) 
	print "svm.SVC (kernel="+kernel_hp+" ,C="+str(c_hp)+", gamma="+str(gamma_hp)+")"
	for key,value in svc_result.iteritems():
		print key +" : "+ str(np.round_(np.mean(value),decimals=5))
	print("\n")

	#create KNN(my implementation) classifier and calculate metrics
	knn_clf = MyKNN(k=10)
	knn_result=cross_validate(knn_clf,data_matrix,category_labels, cv=sk_fold,scoring=metrics,return_train_score=False)
	print "My implementation of KNN(brute force):"
	for key,value in knn_result.iteritems():
		print key +" : "+ str(np.round_(np.mean(value),decimals=5))
	print("\n")

	#Beat the benchmark
	TITLE_WEIGHT= 5
	#preprocess the data differently to achieve better score
	btb_ret_value=btb_data_preprocessing(sys.argv[1],title_weight=TITLE_WEIGHT,n_comp=250,ret_vectorizers=True)
	btb_data_matrix=btb_ret_value[0]
	btb_category_labels=btb_ret_value[1]

	#i chose svc because it was the better scoring classifier
	#calculate metrics
	btb_clf = SVC(kernel=kernel_hp, C=c_hp, gamma=gamma_hp,class_weight='balanced', probability=False)
	btb_result=cross_validate(btb_clf,btb_data_matrix,btb_category_labels, cv=k_fold,scoring=metrics,return_train_score=False, n_jobs=-1) 
	print "(Beat the benchmark)svm.SVC (kernel="+kernel_hp+" ,C="+str(c_hp)+", gamma="+str(gamma_hp)+")"
	for key,value in btb_result.iteritems():
		print key +" : "+ str(np.round_(np.mean(value),decimals=5))
	print("\n")

	#train the classifier with the train data
	#cross_validate() does not train the classifier object passed to it but a copy of it
	btb_clf.fit(btb_data_matrix,btb_category_labels)#refit

	#get the vectorizers and transformers used to fit and trasform the train data
	count_vectorizer=btb_ret_value[2]
	tfidf_transformer=btb_ret_value[3]
	svd=btb_ret_value[4]
	le=btb_ret_value[5]
	#read test data and trasform them using the above vectorizers/transformers
	test_data=pd.read_csv(sys.argv[2],sep="\t")
	test_redu_matrix = test_data_transformation(test_data,count_vectorizer,tfidf_transformer,svd,TITLE_WEIGHT)
	
	#do the class predictions for the test data
	test_category_pred = btb_clf.predict(test_redu_matrix)

	#store predictions to file
	create_pred_file(test_data,test_category_pred,le)

	#store metrics to file
	create_eval_file(mnb_result,rf_result,svc_result,knn_result,btb_result,metrics)	


#transform test data to a vector array using the given vectorizers/transformers
def test_data_transformation(test_data,count_vectorizer,tfidf_transformer,svd,title_weight):
	test_count_matrix = count_vectorizer.transform( title_weight*(test_data['Title']+" ") + test_data['Content'] )
	test_tfidf_matrix = tfidf_transformer.transform(test_count_matrix)
	test_redu_matrix  = svd.transform(test_tfidf_matrix)
	return test_redu_matrix


#create testSet_categories.csv file containing the predictions
def create_pred_file(test_data,test_category_pred,le):
	temp=[]
	for index,pred_cat in enumerate(le.inverse_transform(test_category_pred)):
		id_tag=(test_data.loc[index,['Id']]).values[0]
		temp.append([id_tag,pred_cat])
	df = pd.DataFrame(temp,columns=['Id','Category'])
	df.to_csv('testSet_categories.csv',sep="\t",index=False)


#create EvaluationMetric_10fold.csv file and store metrics there
def create_eval_file(result1,result2,result3,result4,result5,metrics):
	result_data=[]
	for met in metrics:
		key='test_'+met
		result_row = map( lambda x:np.round_(np.mean(x),decimals=5) , [ result1[key], result2[key], result3[key], result4[key], result5[key] ])
		result_data.append(result_row)
	result_df = pd.DataFrame(result_data,index=['Accuracy','Precision','Recall','F-Measure'],columns=['Naive Bayes','Random Forest','SVM','KNN','My Method'])
	result_df.to_csv('EvaluationMetric_10fold.csv',index=True,header=True,sep='\t')



if __name__ == '__main__':
	main()