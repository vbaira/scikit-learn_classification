import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from data_prepr import data_preprocessing


def main():
	#if dataset is not provided on call terminate
	if len(sys.argv)<2:
		print("usage: python lsi_accuracy_plot.py <input_file> ")
		sys.exit()

	#pass dataset and get the matrix containing the data vectors and data targets
	ret_value=data_preprocessing(sys.argv[1],lsi=False)
	data_tfidf_matrix=ret_value[0]
	category_labels=ret_value[1]

	#create k_fold iterator to calculate metrics
	k_fold = KFold(n_splits=10)

	#try lsi for different n_component values and find accuracy for each one to plot a graph
	accuracy_values=[]
	n_comp_values=[]
	for n_comp in range(2,300):
		svd = TruncatedSVD(n_components=n_comp)
		data_redu_matrix = svd.fit_transform(data_tfidf_matrix)

		#create MNB classifier and calculate metrics
		#scale data matrix to positive values because MNB does not accept negative values
		#increasing scaling range increases accuracy up until scale is around 10
		scaler = preprocessing.MinMaxScaler(feature_range=(0,10),copy=True)
		scaled_data_matrix = scaler.fit_transform(data_redu_matrix)
		mnb_clf = MultinomialNB()
		result=cross_validate(mnb_clf,scaled_data_matrix,category_labels, cv=k_fold,scoring='accuracy',return_train_score=False, n_jobs=-1)
		accuracy_values.append( (np.round_(np.mean(result['test_score']),decimals=5)) *100 )
		n_comp_values.append(n_comp)

	#plot the graph
	plt.plot(n_comp_values,accuracy_values)
	plt.xlabel("components")
	plt.ylabel("accuracy")
	plt.savefig("./plot.png")
	plt.show()





if __name__ == '__main__':
	main()