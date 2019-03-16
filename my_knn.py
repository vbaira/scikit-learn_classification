from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.base import BaseEstimator
from collections import Counter
import numpy as np
from heapq import nsmallest

#Brute force Knn estimator(compatible with scikit api)
class MyKNN(BaseEstimator):

	def __init__(self,k=1):
		self.k = k

	def fit(self,X,y):
		#check that X and y have correct shape
		X, y = check_X_y(X,y)
		#store the classes seen during fit
		self.classes_ = unique_labels(y)
		self.X_ = X
		self.y_ = y
		#return the classifier
		return self


	def predict(self, X):
		#check is fit had been called
		check_is_fitted(self, ['X_', 'y_'])
		#input validation
		X = check_array(X)
		#predict class for each test instance
		predictions=[]
		for test_instance in X:
			k_nearest = self.__get_k_nearest_neighbours(test_instance)
			dist,neighb_class = zip(*k_nearest) #unzip
			predicted_class = self.__majority_vote(neighb_class)
			predictions.append(predicted_class)
		return predictions


	#return k nearest neighbours in a list of (neighbour_distance,neighbour_class) tuples
	def __get_k_nearest_neighbours(self,test_instance):
		#distances of test instance from all other data points
		distances = euclidean_distances(self.X_,[test_instance])
		#zip the distance from a point with the class of that point 
		zipped = zip(distances,self.y_)
		#return k smallest distances and their corresponding class
		return nsmallest(self.k,zipped)


	#get a list of neighbour classes and return the most common in the list
	def __majority_vote(self,k_neighbours):
		count = Counter(k_neighbours)
		return count.most_common()[0][0]


	