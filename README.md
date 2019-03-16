# scikit-learn_classification
Python scripts to demonstrate data classification with various methods using the scikit-learn library.Classifiers used are:
- Support Vector Machines
- Random Forests
- Multinomial Naive Bayes
- K-Nearest Neighbor (my implementation using scikit-learn API)


Each script's functionality is described below:

- **classifier_metrics.py:**
~~~
python classifier_metrics.py <train_data> <test_data>
~~~
>script creating the aforementioned classifiers and calculating various metrics for each one using cross validation.
Metric results are printed to an output .csv file .
Afterwards it uses the best classifier to predict the class of the test data points ,printing the results to an output.csv file.

- **data_prepr.py**
>gets imported.Used for data preprocessing .

- **lsi_accuracy_plot.py**
~~~
python lsi_accuracy_plot.py <train_data>
~~~
>Creates a plot presented in a .png image showing the correlation between accuracy and lsi_components using the Multinomian Naive Bayes Classifier

- **my_knn.py**
>gets imported.My implementetion of KNN using the scikit API.

- **svm_parameter_tuning.py:**
~~~
python svm_parameter_tuning.py <train_data>
~~~
>Performs gridsearch to find the best combination of hyperparameters (kernel,gamma,C) for SVC Classifier.
Results get printed on screen but are also saved as a python dictionary at **hyperparameter_values.py**.
