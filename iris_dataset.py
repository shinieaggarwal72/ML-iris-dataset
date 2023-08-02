from sklearn.datasets import load_iris
iris_dataset = load_iris()
#print("Keys of iris dataset: \n{}".format(iris_dataset.keys()))

#print("Target names: \n{}".format(iris_dataset['target_names']))
#print("Feature names: \n{}".format(iris_dataset['feature_names']))
#print("type of data: {}".format(type(iris_dataset['data'])))
#print("Shape of data: {}".format(iris_dataset['data'].shape))
#print("First five columns of data: \n{}".format(iris_dataset['data'][:5]))
#print("Type of target: {}".format(type(iris_dataset['target'])))
#print("Shape of target: {}".format(iris_dataset['target']))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

#print("X_train shape: {}".format(X_train.shape))
#print("y_train shape: {}".format(y_train.shape))
#print("X_test shape: {}".format(X_test.shape))
#print("y_test shape: {}".format(y_test.shape))

#create dataframe from data in X_train
#label the columns using strings in iris_dataset.feature_names
import pandas as pd
iris_dataframe = pd.DataFrame(X_train, columns = iris_dataset.feature_names)

#create scatter matrix from dataframe , color by y_train
import mglearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Combine the features and target variable into one DataFrame
iris_combined = pd.concat([iris_dataframe, pd.DataFrame(y_train, columns=['species'])], axis=1)

# Use seaborn's pairplot to create a scatter matrix
grr = sns.pairplot(iris_combined, hue='species', markers='o', palette='viridis')
plt.show()

#grr = pd.scatter_matrix(iris_dataframe, c= y_train, figsize=(15,15), marker='o', hist_kwds = {'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

#making predictions
import numpy as np
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new shape: {}".format(X_new.shape))
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
#print("Predicted target name: {}".format(iris_dataset['target_names']['prediction']))

#evaluating the model
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
print("test score: {:.2f}".format(knn.score(X_test, y_test)))






