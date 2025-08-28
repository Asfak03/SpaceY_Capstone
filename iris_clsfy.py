from sklearn.datasets import load_iris
iris_dataset = load_iris()
#now let us check the keys of the data set
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print(f"Target names: {iris_dataset['target_names']}")
print(f"Feature names: {iris_dataset['feature_names']}")
print(f"types of data:/n {type(iris_dataset['data'])}")
# now we will use test_split to split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
# now we will check the shape of the training and testing data
print(f"X_train shape: {X_train.shape}")
# create dataframe from data in X_train
import pandas as pd
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset['feature_names'])
print(iris_dataframe.head())
print(f"y_train shape: {y_train.shape}")
print(f"First five target values: {y_train[:5]}")
