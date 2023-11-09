import pandas as pd
import numpy as np
import matplotlib as mp
import seaborn as sb
from sklearn import datasets
import scipy as sp

iris = datasets.load_iris()

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Print the first 3 rows
print(iris_df.head(3))

# Glance over some statistics about the iris-data
print("====================================================================================================")
print("Describing iris-data: \n", iris_df.describe())

# Print the column names of the iris-data
print("====================================================================================================")
print("Columns names in iris-data are: \n", iris_df.columns)

# Plot  the  distribution  of  all  numerical  features  and  the categorical target using matplotlib and observe the plots
print("====================================================================================================")
print("Plotting the distribution of all numerical features and the categorical target using matplotlib")
iris_df.hist()
# mp.pyplot.show()

# Plot a “feature pair-wise” scatter plot to see how the numerical features are correlated to each other and print out the pairwise correlation coefficients between the numerical features
print("====================================================================================================")
print("Plotting a feature pair-wise scatter plot to see how the numerical features are correlated to each other")
sb.pairplot(iris_df)
wait = input("PRESS ENTER TO CONTINUE.")
# mp.pyplot.show()
