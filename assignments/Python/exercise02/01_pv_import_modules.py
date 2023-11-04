import pandas as pd
import numpy as np
import matplotlib as mp
import seaborn as sb
from sklearn import datasets
import scipy as sp

iris = datasets.load_iris()

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

sb.pairplot(iris_df, hue='target')
mp.pyplot.show()
