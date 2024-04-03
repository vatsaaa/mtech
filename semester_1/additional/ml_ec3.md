# Distance Weighted KNN
## Algorithm
    - Assign weights to the neighbors based on their distance from the query point
    - The weight is inversely proportional to the distance
    - The weight is used to calculate the weighted average of the neighbors
    - The class of the query point is assigned based on the weighted average
## Numerical Example
    - Consider a dataset with 5 neighbors
    - Calculate the distance of each neighbor from the query point
    - Assign weights to the neighbors based on their distance
    - Calculate the weighted average of the neighbors
    - Assign the class of the query point based on the weighted average
## Advantages
    - Takes into account the distance of the neighbors from the query point
    - Assigns higher weights to the neighbors that are closer to the query point
    - Reduces the impact of outliers
    - Improves the accuracy of the KNN algorithm
## Disadvantages
    - Computationally expensive: Requires the calculation of distances and weights for each neighbor.
    - May not perform well with high-dimensional data
    - Encounters issues with imbalanced datasets
## Applicability
    - Used in classification problems where the distance of the neighbors from the query point is important
    - Suitable for datasets with a small number of neighbors
    - Effective in reducing the impact of outliers
    - Improves the accuracy of the KNN algorithm
# Locally Weighted Regression
## Algorithm
    - Assign weights to the training examples based on their distance from the query point.
    - The weight is calculated using a kernel function.
    - The weight is used to calculate the locally weighted regression coefficients.
    - The locally weighted regression coefficients are used to predict the output of the query point.
## Algorithm
    - Choose a kernel function.
    - Assign weights to the training examples based on their distance from the query point.
    - The weight is calculated using a kernel function.
    - The weight is used to calculate the locally weighted regression coefficients.
    - The locally weighted regression coefficients are used to predict the output of the query point.
## Numerical Example
    - Consider a dataset with 5 training examples.
    - Calculate the distance of each training example from the query point.
    - Assign weights to the training examples based on their distance.
    - Calculate the locally weighted regression coefficients.
    - Predict the output of the query point using the locally weighted regression coefficients.
## Advantages
    - Takes into account the distance of the training examples from the query point.
    - Assigns higher weights to the training examples that are closer to the query point.
    - Reduces the impact of outliers.
    - Improves the accuracy of the regression model.
    - Difference w.r.t KNN: LWLR assigns weights to the training examples based on their distance from the query point, while KNN assigns weights to the neighbors based on their distance from the query point.
## Disadvantages
    - Computationally expensive: Requires the calculation of distances and weights for each training example.
    - May not perform well with high-dimensional data
    - Issues with imbalanced datasets: For a dataset with more examples of one class than the other, the weights may be biased towards the class with more examples.
## Applicability
    - Used in regression problems where the distance of the training examples from the query point is important
    - Suitable for datasets with a small number of training examples
    - Effective in reducing the impact of outliers
    - Improves the accuracy of the regression model by assigning higher weights to the training examples that are closer to the query point. This ensures that the model is more sensitive to the training examples that are relevant to the query point.
# Radial Basis Functions
## Algorithm
    - A radial basis function (RBF) is a function that assigns a weight to each training example based on its distance from the query point.
    - The weight is calculated using a kernel function.
    - The weight is used to calculate the output of the query point.
    - The output of the query point is a linear combination of the weights and the training examples.
## Algorithm
    - Choose a kernel function.
    - Assign weights to the training examples based on their distance from the query point.
    - The weight is calculated using a kernel function.
    - The weight is used to calculate the output of the query point.
    - The output of the query point is a linear combination of the weights and the training examples.
## Numerical Example
    - Consider a dataset with 5 training examples.
    - Calculate the distance of each training example from the query point.
    - Assign weights to the training examples based on their distance.
    - Calculate the output of the query point using the weights and the training examples.
## Advantages
    - Takes into account the distance of the training examples from the query point.
    - Assigns higher weights to the training examples that are closer to the query point.
    - Reduces the impact of outliers.
    - Improves the accuracy of the regression model.
    - Difference w.r.t KNN: RBF assigns weights to the training examples based on their distance from the query point, while KNN assigns weights to the neighbors based on their distance from the query point.
## Disadvantages
    - Computationally expensive: Requires the calculation of distances and weights for each training example.
    - May not perform well with high-dimensional data
    - Issues with imbalanced datasets: For a dataset with more examples of one class than the other, the weights may be biased towards the class with more examples.
## Applicability
    - Used in regression problems where the distance of the training examples from the query point is important
    - Suitable for datasets with a small number of training examples
    - Effective in reducing the impact of outliers
    - Improves the accuracy of the regression model by assigning higher weights to the training examples that are closer to the query point. This ensures that the model is more sensitive to the training examples that are relevant to the query point.
## Choice of Centers
    - Centers are the points in the input space around which the RBFs are centered.
    - The centers can be chosen randomly, using k-means clustering, or using other methods.
    - The choice of centers can affect the performance of the RBF network by influencing the coverage of the input space. e.g. If the centers are too far apart, the RBF network may not be able to capture the local structure of the data. If the centers are too close together, the RBF network may overfit the data.
## Types of Radial Basis Functions
        - Gaussian RBF: The Gaussian RBF is the most commonly used RBF. It assigns a weight to each training example based on its distance from the query point. The weight is calculated using the Gaussian kernel function.
        - Multiquadric RBF: The Multiquadric RBF assigns a weight to each training example based on its distance from the query point. The weight is calculated using the Multiquadric kernel function.
        - Inverse Multiquadric RBF: The Inverse Multiquadric RBF assigns a weight to each training example based on its distance from the query point. The weight is calculated using the Inverse Multiquadric kernel function.
        - Thin Plate Spline RBF: The Thin Plate Spline RBF assigns a weight to each training example based on its distance from the query point. The weight is calculated using the Thin Plate Spline kernel function.
        - Cubic RBF: The Cubic RBF assigns a weight to each training example based on its distance from the query point. The weight is calculated using the Cubic kernel function.
        - Wendland Compactly Supported RBF: The Wendland Compactly Supported RBF assigns a weight to each training example based on its distance from the query point. The weight is calculated using the Wendland Compactly Supported kernel function.
# Radial Basis Function Networks
# Support Vector Machines
## Support Vectors
    - Support vectors are the data points that lie on the margin or within the margin of the hyperplane.
## Hyperplane
## Margin and its significance
## Types of SVM
### Linear SVM
    - Algorithm
    - Numerical Example
    - Advantages and Disadvantages
    - Applicability
### Slack Variables
### Hard Margin SVM
### Soft Margin SVM
### Non-Linear SVM
    - Algorithm
    - Numerical Example
    - Advantages and Disadvantages
    - Applicability
## Issues with SVM
    - Sensitivity to noise
    - Choice of Kernel
    - Choice of Kernel parameters
    - Optimization criterion
## Purpose
    - Effective classification
    - Handling Non-linear Data
    - Robustness to overfitting
    - Memory Efficiency
    - Flexibility in Kernel Selection
    - Versatility
# Naive Bayes Classifier
## Algorithm
    - NB is a popular and simple probabilistic classifier based on Bayes theorem with naive assumption of the features.
    - Assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.
    - e.g. A fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter. Even if these features depend on each other or upon the existence of the other features.
    - This assumption simplifies computation and makes the model interpretable and computationally efficient.
## Numerical Example
## Advantages
    - Simple and easy to implement
    - Fast and efficient
    - Works well with high-dimensional data
    - Handles missing values well
    - Good for categorical data
    - Good for text classification
## Disadvantages
    - Assumes independence of features
    - Zero frequency problem
    - Resolved with Laplace smoothing: Add 1 to each count and add the number of classes to the denominator.
    - Sensitive to the scale of the data
    - Cannot learn interactions between features
    - Cannot handle continuous data
    - Cannot handle unseen data
## Applicability
    - Text classification: Spam detection, sentiment analysis, document clustering / classification (Topic classification, author identification, language detection), fake news detection, Customer review analysis (Sentiment analysis, opinion mining), Product recommendation(E-commerce product recommendation, movie recommendation)
    - Recommendation systems: Collaborative filtering, content-based filtering
    - Medical diagnosis: Disease prediction, patient classification
    - Fraud detection: Credit card fraud detection, insurance fraud detection
    - Customer segmentation: Market segmentation, customer profiling
# Ensemble Learning
## Bagging
    - Bootstrap Aggregating
    - Random Forest
## Boosting
    - AdaBoost
    - Gradient Boosting
    - XGBoost
## Stacking
### Introduction
    - Ensemble learning technique that combines multiple models via an external meta-model/meta-classifier/meta-regressor to improve the performance of the overall model.
    - The base models are trained on the training data and their predictions are used as input features for the stack generalization meta-model.
    - Base models are of different types, and are trained on different subsets of training data
    - Difference w.r.t bagging and boosting: Combines the predictions of multiple models rather than combining the models themselves.
    - Benefit: Stacking captures the strengths of different models and improves the overall performance of the model as compare to any one of the individual models.
### Algorithm
    - Stacking begins by training multiple diverse base models on the training data. 
    - Common base learners: Decision Trees, Support Vector Machines, Random Forest, Gradient Boosting, Neural Networks, K-Nearest Neighbors, Logistic Regression, Naive Bayes, etc.
    - Holdout data: A portion of the training data is set aside as holdout data to train the meta-model.
    ![Stacking Image 1](Stacking001.jpeg)
### Numerical Example
### Advantages
### Disadvantages
### Applicability

