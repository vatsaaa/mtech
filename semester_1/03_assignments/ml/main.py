from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import f_classif, mutual_info_regression, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.exceptions import UndefinedMetricWarning
import sys, warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UndefinedMetricWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)

import platform, time

if platform.system() == "Windows":
    import psutil
elif platform.system() == "Darwin" or platform.system() == "Linux":
    import resource

def track_time_and_space(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        if platform.system() == "Windows":
            start_memory = psutil.Process().memory_info().rss
            end_memory = psutil.Process().memory_info().rss
        else:
            start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            end_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss

        print(platform.system(), start_memory, end_memory)

        execution_time = end_time - start_time
        memory_usage = (end_memory - start_memory) / 1024  # Convert to kilobytes

        print(f"Execution time: {execution_time} seconds | Memory usage: {memory_usage} KB")

        return result, execution_time, memory_usage

    return wrapper

def load_data(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, index_col = 0, encoding='utf-8')
    df.sort_values(by=["Date", "Time of Day"], ascending = True, inplace = True)
    return df

def plot_class_distribution(y: pd.Series):
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    sns.countplot(x=y, palette='viridis', hue=y, legend=False)
    plt.title('Class Distribution')
    plt.xlabel('Is Fraudulent')
    plt.ylabel('Count')
    plt.show()

def encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame: 
    # Encode categorical features so that they can be used in the SMOTE algorithm
    categorical_features = X.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        encoder = LabelEncoder()
        X[feature] = encoder.fit_transform(X[feature])

    return X

def check_class_imbalance(df: pd.DataFrame, display_percent=True, display_count=True):
    # Separate features (X) from target variable (y)
    df['Is Fraudulent'] = df['Is Fraudulent'].map({0: 'No', 1: 'Yes'})
    X = df.drop("Is Fraudulent", axis=1)
    y = df["Is Fraudulent"]

    X = encode_categorical_features(X)

    plot_class_distribution(y)

    # Display the class distribution percentages
    if display_count:
        print("Class Distribution:")
        print(df['Is Fraudulent'].value_counts())
    
    if display_percent:
        print("\nClass Distribution Percentages:")
        print(df['Is Fraudulent'].value_counts(normalize=True) * 100)
    
    # Convert the target variable back to numeric
    y = y.map({'No': 0, 'Yes': 1})

    # Merge the features and target variable back into a single dataframe
    df = pd.concat([X, y], axis=1)

    return df

def handle_class_imbalance(df: pd.DataFrame):
    # Separate features (X) from target variable (y)
    X = df.drop("Is Fraudulent", axis=1)
    y = df["Is Fraudulent"]

    X = encode_categorical_features(X)

    # Apply SMOTE
    smote = SMOTE(random_state=71)
    X, y = smote.fit_resample(X, y)

    # Convert the target variable back to numeric
    y = y.map({'No': 0, 'Yes': 1})

    # Merge the features and target variable back into a single dataframe
    df = pd.concat([X, y], axis=1)

    return df

def correlatioin_analysis(df: pd.DataFrame):
    correlation_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()

def feature_engineering(df: pd.DataFrame):
    # Date related nominal features
    df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.day_name()
    df['Month'] = pd.to_datetime(df['Date']).dt.month_name()

    # Time of day looks redundant, so we drop it
    df = df.drop(['Date', 'Time of Day'], axis=1)

    return df

def get_column_types(dframe: pd.DataFrame):
  all_features = dframe.columns
  op_features = ["Is Fraudulent"]
  numeric_features = list(set(dframe._get_numeric_data().columns) - set(op_features))
  nominal_and_ordinal_features = list(set(all_features) - set(numeric_features) - set(op_features))
  ordinal_only_features = list(["Merchant Reputation", 'Online Transactions Frequency'])
  nominal_only_features = list(set(nominal_and_ordinal_features) - set(ordinal_only_features))

  return {
      "numeric_features": numeric_features,
      "nominal_only_features": nominal_only_features,
      "ordinal_only_features": ordinal_only_features,
      "nominal_and_ordinal_features": nominal_and_ordinal_features,
      "output_features": op_features
  }

def get_ig_for_features(df: pd.DataFrame):
    X = df.drop(['Is Fraudulent'], axis=1)
    y = df['Is Fraudulent']

    sorted_features = []

    X = encode_categorical_features(X)

    # Apply Information Gain
    ig = mutual_info_regression(X, y) * 100

    feature_scores = {}
    for i in range(len(X.columns)):
        feature_scores[X.columns[i]] = ig[i]

    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

    for feature, score in sorted_features:
        print('Feature:', feature, 'Score:', score)

    return sorted_features, y, X

def plot_ig_for_features(sorted_features):
  fig, ax = plt.subplots()
  y_pos = np.arange(len(sorted_features))
  ax.barh(y_pos, [score for feature, score in sorted_features], align="center")
  ax.set_yticks(y_pos)
  ax.set_yticklabels([feature for feature, score in sorted_features])
#   ax.invert_yaxis()  # Labels read top-to-bottom
  ax.set_xlabel("Importance Score")
  ax.set_title("Feature Importance Scores (Information Gain)")

  # Add importance scores as labels on the horizontal bar chart
  for i, v in enumerate([score for feature, score in sorted_features]):
      ax.text(v + 0.01, i, str(round(v, 3)), color="black", fontweight="bold")

  plt.show()

def split_data_fit_model(df, col_types: dict, model, test_size: float = 0.2):
  # Separate features (X) from target variable (y)
  X = df.drop("Is Fraudulent", axis=1)
  y = df["Is Fraudulent"]

  # Training to testing split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=71)

  # Preprocessing non-numeric features
  preprocessor = ColumnTransformer(
      transformers=[
          ("num", StandardScaler(), col_types["numeric_features"]),
          ("nominal", OneHotEncoder(), col_types["nominal_only_features"]),
          ("ordinal", OrdinalEncoder(), col_types["ordinal_only_features"])
      ])

  pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                            ("pca", PCA(n_components=0.95)),
                            ("feature_selection", SelectKBest(f_classif, k=5)),
                            ("classifier", model)])
  pipeline.fit(X_train, y_train)

  return X_test, y_test, X_train, y_train, pipeline

def predict(xtest, ytest, pipeline):
  ypred = pipeline.predict(xtest)
  cm = confusion_matrix(ytest, ypred)
  y_prob_logreg = pipeline.predict_proba(xtest)[:, 1]
  roc_auc_logreg = roc_auc_score(ytest, y_prob_logreg)

  print(f"ROC AUC: {roc_auc_logreg}")

  return cm, ypred

def print_model_performance(cm, ytest, ypred, model_name: str):
  print("Classification Report:" + model_name)
  print(classification_report(ytest, ypred))

  print("Confusion Matrix:")
  print(cm)

  print("Accuracy Score:")
  print(accuracy_score(ytest, ypred))

  print("\n\n")

def print_model_performance(cm, ytest, ypred, model_name: str):
  print("Classification Report:" + model_name)
  print(classification_report(ytest, ypred))

  print("Confusion Matrix:")
  print(cm)

  print("Accuracy Score:")
  print(accuracy_score(ytest, ypred))

  print("\n\n")

def plot_confusion_matrix(cm, model, model_name: str):
  cmap = sns.color_palette("pastel")

  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
              xticklabels=model.classes_,
              yticklabels=model.classes_)
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix - ' + model_name)
  plt.show()

def plot_roc_auc_curve(xtest, ytest, pipeline, model_name):
    label_encoder = LabelEncoder()
    y_test_numeric = label_encoder.fit_transform(ytest)

    y_prob_logreg = pipeline.predict_proba(xtest)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test_numeric, y_prob_logreg)
    roc_auc_logreg = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc_logreg))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - ' + model_name)
    plt.legend(loc='lower right')
    plt.show()

def create_model(model_name: str) -> object:
    if model_name == "AdaBoostClassifier":
        model = AdaBoostClassifier(random_state=71)
    elif model_name == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(random_state=71)
    elif model_name == "GaussianNB":
        model = GaussianNB(random_state=71)
    elif model_name == "GradientBoostingClassifier":
        model = GradientBoostingClassifier(random_state=71)
    elif model_name == "KNeighborsClassifier":
        model = KNeighborsClassifier(random_state=71)
    elif model_name == "LogisticRegression":
        model = LogisticRegression(random_state=71)
    elif model_name == "RandomForestClassifier":
        model = RandomForestClassifier(random_state=71)
    elif model_name == "SVC":
        model = SVC(random_state=71)
    else:
        raise ValueError("Invalid model name: " + model_name)
    
    return model

@track_time_and_space
def run(df, model, col_types: dict, model_name: str, test_size: float = 0.2, hp_list: dict = None):
    X_test, y_test, X_train, y_train, pipeline = split_data_fit_model(df, col_types, model=model, test_size=test_size)
    cm, ypred = predict(X_test, y_test, pipeline)
    
    if hp_list is not None:
        grid_search = GridSearchCV(estimator=model, param_grid=hp_list, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        print(f"Best parameters for {model_identifier}: {grid_search.best_params_}")
        print(f"Best score for {model_identifier}: {grid_search.best_score_}")

    print_model_performance(cm, y_test, ypred, model_name=model_name)
    plot_confusion_matrix(cm, model=model, model_name=model_name)
    plot_roc_auc_curve(X_test, y_test, pipeline, model_name=model_name)

if __name__ == "__main__":
    pred_models = [
        {
            "name": "AdaBoostClassifier",
            "test_size": [{"default": [20]}, {"others": [15, 25, 30]}],
            "hp_list": {
                "n_estimators": [0, 25, 50, 75, 100],
                "learning_rate": [0.1, 0.35, 0.55, 0.85, 1.00],
                "algorithm": ['SAMME'], 
                "random_state": [71]
            }
        },
        {
            "name": "DecisionTreeClassifier",
            "test_size": [{"default": [20]}, {"others": [15, 25, 30]}],
            "hp_list": {
                "criterion": 'gini', 
                "max_depth": "None", 
                "min_samples_split": 2, 
                "min_samples_leaf": 1, 
                "max_features": None, 
                "random_state": "None", 
                "max_leaf_nodes": "None", 
                "min_impurity_decrease": 0.0, 
                "min_impurity_split": "None", "class_weight": "None", 
                "ccp_alpha": 0.0
            }
        },
        {
            "name": "GaussianNB",
            "test_size": [{"default": [20]}, {"others": [15, 25, 30]}],
            "hp_list": {
                "var_smoothing": "1e-09"
            }
        },
        {
            "name": "GradientBoostingClassifier",
            "test_size": [{"default": [20]}, {"others": [15, 25, 30]}],
            "hp_list": {
                "n_estimators": [100, 200, 300, 400, 500],
                "learning_rate": [0.1, 0.35, 0.55, 0.85, 1.00],
                "max_depth": [3],
                "min_samples_split": [2, 5, 8, 10],
                "min_samples_leaf": [1, 2, 3, 4, 5],
                "subsample": [1.0],
                "max_features": "None",
                "random_state": "None",
                "verbose": [0],
                "warm_start": "False",
                "ccp_alpha": [0.00, 0.25, 0.50, 0.75, 1.00],
                "max_samples": "None",
                "validation_fraction": 0.1,
                "n_iter_no_change": "None",
                "tol": [0.0001],
                "presort": ["deprecated", "auto"]
            }
        },
        {
            "name": "KNeighborsClassifier",
            "test_size": [{"default": [20]}, {"others": [15, 25, 30]}],
            "hp_list": {
                "n_neighbors": [5, 10, 15, 20],
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                "leaf_size": [0, 20, 40, 60],
                "p": [1, 2, 3, 4],
                "metric": ["minkowski", "manhattan", "euclidean", "chebyshev"]
            }
        },
        {
            "name": "LogisticRegression",
            "test_size": [{"default": [20]}, {"others": [15, 25, 30]}],
            "hp_list": {
                "penalty": ["l2", "l1", "elasticnet", "none"],
                "dual": ["False", "True"],
                "tol": [0.0001, 0.0005, 0.0010, 0.0050, 0.0100],
                "C": [1.00, 0.75, 0.5, 0.25, 0.00],
                "fit_intercept": ["True", "False"],
                "intercept_scaling": 1,
                "class_weight": "None",
                "random_state": "None",
                "solver": ["lbfgs", "liblinear", "sag", "saga"],
                "max_iter": [100, 200, 300, 400, 500],
                "multi_class": ["auto", "ovr", "multinomial"],
                "verbose": 0,
                "warm_start": ["False", "True"],
                "n_jobs": ["None", 1],
                "l1_ratio": ["None", "0.5"]
            }
        },
        {
           "name": "RandomForestClassifier",
           "test_size": [{"default": [20]}, {"others": [15, 25, 30]}],
           "hp_list": {
                "n_estimators": [100],
                "criterion": ["gini"],
                "max_depth": "None",
                "min_samples_split": [2],
                "min_samples_leaf": [1],
                "max_features": "auto",
                "bootstrap": "True",
                "oob_score": "False",
                "n_jobs": "None",
                "random_state": "None",
                "verbose": [0],
                "warm_start": "False",
                "class_weight": "None",
                "ccp_alpha": [0.0],
                "max_samples": "None"
            }
        }
    ]

    # Load the data and check for class imbalance
    filename = "data/Group21_Financial_Transactions_Dataset.csv"
    df = load_data(filename=filename)
    check_class_imbalance(df)
    correlatioin_analysis(df)

    # Handle imbalanced class and check that imbalance
    # is resolved and correlation matrix is updated
    df = handle_class_imbalance(df)
    check_class_imbalance(df)
    correlatioin_analysis(df)


    # Drop "Merchant Location History" since it is code only, which cannot
    # be standardised / normalised without mapping code to actual location 
    df = feature_engineering(df).drop(['Merchant Location History'], axis=1)

    # Select features to process and the target variable
    col_types = get_column_types(df)
    sorted_features, y, X = get_ig_for_features(df)
    plot_ig_for_features(sorted_features)

    # Run the models to evaluate for various test sizes and hyperparameters
    for model_info in pred_models:
        test_size = float(model_info["test_size"][0]["default"][0]) / 100
        model_identifier = model_info["name"] + " " + str(model_info["test_size"][0]["default"][0]) 

        model = create_model(model_info["name"])

        print("Running model: ", model_identifier, "...")
        run(df, model, col_types, model_name=model_identifier, test_size=test_size, hp_list=None)
        print("Completed run for model: ", model_identifier)

        # Now run the model for other test sizes
        for os in model_info["test_size"][1]["others"]:
            test_size = float(os) / 100
            model_identifier = model_info["name"] + " " + str(os)

            model = create_model(model_info["name"])
            print("Running model: ", model_identifier, "...")
            run(df, model, col_types, model_name=model_identifier, test_size=test_size, hp_list=model_info["hp_list"])
            print("Completed run for model: ", model_identifier)

