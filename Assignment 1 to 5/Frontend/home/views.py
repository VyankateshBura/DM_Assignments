from django.shortcuts import render
import json
# Create your views here.
from rest_framework.parsers import FileUploadParser
import csv
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.views import View
import csv
import math
from django.http import JsonResponse
from django.views import View
import csv
from django.http import HttpResponse
import json
from django.http import JsonResponse
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import csv
import statistics
import numpy as np
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.http import JsonResponse
from django.views import View
import statistics
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency,zscore,pearsonr
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn import datasets
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.tree import export_text
from django.views.decorators.csrf import csrf_exempt
import logging
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import _tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import tempfile
import shutil


# Function to create and evaluate a Regression classifier
@csrf_exempt
def regression_classifier(request):



    dp = json.loads(request.body)
    df = pd.DataFrame(dp['arrayData'])
    # Assuming 'data' is your DataFrame
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    def apply_zscore(column):
        return zscore_custom(column)

    
    dc = df.copy()
    dc[numerical_columns] = dc[numerical_columns].apply(zscore)

    # Assuming data contains 'target' column with class labels
    # X = [[row[attribute1], row[attribute2]] for row in data]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].to_numpy()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     # Create and fit the Logistic Regression classifier
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(X_train, y_train)

    # Evaluate the Logistic Regression model
    y_pred = logistic_regression_model.predict(X_test)

     # Calculate classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    confusion = confusion_matrix(y_test, y_pred)

    # Return the evaluation results
     # Return the evaluation results as a JSON response
    results = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Confusion Matrix": confusion.tolist(),
    }

    return JsonResponse(results)

# Function to create and evaluate a Naïve Bayesian Classifier
@csrf_exempt
def naive_bayesian_classifier(request):

    dp = json.loads(request.body)
    df = pd.DataFrame(dp['arrayData'])
    # Assuming 'data' is your DataFrame
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    def apply_zscore(column):
        return zscore_custom(column)

    
    dc = df.copy()
    dc[numerical_columns] = dc[numerical_columns].apply(zscore)

    # Assuming data contains 'target' column with class labels
    # X = [[row[attribute1], row[attribute2]] for row in data]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].to_numpy()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit the Naïve Bayesian Classifier (replace with your NB model)
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    # Evaluate the NB model
    y_pred = nb_model.predict(X_test)

    # Calculate evaluation metrics for classification
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # Calculate recognition rate, misclassification rate, sensitivity, and specificity for NB
    recognition_rate = accuracy
    misclassification_rate = 1 - accuracy
    # Number of classes
    num_classes = len(cm)

    sensitivity = []
    specificity = []

    for i in range(num_classes):
        # Sensitivity for class i
        tp = cm[i][i]
        fn = sum(cm[i]) - tp
        sensitivity_i = tp / (tp + fn)
        sensitivity.append(sensitivity_i)

        # Specificity for class i
        tn = sum([sum(cm[j]) - cm[j][i] for j in range(num_classes)]) - (fn + tp)
        fp = sum([cm[j][i] for j in range(num_classes)]) - tp
        specificity_i = tn / (tn + fp)
        specificity.append(specificity_i)

    # Return the evaluation results
    results= {
        "Confusion Matrix": cm.tolist(),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Recognition Rate": recognition_rate,
        "Misclassification Rate": misclassification_rate,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
    }
    return JsonResponse(results)

# Function to create and evaluate a k-NN classifier (vary k)
@csrf_exempt
def knn_classifier(request):
    dp = json.loads(request.body)
    df = pd.DataFrame(dp['arrayData'])
    # Assuming 'data' is your DataFrame
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    def apply_zscore(column):
        return zscore_custom(column)

    
    dc = df.copy()
    dc[numerical_columns] = dc[numerical_columns].apply(zscore)

    # Assuming data contains 'target' column with class labels
    # X = [[row[attribute1], row[attribute2]] for row in data]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    k = dp['k']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features (optional but often recommended for k-NN)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and fit the k-NN classifier
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    # Evaluate the k-NN model
    y_pred = knn_model.predict(X_test)

    # Calculate evaluation metrics for classification
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # Calculate recognition rate, misclassification rate, sensitivity, and specificity for k-NN
    # Calculate recognition rate, misclassification rate, sensitivity, and specificity for NB
    recognition_rate = accuracy
    misclassification_rate = 1 - accuracy
    # Number of classes
    num_classes = len(cm)

    sensitivity = []
    specificity = []

    for i in range(num_classes):
        # Sensitivity for class i
        tp = cm[i][i]
        fn = sum(cm[i]) - tp
        sensitivity_i = tp / (tp + fn)
        sensitivity.append(sensitivity_i)

        # Specificity for class i
        tn = sum([sum(cm[j]) - cm[j][i] for j in range(num_classes)]) - (fn + tp)
        fp = sum([cm[j][i] for j in range(num_classes)]) - tp
        specificity_i = tn / (tn + fp)
        specificity.append(specificity_i)


    # Return the evaluation results
    results= {
        "Confusion Matrix": cm.tolist(),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Recognition Rate": recognition_rate,
        "Misclassification Rate": misclassification_rate,
        "Sensitivity": sensitivity,
        "Specificity": specificity
    }
    return JsonResponse(results)


# Function to create and evaluate a Three-layer Artificial Neural Network (ANN) classifier
@csrf_exempt
def neural_network_classifier(request):
    dp = json.loads(request.body)
    df = pd.DataFrame(dp['arrayData'])
    # Assuming 'data' is your DataFrame
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    def apply_zscore(column):
        return zscore_custom(column)

    
    dc = df.copy()
    dc[numerical_columns] = dc[numerical_columns].apply(zscore)

    # Assuming data contains 'target' column with class labels
    # X = [[row[attribute1], row[attribute2]] for row in data]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features (recommended for ANNs)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and fit the ANN classifier (replace with your ANN model)
    ann_model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
    ann_model.fit(X_train, y_train)

    # Evaluate the ANN model
    y_pred = ann_model.predict(X_test)

    # Calculate evaluation metrics for classification
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    # Return the evaluation results
    results= {
        "Confusion Matrix": cm.tolist(),
        "Accuracy": accuracy,
        "Precision": precision,
        "F1-Score": f1,
        "Recall": recall,
        
    }
    return JsonResponse(results)

# Define a function to extract rules from the tree
def tree_to_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            left_rule = f'if {name} <= {threshold:.2f}:'
            right_rule = f'else:  # if {name} > {threshold:.2f}'
            
            left_rules = recurse(tree_.children_left[node])
            right_rules = recurse(tree_.children_right[node])
            
            return [left_rule] + left_rules + [right_rule] + right_rules
        else:
            return [f'return {tree_.value[node]}']

    rules = recurse(0)
    return rules










def parse_tree_structure(tree_text):
    lines = tree_text.split('\n')
    root = {'name': 'Root', 'children': []}
    stack = [(0, root)]

    for line in lines:
        # Skip empty lines or lines with no '|' characters
        if not line.strip() or '|' not in line:
            continue

        depth = 0
        while depth < len(line) and line[depth] == '|':
            depth += 1

        # Extract the class label or condition from the line
        parts = line.strip().split()
        if len(parts) > 1:
            node = {'name': ' '.join(parts[1:])}  # Use everything after the first word
        else:
            node = {'name': parts[0]}

        parent_depth, parent = stack[depth - 1]
        parent.setdefault('children', []).append(node)
        stack.append((depth, node))

    return root



#DecisionTree Classifier
@csrf_exempt
def decision_tree_classifier(request):
    dp = json.loads(request.body)
    df = pd.DataFrame(dp['arrayData'])
    # Assuming 'data' is your DataFrame
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    def apply_zscore(column):
        return zscore_custom(column)

    
    dc = df.copy()
    dc[numerical_columns] = dc[numerical_columns].apply(zscore)

    # Assuming data contains 'target' column with class labels
    # X = [[row[attribute1], row[attribute2]] for row in data]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].to_numpy()
    # print(X,y)
    # Create and fit the Decision Tree classifier
     # Initialize a dictionary to store the results and metrics
    results = {}

    # Implement decision tree classifiers with different attribute selection measures
    attribute_measures = ["gini", "log_loss", "entropy"]
    i=1
    for measure in attribute_measures:
        # Create and fit the Decision Tree classifier with the selected measure
        clf = DecisionTreeClassifier(criterion=measure)
        clf.fit(X, y)

        # Visualize the tree (You can use libraries like graphviz or dtreeviz)
         # Create a temporary file to save the decision tree plot
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            # Plot the decision tree
            plt.figure(figsize=(20, 10))  # Adjust figure size as needed
            plot_tree(clf, filled=True, feature_names=list(X.columns), class_names=list(map(str, clf.classes_)), rounded=True)
            plt.savefig(temp_file.name)

        image_path = temp_file.name

        # Specify the source image file path
        source_image_path = image_path

        # Specify the destination directory where you want to save the image
        destination_directory = f"D:\\Projects\\Data Mining Assignment\\mern_starter-kit\\src\\assets\\DecisionTree{i}.png"

        # Use shutil.copy to copy the image to the destination directory
        shutil.copy(source_image_path, destination_directory)

        print("Image file saved successfully to:", destination_directory)

        # Calculate metrics
        y_pred = clf.predict(X)
        cm = confusion_matrix(y, y_pred)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred,average='weighted')
        recall = recall_score(y, y_pred,average='weighted')


        i+=1
        # Store the results and metrics in the dictionary
        results[measure] = {
            'tree_image_path': image_path,  # Replace with actual tree structure
            'confusion_matrix': cm.tolist(),
            'measure':measure,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
    

    return JsonResponse(results)


@csrf_exempt
def Rule_based_classifier(request):
    dp = json.loads(request.body)
    df = pd.DataFrame(dp['arrayData'])
    numerical_columns = df.select_dtypes(include=[np.number]).columns

    # Standardize the numerical columns using z-score
    dc = df.copy()
    dc[numerical_columns] = dc[numerical_columns].apply(zscore)
    df = dc

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].to_numpy()

    # Create and fit the Decision Tree classifier
    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(X, y)

    # Extract rules from the decision tree using export_text
    rules = export_text(clf, feature_names=list(X.columns), show_weights=True)

    # Call the function to extract rules
    # print("Features",list(X.columns))
    
    ans = tree_to_rules(clf, feature_names=list(X.columns))
    print("Tree ****************************************:",json.dumps(ans))
    # Calculate coverage, accuracy, and toughness
    coverage = 1-calculate_coverage(clf, X)
    accuracy = calculate_accuracy(clf, X, y)
    toughness = calculate_toughness(rules)

    # Print and return the results
    results = {
        "Coverage": coverage,
        "Accuracy": accuracy,
        "Toughness": toughness,
        "Tree":json.dumps(ans)
    }
    print("Coverage:", results["Coverage"])
    print("Accuracy:", results["Accuracy"])
    print("Toughness:", results["Toughness"])

    return JsonResponse(results)

def calculate_coverage(tree, X):
    leaf_indices = tree.apply(X)
    unique_leaves = np.unique(leaf_indices)
    return len(unique_leaves) / len(X)

def calculate_accuracy(tree, X, y):
    predicted_classes = tree.predict(X)
    return np.mean(predicted_classes == y)

def calculate_toughness(rules):
    # Calculate toughness (size) based on the extracted rules
    # Count the number of conditions in each rule to determine toughness
    rule_lines = rules.split('\n')
    toughness = 0
    for rule_line in rule_lines:
        rule = rule_line.strip()
        if rule:
            conditions = rule.split('(')[0].strip()
            if conditions:
                # Split conditions by "AND" to count the number of conditions
                conditions_list = conditions.split(" AND ")
                toughness = max(toughness, len(conditions_list))
    return toughness






# Zscore normalization function

def zscore_custom(data):
    data = np.asarray(data)
    data = data.astype(float)  # Convert data to float
    if np.isnan(data).all():
        return np.zeros_like(data)  # Return zeros if all values are NaN
    mean = np.nanmean(data)
    std = np.nanstd(data)
    if std == 0:
        return np.zeros_like(data)  # Return zeros if standard deviation is zero
    z_scores = (data - mean) / std
    return z_scores


@csrf_exempt
def calculate_contingency_table(request):
    if request.method == 'POST':
        try:
            # Load JSON object containing CSV data
            data = json.loads(request.body)
            
            # Convert JSON object to DataFrame
            df = pd.DataFrame(data['arrayData'])
            col1 = data['col1']
            col2 = data['col2']
            # print(df,col1,col2)
            # Calculate the contingency table
            contingency_table = pd.crosstab(df[col1], df[col2])
            
            chi2, p, dof, expected = chi2_contingency(contingency_table)

            # print(chi2.dtypes)
            # Set the significance level
            alpha = 0.05
            
            # Check if the relationship is significant
            if p < alpha:
                relationship_status = "Related (significant)"
            else:
                relationship_status = "Not related (insignificant)"
            
            response_data = {
                'message': 'Contingency table calculated successfully.',
                'contingency_table': contingency_table.to_dict(),
                'chi2':chi2,
                'p':p,
                'dof':dof,
                'expected':expected.tolist(),
                'status':relationship_status
            }

            return JsonResponse(response_data)
        except Exception as e:
            logging.exception("An error occurred:")
            return JsonResponse({'message': f'Error: {str(e)}'}, status=500)

    return HttpResponse("POST request required.")

@csrf_exempt
def zscoreCalc(request):
    if request.method == 'POST':
        # Convert JSON object to a Python list of dictionaries
        data = json.loads(request.body)
        df = pd.DataFrame(data['arrayData'])
        # Assuming 'data' is your DataFrame
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        def apply_zscore(column):
            return zscore_custom(column)

        # Apply Z-score normalization to numerical columns
        zscore_df = df.copy()
        zscore_df[numerical_columns] = zscore_df[numerical_columns].apply(zscore)


        # Apply the zscore_custom function to numerical columns using apply
        zscore_estimated = df.copy()
        zscore_estimated[numerical_columns] = zscore_estimated[numerical_columns].apply(apply_zscore)


        # print("Calculated zscore normalized table ")
        # print(zscore_estimated)
        # print("\n\nBuiltin function zscore normalized table")
        # print(zscore_df)
        # Convert DataFrames to dictionaries
        zscore_df_dict = zscore_df.to_dict(orient='split')
        zscore_estimated_dict = zscore_estimated.to_dict(orient='split')

        # For example, you can save it to a model, perform calculations, etc.
        response_data = {
            'message': 'Data received and processed successfully.',
            'data': [
                {
                    'df_type': 'zscore_df',
                    'data': zscore_df_dict
                },
                {
                    'df_type': 'zscore_estimated',
                    'data': zscore_estimated_dict
                }
            ]
        }

        # Convert response_data to JSON and send as the response
        return JsonResponse(response_data)

    return HttpResponse("Request received")


def min_max_normalize(column):
    if column.dtype in [np.number, np.float64]:
        return (column - column.min()) / (column.max() - column.min())
    else:
        return column

def decimal_scale_normalize(column, scaling_factor):
    if column.dtype in [np.number, np.float64]:
        return column / scaling_factor
    else:
        return column


@csrf_exempt
def minMaxNormalization(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        df = pd.DataFrame(data['arrayData'])
        # Define a function to convert values to numeric if possible
        def convert_to_numeric(value):
            try:
                return pd.to_numeric(value)
            except:
                return value

        # Apply the function to all columns
        result_df = df.applymap(convert_to_numeric)
        print(result_df)

        # Assuming 'data' is your DataFrame
        numerical_columns = result_df.select_dtypes(include=[np.number]).columns


         # Using built-in MinMaxScaler
        scaler = MinMaxScaler()
        min_max_scaled_df_builtin = result_df.copy()
        min_max_scaled_df_builtin[numerical_columns] = scaler.fit_transform(min_max_scaled_df_builtin[numerical_columns])

        # Using traditional methods
        min_max_scaled_df_traditional = result_df.copy()

        for col in result_df.columns:
            min_max_scaled_df_traditional[col] = min_max_normalize(result_df[col])
        # print("Calculated minmax normalized table ")
        # print( min_max_scaled_df_builtin.to_dict(orient='split'))
        # print("\n\nBuiltin function minmax normalized table")
        # print(min_max_scaled_df_traditional.to_dict(orient='split'))
        response_data = {
            'message': 'Data received and processed successfully.',
            'data': [
                {
                    'df_type': 'min_max_scaled_builtin',
                    'data': min_max_scaled_df_builtin.to_dict(orient='split')
                },
                {
                    'df_type': 'min_max_scaled_traditional',
                    'data':  min_max_scaled_df_traditional.to_dict(orient='split')
                }
            ]
        }
        
        return JsonResponse(response_data)
    
    return HttpResponse("Request received")








@csrf_exempt
def decimalScalingNormalization(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        df = pd.DataFrame(data['arrayData'])

        # Define a function to convert values to numeric if possible
        def convert_to_numeric(value):
            try:
                return pd.to_numeric(value)
            except:
                return value

        # Apply the function to all columns
        result_df = df.applymap(convert_to_numeric)
        
        # Assuming you have a DataFrame 'df'
        # data_types = result_df.dtypes

        # print(result_df)

        # Remove rows with missing values
        result_df = result_df.dropna()
        numerical_columns = result_df.select_dtypes(include=[np.number]).columns

        # Using traditional Decimal Scaling
        decimal_scaled_traditional = result_df.copy()
        
        # print(result_df[numerical_columns].apply(count_digits_before_decimal))
        for col in result_df[numerical_columns].columns: 
            scale_factor = 10 ** (len(str(result_df[col].abs().max().astype(int)))) 
            decimal_scaled_traditional[col] = decimal_scale_normalize(decimal_scaled_traditional[col],scale_factor)
 
        
        # Using built-in StandardScaler for comparison
        scaler = StandardScaler()
        standard_scaled = result_df.copy()
        print("Standard Scaled\n",standard_scaled[numerical_columns].shape)
        standard_scaled[numerical_columns] = scaler.fit_transform(standard_scaled[numerical_columns])
        # print("Calculated Decimal Scaling normalized table ")
        # print(decimal_scaled_traditional.to_dict())
        # print("\n\nBuiltin function Decimal Scaling normalized table")
        # print(standard_scaled.to_dict(orient='split'))
        response_data = {
            'message': 'Data received and processed successfully.',
            'data':  [
                {
                    'df_type': 'decimal_scaled',
                    'data': decimal_scaled_traditional.to_dict(orient='split')
                },
                {
                    'df_type': 'standard_scaled_builtin',
                    'data':  standard_scaled.to_dict(orient='split')
                }
            ]
        }
        
        return JsonResponse(response_data)
    
    return HttpResponse("Request received")
        
@csrf_exempt
def correlation_analysis(request):
    if request.method == 'POST':
        # Assuming you have a form with attribute names as input fields (e.g., 'attribute1' and 'attribute2')
        data = json.loads(request.body)
        attribute1_name = data['attribute1']
        attribute2_name = data['attribute2']

        # Assuming you have a dataset (DataFrame) containing the relevant data
        # Replace 'your_dataset' with your actual dataset
        # Make sure the dataset contains columns with the specified attribute names
        df = pd.DataFrame(data['arrayData'])
        def convert_to_numeric(value):
            try:
                return pd.to_numeric(value)
            except:
                return value

        # Apply the function to all columns
        your_dataset = df.applymap(convert_to_numeric)
        # your_dataset = pd.DataFrame(data['arrayData'])  # Load your dataset here

        if attribute1_name in your_dataset.columns and attribute2_name in your_dataset.columns:
            # Extract the selected attributes as Series
            attribute1 = your_dataset[attribute1_name]
            attribute2 = your_dataset[attribute2_name]

            # Calculate the Pearson correlation coefficient and covariance
            correlation_coefficient, _ = pearsonr(attribute1, attribute2)
            covariance = np.cov(attribute1, attribute2)[0, 1]

            # Determine the conclusion
            if correlation_coefficient > 0:
                conclusion = "There is a positive correlation between the selected attributes."
            elif correlation_coefficient < 0:
                conclusion = "There is a negative correlation between the selected attributes."
            else:
                conclusion = "There is no linear correlation between the selected attributes."


            response_data = {
                'message': 'Data received and processed successfully.',
                'data':  [
                    {
                        'df_type': 'attribute1_name',
                        'data': attribute1_name
                    },
                    {
                        'df_type': 'attribute2_name',
                        'data':  attribute2_name
                    },
                    {
                        'df_type': 'correlation_coefficient',
                        'data': correlation_coefficient
                    },
                    {
                        'df_type': 'covariance',
                        'data': covariance
                    },
                    {
                        'df_type': 'conclusion',
                        'data': conclusion
                    }
                ]
            }

            return JsonResponse(response_data)
    
    return HttpResponse("Request received")
