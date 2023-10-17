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
from sklearn.cluster import KMeans
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
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from random import sample
from kmedoids import KMedoids
from sklearn.cluster import Birch
import os


def agnesAlgo(data, k):
    
    try:
        data = np.array(data, dtype=np.float64)  # Convert data to a numpy array of type float64
        num_points = len(data)
        clusters = [[i] for i in range(num_points)]
        distances = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(i + 1, num_points):
                distances[i, j] = distances[j, i] = euclidean(data[i], data[j])

        while len(clusters) > k:
            min_distance = np.inf
            merge_clusters = None
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    for point1 in clusters[i]:
                        for point2 in clusters[j]:
                            if point1 < num_points and point2 < num_points:  # Ensure indices are within bounds
                                distance = distances[point1][point2]
                                if distance < min_distance:
                                    min_distance = distance
                                    merge_clusters = (i, j)
                                    
            if merge_clusters is None:
                break
            i, j = merge_clusters
            clusters[i].extend(clusters[j])
            del clusters[j]
            num_points -= 1  # Update the number of points after deleting a cluster

            updated_distances = np.zeros((num_points, num_points))
            for p in range(num_points):
                for q in range(p + 1, num_points):
                    p_orig = clusters[i][p]
                    q_orig = clusters[i][q]
                    updated_distances[p, q] = updated_distances[q, p] = distances[p_orig][q_orig]
            
            distances = np.delete(distances, j, 0)
            distances = np.delete(distances, j, 1)
            distances = updated_distances

        return clusters, distances
    except Exception as e:
        print(e)

# Plotting the dendrogram
def plot_dendrogram(clusters, distances):
    plt.figure(figsize=(10, 7))
    plt.title("Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    for cluster in clusters:
        for point in cluster:
            plt.annotate(str(point), (point, 0), textcoords="offset points", xytext=(0, 10), ha='center')
    for i in range(len(distances)):
        for j in range(i + 1, len(distances)):
            plt.plot([i, j], [distances[i, j], distances[i, j]], c='b')
    plt.show()
    image_path = 'C:/Users/Admin/Desktop/Test/DM_Assignments/Assignment 1 to 5/Backend/src/Pages/Hierarchial/dendrogram.png'
    plt.savefig(image_path)
    plt.close()

@csrf_exempt
def agnes(request):
    
    try:
        
        dp = json.loads(request.body)
        df = pd.DataFrame(dp['arrayData'])
        # Assuming 'data' is your DataFrame
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        def apply_zscore(column):
            return zscore_custom(column)

        dc = df.copy()
        dc[numerical_columns] = dc[numerical_columns].apply(apply_zscore)

        # Assuming data contains 'target' column with class labels
        # X = [[row[attribute1], row[attribute2]] for row in data]````
        X = df.iloc[:, :-1]
        k = 3
        data = np.array(X, dtype=np.float64)
        result_clusters, distances = agnesAlgo(data, k)
        print("Here is the cluster ",result_clusters,distances)
        
        # Agnes Agglomerative Clustering
        linked = linkage(X, 'ward')
        labelList = range(1, len(X) + 1)

        plt.figure(figsize=(10, 7))
        dendrogram(linked,
                   orientation='top',
                   labels=labelList,
                   distance_sort='descending',
                   show_leaf_counts=True)
        
        
        # Save the image to a specific location
        # image_path = os.path.join('path_to_your_directory', 'dendrogram.png')
        # plot_dendrogram(result_clusters, distances)
        image_path = 'C:/Users/Admin/Desktop/Test/DM_Assignments/Assignment 1 to 5/Backend/src/Pages/Hierarchial/AGNES/dendrogram.png'
        plt.savefig(image_path)
        plt.close()

        
        results = {
            "ClusterNumber":'result_clusters'
        }
        return JsonResponse({"result":"Agnes Agglomerative Clustering completed"})
        
    except Exception as e:
        print(e)
        return JsonResponse({'message': f'Error: {str(e)}'}, status=500)
    
    
@csrf_exempt
def diana(request):
    try:
        
        dp = json.loads(request.body)
        df = pd.DataFrame(dp['arrayData'])
        # Assuming 'data' is your DataFrame
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        def apply_zscore(column):
            return zscore_custom(column)

        
        dc = df.copy()
        dc[numerical_columns] = dc[numerical_columns].apply(apply_zscore)

        # Assuming data contains 'target' column with class labels
        # X = [[row[attribute1], row[attribute2]] for row in data]````
        X = df.iloc[:, :-1]
        k = 3
        data = np.array(X, dtype=np.float64)
        # result_clusters, distances = agnesAlgo(data, k)
        # print(result_clusters)
        
        
        # Agnes Agglomerative Clustering
        linked = linkage(X, 'ward')
        labelList = range(1, len(X) + 1)

        plt.figure(figsize=(10, 7))
        dendrogram(linked,
                   orientation='top',
                   labels=labelList,
                   distance_sort='descending',
                   show_leaf_counts=True)
        
        
        # Save the image to a specific location
        # image_path = os.path.join('path_to_your_directory', 'dendrogram.png')
        # plot_dendrogram(result_clusters, distances)
        image_path = 'C:/Users/Admin/Desktop/Test/DM_Assignments/Assignment 1 to 5/Backend/src/Pages/Hierarchial/DIANA/dendrogram.png'
        plt.savefig(image_path)
        plt.close()

        
        results = {
            "ClusterNumber":'result_clusters'
        }
        return JsonResponse({"result":"Agnes Agglomerative Clustering completed"})
        
    except Exception as e:
        print(e)
        return JsonResponse({'message': f'Error: {str(e)}'}, status=500)




class KMeansScratch:
    def __init__(self, k=3, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, data):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]

        for _ in range(self.max_iterations):
            self.classifications = {}
            for i in range(self.k):
                self.classifications[i] = []

            for features in data:
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(features)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > 0.001:
                    optimized = False

            if optimized:
                break

        return self.classifications, self.centroids
    
    
    
    
def convert_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def apply_zscore(column):
    return zscore_custom(column) 
@csrf_exempt
def K_Means(request):
    
    try:
        
        dp = json.loads(request.body)
        df = pd.DataFrame(dp['arrayData'])
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        K_cluster = dp['k']
        print("The number of cluster = ",K_cluster)
        dc = df.copy()
        dc[numerical_columns] = dc[numerical_columns].apply(apply_zscore)

        X = df.iloc[:, :-1]
        data = np.array(X, dtype=np.float64)

        # K-means scratch implementation
        kmeans_scratch = KMeansScratch(k=K_cluster)
        results_scratch = kmeans_scratch.fit(data)
        data_serializable_scratch = json.dumps(results_scratch, default=convert_to_list, indent=2)

        # K-means builtin implementation
        kmeans_builtin = KMeans(n_clusters=K_cluster)
        results_builtin = kmeans_builtin.fit_predict(data)
        data_serializable_builtin = json.dumps(results_builtin, default=convert_to_list, indent=2)

        # Generate the scatter plot
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=results_builtin, s=50, cmap='viridis')
        centers = kmeans_builtin.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

        # Save the image
        plt.savefig('C:/Users/Admin/Desktop/Test/DM_Assignments/Assignment 1 to 5/Backend/src/Pages/K-Means/kmeans_clusters.png')
        plt.close()

        results = {
            "kmeans_scratch": data_serializable_scratch,
            "kmeans_builtin": data_serializable_builtin
        }

        return JsonResponse({"result": "K-means Clustering completed", "data": results})
        
    except Exception as e:
        print(e)
        return JsonResponse({'message': f'Error: {str(e)}'}, status=500)
    
    
# class KMedoids:
#     def __init__(self, k=2, max_iterations=100):
#         self.k = k
#         self.max_iterations = max_iterations

#     def fit(self, data):
#         n, _ = data.shape
#         medoids = sample(range(n), self.k)

#         for _ in range(self.max_iterations):
#             clusters = [[] for _ in range(self.k)]

#             for i in range(n):
#                 distances = pairwise_distances(data, [data[i]], metric='euclidean').ravel()
#                 cluster = np.argmin(distances)
#                 clusters[cluster].append(i)

#             new_medoids = []
#             for cluster in clusters:
#                 cluster_distances = pairwise_distances(data[cluster], metric='euclidean')
#                 total_distance = cluster_distances.sum(axis=1)
#                 min_index = cluster[np.argmin(total_distance)]
#                 new_medoids.append(min_index)

#             if set(medoids) == set(new_medoids):
#                 break
#             medoids = new_medoids

#         self.labels_ = np.zeros(n)
#         for i, cluster in enumerate(clusters):
#             self.labels_[cluster] = i

#         self.cluster_centers_ = data[medoids]
#         return self


# def k_medoids_scratch(D, k=3, max_iterations=100):
#     n, _ = D.shape
#     M = np.array(D[np.random.choice(n, k, replace=False)])

#     for _ in range(max_iterations):
#         D_M = cdist(D, M)
#         C = np.argmin(D_M, axis=1)

#         M_new = np.array([D[C == i].mean(axis=0) for i in range(k)])
#         if np.all(M == M_new):
#             break
#         M = M_new

#     return C, M

def k_medoids_scratch(X, k=2, max_iterations=100):
    n = X.shape[0]  # Number of data points
    m = X.shape[1]  # Number of features

    # Initialize randomly
    medoids = sample(range(n), k)

    for _ in range(max_iterations):
        clusters = {}
        for i in range(k):
            clusters[i] = []

        for point_idx in range(n):
            distances = [np.linalg.norm(X[point_idx] - X[m_idx]) for m_idx in medoids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point_idx)

        new_medoids = []

        for i in range(k):
            cluster_points = X[clusters[i]]
            costs = [sum(np.linalg.norm(cluster_points - cluster_points[j], axis=1)) for j in range(len(clusters[i]))]
            new_medoid_idx = clusters[i][np.argmin(costs)]
            new_medoids.append(new_medoid_idx)

        if set(medoids) == set(new_medoids):
            break

        medoids = new_medoids

    return medoids, clusters

@csrf_exempt
def K_Medoids(request):
    
    try:
        dp = json.loads(request.body)
        df = pd.DataFrame(dp['arrayData'])
        K_clusters = dp['k']
        # Assuming 'data' is your DataFrame
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        def apply_zscore(column):
            return zscore_custom(column)

        
        dc = df.copy()
        dc[numerical_columns] = dc[numerical_columns].apply(apply_zscore)

        # Assuming data contains 'target' column with class labels
        # X = [[row[attribute1], row[attribute2]] for row in data]````
        X = df.iloc[:, :-1]
        k = 3
        data = np.array(X, dtype=np.float64)
        # print(data)

        # Using K-Medoids from scratch
        medoids_scratch, clusters_scratch  = k_medoids_scratch(data, K_clusters)

        # Using built-in K-Medoids
        # Compute the distance matrix
        D = pairwise_distances(X, metric='euclidean')

        # Initialize KMedoids
        kmedoids = KMedoids(n_clusters=K_clusters, random_state=0)

        # Fit the model
        kmedoids.fit(D)

        # Get cluster labels
        clusters_builtin = kmedoids.labels_.tolist()
        
        # Plotting clusters from built-in KMedoids
        # plt.figure(figsize=(8, 6))
        # for i in range(k):
        #     cluster_data = X[np.array(clusters_builtin) == i]
        #     plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i+1}')

        # # Plotting medoids
        # medoids_data = X[medoids_builtin]
        # plt.scatter(medoids_data[:, 0], medoids_data[:, 1], c='black', marker='x', label='Medoids')

        # plt.title('Clusters from Built-in KMedoids')
        # plt.xlabel('Feature 1')
        # plt.ylabel('Feature 2')
        # plt.legend()

        # # Save the image
        # plt.savefig('clusters_plot.png')

        # # Show the plot
        # plt.close()

        # Get cluster medoids
        medoids_builtin = kmedoids.medoid_indices_.tolist()
        clusters_builtin_final=[]
        for x in medoids_builtin:
            cl=[]
            i=1
            print(x)
            for y in clusters_builtin:
                
                if x==medoids_builtin[y]:
                    cl.append(i)
                i+=1
            clusters_builtin_final.append(cl)
                
        print(clusters_builtin_final)
        
        # print(type(medoids_scratch),type(clusters_scratch))
        serialized_object_scratch= json.dumps({'list1': medoids_scratch, 'list2': clusters_scratch})
        serialized_object_builtin = json.dumps({'list1': medoids_builtin, 'list2': clusters_builtin_final})
        
        # print(serialized_object_scratch)
        results = {
            "clusters_scratch":  serialized_object_scratch,
            "clusters_builtin":  serialized_object_builtin
        }
        return JsonResponse({"result":"K Medoids Clustering completed","data":results})
        
    except Exception as e:
        print(e)
        return JsonResponse({'message': f'Error: {str(e)}'}, status=500)
    
    




class BIRCH:
    def __init__(self, threshold, branching_factor, n_clusters):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.root = None

    def fit(self, X):
        self.root = Node(self.branching_factor, self.threshold)
        self.root.build_subclusters(X)

    def predict(self, X):
        labels = []
        for point in X:
            labels.append(self.root.predict(point))
        return labels

class Node:
    def __init__(self, branching_factor, threshold):
        self.branching_factor = branching_factor
        self.threshold = threshold
        self.n = 0
        self.ls = 0
        self.ss = 0
        self.children = []
        self.subclusters = []

    def add_child(self, node):
        self.children.append(node)

    def build_subclusters(self, X):
        self.n = X.shape[0]
        self.ls = np.sum(X, axis=0)
        self.ss = np.sum(np.square(X), axis=0)

        self.subclusters.append(Subcluster(X))
        if self.n > self.branching_factor:
            centroids = [subcluster.centroid for subcluster in self.subclusters]
            centroids = np.array(centroids)
            while len(self.subclusters) > self.branching_factor:
                closest_pair = self.find_closest_pair(centroids)
                new_subcluster = closest_pair[0].merge(closest_pair[1])
                self.subclusters.remove(closest_pair[0])
                self.subclusters.remove(closest_pair[1])
                self.subclusters.append(new_subcluster)
                centroids = [subcluster.centroid for subcluster in self.subclusters]
                centroids = np.array(centroids)
            for subcluster in self.subclusters:
                child = Node(self.branching_factor, self.threshold)
                child.build_subclusters(subcluster.points)
                self.add_child(child)

    def find_closest_pair(self, centroids):
        min_dist = np.inf
        closest_pair = None
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (self.subclusters[i], self.subclusters[j])
        return closest_pair

    def predict(self, point):
        if self.children:
            closest_child = min(self.children, key=lambda child: np.linalg.norm(child.subclusters[0].centroid - point))
            return closest_child.predict(point)
        else:
            return self.subclusters[0].centroid


class Subcluster:
    def __init__(self, points):
        self.points = points
        self.n = points.shape[0]
        self.ls = np.sum(points, axis=0)
        self.ss = np.sum(np.square(points), axis=0)
        self.centroid = self.ls / self.n

    def merge(self, other_subcluster):
        merged_points = np.concatenate((self.points, other_subcluster.points), axis=0)
        return Subcluster(merged_points)
@csrf_exempt
def birchAlgo(request):
    
    try:
        dp = json.loads(request.body)
        df = pd.DataFrame(dp['arrayData'])
        # Assuming 'data' is your DataFrame
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        def apply_zscore(column):
            return zscore_custom(column)

        
        dc = df.copy()
        dc[numerical_columns] = dc[numerical_columns].apply(apply_zscore)

        # Assuming data contains 'target' column with class labels
        # X = [[row[attribute1], row[attribute2]] for row in data]````
        X = df.iloc[:, :-1]
        k = 3
        data = np.array(X, dtype=np.float64)
        birch_custom = BIRCH(threshold=0.5, branching_factor=10, n_clusters=3)
        birch_custom.fit(X)
        labels_custom = birch_custom.predict(X)
        print(labels_custom)
        
        
        birch_builtin = Birch(threshold=0.5, branching_factor=10, n_clusters=3)
        birch_builtin.fit(X)
        labels_builtin = birch_builtin.predict(X)
        print(labels_builtin)     
        results = {
            "ClusterNumber":'result_clusters'
        }
        return JsonResponse({"result":"Agnes Agglomerative Clustering completed","data":results})
        
    except Exception as e:
        print(e)
        return JsonResponse({'message': f'Error: {str(e)}'}, status=500)
    
  
@csrf_exempt
def DBSCAN(request):
    
    try:
        dp = json.loads(request.body)
        df = pd.DataFrame(dp['arrayData'])
        # Assuming 'data' is your DataFrame
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        def apply_zscore(column):
            return zscore_custom(column)

        
        dc = df.copy()
        dc[numerical_columns] = dc[numerical_columns].apply(apply_zscore)

        # Assuming data contains 'target' column with class labels
        # X = [[row[attribute1], row[attribute2]] for row in data]````
        X = df.iloc[:, :-1]
        k = 3
        data = np.array(X, dtype=np.float64)
        result_clusters, distances = agnesAlgo(data, k)
        print(result_clusters)
        
        
        # Agnes Agglomerative Clustering
        # linked = linkage(X, 'ward')
        # labelList = range(1, len(X) + 1)

        # plt.figure(figsize=(10, 7))
        # dendrogram(linked,
        #            orientation='top',
        #            labels=labelList,
        #            distance_sort='descending',
        #            show_leaf_counts=True)
        
        
        # Save the image to a specific location
        # image_path = os.path.join('path_to_your_directory', 'dendrogram.png')
        # plot_dendrogram(result_clusters, distances)
        # plt.savefig(image_path)
        # plt.close()

        
        results = {
            "ClusterNumber":'result_clusters'
        }
        return JsonResponse({"result":"Agnes Agglomerative Clustering completed","data":results})
        
    except Exception as e:
        print(e)
        return JsonResponse({'message': f'Error: {str(e)}'}, status=500)
    
      

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
    attribute_measures = ["gini", "entropy"]
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
    
        destination_directory = f"C:/Users/Admin/Desktop/Test/DM_Assignments/Assignment 1 to 5/Backend/src/Pages/RuleBased/DecisionTree/DecisionTree{i}.png"

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
