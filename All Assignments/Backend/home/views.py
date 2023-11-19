from django.shortcuts import render
import json
import urllib
from sklearn.decomposition import PCA
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
import sklearn
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
import matplotlib
matplotlib.use('Agg')
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
from sklearn.tree import _tree
from sklearn.tree import export_graphviz
import graphviz
# from anytree import Node, RenderTree
from .BIRCH import birch
from .DBSCAN import dbscan
from .APRIORI import generate_rules,APRIORI
from sklearn.cluster import DBSCAN
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
import requests
from bs4 import BeautifulSoup
import networkx as nx


def web_crawler(seed_url, method='dfs',max_depth=3):
    # Implement DFS or BFS crawler logic here
    # Return a list of crawled links
    # For simplicity, I'm using a basic example of BFS crawler
    visited = set()
    queue = [(seed_url,0)]
    links = []

    while queue:
        current_url, depth = queue.pop(0)


        if current_url not in visited and depth<=max_depth:
            try:
                response = requests.get(current_url)
                soup = BeautifulSoup(response.text, 'html.parser')
                links_on_page = [a['href'] for a in soup.find_all('a', href=True)]
                links.extend(links_on_page)
                visited.add(current_url)
                queue.extend([(link, depth + 1) for link in links_on_page])
            except Exception as e:
                print(f"Error crawling {current_url}: {e}")

    return links

@csrf_exempt
def crawl(request):
    try:
        dp = json.loads(request.body)
        seed_url = dp['seed_url']
        # print(seed_url)
        if seed_url:
            links = web_crawler(seed_url)
            # print(links)
            return JsonResponse({'links': links})
    except Exception as e:
        print(e)
        return JsonResponse({"msg":"Error occurred "})


@csrf_exempt
def calculate_hits(request):
    try:
        # Path to the CSV file
        csv_file_path = 'D:/Projects/Data Mining Assignment/All Assignments/Backend/home/NodeEdges.csv'

        # Read edges from the CSV file
        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # Skip the header row
            edges = [(int(row[0]), int(row[1])) for row in csv_reader]

        # Create a directed graph using NetworkX
        G = nx.DiGraph(edges)

        # Run the HITS algorithm
        hubs, authorities = nx.hits(G)

        # Get the top 10 authorities and hubs
        top_authorities = sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:10]
        top_hubs = sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:10]

        # Convert the graph to an adjacency matrix
        adjacency_matrix = nx.to_numpy_matrix(G)

        # Tabulate the results
        results = {
            'adjacency_matrix': adjacency_matrix.tolist(),
            'authority_rank': top_authorities,
            'hub_rank': top_hubs,
        }
        return JsonResponse({'msg':'Request processed.....','data':results})
    except Exception as e:
        print(e)
        return JsonResponse({"msg":"Error occurred "})


@csrf_exempt
def calculate_pagerank(request):
    try:
        # Read CSV file and create a directed graph
        G = nx.DiGraph()
        with open("D:/Projects/Data Mining Assignment/All Assignments/Backend/home/NodeEdges.csv",mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                from_node = row['FromNodeId']
                to_node = row['ToNodeId']
                G.add_edge(from_node, to_node)

        # Calculate PageRank
        pagerank_scores = nx.pagerank(G)

        # Get the 10 pages with the highest rank
        top_pages = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        # Tabulate the results containing the adjacency matrix and rank of pages
        adjacency_matrix = nx.adjacency_matrix(G).todense().tolist()
        rank_table = [{'Page': page, 'Rank': rank} for page, rank in top_pages]

        return JsonResponse({'adjacency_matrix': adjacency_matrix, 'rank_table': rank_table})
        return JsonResponse({'msg':'Request processed.....'})
    except Exception as e:
        print(e)
        return JsonResponse({"msg":"Error occurred "})

@csrf_exempt
def AprioriAlgo(request):
    try:
    
        # Step 1: Load and preprocess the dataset
        # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data"
        column_names = [
            "Class Name",
            "handicapped-infants",
            "water-project-cost-sharing",
            "adoption-of-the-budget-resolution",
            "physician-fee-freeze",
            "el-salvador-aid",
            "religious-groups-in-schools",
            "anti-satellite-test-ban",
            "aid-to-nicaraguan-contras",
            "mx-missile",
            "immigration",
            "synfuels-corporation-cutback",
            "education-spending",
            "superfund-right-to-sue",
            "crime",
            "duty-free-exports",
            "export-administration-act-south-africa",
        ]
        # warnings.filterwarnings("ignore", category=DeprecationWarning)
        # df = pd.read_csv(url, header=None, names=column_names)

        # # Using only 15% of the data for processing
        # df = df.sample(frac=0.25, random_state=1)

        # df = df.fillna(df.mode().iloc[0])
        # df = df.replace({"y": 1, "n": 0, "?": 0})

        # Step 2: Implement the Apriori algorithm with varying support, confidence, and maximum rule length
        supports = [0.5]  # Vary support values
        confidences = [0.6]  # Vary confidence values
        max_len = 2  # Maximum rule length

        results = []
     
        # df.to_csv("data.csv",index=False)

        df = pd.read_csv("D:/Projects/Data Mining Assignment/All Assignments/Backend/data.csv", header=None, names=column_names)
        

        for support in supports:
            for confidence in confidences:
                
                frequent_itemsets = APRIORI(df,support, max_len,column_names)
                # print(frequent_itemsets)
                test_data = dict()
                
                for itemset in frequent_itemsets:
                    test_data.update(itemset)
                rules = generate_rules(test_data, min_confidence=confidence)
                
                # Using the Apriori algorithm
                # frequent_itemsets = apriori(df.drop("Class Name", axis=1), support, max_len)
              
                # # Using the association rules function
                # rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)
                # print(rules)
                # Convert frozensets to lists for JSON serialization
                # rules['antecedents'] = rules['antecedents'].apply(list)
                # rules['consequents'] = rules['consequents'].apply(list)

                # # Calculate the Kulczynski measure manually
                # rules['kulczynski'] = (rules['support'] + rules['confidence']) / 2
                # print("Support",support)
                # print("confidence",confidence)
                # print("frequent_itemsets",len(frequent_itemsets))
                # print("total_rules",len(rules))
                # print("interestingness_measures",rules.to_dict(orient="records"))
                results.append({
                    "frequent_itemsets": len(frequent_itemsets),
                    "total_rules": len(rules),
                    "Rules": rules
                })
              
            
                print(rules)
        return JsonResponse({"results": results})




        # Step 3: Tabulate the results for frequent item sets and the total number of generated rules
        # results_df = pd.DataFrame(results, columns=["Support", "Confidence", "Frequent Itemsets", "Total Rules"])
        # print(results_df)

        # interestingness_measures = ["lift", "leverage", "conviction", "confidence", "kulczynski", "cosine"]

        # for support, confidence, _, _ in results:
        #     frequent_itemsets = apriori(df.drop("Class Name", axis=1), min_support=support, use_colnames=True, max_len=max_len)
        #     rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)
        #     print(f"\nSupport: {support}, Confidence: {confidence}\n")
            # print(rules)
            # print(type(rules))
            # for measure in interestingness_measures:
            #     if measure == 'confidence':
            #         continue  # Skip calculating confidence again
            #     if measure == 'lift':
            #         rules[measure] = rules['lift']
            #     elif measure == 'leverage':
            #         rules[measure] = rules['leverage']
            #     elif measure == 'conviction':
            #         rules[measure] = rules['conviction']
            #     elif measure == 'kulczynski':
            #         rules[measure] = (rules['support'] + rules['confidence']) / 2
            #     elif measure == 'cosine':
            #         rules[measure] = rules['cosine']

            # print(rules)

        
        return JsonResponse({"msg":"Apriori Algo Executed.... "})

    except Exception as e:
        print(e)
        return JsonResponse({"msg":"Error occurred "})







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
    image_path = 'D:/Projects/Data Mining Assignment/All Assignments/Frontend/src/Pages/Hierarchial/dendrogram.png'
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
        image_path = 'D:/Projects/Data Mining Assignment/All Assignments/Frontend/src/Pages/Hierarchial/AGNES/dendrogram.png'
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
        image_path = 'D:/Projects/Data Mining Assignment/All Assignments/Frontend/src/Pages/Hierarchial/DIANA/dendrogram.png'
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
        plt.savefig('D:/Projects/Data Mining Assignment/All Assignments/Frontend/src/Pages/K-Means/kmeans_clusters.png')
        plt.close()

        results = {
            "kmeans_scratch": data_serializable_scratch,
            "kmeans_builtin": data_serializable_builtin
        }

        return JsonResponse({"result": "K-means Clustering completed", "data": results})
        
    except Exception as e:
        print(e)
        return JsonResponse({'message': f'Error: {str(e)}'}, status=500)
    


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
    
    




# class BIRCH:
#     def __init__(self, threshold, branching_factor, n_clusters):
#         self.threshold = threshold
#         self.branching_factor = branching_factor
#         self.n_clusters = n_clusters
#         self.root = None
#         self.subclusters = []

#     def fit(self, X):
#         self.root = SubclusterNode(X[0])
#         self.root.is_leaf = True
#         self.root.N = 1
#         self.subclusters.append(self.root)
#         for i in range(1, X.shape[0]):
#             self.insert_new_point(X[i])

#     def insert_new_point(self, point):
#         self.root = self.insert(self.root, point)

#     def insert(self, node, point):
#         if node.is_leaf:
#             if node.N < self.branching_factor:
#                 node.insert(point)
#                 return node
#             else:
#                 subcluster = SubclusterNode(point)
#                 subcluster.N = 1
#                 self.subclusters.append(subcluster)
#                 return subcluster
#         else:
#             best_subcluster = None
#             best_distance = np.inf
#             for child in node.children:
#                 distance = np.linalg.norm(point - child.centroid)
#                 if distance < best_distance:
#                     best_distance = distance
#                     best_subcluster = child
#             if best_subcluster is not None:
#                 updated_subcluster = self.insert(best_subcluster, point)
#                 node.update_centroid()
#                 return node
#             else:
#                 # Handle the case when best_subcluster is None
#                 return node  # or perform some other action as needed
#     def print_tree(self, node, level=0):
#         if node:
#             print('  ' * level + str(node.centroid))
#             if not node.is_leaf:
#                 for child in node.children:
#                     self.print_tree(child, level + 1)


# class SubclusterNode:
#     def __init__(self, point):
#         self.points = [point]
#         self.N = 0
#         self.centroid = point
#         self.children = []
#         self.is_leaf = False

#     def insert(self, point):
#         self.points.append(point)
#         self.N += 1
#         self.update_centroid()

#     def update_centroid(self):
#         self.centroid = np.mean(self.points, axis=0)


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

        # # Assuming data contains 'target' column with class labels
        # # X = [[row[attribute1], row[attribute2]] for row in data]````
        # X = df.iloc[:, :-1]
        # data = np.array(X, dtype=np.float64)
        # birch_custom = BIRCH(threshold=500, branching_factor=2, n_clusters=3)
        # birch_custom.fit(data)
        # # Define the tree structure
        # root = Node(str(birch_custom.root.centroid))
        # for child in birch_custom.root.children:
        #     child_node = Node(str(child.centroid), parent=root)
        #     for subchild in child.children:
        #         subchild_node = Node(str(subchild.centroid), parent=child_node)

        # # Print the tree structure
        # for pre, fill, node in RenderTree(root):
        #     print("%s%s" % (pre, node.name))
        
        
        # # Create a basic plot for the tree structure
        # fig, ax = plt.subplots(figsize=(6, 6))
        # for pre, _, node in RenderTree(root):
        #     x, y = len(pre) / 2, node.depth * -0.5
        #     ax.text(x, y, node.name, ha='center', va='center', bbox=dict(facecolor='lightblue', alpha=0.5, boxstyle='round'))

        # ax.set_axis_off()
        # plt.savefig('D:/Projects/Data Mining Assignment/Assignment 1 to 5/Frontend/test.png')
        # Import required libraries and modules


        '''Birch.py file functions'''
       
        # model = birch(data,6,50)
        # model.process()

        # clusters = model.get_cf_cluster()
        # print(clusters)
        
        # Creating the BIRCH clustering model
        # model = Birch(branching_factor = 50, n_clusters = None, threshold = 1.5)

        # # Fit the data (Training)
        # model.fit(data)

      

        # Print the CF Tree
        # def print_tree(node, depth=0):
        #     if hasattr(node, 'subcluster_centers_'):
        #         print('  ' * depth, f"Subcluster with {len(node.subcluster_centers_)} subclusters")
        #     else:
        #         print('  ' * depth, "CF node")

        #     if hasattr(node, 'children_'):
        #         for child in node.children_:
        #             print_tree(child, depth + 1)


        # print_tree(model.root_)

        # # Creating a scatter plot
        # plt.scatter(X[:, 0], X[:, 1], c = pred, cmap = 'rainbow', alpha = 0.7, edgecolors = 'b')
        # image_path = os.path.join('D:/Projects/Data Mining Assignment/All Assignments/Backend/src/Pages/BIRCH', 'dendrogram.png')
        # plt.savefig(image_path)
        # plt.close()
       
         # Assuming data contains 'target' column with class labels
        X = df.iloc[:, :-1]
        data = np.array(X, dtype=np.float64)
        
        # Standardize the data
        Standardized_data = StandardScaler().fit_transform(data)


        # Apply PCA for visualization purposes (reduce to 2 dimensions)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(Standardized_data)

        # Birch Clustering
        birch_cluster = Birch(threshold=0.5, n_clusters=None)
        birch_labels = birch_cluster.fit_predict(Standardized_data)


        # Visualize the clusters
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=birch_labels, cmap='viridis', edgecolor='k')
        plt.title('Birch Clustering')


        # Save the images
        plt.savefig('D:/Projects/Data Mining Assignment/All Assignments/Frontend/src/Pages/BIRCH/birch_clusters.png', bbox_inches='tight')
        plt.close()
                # Plot result (you may need to adapt this part based on how you want to visualize BIRCH results)
        # ...

        # Convert the tuple to a list for serialization
        # colors = [list(color) for color in colors]

        # # Convert int64 type to regular integer
        # n_clusters_ = int(n_clusters_)

        # # Convert the set to a list
        # unique_labels = list(set(labels))

        # # Create a dictionary with the data
        # data = {"unique_labels": unique_labels,"clusters":n_clusters_}

        # # Convert the data to JSON, ensuring that int64 values are converted to integers
        # def convert_to_json_serializable(obj):
        #     if isinstance(obj, np.int64):
        #         return int(obj)
        #     raise TypeError("Type not serializable")

        # serialized_data = json.dumps(data, default=convert_to_json_serializable)
            
        # print(serialized_data)
        return JsonResponse({"result":"BIRCH completed","data":'serialized_data'})
        
    except Exception as e:
        print(e)
        return JsonResponse({'message': f'Error: {str(e)}'}, status=500)
    
  
@csrf_exempt
def DBSCANAlgo(request):
    
    try:
        dp = json.loads(request.body)
        df = pd.DataFrame(dp['arrayData'])
        # Assuming 'data' is your DataFrame
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        def apply_zscore(column):
            return zscore_custom(column)

        print("eps",dp['eps'])
        print("Min samples",dp['min_samples'])
        dc = df.copy()
        dc[numerical_columns] = dc[numerical_columns].apply(apply_zscore)

        # Assuming data contains 'target' column with class labels
        # X = [[row[attribute1], row[attribute2]] for row in data]````
        X = df.iloc[:, :-1]
        k = 3
        data = np.array(X, dtype=np.float64)
        # result_clusters, distances = agnesAlgo(data, k)
        # print(result_clusters)
        
        # Standardize the data
        Standardized_data = StandardScaler().fit_transform(data)
        # model = dbscan()

        # cl =model.predict(Standardized_data)
        # print(cl)
        
        # print(sklearn.__version__)
        # Compute DBSCAN
        dbscan = DBSCAN(eps=dp['eps'], min_samples=dp['min_samples'])
        dbscan.fit(Standardized_data)

        # Extract the labels and core sample indices
        labels = dbscan.labels_
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[dbscan.core_sample_indices_] = True

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        # Plot result
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        print(n_clusters_,unique_labels,colors)
        # Plot result
        # Create the plot
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = Standardized_data[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)

            xy = Standardized_data[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        # plt.show()
        plt.savefig('D:/Projects/Data Mining Assignment/All Assignments/Frontend/src/Pages/DBSCAN/DBSCAN.png')
   
        

       # Convert the tuple to a list for serialization
        colors = [list(color) for color in colors]

        # Convert int64 type to regular integer
        n_clusters_ = int(n_clusters_)

        # Convert the set to a list
        unique_labels = list(unique_labels)

        # Create a dictionary with the data
        data = {"unique_labels": unique_labels, "colors": colors}

        # Convert the data to JSON, ensuring that int64 values are converted to integers
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.int64):
                return int(obj)
            raise TypeError("Type not serializable")

        serialized_data = json.dumps(data, default=convert_to_json_serializable)
        return JsonResponse({"result":"DBSCAN Clustering completed","data":serialized_data})
        
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
    
        destination_directory = f"D:/Projects/Data Mining Assignment/All Assignments/Frontend/src/Pages/RuleBased/DecisionTree/DecisionTree{i}.png"

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
