from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import json
from django.views.decorators.csrf import csrf_exempt
from io import TextIOWrapper
import numpy as np
import csv
import networkx as nx

# Create your views here.
# def findMin(centroids,data):
#     for row in centroids:



# class KMeans:

#     def __init__(self,k,max_iterations) -> None:
#         self.k = k
#         self.iterations = max_iterations

    
#     def fit(self,data):
#         self.centroids = data.sample(self.k)
        
#         for _ in range(self.iterations):
#             self.classifications = {}
#             for i in range(self.k):
#                 self.classifications[i]=[]
def hits_algorithm2(graph,max_iter=10,tol=1e-6):


    hubs = np.ones(graph.nodes)
    authorities = np.ones(graph.nodes)

    for _ in range(max_iter):

        new_authorities = np.dot(graph.adjacency_matrix().T,hubs)

        new_authorities /= np.linalg.norm(new_authorities,2)


        new_hubs = np.dot(graph.adjacency_matrix().T,new_authorities)

        new_hubs /= np.linalg.norm(new_hubs,2)

        if np.linalg.norm(new_hubs-new_authorities,2)<tol or np.linalg.norm(hubs-authorities,2)<tol:
            break

        hubs = new_hubs
        authorities = new_authorities

    

    return hubs,authorities



def hits_algorithm(graph, max_iter=100, tol=1e-6):
    # Initialize hub and authority scores
    hubs = {node: 1.0 for node in graph.nodes()}
    authorities = {node: 1.0 for node in graph.nodes()}

    for _ in range(max_iter):
        # Update authority scores
        new_authorities = {node: sum(hubs[neighbors] for neighbors in graph.neighbors(node))
                           for node in graph.nodes()}

        # Normalize authority scores
        norm_authorities = sum(val**2 for val in new_authorities.values())**0.5
        authorities = {node: score / norm_authorities for node, score in new_authorities.items()}

        # Update hub scores
        new_hubs = {node: sum(authorities[neighbor] for neighbor in graph.neighbors(node))
                     for node in graph.nodes()}

        # Normalize hub scores
        norm_hubs = sum(val**2 for val in new_hubs.values())**0.5
        hubs = {node: score / norm_hubs for node, score in new_hubs.items()}

    return hubs, authorities
      



@csrf_exempt
def HITSAlgo(request):

    try:
        if 'data' in request.FILES:
            newFile = request.FILES['data']
          
            # # Use TextIOWrapper to handle the uploaded file content
            file_wrapper = TextIOWrapper(newFile.file, encoding=request.encoding) 
      
            # df = pd.DataFrame(file_wrapper)
            csv_reader = csv.reader(file_wrapper)
            next(csv_reader)
            edges = [(int(row[0]),int(row[1])) for row in csv_reader]
            
            # print(edges)

            G = nx.DiGraph(edges)

            hubs,authorities = hits_algorithm(G)

            
            top_hubs = np.argsort(hubs)[::-1][:10]

            for i,node in enumerate(hubs):
                print(i," ",node)
        # print("Request recieved...")

    except Exception as e:
        print(e)

    return JsonResponse({'msg':'Response Received'})


