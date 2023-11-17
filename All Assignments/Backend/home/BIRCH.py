

import numpy as np

from pyclustering.cluster.agglomerative import agglomerative, type_link
from pyclustering.cluster.encoder import cluster_encoder, type_encoding

from pyclustering.container.cftree import cftree, measurement_type


class birch:

    def __init__(self, data, number_clusters, branching_factor=50, max_node_entries=200, diameter=0.5,
                 type_measurement=measurement_type.CENTROID_EUCLIDEAN_DISTANCE,
                 entry_size_limit=500,
                 diameter_multiplier=1.5,
                 ccore=True):
        self.__pointer_data = data
        self.__number_clusters = number_clusters      
        self.__measurement_type = type_measurement
        self.__entry_size_limit = entry_size_limit
        self.__diameter_multiplier = diameter_multiplier
        self.__ccore = ccore

        self.__verify_arguments()

        self.__features = None
        self.__tree = cftree(branching_factor, max_node_entries, diameter, type_measurement)
        
        self.__clusters = []
        self.__cf_clusters = []


    # def process(self):
 
    #     self.__insert_data()
    #     self.__extract_features()

    #     cf_data = [feature.get_centroid() for feature in self.__features]

    #     algorithm = agglomerative(cf_data, self.__number_clusters, type_link.SINGLE_LINK).process()
    #     self.__cf_clusters = algorithm.get_clusters()

    #     cf_labels = cluster_encoder(type_encoding.CLUSTER_INDEX_LIST_SEPARATION, self.__cf_clusters, cf_data).\
    #         set_encoding(type_encoding.CLUSTER_INDEX_LABELING).get_clusters()

    #     self.__clusters = [[] for _ in range(len(self.__cf_clusters))]
    #     for index_point in range(len(self.__pointer_data)):
    #         index_cf_entry = numpy.argmin(numpy.sum(numpy.square(
    #             numpy.subtract(cf_data, self.__pointer_data[index_point])), axis=1))
    #         index_cluster = cf_labels[index_cf_entry]
    #         self.__clusters[index_cluster].append(index_point)

    #     return self


    def get_clusters(self):
 
        print(self._clusters)
        return self.__clusters


    def get_cf_entries(self):
   
        return self.__features


    def get_cf_cluster(self):

        return self.__cf_clusters


    def get_cluster_encoding(self):

        return type_encoding.CLUSTER_INDEX_LIST_SEPARATION


    def __verify_arguments(self):

        if np.any(len(self.__pointer_data)) == 0:
            raise ValueError("Input data is empty (size: '%d')." % len(self.__pointer_data))

        if self.__number_clusters <= 0:
            raise ValueError("Amount of cluster (current value: '%d') for allocation should be greater than 0." %
                             self.__number_clusters)

        if self.__entry_size_limit <= 0:
            raise ValueError("Limit entry size (current value: '%d') should be greater than 0." %
                             self.__entry_size_limit)


    # def __extract_features(self):

        
    #     self.__features = []
        
    #     if len(self.__tree.leafes) == 1:
    #         # parameters are too general, copy all entries
    #         for entry in self.__tree.leafes[0].entries:
    #             self.__features.append(entry)

    #     else:
    #         # copy all leaf clustering features
    #         for leaf_node in self.__tree.leafes:
    #             self.__features += leaf_node.entries


    # def __insert_data(self):
        
    #     for index_point in range(0, len(self.__pointer_data)):
    #         point = self.__pointer_data[index_point]
    #         self.__tree.insert_point(point)
            
    #         if self.__tree.amount_entries > self.__entry_size_limit:
    #             self.__tree = self.__rebuild_tree(index_point)
    
    
    def __insert_data(self):
        for index_point in range(0, len(self.__pointer_data)):
            point = self.__pointer_data[index_point]
            self.__tree.insert_point(point)

            if np.any(self.__tree.amount_entries > self.__entry_size_limit):
                self.__tree = self.__rebuild_tree(index_point)

    def __extract_features(self):
        self.__features = []

        if np.all(len(self.__tree.leafes)) == 1:
            # parameters are too general, copy all entries
            for entry in self.__tree.leafes[0].entries:
                self.__features.append(entry)
        else:
            # copy all leaf clustering features
            for leaf_node in self.__tree.leafes:
                self.__features += leaf_node.entries

    def __rebuild_tree(self, index_point):
        rebuild_result = False
        increased_diameter = self.__tree.threshold * self.__diameter_multiplier
        tree = None

        while not any([not rebuild_result]):
            # increase diameter and rebuild tree
            if increased_diameter == 0.0:
                increased_diameter = 1.0

            # build tree with updated parameters
            tree = cftree(
                self.__tree.branch_factor,
                self.__tree.max_entries,
                increased_diameter,
                self.__tree.type_measurement
            )

            for idx in range(0, index_point + 1):
                point = self.__pointer_data[idx]
                tree.insert_point(point)

                if np.any(tree.amount_entries > self.__entry_size_limit):
                    increased_diameter *= self.__diameter_multiplier
                    continue

            # Re-build is successful.
            rebuild_result = True

        return tree

    def process(self):
        self.__insert_data()
        self.__extract_features()

        cf_data = [feature.get_centroid() for feature in self.__features]

        algorithm = agglomerative(cf_data, self.__number_clusters, type_link.SINGLE_LINK).process()
        self.__cf_clusters = algorithm.get_clusters()

        cf_labels = cluster_encoder(type_encoding.CLUSTER_INDEX_LIST_SEPARATION, self.__cf_clusters, cf_data). \
            set_encoding(type_encoding.CLUSTER_INDEX_LABELING).get_clusters()

        self.__clusters = [[] for _ in range(len(self.__cf_clusters))]
        for index_point in range(np.any(len(self.__pointer_data))):
            index_cf_entry = np.argmin(np.sum(np.square(np.subtract(cf_data, self.__pointer_data[index_point])), axis=1))
            index_cluster = cf_labels[index_cf_entry]
            self.__clusters[index_cluster].append(index_point)

        return self