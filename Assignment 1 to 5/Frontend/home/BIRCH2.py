import numpy as np

class CFVector:
    def __init__(self, n=0, LS=None, SS=None,n_features=4):
        self.n = n
        self.LS = np.zeros(n_features) if LS is None else LS
        self.SS = np.zeros(n_features) if SS is None else SS

class Node:
    def __init__(self, parent=None, is_leaf=False):
        self.parent = parent
        self.is_leaf = is_leaf
        self.children = []
        self.cf_vector = CFVector()
        self.radius = 0

class CFTree:
    def __init__(self, branching_factor=50, threshold=0.7, is_leaf=True):
        self.root = Node(is_leaf=is_leaf)
        self.branching_factor = branching_factor
        self.threshold = threshold

    def insert(self, point):
        # If the tree is empty, create a new leaf node
        if len(self.root.children) == 0:
            leaf_node = Node(parent=self.root, is_leaf=True)
            leaf_node.cf_vector = CFVector(n=1, LS=point, SS=np.square(point))
            self.root.children.append(leaf_node)
            return

        # Find the leaf node to insert the point
        leaf_node = self._find_leaf_node(self.root, point)

        # Insert the point into the leaf node
        self._insert_into_node(leaf_node, point)

    def _find_leaf_node(self, node, point):
        # If the node is a leaf, return it
        if node.is_leaf:
            return node

        # If the node is not a leaf, find the closest child
        min_distance = float('inf')
        closest_child = None
        for child in node.children:
            distance = self._distance(child.cf_vector, point)
            if distance < min_distance:
                min_distance = distance
                closest_child = child

        # Recursively find the leaf node
        return self._find_leaf_node(closest_child, point)

    def _insert_into_node(self, node, point):
        # Update the node's CF vector
        node.cf_vector.n += 1
        node.cf_vector.LS += point
        node.cf_vector.SS += point**2

        # If the node's radius exceeds the threshold, split the node
        node.radius = self._compute_radius(node.cf_vector)
        if node.radius > self.threshold:
            self._split_node(node)
    
class CFTree(CFTree):
    def _distance(self, cf_vector, point):
        centroid = cf_vector.LS / cf_vector.n
        return np.linalg.norm(centroid - point)

    def _compute_radius(self, cf_vector):
        centroid = cf_vector.LS / cf_vector.n
        return np.sqrt(cf_vector.SS / cf_vector.n - centroid**2)


    def _split_node(self, node):
        # Create two new nodes
        new_node1 = Node(parent=node.parent, is_leaf=node.is_leaf)
        new_node2 = Node(parent=node.parent, is_leaf=node.is_leaf)

        # Redistribute the data points between the two new nodes
        points = [child.cf_vector.LS / child.cf_vector.n for child in node.children]
        centroids = [sum(point) / len(point) for point in points]
        points_partition = [points[i] for i in range(len(points)) if centroids[i] < np.median(centroids)]
        for point in points_partition:
            self._insert_into_node(new_node1, point)
        for point in points:
            if point not in points_partition:
                self._insert_into_node(new_node2, point)

        # Replace the original node with the two new nodes in the parent node
        index = node.parent.children.index(node)
        node.parent.children[index] = new_node1
        node.parent.children.insert(index+1, new_node2)

        # If the parent node now has too many children, split the parent node
        if len(node.parent.children) > self.branching_factor:
            self._split_node(node.parent)

    def global_clustering(self, n_clusters):
        # Get the leaf nodes
        leaf_nodes = self._get_leaf_nodes(self.root)

        # Initialize each leaf node as a separate cluster
        clusters = [[leaf_node] for leaf_node in leaf_nodes]

        # Iteratively merge the closest pair of clusters until the desired number of clusters is reached
        while len(clusters) > n_clusters:
            # Find the closest pair of clusters
            min_distance = float('inf')
            closest_pair = None
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    distance = self._compute_cluster_distance(clusters[i], clusters[j])
                    if distance < min_distance:
                        min_distance = distance
                        closest_pair = (i, j)

            # Merge the closest pair of clusters
            clusters[closest_pair[0]].extend(clusters[closest_pair[1]])
            del clusters[closest_pair[1]]

        # Return the clusters
        return clusters

    def _get_leaf_nodes(self, node):
        # If the node is a leaf, return it
        if node.is_leaf:
            return [node]

        # If the node is not a leaf, get the leaf nodes from the children
        leaf_nodes = []
        for child in node.children:
            leaf_nodes.extend(self._get_leaf_nodes(child))

        return leaf_nodes

    def _compute_cluster_distance(self, cluster1, cluster2):
        # Compute the distance between two clusters as the average distance between their centroids
        centroids1 = [node.cf_vector.LS / node.cf_vector.n for node in cluster1]
        centroids2 = [node.cf_vector.LS / node.cf_vector.n for node in cluster2]
        return sum(self._distance(centroid1, centroid2) for centroid1 in centroids1 for centroid2 in centroids2) / (len(centroids1) * len(centroids2))

