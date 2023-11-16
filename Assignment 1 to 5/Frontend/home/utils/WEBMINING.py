import requests
from bs4 import BeautifulSoup
from collections import deque

def BFS_crawler(seed_url):
    queue = deque([seed_url])
    visited = set()

    while queue:
        current_url = queue.popleft()
        if current_url in visited:
            continue

        print(f"Crawling: {current_url}")
        visited.add(current_url)

        # Fetch links from the current_url
        response = requests.get(current_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = [link.get('href') for link in soup.find_all('a') if link.get('href') is not None]

        # Process the fetched links
        print(f"Links obtained from {current_url}:")
        for link in links:
            print(link)
            if link not in visited:
                queue.append(link)

seed_url = "https://example.com"  # Replace with your seed URL
BFS_crawler(seed_url)



import numpy as np

def calculate_PageRank(adjacency_matrix, damping_factor=0.85, iterations=10):
    n = len(adjacency_matrix)
    page_rank = np.ones(n) / n

    for _ in range(iterations):
        new_page_rank = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if adjacency_matrix[j, i] != 0:
                    new_page_rank[i] += page_rank[j] / float(np.sum(adjacency_matrix[j, :]))

        page_rank = (1 - damping_factor) / n + damping_factor * new_page_rank

    return page_rank

# Your code to read the web graph dataset and build the adjacency matrix here
# Example adjacency matrix (replace with your data)
adjacency_matrix = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]])
page_rank = calculate_PageRank(adjacency_matrix, damping_factor=0.85, iterations=10)

# Output the page ranks
for i, rank in enumerate(page_rank):
    print(f"Page {i} Rank: {rank}")


import numpy as np

def calculate_HITS(adjacency_matrix, iterations=10):
    n = len(adjacency_matrix)

    authority = np.ones(n)
    hub = np.ones(n)

    for _ in range(iterations):
        new_authority = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if adjacency_matrix[j, i] != 0:
                    new_authority[i] += hub[j]

        authority = new_authority

        new_hub = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if adjacency_matrix[i, j] != 0:
                    new_hub[i] += authority[j]

        hub = new_hub

    return authority, hub

# Your code to read the web graph dataset and build the adjacency matrix here
# Example adjacency matrix (replace with your data)
adjacency_matrix = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]])
authority, hub = calculate_HITS(adjacency_matrix, iterations=10)

# Output the authorities and hubs
print(f"Authorities: {authority}")
print(f"Hubs: {hub}")
