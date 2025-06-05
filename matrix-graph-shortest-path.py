import time
import heapq
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class DisjointSet:
    def __init__(self, vertices):
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices}

    def find(self, vertex):
        if self.parent[vertex] != vertex:
            self.parent[vertex] = self.find(self.parent[vertex])
        return self.parent[vertex]

    def union(self, root1, root2):
        root1 = self.find(root1)
        root2 = self.find(root2)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1

def kruskal(vertices, edges):
    start_time = time.time()
    mst = []
    disjoint_set = DisjointSet(vertices)
    edges.sort(key=lambda x: x[2])  # Sorting edges by weight
    for u, v, weight in edges:
        if disjoint_set.find(u) != disjoint_set.find(v):
            disjoint_set.union(u, v)
            mst.append((u, v, weight))
    end_time = time.time()
    return mst, end_time - start_time

def prims(vertices, adjacency_list, start_vertex):
    start_time = time.time()
    min_heap = []
    in_mst = {start_vertex}
    mst = []
    for next_vertex, weight in adjacency_list[start_vertex]:
        heapq.heappush(min_heap, (weight, start_vertex, next_vertex))
    while min_heap:
        weight, u, v = heapq.heappop(min_heap)
        if v not in in_mst:
            in_mst.add(v)
            mst.append((u, v, weight))
            for next_vertex, weight in adjacency_list[v]:
                if next_vertex not in in_mst:
                    heapq.heappush(min_heap, (weight, v, next_vertex))
    end_time = time.time()
    return mst, end_time - start_time

def load_graph_example(example_number):
    examples = {
        1: {
            'vertices': [1, 2, 3, 4, 5],
            'edges': [(1, 2, 10), (2, 3, 15), (1, 3, 5), (4, 5, 10), (2, 5, 5), (3, 4, 10)],
            'adjacency_list': {
                1: [(2, 10), (3, 5)],
                2: [(1, 10), (3, 15), (5, 5)],
                3: [(1, 5), (2, 15), (4, 10)],
                4: [(3, 10), (5, 10)],
                5: [(4, 10), (2, 5)]

            },
            'start_vertex': 1
        },
        2: {
            'vertices': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'edges': [
                (1, 2, 7), (1, 3, 9), (1, 6, 14), (2, 3, 10), (2, 4, 15), (3, 6, 2), (3, 4, 11),
                (4, 5, 6), (5, 6, 9), (5, 7, 5), (6, 7, 12), (7, 8, 7), (7, 9, 6), (8, 9, 4),
                (8, 10, 3), (9, 10, 9), (7, 10, 10), (5, 10, 8), (1, 10, 5)
            ],
            'adjacency_list': {
                1: [(2, 7), (3, 9), (6, 14), (10, 5)],
                2: [(1, 7), (3, 10), (4, 15)],
                3: [(1, 9), (2, 10), (4, 11), (6, 2)],
                4: [(2, 15), (3, 11), (5, 6)],
                5: [(4, 6), (6, 9), (7, 5), (10, 8)],
                6: [(1, 14), (3, 2), (5, 9), (7, 12)],
                7: [(5, 5), (6, 12), (8, 7), (9, 6), (10, 10)],
                8: [(7, 7), (9, 4), (10, 3)],
                9: [(7, 6), (8, 4), (10, 9)],
                10: [(1, 5), (5, 8), (7, 10), (8, 3), (9, 9)]
            },
            'start_vertex': 1
        },
        3: {
            'vertices': list(range(1, 101)),  # Vertices are numbered from 1 to 100
            'edges': [],
            'adjacency_list': {i: [] for i in range(1, 101)},
            'start_vertex': 1
        },
        # Additional examples can be added here
    }

    # Only generate edges for the third example if it's selected
    if example_number == 3:
        num_vertices = 100
        density = 0.05  # Moderate density
        max_weight = 100
        vertices = examples[3]['vertices']
        edges = []
        adjacency_list = {i: [] for i in vertices}

        # Create a connected graph first
        for i in range(1, num_vertices):
            weight = random.randint(1, max_weight)
            edges.append((i, i + 1, weight))
            adjacency_list[i].append((i + 1, weight))
            adjacency_list[i + 1].append((i, weight))

        # Add additional edges to increase density
        additional_edges_count = int(num_vertices * (num_vertices - 1) / 2 * density) - (num_vertices - 1)
        for _ in range(additional_edges_count):
            u = random.randint(1, num_vertices)
            v = random.randint(1, num_vertices)
            while u == v or (v, u, weight) in edges or (u, v, weight) in edges:
                u = random.randint(1, num_vertices)
                v = random.randint(1, num_vertices)
            weight = random.randint(1, max_weight)
            edges.append((u, v, weight))
            adjacency_list[u].append((v, weight))
            adjacency_list[v].append((u, weight))

        examples[3]['edges'] = edges
        examples[3]['adjacency_list'] = adjacency_list

    return examples.get(example_number, None)

def plot_graph(vertices, edges, mst_edges, title):
    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_weighted_edges_from(edges)

    pos = nx.spring_layout(G)  # positions for all nodes

    # Draw the graph with all edges
    nx.draw_networkx_nodes(G, pos, node_size=700)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1, alpha=0.5, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d for u, v, d in edges}, font_color='red')

    # Draw the MST edges with different properties
    nx.draw_networkx_edges(G, pos, edgelist=mst_edges, width=2, alpha=0.7, edge_color='blue')
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

    plt.title(title)
    plt.axis('off')  # Turn off the axis
    plt.show()

def visualize_kruskals(vertices, edges, mst_edges):
    plot_graph(vertices, edges, mst_edges, "Kruskal's MST")

def visualize_prims(vertices, edges, mst_edges):
    plot_graph(vertices, edges, mst_edges, "Prim's MST")

def time_complexity_plots():
    ns = [5, 10, 100]
    times_kruskal = []
    times_prims = []
    expected_times = []

    for n in ns:
        vertices = list(range(n))
        edges = [(i, (i+1)%n, random.randint(1, 100)) for i in range(n)]
        adjacency_list = {i: [(j, random.randint(1, 100)) for j in vertices if i != j] for i in vertices}

        # Measure times
        _, time_kruskal = kruskal(vertices, edges)
        _, time_prims = prims(vertices, adjacency_list, 0)

        # Store times
        times_kruskal.append(time_kruskal)
        times_prims.append(time_prims)
        expected_times.append(n * np.log(n) / 20000)  # Scaling for visibility

    plt.figure(figsize=(10, 5))
    plt.plot(ns, times_kruskal, label="Kruskal's Time", marker='o')
    plt.plot(ns, times_prims, label="Prim's Time", marker='o')
    plt.plot(ns, expected_times, label="Expected O(n log n)", linestyle='--')
    plt.xlabel("Number of vertices (n)")
    plt.ylabel("Time (seconds)")
    plt.title("Algorithm Time Complexity Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    while True:
        print("\nAvailable Examples:")
        print("1. Small example graph")
        print("2. Medium example graph")
        print("3. Large example graph")
        print("0. Exit")
        choice = input("Select an example or 0 to exit: ")

        if choice == '0':
            break
        elif choice.isdigit() and int(choice) in range(1, 4):
            example = load_graph_example(int(choice))
            if example:
                vertices = example['vertices']
                edges = example['edges']
                adjacency_list = example['adjacency_list']
                start_vertex = example['start_vertex']

                mst_kruskal, time_kruskal = kruskal(vertices, edges)
                mst_prims, time_prims = prims(vertices, adjacency_list, start_vertex)

                print("\nKruskal's MST:", mst_kruskal)
                print("Kruskal's Execution time: {:.9f} seconds".format(time_kruskal))
                print("Prim's MST:", mst_prims)
                print("Prim's Execution time: {:.9f} seconds".format(time_prims))



                visualize_kruskals(vertices, edges, mst_kruskal)
                visualize_prims(vertices, edges, mst_prims)

                time_complexity_plots()
            else:
                print("Invalid  number.")
        else:
            print


if __name__ == '__main__':
    main()
