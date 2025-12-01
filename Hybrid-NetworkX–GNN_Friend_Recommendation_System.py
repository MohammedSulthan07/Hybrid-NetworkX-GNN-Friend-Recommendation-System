# ============================================================
#  🌐 Hybrid NetworkX – GNN Framework with Visualization
#  (Colab-Compatible Version of SNAP.py + GNN Friend Recommender)
# ============================================================

import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
import numpy as np
import random
import matplotlib.pyplot as plt

# ------------------------------
# Step 1: Build Graph
# ------------------------------
def build_graph(num_nodes=50, edge_prob=0.05):
    """
    Creates a synthetic social graph.
    Nodes = users, Edges = friendships.
    """
    G = nx.barabasi_albert_graph(num_nodes, 3)
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

# ------------------------------
# Step 2: Convert to PyTorch Geometric Data
# ------------------------------
def nx_to_pyg(G):
    data = from_networkx(G)
    data.x = torch.randn((G.number_of_nodes(), 8))  # Random node features
    return data

# ------------------------------
# Step 3: Define GCN Model
# ------------------------------
class GCNModel(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# ------------------------------
# Step 4: Train GCN
# ------------------------------
def train_gcn(data, epochs=100):
    model = GCNModel(data.num_features, 16, 8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        src, dst = data.edge_index
        loss = F.mse_loss(out[src], out[dst])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")

    embeddings = model(data).detach()
    return embeddings

# ------------------------------
# Step 5: Recommend Friends
# ------------------------------
def recommend_friend(node_id, embeddings, k=3):
    sims = torch.matmul(embeddings[node_id], embeddings.T)
    sims[node_id] = -1e9  # ignore self
    topk = torch.topk(sims, k)
    return topk.indices.tolist()

# ------------------------------
# Step 6: Visualize Graph + Recommendations
# ------------------------------
def visualize_graph(G, embeddings, highlight_node=None, recommendations=None):
    plt.figure(figsize=(7, 7))
    pos = nx.spring_layout(G, seed=42)

    # Base graph color
    node_colors = ['skyblue'] * G.number_of_nodes()

    if highlight_node is not None:
        node_colors[highlight_node] = 'green'  # target node
        if recommendations:
            for r in recommendations:
                node_colors[r] = 'orange'  # recommended friends

    nx.draw_networkx(
        G, pos, with_labels=True, node_color=node_colors,
        node_size=500, font_size=9, edge_color='gray'
    )
    plt.title("Friend Recommendation Visualization", fontsize=13)
    plt.axis('off')
    plt.show()

# ------------------------------
# Step 7: Main Execution
# ------------------------------
if __name__ == "__main__":
    print("=== Building Graph ===")
    G = build_graph(num_nodes=40, edge_prob=0.08)

    print("\n=== Converting to PyTorch Geometric Data ===")
    data = nx_to_pyg(G)

    print("\n=== Training GCN Model ===")
    embeddings = train_gcn(data, epochs=100)

    print("\n=== Friend Recommendations ===")
    test_node = 7
    recs = recommend_friend(test_node, embeddings, k=3)
    print(f"Recommended friends for Node {test_node}: {recs}")

    visualize_graph(G, embeddings, highlight_node=test_node, recommendations=recs)
