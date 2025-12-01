### 🧠 Hybrid NetworkX – GNN Friend Recommendation System
A Graph Neural Network (GCN) Based Social Recommendation Engine with Visualization

This project combines NetworkX, PyTorch Geometric, and Graph Convolutional Networks (GCNs) to build an intelligent friend recommendation system similar to those used in modern social media platforms.
It generates a synthetic social graph, learns node embeddings using a GCN, and visualizes both the network and the recommended connections.

---

### 🚀 Features:

📌 Graph Generation using Barabási–Albert scale-free model

🔄 NetworkX → PyTorch Geometric Conversion

🧩 Custom GCN Model (2-layer)

🧠 Unsupervised Training using edge-based consistency loss

🔍 Friend Recommendation using embedding similarity

🎨 Graph Visualization with highlighted recommended nodes

☁️ Colab-Compatible — runs smoothly on CPU

---

### 🛠 Tech Stack:

| Type                     | Tools                   |
| ------------------------ | ----------------------- |
| **Programming Language** | Python                  |
| **Graph Library**        | NetworkX                |
| **Deep Learning**        | PyTorch                 |
| **Graph ML Framework**   | PyTorch Geometric (PyG) |
| **Math & Utils**         | NumPy                   |
| **Visualization**        | Matplotlib, NetworkX    |

---

### 📦 Installation:

## 1️⃣ Clone the repository:
```
git clone https://github.com/MohammedSulthan07/Hybrid-NetworkX-GNN-Friend-Recommendation-System/tree/main
cd Hybrid-NetworkX-GNN-Friend-Recommendation-System

```

## 2️⃣ Install dependencies:
```
pip install networkx torch torchvision torchaudio torch-geometric matplotlib numpy
```

---

### ▶️ How It Works:

## 1. Build a Social Graph:

Creates a synthetic network of users using BA model.
```
G = build_graph(num_nodes=40)
```

## 2. Convert to PyTorch Geometric Format:

```
data = nx_to_pyg(G)
```

## 3. Train the GCN Model:

Learns embeddings that capture graph structure.
```
embeddings = train_gcn(data, epochs=100)
```

## 4. Generate Friend Recommendations:
```
recommend_friend(7, embeddings, k=3)
```

## 5. Visualize the Graph & Recommendations:
```
visualize_graph(G, embeddings, highlight_node=7, recommendations=recs)
```

---

### 🔍 Friend Recommendation Logic:

The GCN learns low-dimensional node embeddings.
We use cosine similarity via dot-product:

```
similarity = embedding[node] · embedding[others]
```
Top-k most similar nodes → recommended friends.

---

### 📊 Example Output:
```
=== Building Graph ===
Graph created with 40 nodes and 111 edges.

=== Converting to PyTorch Geometric Data ===

=== Training GCN Model ===
Epoch 000, Loss: 0.0151
Epoch 010, Loss: 0.0013
Epoch 020, Loss: 0.0004
Epoch 030, Loss: 0.0002
Epoch 040, Loss: 0.0001
Epoch 050, Loss: 0.0001
Epoch 060, Loss: 0.0000
Epoch 070, Loss: 0.0000
Epoch 080, Loss: 0.0000
Epoch 090, Loss: 0.0000

=== Friend Recommendations ===
Recommended friends for Node 7: [35, 9, 29]
```
Visualization shows:

* 🟢 Target node
* 🟠 Recommended nodes
* 🔵 Others

---

### 💡 Future Improvements:

* 🔮 Graph Attention Networks (GAT)
* 📈 Trainable link prediction head
*🧪 Datasets like Cora, PubMed
*🌐 Deploy as a web API / Streamlit app
