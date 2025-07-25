from a_extract_node_and_edge import extract_graph_data
from b_graph import build_graph
from c_gcn import SlotGCN
import torch
import pickle
import networkx as nx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(json_path):
    # Extract graph data from JSON
    slot_list, coexistence_edges, domain_edges = extract_graph_data(json_path)

    # Build the graph
    graph_data, slot2idx = build_graph(slot_list, coexistence_edges, domain_edges)

    # Move data to GPU if needed
    graph_data = graph_data.to(device)

    # Initialize the GCN model
    in_channels = 384
    hidden_channels = 64
    out_channels = 32
    model = SlotGCN(in_channels, hidden_channels, out_channels).to(device)

    # Forward pass (inference)
    model.eval()
    with torch.no_grad():
        output = model(graph_data)  # [num_nodes, out_channels]

    # Save output embeddings and graph structure
    save_graph(slot_list, coexistence_edges, domain_edges, output, slot2idx)

    return output, slot2idx

def save_graph(slot_list, coexistence_edges, domain_edges, output, slot2idx, path="graph/slot_graph.pkl"):
    # Create NetworkX graph
    G = nx.Graph()
    for slot in slot_list:
        G.add_node(slot)

    for (u, v), weight in coexistence_edges.items():
        G.add_edge(u, v, type="coexistence", weight=weight)
    for (u, v), weight in domain_edges.items():
        if G.has_edge(u, v):
            G[u][v]["type"] = "both"
            G[u][v]["weight"] += weight
        else:
            G.add_edge(u, v, type="domain", weight=weight)

    # Add embeddings to nodes
    output = output.cpu()
    for slot, idx in slot2idx.items():
        G.nodes[slot]["embedding"] = output[idx].tolist()

    # Save using pickle
    with open(path, "wb") as f:
        pickle.dump(G, f)

    print(f"Graph saved to {path}")

if __name__ == "__main__":
    json_path = "data/mw24/train_dials.json"  # Update if needed
    output, slot2idx = main(json_path)
    print("Model output shape:", output.shape)
