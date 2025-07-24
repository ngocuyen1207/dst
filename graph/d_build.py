from a_extract_node_and_edge import extract_graph_data
from b_graph import build_graph
from c_gcn import SlotGCN
import torch

def main(json_path):
    # Extract graph data from JSON
    slot_list, coexistence_edges, domain_edges = extract_graph_data(json_path)

    # Build the graph
    graph_data, slot2idx = build_graph(slot_list, coexistence_edges, domain_edges)

    # Initialize the GCN model
    in_channels = 128
    hidden_channels = 64  # Example hidden size
    out_channels = 32  # Output size is the number of slots
    model = SlotGCN(in_channels, hidden_channels, out_channels)

    # Forward pass through the model
    output = model(graph_data)

    return output, slot2idx

if __name__ == "__main__":
    json_path = "data/mw24/train_dials.json"  # Update with your JSON file path
    output, slot2idx = main(json_path)
    print("Model output:", output)
    print("Slot to index mapping:", slot2idx)