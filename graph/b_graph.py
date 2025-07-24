import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer

def build_graph(slot_list, coexistence_edges, domain_edges):
    slot2idx = {slot: i for i, slot in enumerate(slot_list)}
    num_slots = len(slot_list)

    # Semantic embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    slot_embeddings = model.encode(slot_list, convert_to_tensor=True)  # shape: [num_slots, hidden_size]

    edge_index = []

    # Add coexistence and domain edges
    for edge_dict in [coexistence_edges, domain_edges]:
        for (s1, s2), _ in edge_dict.items():
            i, j = slot2idx[s1], slot2idx[s2]
            edge_index.append([i, j])
            edge_index.append([j, i])  # undirected

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # shape: [2, num_edges]

    graph_data = Data(x=slot_embeddings, edge_index=edge_index)
    return graph_data, slot2idx
