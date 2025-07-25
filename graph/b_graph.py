import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer

def build_graph(slot_list, coexistence_edges, domain_edges, device='cpu', coexistence_weight=2.0, domain_weight=1.0):
    slot2idx = {slot: i for i, slot in enumerate(slot_list)}

    # Encode slot names
    model = SentenceTransformer("all-MiniLM-L6-v2")
    slot_embeddings = model.encode(slot_list, convert_to_tensor=True)
    slot_embeddings = slot_embeddings.clone().detach().to(device).requires_grad_(True)

    edge_index = []
    edge_weight = []

    # Add coexistence edges with higher weight
    for (s1, s2), _ in coexistence_edges.items():
        i, j = slot2idx[s1], slot2idx[s2]
        edge_index += [[i, j], [j, i]]
        edge_weight += [coexistence_weight, coexistence_weight]

    # Add domain edges with default weight
    for (s1, s2), _ in domain_edges.items():
        i, j = slot2idx[s1], slot2idx[s2]
        edge_index += [[i, j], [j, i]]
        edge_weight += [domain_weight, domain_weight]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    edge_attr = torch.tensor(edge_weight, dtype=torch.float).to(device)

    graph_data = Data(x=slot_embeddings, edge_index=edge_index, edge_attr=edge_attr).to(device)
    return graph_data, slot2idx
