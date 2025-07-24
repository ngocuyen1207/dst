import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from models.slot_embedder import SlotEmbedder
from models.graph_encoder import SimpleGNNLayer
from models.dynamic_graph_builder import DynamicGraph
from utils.ontology_utils import parse_ontology
from tqdm import tqdm

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ONTOLOGY_PATH = "data/mw21/ontology.json"
TRAIN_PATH = "data/mw21/train_dials.json"
BATCH_SIZE = 1
EMBED_DIM = 384  # Match your sentence transformer

# === Dataset Class ===
class DSTDataset(Dataset):
    def __init__(self, dialogues):
        self.data = dialogues

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dial = self.data[idx]
        turns = dial["turns"]
        belief = dial["turns"][-1].get("belief_state", [])
        active_slots = [b["slots"][0][0] for b in belief if b["slots"]]
        return turns, active_slots

# === Load Data ===
with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    train_data = json.load(f)

ontology = parse_ontology(ONTOLOGY_PATH)
all_slots = list(ontology.keys())

# === Initialize Modules ===
embedder = SlotEmbedder()
gnn = SimpleGNNLayer(EMBED_DIM, EMBED_DIM).to(DEVICE)
graph_builder = DynamicGraph(slot_descriptions={slot: slot.replace("-", " ") for slot in all_slots},
                             embedder=embedder)

dataset = DSTDataset(train_data)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Training Loop (Stub) ===
optimizer = torch.optim.Adam(gnn.parameters(), lr=1e-4)

for epoch in range(3):
    total_loss = 0
    for turns, active_slots in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        batch_loss = 0

        for b in range(len(turns)):
            slots = active_slots[b]
            nodes, edges, node_feats = graph_builder.build_graph(slots)
            edge_index = torch.tensor(edges, dtype=torch.long).T.to(DEVICE)
            node_feats = node_feats.to(DEVICE)

            out = gnn(node_feats, edge_index)

            # === Placeholder: Compute DST belief prediction and loss ===
            # For now, we just do dummy loss
            dummy_target = torch.ones_like(out)
            loss = torch.nn.functional.mse_loss(out, dummy_target)

            batch_loss += loss

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
