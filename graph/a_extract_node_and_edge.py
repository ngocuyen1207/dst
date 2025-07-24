from collections import defaultdict
import json

def extract_graph_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        dials = json.load(f)

    slot_set = set()
    coexistence_edges = defaultdict(int)
    domain_edges = defaultdict(int)

    for dial in dials:
        slot_occurrences = set()

        for turn in dial.get("dialogue", []):
            belief_state = turn.get("belief_state", [])

            # Extract slot names from belief_state format: list of dicts with "slots": [[slot, value], ...]
            slots = set()
            for entry in belief_state:
                for s in entry.get("slots", []):
                    slot_name = s[0].lower().strip()
                    slots.add(slot_name)
                    slot_set.add(slot_name)

            # Add coexistence edges for this turn
            slots = list(slots)
            for i in range(len(slots)):
                for j in range(i + 1, len(slots)):
                    pair = tuple(sorted((slots[i], slots[j])))
                    coexistence_edges[pair] += 1

            slot_occurrences.update(slots)

        # Domain-based edges
        domains = dial.get("domains", [])  # ['hotel', ...]
        for domain in domains:
            domain_slots = [s for s in slot_occurrences if s.startswith(domain)]
            for i in range(len(domain_slots)):
                for j in range(i + 1, len(domain_slots)):
                    pair = tuple(sorted((domain_slots[i], domain_slots[j])))
                    domain_edges[pair] += 1

    return sorted(slot_set), coexistence_edges, domain_edges
