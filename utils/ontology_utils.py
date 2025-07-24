import json

def parse_ontology(path):
    with open(path, 'r') as f:
        ontology = json.load(f)

    domain_slot_map = {}
    for slot_key in ontology:
        domain, slot = slot_key.split('-')
        domain_slot_map.setdefault(domain, []).append(slot)
    return domain_slot_map
