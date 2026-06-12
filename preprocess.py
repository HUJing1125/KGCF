import json
import os
from typing import Set, Dict, List, Tuple

def load_entity_names(file_path: str) -> Dict[str, str]:
    name_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                eid, name = parts
                name_dict[eid] = name
    return name_dict

def load_entity_descriptions(file_path: str) -> Dict[str, str]:
    desc_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
            eid, desc = parts
            if desc.startswith('"') and desc.endswith('"'):
                desc = desc[1:-1]
            desc_dict[eid] = desc
    return desc_dict

def load_relation_names(file_path: str) -> Dict[str, str]:
    rel_dict = {}
    if not os.path.exists(file_path):
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                rid, name = parts
                rel_dict[rid] = name
    return rel_dict

def collect_ids_from_triples(file_paths: List[str]) -> Tuple[Set[str], Set[str]]:
    entities = set()
    relations = set()
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) != 3:
                    continue
                h, r, t = parts
                entities.add(h)
                entities.add(t)
                relations.add(r)
    return entities, relations

def main():
    data_dir = "./FB15k237"
    entity_name_file = os.path.join(data_dir, "FB15k_mid2name.txt")
    entity_desc_file = os.path.join(data_dir, "FB15k_mid2description.txt")
    relation_name_file = os.path.join(data_dir, "relation2text.txt")
    train_file = os.path.join(data_dir, "train.txt")
    valid_file = os.path.join(data_dir, "valid.txt")
    test_file = os.path.join(data_dir, "test.txt")
    output_json = os.path.join(data_dir, "node_info.json")

    entity_name = load_entity_names(entity_name_file)
    print(f"Load {len(entity_name)} entities")

    entity_desc = load_entity_descriptions(entity_desc_file)
    print(f"Load {len(entity_desc)} entity description")


    relation_name = load_relation_names(relation_name_file)
    print(f"Load {len(relation_name)} relations")


    triple_files = [train_file, valid_file, test_file]
    entities, relations = collect_ids_from_triples(triple_files)
    print(f"Collect {len(entities)} entities，{len(relations)} relations from triples")


    nodes = []


    for eid in entities:
        name = entity_name.get(eid, eid)
        desc = entity_desc.get(eid, "")
        nodes.append({
            "id": eid,
            "name": name,
            "description": desc
        })


    for rid in relations:
        name = relation_name.get(rid, rid)
        nodes.append({
            "id": rid,
            "name": name,
            "description": ""
        })

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)

    print(f"Save {len(nodes)} nodes（{len(entities)} entities + {len(relations)} relations）to {output_json}")

if __name__ == "__main__":
    main()