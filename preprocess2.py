import json
import os
from typing import Set, Tuple, Dict, List

def load_entity_text(file_path: str) -> Dict[str, Tuple[str, str]]:
    entity_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
            eid, text = parts
            # 拆分名称和描述（取第一个逗号）
            if ',' in text:
                name_part, desc_part = text.split(',', 1)
                name = name_part.strip()
                description = desc_part.strip()
            else:
                name = text.strip()
                description = ""
            entity_dict[eid] = (name, description)
    return entity_dict

def load_relation_text(file_path: str) -> Dict[str, str]:
    rel_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
            rid, name = parts
            rel_dict[rid] = name.strip()
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

    data_dir = "./WN18RR"                # 数据所在目录
    entity_file = os.path.join(data_dir, "entity2text.txt")
    relation_file = os.path.join(data_dir, "relation2text.txt")
    train_file = os.path.join(data_dir, "train.txt")
    valid_file = os.path.join(data_dir, "valid.txt")
    test_file = os.path.join(data_dir, "test.txt")
    output_json = os.path.join(data_dir, "node_info.json")



    entity_info = load_entity_text(entity_file)
    relation_info = load_relation_text(relation_file)


    triple_files = [train_file, valid_file, test_file]
    entities, relations = collect_ids_from_triples(triple_files)
    print(f"Find {len(entities)} entities，{len(relations)} relations")


    nodes = []


    for eid in entities:
        if eid in entity_info:
            name, desc = entity_info[eid]
        else:
            name = eid
            desc = ""
        nodes.append({
            "id": eid,
            "name": name,
            "description": desc
        })

    for rid in relations:
        if rid in relation_info:
            name = relation_info[rid]
        else:
            name = rid
        nodes.append({
            "id": rid,
            "name": name,
            "description": ""
        })


    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)

    print(f"Save {len(nodes)} nodes to {output_json}")

if __name__ == "__main__":
    main()