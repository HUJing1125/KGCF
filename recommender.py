import json
import pickle
import torch
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict

from MGAA import MGAAModel
from train import RelationEmbedding, KRLModel

DATA_DIR = Path("./data")
CKG_DIR = DATA_DIR / "ckg_ml1m"
MODEL_DIR = DATA_DIR / "mtrainmodel"

TEST_FILE = CKG_DIR / "test.txt"
REL2ID_FILE = MODEL_DIR / "relation_to_id.json"
NEIGHBOR_FILE = MODEL_DIR / "neighbor_dict.pkl"
EMBEDDING_FILE = "./semantic_embeddings/ml_embeddings.h5"
MODEL_FILE = MODEL_DIR / "best_model.pt"

# Hyperparameters must be exactly the same as training
EMBEDDING_DIM = 384
HIDDEN_DIM = 256
OUTPUT_DIM = 128
NUM_LAYERS = 2
HEADS = 4
DROPOUT = 0.3

TOP_K = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_data():
    # Relation mapping
    with open(REL2ID_FILE, 'r') as f:
        relation_to_id = json.load(f)
    num_relations = len(relation_to_id)
    relation_set = set(relation_to_id.keys())

    # Semantic embeddings
    with h5py.File(EMBEDDING_FILE, 'r') as hf:
        ids = [x.decode('utf-8').strip() for x in hf['ids'][:]]
        embeddings = hf['embeddings'][:]
    node_features = {eid: embeddings[i].astype(np.float32) for i, eid in enumerate(ids)}
    embedding_dim = embeddings.shape[1]

    # Only format cleaning, no deletion of users
    expected_dim = embedding_dim
    for eid, feat in node_features.items():
        if isinstance(feat, set):
            continue
        feat = np.array(feat, dtype=np.float32).reshape(-1)
        if feat.shape[0] != expected_dim:
            new_feat = np.zeros(expected_dim, dtype=np.float32)
            copy_len = min(feat.shape[0], expected_dim)
            new_feat[:copy_len] = feat[:copy_len]
            feat = new_feat
        node_features[eid] = feat.astype(np.float32)

    print(f"Loaded {len(node_features)} entity features.")

    # Fill missing entities in test set (must be done before model initialization)
    test_triples = load_triples(TEST_FILE)
    all_entities = set()
    for h, r, t in test_triples:
        all_entities.update([h, t])
    missing = [e for e in all_entities if e not in node_features]
    if missing:
        print(f"Filling {len(missing)} missing test entities with zeros.")
        for e in missing:
            node_features[e] = np.zeros(embedding_dim, dtype=np.float32)

    # Neighbor cache
    with open(NEIGHBOR_FILE, 'rb') as f:
        neighbor_dict = pickle.load(f)

    # Model
    entity_encoder = MGAAModel(
        embedding_dim=embedding_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_relations=num_relations,
        num_layers=NUM_LAYERS,
        heads=HEADS,
        dropout=DROPOUT,
        device=DEVICE
    )
    rel_init = {rel: node_features.get(rel, np.zeros(embedding_dim, dtype=np.float32))
                for rel in relation_set}
    relation_embedding = RelationEmbedding(
        relation_ids=list(relation_set),
        init_embeddings=rel_init,
        in_dim=embedding_dim,
        out_dim=OUTPUT_DIM
    ).to(DEVICE)
    model = KRLModel(entity_encoder, relation_embedding, OUTPUT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.eval()
    return model, neighbor_dict, node_features, relation_to_id

def load_triples(file_path):
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) != 3: continue
            triples.append(parts)
    return triples

def load_test_interactions(test_file):
    user_items = defaultdict(set)
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) != 3: continue
            user, _, item = parts
            user_items[user].add(item)
    return user_items

def evaluate_test(model, test_interactions, all_items, neighbor_dict, node_features, relation_to_id, device, k):
    model.eval()
    with torch.no_grad():
        item_emb_dict = model.entity_encoder(all_items, neighbor_dict, node_features, relation_to_id)
        item_embs = torch.stack([item_emb_dict[i] for i in all_items]).to(device)

        precisions, recalls, f1s = [], [], []
        all_recommended = set()

        for user, gt_set in test_interactions.items():
            # Encode user using the model (even if embedding is zero, neighbor aggregation yields meaningful representation)
            user_emb_dict = model.entity_encoder([user], neighbor_dict, node_features, relation_to_id)
            user_emb = user_emb_dict[user]

            scores = torch.mm(user_emb.unsqueeze(0), item_embs.t()).squeeze(0)
            top_indices = torch.topk(scores, k=min(k, len(all_items))).indices
            rec_items = [all_items[idx] for idx in top_indices.cpu().tolist()]

            all_recommended.update(rec_items)
            rec_set = set(rec_items)
            intersect = rec_set & gt_set
            precision = len(intersect) / k
            recall = len(intersect) / len(gt_set) if gt_set else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0
    avg_f1 = np.mean(f1s) if f1s else 0.0
    ad = len(all_recommended)

    results = {
        "K": k,
        "Precision": round(avg_precision, 4),
        "Recall": round(avg_recall, 4),
        "F1": round(avg_f1, 4),
        "Aggregate_Diversity": ad
    }
    output_file = MODEL_DIR / "test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")
    return results

def main():
    model, neighbor_dict, node_features, relation_to_id = load_model_and_data()

    #book_ids = [eid for eid in node_features if eid.startswith("book_")]
    #print(f"Total candidate books: {len(book_ids)}")

    movie_ids = [eid for eid in node_features if eid.startswith("movie_")]
    print(f"Total candidate books: {len(movie_ids)}")

    test_interactions = load_test_interactions(TEST_FILE)
    print(f"Loaded {len(test_interactions)} test users.")

    results = evaluate_test(
        model, test_interactions, movie_ids, neighbor_dict, node_features, relation_to_id, DEVICE, k=TOP_K
    )
    print("\n===== Evaluation Results =====")
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()