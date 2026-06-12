# test_fb15k237.py
import json
import pickle
import torch
import numpy as np
import h5py
from collections import defaultdict
from tqdm import tqdm
from typing import List, Tuple, Set, Dict

from MGAA import MGAAModel
from train_utils import load_triples

# ---------------------------- Model Definition (consistent with training) ----------------------------
class RelationEmbedding(torch.nn.Module):
    def __init__(self, relation_ids, init_embeddings, in_dim, out_dim):
        super().__init__()
        self.relation_ids = relation_ids
        self.num_relations = len(relation_ids)
        emb_array = np.zeros((self.num_relations, in_dim), dtype=np.float32)
        for i, rel in enumerate(relation_ids):
            emb_array[i] = init_embeddings.get(rel, np.zeros(in_dim))
        self.embeddings = torch.nn.Parameter(torch.tensor(emb_array), requires_grad=True)
        self.proj = torch.nn.Linear(in_dim, out_dim)

    def forward(self, rel_names):
        indices = [self.relation_ids.index(r) for r in rel_names]
        x = self.embeddings[torch.tensor(indices, device=self.embeddings.device)]
        return self.proj(x)


class KRLModel(torch.nn.Module):
    def __init__(self, entity_encoder, relation_embedding, common_dim):
        super().__init__()
        self.entity_encoder = entity_encoder
        self.relation_embedding = relation_embedding
        self.common_dim = common_dim

    def forward(self, head_ids, rel_names, tail_ids, neighbor_dict, node_features, relation_to_id):
        all_ents = set(head_ids) | set(tail_ids)
        emb_dict = self.entity_encoder(list(all_ents), neighbor_dict, node_features, relation_to_id)
        h_emb = torch.stack([emb_dict[h] for h in head_ids])
        t_emb = torch.stack([emb_dict[t] for t in tail_ids])
        r_emb = self.relation_embedding(rel_names)
        # DistMult score: higher is better
        scores = (h_emb * r_emb * t_emb).sum(dim=-1)
        return scores


# ---------------------------- Evaluation Function (based on scores) ----------------------------
def evaluate_filtered(
        model,
        test_triples: List[Tuple[str, str, str]],
        all_correct: Set[Tuple[str, str, str]],
        all_entities: List[str],
        neighbor_dict: Dict[str, List[Tuple[str, str, str]]],
        node_features: Dict[str, np.ndarray],
        relation_to_id: Dict[str, int],
        device: torch.device,
        batch_size: int = 128,
        filter_self: bool = True
):
    model.eval()
    filter_dict = defaultdict(set)
    for h, r, t in all_correct:
        filter_dict[(h, r)].add(t)

    entity_to_idx = {eid: idx for idx, eid in enumerate(all_entities)}
    num_entities = len(all_entities)

    with torch.no_grad():
        entity_emb_dict = model.entity_encoder(
            all_entities, neighbor_dict, node_features, relation_to_id
        )
        entity_embs = torch.stack([entity_emb_dict[eid] for eid in all_entities]).to(device)

    ranks = []
    for i in tqdm(range(0, len(test_triples), batch_size), desc="Evaluating"):
        batch = test_triples[i:i+batch_size]
        heads, rels, tails = zip(*batch)

        head_embs = torch.stack([entity_emb_dict[h] for h in heads]).to(device)
        rel_embs = model.relation_embedding(list(rels)).to(device)

        # Score matrix (B, N) = sum(h * r * t_emb)
        scores = (head_embs.unsqueeze(1) * rel_embs.unsqueeze(1) * entity_embs.unsqueeze(0)).sum(dim=-1)

        for j, (h, r, t) in enumerate(batch):
            filter_tails = filter_dict.get((h, r), set()).copy()
            if filter_self and t in filter_tails:
                filter_tails.remove(t)

            scores_j = scores[j].clone()
            for ft in filter_tails:
                idx = entity_to_idx.get(ft)
                if idx is not None:
                    scores_j[idx] = -float('inf')   # Mask other correct entities

            sorted_indices = torch.argsort(scores_j, descending=True)
            true_idx = entity_to_idx[t]
            rank = (sorted_indices == true_idx).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)

    ranks = np.array(ranks)
    mrr = np.mean(1.0 / ranks)
    hits1 = np.mean(ranks <= 1)
    hits3 = np.mean(ranks <= 3)
    hits10 = np.mean(ranks <= 10)
    return {'MRR': mrr, 'Hits@1': hits1, 'Hits@3': hits3, 'Hits@10': hits10}


# ---------------------------- Main Program ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./FB15k237"
    model_dir = "./trainmodelmovie"   # 与训练代码中的 model_dir 一致

    # Load test triples
    test_path = f"{data_dir}/test.txt"
    test_triples = load_triples(test_path)
    print(f"Loaded {len(test_triples)} test triples")

    # All correct triples (for filtering)
    train_path = f"{data_dir}/train.txt"
    valid_path = f"{data_dir}/valid.txt"
    train_triples = load_triples(train_path)
    valid_triples = load_triples(valid_path) if valid_path else []
    all_correct = set(train_triples + valid_triples + test_triples)

    # All entities
    all_entities = list(set(e for triple in all_correct for e in (triple[0], triple[2])))
    print(f"Total entities: {len(all_entities)}")

    # Load semantic embeddings
    h5_path = "./semantic_embeddings/FB_embeddings.h5"
    with h5py.File(h5_path, 'r') as f:
        ids = [x.decode('utf-8') for x in f['ids'][:]]
        embeddings = f['embeddings'][:]
    node_features = {eid: embeddings[i].astype(np.float32) for i, eid in enumerate(ids)}
    embedding_dim = embeddings.shape[1]
    for eid in all_entities:
        if eid not in node_features:
            node_features[eid] = np.zeros(embedding_dim, dtype=np.float32)

    # Load neighbor dictionary
    neighbor_path = f"{model_dir}/neighbor_dict.pkl"
    with open(neighbor_path, "rb") as f:
        neighbor_dict = pickle.load(f)
    for eid in all_entities:
        if eid not in neighbor_dict:
            neighbor_dict[eid] = []

    # Load relation mapping
    relation_path = f"{model_dir}/relation_to_id.json"
    with open(relation_path, "r") as f:
        relation_to_id = json.load(f)
    num_relations = len(relation_to_id)
    relation_set = set(relation_to_id.keys())
    print(f"Loaded {num_relations} relations")

    # Rebuild model (hyperparameters must match training)
    entity_encoder = MGAAModel(
        embedding_dim=embedding_dim,
        hidden_dim=512,
        output_dim=256,      # 与训练时的 output_dim 一致
        num_relations=num_relations,
        num_layers=2,
        heads=4,
        dropout=0.3,
        device=device
    )
    rel_init = {rel: node_features.get(rel, np.zeros(embedding_dim, dtype=np.float32)) for rel in relation_set}
    relation_embedding = RelationEmbedding(
        relation_ids=list(relation_set),
        init_embeddings=rel_init,
        in_dim=embedding_dim,
        out_dim=256
    ).to(device)
    model = KRLModel(entity_encoder, relation_embedding, 256).to(device)

    # Load best model weights
    best_model_path = f"{model_dir}/best_model.pt"
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded best model from {best_model_path}")

    # Evaluate on test set
    metrics = evaluate_filtered(
        model=model,
        test_triples=test_triples,
        all_correct=all_correct,
        all_entities=all_entities,
        neighbor_dict=neighbor_dict,
        node_features=node_features,
        relation_to_id=relation_to_id,
        device=device,
        batch_size=128,
        filter_self=True
    )

    print("\n===== Test Results =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Save results
    with open(f"{model_dir}/test_results.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()