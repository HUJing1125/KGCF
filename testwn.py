# test.py
import json
import pickle
import torch
import numpy as np
import h5py
from collections import defaultdict
from tqdm import tqdm
from typing import List, Tuple, Set, Dict

from ONS import BatchNeighborSampler, build_graph_from_triples
from MGAA import MGAAModel
from train_utils import load_triples

# ---------------------------- Relation Embedding Module (same as training) ----------------------------
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
        return self.proj(x)   # output phase angle, dimension out_dim

# ---------------------------- RotatE Model (same as training) ----------------------------
class KRLModel(torch.nn.Module):
    def __init__(self, entity_encoder, relation_embedding, common_dim, gamma=24.0):
        super().__init__()
        self.entity_encoder = entity_encoder
        self.relation_embedding = relation_embedding
        self.common_dim = common_dim
        self.complex_proj = torch.nn.Linear(common_dim, common_dim * 2)
        self.gamma = torch.nn.Parameter(torch.tensor(gamma, dtype=torch.float))

    def rotate_score(self, h_emb, r_emb, t_emb):
        re_h, im_h = torch.chunk(h_emb, 2, dim=-1)
        re_t, im_t = torch.chunk(t_emb, 2, dim=-1)
        cos_r = torch.cos(r_emb)
        sin_r = torch.sin(r_emb)
        re_rot = re_h * cos_r - im_h * sin_r
        im_rot = re_h * sin_r + im_h * cos_r
        diff_re = re_rot - re_t
        diff_im = im_rot - im_t
        dist = torch.abs(diff_re) + torch.abs(diff_im)
        dist_sum = dist.sum(dim=-1)
        return self.gamma - dist_sum

    def forward(self, head_ids, rel_names, tail_ids, neighbor_dict, node_features, relation_to_id):
        all_entities = set(head_ids) | set(tail_ids)
        entity_emb_dict = self.entity_encoder(
            list(all_entities), neighbor_dict, node_features, relation_to_id
        )
        h_emb_real = torch.stack([entity_emb_dict[h] for h in head_ids])
        t_emb_real = torch.stack([entity_emb_dict[t] for t in tail_ids])
        r_emb = self.relation_embedding(rel_names)
        h_emb_cplx = self.complex_proj(h_emb_real)
        t_emb_cplx = self.complex_proj(t_emb_real)
        scores = self.rotate_score(h_emb_cplx, r_emb, t_emb_cplx)
        return scores

# ---------------------------- Evaluation Function (RotatE version) ----------------------------
def evaluate_filtered(model, test_triples, all_correct, all_entities,
                      neighbor_dict, node_features, relation_to_id, device,
                      batch_size=256, filter_self=True):
    model.eval()
    filter_dict = defaultdict(set)
    for h, r, t in all_correct:
        filter_dict[(h, r)].add(t)

    entity_to_idx = {eid: idx for idx, eid in enumerate(all_entities)}
    num_entities = len(all_entities)

    with torch.no_grad():
        entity_emb_dict_real = model.entity_encoder(
            all_entities, neighbor_dict, node_features, relation_to_id
        )
        entity_embs_real = torch.stack([entity_emb_dict_real[eid] for eid in all_entities])
        entity_embs_cplx = model.complex_proj(entity_embs_real).to(device)

    ranks = []
    for i in range(0, len(test_triples), batch_size):
        batch = test_triples[i:i+batch_size]
        heads, rels, tails = zip(*batch)

        head_embs_real = torch.stack([entity_emb_dict_real[h] for h in heads])
        head_embs_cplx = model.complex_proj(head_embs_real).to(device)
        rel_embs = model.relation_embedding(list(rels)).to(device)

        B = head_embs_cplx.size(0)
        N = num_entities
        h_exp = head_embs_cplx.unsqueeze(1)
        r_exp = rel_embs.unsqueeze(1)
        t_exp = entity_embs_cplx.unsqueeze(0)
        re_h, im_h = torch.chunk(h_exp, 2, dim=-1)
        re_t, im_t = torch.chunk(t_exp, 2, dim=-1)
        cos_r = torch.cos(r_exp)
        sin_r = torch.sin(r_exp)
        re_rot = re_h * cos_r - im_h * sin_r
        im_rot = re_h * sin_r + im_h * cos_r
        diff_re = re_rot - re_t
        diff_im = im_rot - im_t
        dist = torch.abs(diff_re) + torch.abs(diff_im)
        dist_sum = dist.sum(dim=-1)
        scores = model.gamma - dist_sum

        for j, (h, r, t) in enumerate(batch):
            filter_tails = filter_dict.get((h, r), set()).copy()
            if filter_self and t in filter_tails:
                filter_tails.remove(t)
            scores_j = scores[j].clone()
            for ft in filter_tails:
                idx = entity_to_idx.get(ft)
                if idx is not None:
                    scores_j[idx] = -float('inf')
            sorted_indices = torch.argsort(scores_j, descending=True)
            true_idx = entity_to_idx[t]
            rank = (sorted_indices == true_idx).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)

    ranks = np.array(ranks)
    mrr = np.mean(1.0 / ranks)
    hits1 = np.mean(ranks <= 1)
    hits3 = np.mean(ranks <= 3)
    hits10 = np.mean(ranks <= 10)
    return mrr, hits1, hits3, hits10

# ---------------------------- Main Test ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./WN18RR"
    model_dir = "./trainmodel"

    # 1. Load test triples
    test_path = f"{data_dir}/test.txt"
    test_triples = load_triples(test_path)
    print(f"Loaded {len(test_triples)} test triples")

    # 2. All correct triples for filtering (train + valid + test)
    train_path = f"{data_dir}/train.txt"
    valid_path = f"{data_dir}/valid.txt"
    train_triples = load_triples(train_path)
    valid_triples = load_triples(valid_path) if valid_path else []
    all_correct = set(train_triples + valid_triples + test_triples)

    # 3. All entities
    all_entities = list(set(e for triple in all_correct for e in (triple[0], triple[2])))
    print(f"Total entities: {len(all_entities)}")

    # 4. Load semantic embeddings
    h5_path = "./semantic_embeddings/WN_embeddings.h5"
    with h5py.File(h5_path, 'r') as f:
        ids = [x.decode('utf-8') for x in f['ids'][:]]
        embeddings = f['embeddings'][:]
    node_features = {eid: embeddings[i].astype(np.float32) for i, eid in enumerate(ids)}
    embedding_dim = embeddings.shape[1]
    for eid in all_entities:
        if eid not in node_features:
            node_features[eid] = np.zeros(embedding_dim, dtype=np.float32)

    # 5. Load neighbor dictionary (must have been saved during training)
    neighbor_path = f"{model_dir}/neighbor_dict.pkl"
    with open(neighbor_path, "rb") as f:
        neighbor_dict = pickle.load(f)
    for eid in all_entities:
        if eid not in neighbor_dict:
            neighbor_dict[eid] = []

    # 6. Load relation mapping
    relation_path = f"{model_dir}/relation_to_id.json"
    with open(relation_path, "r") as f:
        relation_to_id = json.load(f)
    num_relations = len(relation_to_id)
    relation_set = set(relation_to_id.keys())
    print(f"Loaded {num_relations} relations")

    # 7. Build model (must match training hyperparameters)
    entity_encoder = MGAAModel(
        embedding_dim=embedding_dim,
        hidden_dim=512,          # same as training
        output_dim=256,          # common_dim
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
    model = KRLModel(entity_encoder, relation_embedding, 256, gamma=24.0).to(device)

    # 8. Load best model weights
    best_model_path = f"{model_dir}/best_model.pt"
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded best model from {best_model_path}")

    # 9. Evaluate
    mrr, h1, h3, h10 = evaluate_filtered(
        model=model,
        test_triples=test_triples,
        all_correct=all_correct,
        all_entities=all_entities,
        neighbor_dict=neighbor_dict,
        node_features=node_features,
        relation_to_id=relation_to_id,
        device=device,
        batch_size=64,
        filter_self=True
    )

    print("\n===== Test Results =====")
    print(f"MRR: {mrr:.4f}")
    print(f"Hits@1: {h1:.4f}")
    print(f"Hits@3: {h3:.4f}")
    print(f"Hits@10: {h10:.4f}")

    # Save results
    results = {"MRR": mrr, "Hits@1": h1, "Hits@3": h3, "Hits@10": h10}
    with open(f"{model_dir}/test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {model_dir}/test_results.json")

if __name__ == "__main__":
    main()