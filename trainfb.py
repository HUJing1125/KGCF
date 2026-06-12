
import os
import json
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import h5py
from collections import defaultdict

from ONS import BatchNeighborSampler, build_graph_from_triples
from MGAA import MGAAModel
from train_utils import load_triples

# ---------------------------- Relation Embedding Module ----------------------------
class RelationEmbedding(nn.Module):
    def __init__(self, relation_ids, init_embeddings, in_dim, out_dim):
        super().__init__()
        self.relation_ids = relation_ids
        self.num_relations = len(relation_ids)
        emb_array = np.zeros((self.num_relations, in_dim), dtype=np.float32)
        for i, rel in enumerate(relation_ids):
            emb_array[i] = init_embeddings.get(rel, np.zeros(in_dim))
        self.embeddings = nn.Parameter(torch.tensor(emb_array), requires_grad=True)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, rel_names):
        indices = [self.relation_ids.index(r) for r in rel_names]
        x = self.embeddings[torch.tensor(indices, device=self.embeddings.device)]
        return self.proj(x)   # output phase angle, dimension out_dim

# ---------------------------- Complete Model (RotatE) ----------------------------
class KRLModel(nn.Module):
    def __init__(self, entity_encoder, relation_embedding, common_dim, gamma=24.0):
        super().__init__()
        self.entity_encoder = entity_encoder
        self.relation_embedding = relation_embedding
        self.common_dim = common_dim
        self.complex_proj = nn.Linear(common_dim, common_dim * 2)
        # Learnable gamma parameter, initialized to 24.0
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float))

    def rotate_score(self, h_emb, r_emb, t_emb):
        """
        h_emb, t_emb: complex vectors, shape (batch, common_dim*2)
        r_emb: relation phase angles, shape (batch, common_dim)
        returns score: gamma - L1_distance (higher is better)
        """
        re_h, im_h = torch.chunk(h_emb, 2, dim=-1)
        re_t, im_t = torch.chunk(t_emb, 2, dim=-1)
        cos_r = torch.cos(r_emb)
        sin_r = torch.sin(r_emb)

        # Rotation: (re_h + i*im_h) * (cos_r + i*sin_r)
        re_rot = re_h * cos_r - im_h * sin_r
        im_rot = re_h * sin_r + im_h * cos_r

        # L1 distance
        diff_re = re_rot - re_t
        diff_im = im_rot - im_t
        dist = torch.abs(diff_re) + torch.abs(diff_im)   # (batch, common_dim)
        dist_sum = dist.sum(dim=-1)                      # (batch,)
        score = self.gamma - dist_sum
        return score

    def forward(self, head_ids, rel_names, tail_ids, neighbor_dict, node_features, relation_to_id):
        # Get all entities
        all_entities = set(head_ids) | set(tail_ids)
        entity_emb_dict = self.entity_encoder(
            list(all_entities), neighbor_dict, node_features, relation_to_id
        )
        # Head and tail real embeddings
        h_emb_real = torch.stack([entity_emb_dict[h] for h in head_ids])  # (B, common_dim)
        t_emb_real = torch.stack([entity_emb_dict[t] for t in tail_ids])
        # Project to complex space
        h_emb_cplx = self.complex_proj(h_emb_real)   # (B, common_dim*2)
        t_emb_cplx = self.complex_proj(t_emb_real)
        # Relation embedding (phase angle)
        r_emb = self.relation_embedding(rel_names)   # (B, common_dim)
        # Compute scores
        scores = self.rotate_score(h_emb_cplx, r_emb, t_emb_cplx)
        return scores

# ---------------------------- Dataset ----------------------------
class TripletDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        return h, r, t

def collate_fn(batch):
    heads, rels, tails = zip(*batch)
    return list(heads), list(rels), list(tails)

# ---------------------------- Evaluation Function (RotatE version) ----------------------------
def evaluate_filtered(model, test_triples, all_correct, all_entities,
                      neighbor_dict, node_features, relation_to_id, device,
                      batch_size=256, filter_self=True):
    """
    Filtered link prediction evaluation using RotatE score (gamma - L1 distance)
    """
    model.eval()
    filter_dict = defaultdict(set)
    for h, r, t in all_correct:
        filter_dict[(h, r)].add(t)

    entity_to_idx = {eid: idx for idx, eid in enumerate(all_entities)}
    num_entities = len(all_entities)

    with torch.no_grad():
        # Precompute complex embeddings for all entities (real -> complex projection)
        entity_emb_dict_real = model.entity_encoder(
            all_entities, neighbor_dict, node_features, relation_to_id
        )
        # Project real embeddings to complex space
        entity_embs_real = torch.stack([entity_emb_dict_real[eid] for eid in all_entities])
        entity_embs_cplx = model.complex_proj(entity_embs_real).to(device)  # (N, common_dim*2)

    ranks = []
    for i in range(0, len(test_triples), batch_size):
        batch = test_triples[i:i+batch_size]
        heads, rels, tails = zip(*batch)

        # Head complex embeddings
        head_embs_real = torch.stack([entity_emb_dict_real[h] for h in heads])
        head_embs_cplx = model.complex_proj(head_embs_real).to(device)   # (B, common_dim*2)
        # Relation phase angles
        rel_embs = model.relation_embedding(list(rels)).to(device)        # (B, common_dim)

        # Score matrix (B, N)
        B = head_embs_cplx.size(0)
        N = num_entities
        # Expand dimensions
        h_exp = head_embs_cplx.unsqueeze(1)     # (B, 1, common_dim*2)
        r_exp = rel_embs.unsqueeze(1)           # (B, 1, common_dim)
        t_exp = entity_embs_cplx.unsqueeze(0)   # (1, N, common_dim*2)
        # Split real and imaginary parts
        re_h, im_h = torch.chunk(h_exp, 2, dim=-1)   # (B, 1, common_dim)
        re_t, im_t = torch.chunk(t_exp, 2, dim=-1)   # (1, N, common_dim)
        cos_r = torch.cos(r_exp)   # (B, 1, common_dim)
        sin_r = torch.sin(r_exp)
        # Rotation
        re_rot = re_h * cos_r - im_h * sin_r
        im_rot = re_h * sin_r + im_h * cos_r
        # Distance
        diff_re = re_rot - re_t
        diff_im = im_rot - im_t
        dist = torch.abs(diff_re) + torch.abs(diff_im)   # (B, N, common_dim)
        dist_sum = dist.sum(dim=-1)                      # (B, N)
        scores = model.gamma - dist_sum                  # (B, N)

        for j, (h, r, t) in enumerate(batch):
            filter_tails = filter_dict.get((h, r), set()).copy()
            if filter_self and t in filter_tails:
                filter_tails.remove(t)

            scores_j = scores[j].clone()
            for ft in filter_tails:
                idx = entity_to_idx.get(ft)
                if idx is not None:
                    scores_j[idx] = -float('inf')   # mask other correct entities

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

# ---------------------------- Main Training Function ----------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./FB15k237"
    model_dir = "./trainmodelmovie"
    os.makedirs(model_dir, exist_ok=True)

    train_path = f"{data_dir}/train.txt"
    valid_path = f"{data_dir}/valid.txt"
    test_path = f"{data_dir}/test.txt"
    h5_path = "./semantic_embeddings/FB_embeddings.h5"

    # Hyperparameters (tuned for WN18RR)
    embedding_dim = 384
    hidden_dim = 512
    output_dim = 256          # corresponds to common_dim
    num_layers = 2
    heads = 4
    dropout = 0.3
    T = 30
    walk_times = 10
    walk_length = 3
    lambda_weight = 0.7
    batch_size = 64
    learning_rate = 5e-4
    num_epochs = 200
    num_neg = 5               # number of negative samples per positive (for multi-neg sampling)
    warmup_epochs = 5
    patience = 10
    gamma_init = 12.0         # initial gamma value

    # Load data
    train_triples = load_triples(train_path)
    valid_triples = load_triples(valid_path)
    test_triples = load_triples(test_path)
    all_triples = train_triples + valid_triples + test_triples
    all_correct = set(all_triples)

    all_entities = list(set(e for triple in all_triples for e in (triple[0], triple[2])))
    print(f"Total entities: {len(all_entities)}")

    # Semantic vectors
    with h5py.File(h5_path, 'r') as f:
        ids = [x.decode('utf-8') for x in f['ids'][:]]
        embeddings = f['embeddings'][:]
    node_features = {eid: embeddings[i].astype(np.float32) for i, eid in enumerate(ids)}
    for eid in all_entities:
        if eid not in node_features:
            node_features[eid] = np.zeros(embedding_dim, dtype=np.float32)

    # Build graph and sample neighbors
    graph = build_graph_from_triples(train_path)
    sampler = BatchNeighborSampler(
        graph=graph,
        embeddings=node_features,
        lambda_weight=lambda_weight,
        T=T,
        walk_times=walk_times,
        walk_length=walk_length
    )
    neighbor_cache = os.path.join(model_dir, "neighbor_dict.pkl")
    if os.path.exists(neighbor_cache):
        with open(neighbor_cache, "rb") as f:
            neighbor_dict = pickle.load(f)
        print("Loaded neighbor dict from cache")
    else:
        print("Sampling neighbors for all entities...")
        neighbor_dict = sampler.sample_batch(all_entities)
        with open(neighbor_cache, "wb") as f:
            pickle.dump(neighbor_dict, f)
    for eid in all_entities:
        if eid not in neighbor_dict:
            neighbor_dict[eid] = []

    # Relation mapping
    relation_set = set()
    for h, r, t in train_triples:
        relation_set.add(r)
        relation_set.add(f"inv_{r}")
    for eid in neighbor_dict:
        for _, rel, _ in neighbor_dict[eid]:
            relation_set.add(rel)
    relation_set.add("self_loop")
    relation_to_id = {rel: i for i, rel in enumerate(sorted(relation_set))}
    num_relations = len(relation_to_id)
    with open(os.path.join(model_dir, "relation_to_id.json"), "w") as f:
        json.dump(relation_to_id, f, indent=2)
    print(f"Total relations: {num_relations}")

    # Initialize model (pass gamma_init)
    entity_encoder = MGAAModel(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_relations=num_relations,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout,
        device=device
    )
    rel_init = {rel: node_features.get(rel, np.zeros(embedding_dim, dtype=np.float32)) for rel in relation_set}
    relation_embedding = RelationEmbedding(
        relation_ids=list(relation_set),
        init_embeddings=rel_init,
        in_dim=embedding_dim,
        out_dim=output_dim
    ).to(device)
    model = KRLModel(entity_encoder, relation_embedding, output_dim, gamma=gamma_init).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)

    # Resume training
    start_epoch = 0
    best_mrr = 0.0
    patience_counter = 0
    checkpoint_path = os.path.join(model_dir, "checkpoint.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_mrr = checkpoint.get('best_mrr', 0.0)
        print(f"Resumed from epoch {start_epoch}, best MRR: {best_mrr:.4f}")

    train_dataset = TripletDataset(train_triples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        # Learning rate warmup
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * lr_scale

        for heads, rels, tails in pbar:
            # Positive scores
            pos_scores = model(heads, rels, tails, neighbor_dict, node_features, relation_to_id)

            # ---------- Multi-negative sampling (harder negatives) ----------
            all_neg_scores = []
            for _ in range(num_neg):
                # Generate a batch of random negative samples
                neg_tails = []
                for t in tails:
                    neg = t
                    while neg == t:
                        neg = random.choice(all_entities)
                    neg_tails.append(neg)
                # Compute scores for this batch of negatives
                neg_scores = model(heads, rels, neg_tails, neighbor_dict, node_features, relation_to_id)
                all_neg_scores.append(neg_scores)
            # Take the highest scoring negative as training negative (hardest to distinguish)
            neg_scores = torch.stack(all_neg_scores).max(dim=0)[0]
            # ----------------------------------------

            # BPR loss
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.6f}")

        if epoch >= warmup_epochs:
            scheduler.step()

        # Validate every 3 epochs
        if (epoch + 1) % 1 == 0:
            mrr, h1, h3, h10 = evaluate_filtered(
                model=model,
                test_triples=valid_triples,
                all_correct=set(train_triples + valid_triples),
                all_entities=all_entities,
                neighbor_dict=neighbor_dict,
                node_features=node_features,
                relation_to_id=relation_to_id,
                device=device,
                batch_size=32,
                filter_self=True
            )
            print(f"Validation MRR: {mrr:.4f}, Hits@1: {h1:.4f}, Hits@3: {h3:.4f}, Hits@10: {h10:.4f}")

            if mrr > best_mrr:
                best_mrr = mrr
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
                print(f"New best model saved with MRR {best_mrr:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered.")
                    break

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_mrr': best_mrr,
        }, checkpoint_path)

    print("Training finished.")

if __name__ == "__main__":
    train()