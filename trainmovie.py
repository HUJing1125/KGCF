
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

from ONS import BatchNeighborSampler, build_graph_from_edges_file
from MGAA import MGAAModel
from train_utils import load_triples   # Custom utility function

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
        return self.proj(x)


# ---------------------------- Complete Model ----------------------------
class KRLModel(nn.Module):
    def __init__(self, entity_encoder, relation_embedding, common_dim):
        super().__init__()
        self.entity_encoder = entity_encoder
        self.relation_embedding = relation_embedding
        self.common_dim = common_dim

    def forward(self, head_ids, rel_names, tail_ids, neighbor_dict, node_features, relation_to_id):
        # Collect all entities
        all_entities = set(head_ids) | set(tail_ids)
        entity_emb_dict = self.entity_encoder(
            list(all_entities), neighbor_dict, node_features, relation_to_id
        )
        h_emb = torch.stack([entity_emb_dict[h] for h in head_ids])
        t_emb = torch.stack([entity_emb_dict[t] for t in tail_ids])
        r_emb = self.relation_embedding(rel_names)
        # DistMult score: sum(h * r * t)
        scores = (h_emb * r_emb * t_emb).sum(dim=-1)
        return scores   # Higher is better


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


# ---------------------------- Validation Evaluation (Recommendation Metrics) ----------------------------
def evaluate_validation(model, valid_triples, all_items, neighbor_dict, node_features, relation_to_id, device, k=20):
    """
    Validation evaluation: compute Precision@K, Recall@K, F1@K (only for users appearing in training set)
    """
    model.eval()
    # Build ground truth per user
    user_gt = defaultdict(set)
    for u, _, i in valid_triples:
        user_gt[u].add(i)

    with torch.no_grad():
        item_emb_dict = model.entity_encoder(all_items, neighbor_dict, node_features, relation_to_id)
        item_embs = torch.stack([item_emb_dict[i] for i in all_items]).to(device)

        precisions, recalls, f1s = [], [], []
        for user, gt_set in user_gt.items():
            # Get user embedding
            user_emb_dict = model.entity_encoder([user], neighbor_dict, node_features, relation_to_id)
            user_emb = user_emb_dict[user]

            scores = torch.mm(user_emb.unsqueeze(0), item_embs.t()).squeeze(0)
            top_indices = torch.topk(scores, k=min(k, len(all_items))).indices
            rec_items = [all_items[idx] for idx in top_indices.cpu().tolist()]

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
    return avg_precision, avg_recall, avg_f1


# ---------------------------- Main Training Function ----------------------------
def train():
    # ==================== Configuration ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./data"
    ckg_dir = data_dir + "/ckg_ml1m"
    model_dir = data_dir + "/mtrainmodel"
    os.makedirs(model_dir, exist_ok=True)

    # File paths
    edge_file = f"{ckg_dir}/graph_edges.txt"
    train_file = f"{ckg_dir}/train.txt"
    valid_file = f"{ckg_dir}/valid.txt"
    test_file = f"{ckg_dir}/test.txt"
    h5_path = "./semantic_embeddings/ml_embeddings.h5"
    rel2id_file = f"{model_dir}/relation_to_id.json"

    # Hyperparameters (recommended configuration)
    embedding_dim = 384          # SBERT dimension
    hidden_dim = 256
    output_dim = 128
    num_layers = 2
    heads = 4
    dropout = 0.2
    lambda_weight = 0.5         # Higher semantic weight (0.3 means 30% structural, 70% semantic)
    T = 10
    walk_times = 10
    walk_length = 2
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 200
    warmup_epochs = 5
    patience = 10
    top_k = 20

    # ---------------------------- Load Data ----------------------------
    train_triples = load_triples(train_file)
    valid_triples = load_triples(valid_file)
    # Note: test_triples are used for final testing, not needed here

    # All entities (for completing semantic vectors)
    all_triples = train_triples + valid_triples
    all_entities = list(set(e for triple in all_triples for e in (triple[0], triple[2])))
    print(f"Total entities: {len(all_entities)}")

    # Load semantic embeddings
    with h5py.File(h5_path, 'r') as f:
        ids = [x.decode('utf-8') for x in f['ids'][:]]
        embeddings = f['embeddings'][:]
    node_features = {eid: embeddings[i].astype(np.float32) for i, eid in enumerate(ids)}
    # Fill missing entities
    for eid in all_entities:
        if eid not in node_features:
            node_features[eid] = np.zeros(embedding_dim, dtype=np.float32)

    # Build graph using edge file
    print("Building graph from edges file...")
    graph = build_graph_from_edges_file(edge_file)
    sampler = BatchNeighborSampler(
        graph=graph,
        embeddings=node_features,
        lambda_weight=lambda_weight,
        T=T,
        walk_times=walk_times,
        walk_length=walk_length
    )
    # Sample neighbors for all entities
    all_nodes = list(node_features.keys())
    neighbor_cache = f"{model_dir}/neighbor_dict.pkl"
    if os.path.exists(neighbor_cache):
        with open(neighbor_cache, "rb") as f:
            neighbor_dict = pickle.load(f)
        print("Loaded neighbor dict from cache")
    else:
        print("Sampling neighbors for all nodes...")
        neighbor_dict = sampler.sample_batch(all_nodes)   # Please confirm method name
        with open(neighbor_cache, "wb") as f:
            pickle.dump(neighbor_dict, f)
    # Ensure each node has a neighbor list
    for node in all_nodes:
        if node not in neighbor_dict:
            neighbor_dict[node] = []

    # ---------------------------- Relation Mapping ----------------------------
    relation_set = set()
    for h, r, t in train_triples:
        relation_set.add(r)
        relation_set.add(f"inv_{r}")
    for node, neighs in neighbor_dict.items():
        for _, rel, _ in neighs:
            relation_set.add(rel)
    relation_set.add("self_loop")
    relation_to_id = {rel: i for i, rel in enumerate(sorted(relation_set))}
    num_relations = len(relation_to_id)
    with open(rel2id_file, "w") as f:
        json.dump(relation_to_id, f, indent=2)
    print(f"Total relations: {num_relations}")

    # ---------------------------- Model Initialization ----------------------------
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
    model = KRLModel(entity_encoder, relation_embedding, output_dim).to(device)

    # ---------------------------- Optimizer and Scheduler ----------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)

    # ---------------------------- Resume Training ----------------------------
    start_epoch = 0
    best_f1 = 0.0
    patience_counter = 0
    checkpoint_path = f"{model_dir}/checkpoint.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        print(f"Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")

    # ---------------------------- Dataset and DataLoader ----------------------------
    train_dataset = TripletDataset(train_triples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # All candidate items (for negative sampling and evaluation)
    all_items = [eid for eid in node_features if eid.startswith("movie_")]
    print(f"Total items: {len(all_items)}")

    # ---------------------------- Training Loop ----------------------------
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        # Linear learning rate increase during warmup
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * lr_scale

        for heads, rels, tails in pbar:
            # Positive sample scores
            pos_scores = model(heads, rels, tails, neighbor_dict, node_features, relation_to_id)

            # Simple random negative sampling: for each positive, randomly pick a different item
            neg_tails = []
            for t in tails:
                neg = t
                while neg == t:
                    neg = random.choice(all_items)
                neg_tails.append(neg)
            neg_scores = model(heads, rels, neg_tails, neighbor_dict, node_features, relation_to_id)

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

        # Update learning rate after warmup
        if epoch >= warmup_epochs:
            scheduler.step()

        # Validate every 3 epochs
        if (epoch + 1) % 3 == 0:
            precision, recall, f1 = evaluate_validation(
                model, valid_triples, all_items, neighbor_dict, node_features, relation_to_id, device, k=top_k
            )
            print(f"Validation @{top_k}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                torch.save(model.state_dict(), f"{model_dir}/best_model.pt")
                print(f"New best model saved with F1={f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {patience} consecutive no improvements.")
                    break

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_f1,
        }, checkpoint_path)

    print("Training finished.")


if __name__ == "__main__":
    train()