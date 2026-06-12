# train_book_simple.py
import json
import os
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import h5py
from collections import defaultdict

from ONS import BatchNeighborSampler, build_graph_from_edges_file
from MGAA import MGAAModel
from train import RelationEmbedding, KRLModel

DATA_DIR = Path("./data")
CKG_DIR = DATA_DIR / "ckg_book"
MODEL_DIR = DATA_DIR / "btrainmodel"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EDGE_FILE = CKG_DIR / "graph_edges.txt"
REL2ID_FILE = MODEL_DIR / "relation_to_id.json"
TRAIN_FILE = CKG_DIR / "train.txt"
VALID_FILE = CKG_DIR / "valid.txt"
TEST_FILE = CKG_DIR / "test.txt"
EMBEDDING_H5 = "./semantic_embeddings/book_embeddings.h5"

# ---------- Fixed hyperparameters ----------
EMBEDDING_DIM = 384
HIDDEN_DIM = 256          # increase capacity
OUTPUT_DIM = 128
NUM_LAYERS = 2
HEADS = 4                  # increase number of heads
DROPOUT = 0.3              # moderate regularization
LAMBDA_WEIGHT = 0.5        # higher semantic weight
T = 20                     # increase neighbor sampling
WALK_TIMES = 10
WALK_LENGTH = 3
BATCH_SIZE = 64            # reduce batch size for stable gradients
LEARNING_RATE = 1e-3       # lower learning rate
NUM_EPOCHS = 200
TOP_K = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATIENCE = 10
WARMUP_EPOCHS = 5
MIN_LR = 1e-5

# ------------------------------- Data Loading -------------------------------
def load_triples(file_path):
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) != 3: continue
            triples.append(tuple(parts))
    return triples

class TripletDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples
    def __len__(self):
        return len(self.triples)
    def __getitem__(self, idx):
        return self.triples[idx]

def collate_fn(batch):
    heads, rels, tails = zip(*batch)
    return list(heads), list(rels), list(tails)

# ------------------------------- Validation (common users only) -------------------------------
def evaluate_validation(model, train_triples, valid_triples, all_items, neighbor_dict,
                        node_features, relation_to_id, device, k=20):
    model.eval()
    train_users = {u for u, _, _ in train_triples}
    user_gt = defaultdict(set)
    for u, _, i in valid_triples:
        user_gt[u].add(i)
    common_users = [u for u in user_gt if u in train_users]
    if not common_users:
        return 0.0, 0.0, 0.0
    print(f"Evaluating on {len(common_users)} common users (out of {len(user_gt)} total)")

    with torch.no_grad():
        item_emb_dict = model.entity_encoder(all_items, neighbor_dict, node_features, relation_to_id)
        item_embs = torch.stack([item_emb_dict[i] for i in all_items]).to(device)

        precisions, recalls, f1s = [], [], []
        for user in common_users:
            gt_set = user_gt[user]
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

    return np.mean(precisions), np.mean(recalls), np.mean(f1s)

# ------------------------------- Main Training Function -------------------------------
def main():
    # Load relation mapping
    with open(REL2ID_FILE, 'r') as f:
        relation_to_id = json.load(f)
    num_relations = len(relation_to_id)
    relation_set = set(relation_to_id.keys())
    print(f"Loaded {num_relations} relations.")

    # Load semantic embeddings
    with h5py.File(EMBEDDING_H5, 'r') as hf:
        ids = [x.decode('utf-8') for x in hf['ids'][:]]
        embeddings = hf['embeddings'][:]
    node_features = {eid: embeddings[i].astype(np.float32) for i, eid in enumerate(ids)}
    embedding_dim = embeddings.shape[1]

    # Build graph and sample neighbors
    print("Building graph from edges file...")
    graph = build_graph_from_edges_file(EDGE_FILE)
    sampler = BatchNeighborSampler(
        graph=graph,
        embeddings=node_features,
        lambda_weight=LAMBDA_WEIGHT,
        T=T,
        walk_times=WALK_TIMES,
        walk_length=WALK_LENGTH
    )
    all_nodes = list(node_features.keys())
    neighbor_cache = MODEL_DIR / "neighbor_dict.pkl"
    if neighbor_cache.exists():
        with open(neighbor_cache, 'rb') as f:
            neighbor_dict = pickle.load(f)
        print("Loaded neighbor dict from cache.")
    else:
        print("Sampling neighbors for all nodes...")
        neighbor_dict = sampler.sample_batch(all_nodes)
        with open(neighbor_cache, 'wb') as f:
            pickle.dump(neighbor_dict, f)
    for node in all_nodes:
        if node not in neighbor_dict:
            neighbor_dict[node] = []

    # Load triples
    train_triples = load_triples(TRAIN_FILE)
    valid_triples = load_triples(VALID_FILE)
    all_items = [eid for eid in all_nodes if eid.startswith("book_")]
    print(f"Total items: {len(all_items)}")

    # Fill missing entity vectors
    all_entities = set()
    for triples in [train_triples, valid_triples]:
        for h, r, t in triples:
            all_entities.update([h, r, t])
    missing = [e for e in all_entities if e not in node_features]
    if missing:
        print(f"WARNING: {len(missing)} entities missing, fill with zeros.")
        for e in missing:
            node_features[e] = np.zeros(embedding_dim, dtype=np.float32)

    # Initialize model
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
    rel_init = {rel: node_features.get(rel, np.zeros(embedding_dim, dtype=np.float32)) for rel in relation_set}
    relation_embedding = RelationEmbedding(
        relation_ids=list(relation_set),
        init_embeddings=rel_init,
        in_dim=embedding_dim,
        out_dim=OUTPUT_DIM
    ).to(DEVICE)
    model = KRLModel(entity_encoder, relation_embedding, OUTPUT_DIM).to(DEVICE)

    train_dataset = TripletDataset(train_triples)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler (warmup + cosine decay)
    def warmup_cosine_scheduler(epoch, warmup_epochs, total_epochs, base_lr, min_lr):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return max(min_lr / base_lr, cosine_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: warmup_cosine_scheduler(e, WARMUP_EPOCHS, NUM_EPOCHS, LEARNING_RATE, MIN_LR)
    )

    start_epoch = 0
    best_f1 = 0.0
    patience_counter = 0
    checkpoint_path = MODEL_DIR / "checkpoint.pt"
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_f1 = ckpt.get('best_f1', 0.0)
        print(f"Resumed from epoch {start_epoch}, best F1={best_f1:.4f}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

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

        # Validate after each epoch
        precision, recall, f1 = evaluate_validation(
            model, train_triples, valid_triples, all_items, neighbor_dict,
            node_features, relation_to_id, DEVICE, k=TOP_K
        )
        print(f"Validation @{TOP_K}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
            print(f"New best model saved with F1={f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_f1,
        }, checkpoint_path)

        scheduler.step()

    print("Training finished.")

if __name__ == "__main__":
    main()