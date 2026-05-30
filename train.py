# train.py
import os
import json
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import h5py
from collections import defaultdict


from ONS import BatchNeighborSampler, build_graph_from_triples
from MGAA import MGAAModel
from train_utils import load_triples

# ---------------------------- 关系嵌入模块 ----------------------------
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


# ---------------------------- 完整模型 ----------------------------
class KRLModel(nn.Module):
    def __init__(self, entity_encoder, relation_embedding, embedding_dim):
        super().__init__()
        self.entity_encoder = entity_encoder
        self.relation_embedding = relation_embedding
        self.embedding_dim = embedding_dim

    def forward(self, head_ids, rel_names, tail_ids, neighbor_dict, node_features, relation_to_id):
        all_entities = set(head_ids) | set(tail_ids)
        entity_emb_dict = self.entity_encoder(
            list(all_entities), neighbor_dict, node_features, relation_to_id
        )
        h_emb = torch.stack([entity_emb_dict[h] for h in head_ids])
        t_emb = torch.stack([entity_emb_dict[t] for t in tail_ids])
        r_emb = self.relation_embedding(rel_names)

        dist = (h_emb*t_emb*r_emb).sum(dim=-1)
        return dist


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


def evaluate_filtered(
        model,
        test_triples,
        all_correct,
        all_entities,
        neighbor_dict,
        node_features,
        relation_to_id,
        device,
        batch_size=256,
        filter_self=True
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
        entity_embs_cpu = torch.stack([entity_emb_dict[eid].cpu() for eid in all_entities])  # (N, dim)

    ranks = []
    for i in range(0, len(test_triples), batch_size):
        batch = test_triples[i:i+batch_size]
        heads, rels, tails = zip(*batch)


        head_embs = torch.stack([entity_emb_dict[h] for h in heads]).to(device)
        rel_embs = model.relation_embedding(list(rels)).to(device)


        combined = (head_embs + rel_embs).cpu()  # (B, dim)
        dists = torch.cdist(combined, entity_embs_cpu, p=2)  # (B, N)

        for j, (h, r, t) in enumerate(batch):
            filter_tails = filter_dict.get((h, r), set()).copy()
            if filter_self and t in filter_tails:
                filter_tails.remove(t)

            dists_j = dists[j].clone()
            for ft in filter_tails:
                idx = entity_to_idx.get(ft)
                if idx is not None:
                    dists_j[idx] = float('inf')

            sorted_indices = torch.argsort(dists_j)
            true_idx = entity_to_idx[t]
            rank = (sorted_indices == true_idx).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)

    ranks = np.array(ranks)
    mrr = np.mean(1.0 / ranks)
    hits1 = np.mean(ranks <= 1)
    hits3 = np.mean(ranks <= 3)
    hits10 = np.mean(ranks <= 10)
    return mrr, hits1, hits3, hits10


# ---------------------------- 训练主函数 ----------------------------
def train():
    # ==================== 配置参数 ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./WN18RR"
    model_dir = "./trainmodel"
    os.makedirs(model_dir, exist_ok=True)

    # 文件路径
    train_path = f"{data_dir}/train.txt"
    valid_path = f"{data_dir}/valid.txt"
    test_path = f"{data_dir}/test.txt"
    h5_path = "./semantic_embeddings/WN_embeddings.h5"

    # 超参数
    embedding_dim = 384
    hidden_dim = 512
    output_dim = 256
    num_layers = 3
    heads = 8
    dropout = 0.2
    T = 30
    walk_times = 10
    walk_length = 3
    lambda_weight = 0.5
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = 200
    margin = 5.0
    use_bpr = True


    train_triples = load_triples(train_path)
    valid_triples = load_triples(valid_path)
    test_triples = load_triples(test_path)
    all_triples = train_triples + valid_triples + test_triples
    all_correct = set(all_triples)


    all_entities = list(set(e for triple in all_triples for e in (triple[0], triple[2])))
    print(f"Total entities: {len(all_entities)}")


    with h5py.File(h5_path, 'r') as f:
        ids = [x.decode('utf-8') for x in f['ids'][:]]
        embeddings = f['embeddings'][:]
    node_features = {eid: embeddings[i].astype(np.float32) for i, eid in enumerate(ids)}

    for eid in all_entities:
        if eid not in node_features:
            node_features[eid] = np.zeros(embedding_dim, dtype=np.float32)


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
    sample_entity = all_entities[0]
    print(f"Sample entity {sample_entity} embedding norm: {np.linalg.norm(node_features[sample_entity]):.4f}")
    print(f"Sample entity neighbors: {len(neighbor_dict.get(sample_entity, []))}")


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

    rel_init = {}
    for rel in relation_set:
        rel_init[rel] = node_features.get(rel, np.zeros(embedding_dim, dtype=np.float32))
    relation_embedding = RelationEmbedding(
        relation_ids=list(relation_set),
        init_embeddings=rel_init,
        in_dim=embedding_dim,
        out_dim=output_dim
    ).to(device)
    model = KRLModel(entity_encoder, relation_embedding, output_dim).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch = 0
    best_mrr = 0.0
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

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for heads, rels, tails in pbar:
            num_neg = 5
            neg_dists = []
            for _ in range(num_neg):
                neg_tails = tails.copy()
                random.shuffle(neg_tails)
                for i in range(len(neg_tails)):
                    if neg_tails[i] == tails[i]:
                        neg_tails[i] = random.choice([e for e in all_entities if e != tails[i]])
                neg_dist = model(heads, rels, neg_tails, neighbor_dict, node_features, relation_to_id)
                neg_dists.append(neg_dist)

            pos_dist = model(heads, rels, tails, neighbor_dict, node_features, relation_to_id)
            neg_dist = torch.min(torch.stack(neg_dists), dim=0)[0]
            loss = -torch.log(torch.sigmoid(neg_dist - pos_dist)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.6f}")

        if (epoch + 1) % 5 == 0:
            mrr, h1, h3, h10 = evaluate_filtered(
                model=model,
                test_triples=valid_triples,
                all_correct=set(train_triples + valid_triples),
                all_entities=all_entities,
                neighbor_dict=neighbor_dict,
                node_features=node_features,
                relation_to_id=relation_to_id,
                device=device,
                batch_size=128,
                filter_self=True
            )
            print(f"Validation MRR: {mrr:.4f}, Hits@1: {h1:.4f}, Hits@3: {h3:.4f}, Hits@10: {h10:.4f}")

            if mrr > best_mrr:
                best_mrr = mrr
                torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
                print(f"New best model saved with MRR {best_mrr:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_mrr': best_mrr,
        }, checkpoint_path)



if __name__ == "__main__":
    train()