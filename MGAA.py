import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RelationAwareGATStack(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_relations: int,
        relation_emb_dim: int,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_layer_norm: bool = True,
        use_residual: bool = True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual


        self.relation_emb = nn.Embedding(num_relations, relation_emb_dim)


        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        current_dim = hidden_channels
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            out_dim = out_channels if is_last else hidden_channels
            concat = not is_last

            gat = GATConv(
                in_channels=current_dim,
                out_channels=out_dim,
                heads=heads,
                concat=concat,
                dropout=dropout,
                add_self_loops=True,
                edge_dim=relation_emb_dim
            )
            self.gat_layers.append(gat)


            actual_out_dim = out_dim * heads if concat else out_dim
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(actual_out_dim))
            else:
                self.norms.append(nn.Identity())


            if use_residual and current_dim != actual_out_dim:
                self.residual_projs.append(nn.Linear(current_dim, actual_out_dim))
            else:
                self.residual_projs.append(nn.Identity())

            current_dim = actual_out_dim

        self.current_dim = current_dim
        self.activation = nn.ELU()
        self.final_norm = nn.LayerNorm(self.current_dim) if use_layer_norm else nn.Identity()

    def forward(self, x, edge_index, edge_attr):

        edge_emb = self.relation_emb(edge_attr)   # (E, relation_emb_dim)

        x = self.input_proj(x)

        for i, (gat, norm, res_proj) in enumerate(zip(self.gat_layers, self.norms, self.residual_projs)):
            x_new = gat(x, edge_index, edge_emb)
            x_new = self.activation(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x_new = norm(x_new)

            if self.use_residual:
                x = res_proj(x) + x_new
            else:
                x = x_new

        x = self.final_norm(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class MGAAModel(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_relations: int,
        relation_emb_dim: int = None,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        device: str = "cuda"
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device

        if relation_emb_dim is None:
            relation_emb_dim = hidden_dim

        self.gat_encoder = RelationAwareGATStack(
            in_channels=embedding_dim,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            num_relations=num_relations,
            relation_emb_dim=relation_emb_dim,
            heads=heads,
            num_layers=num_layers,
            dropout=dropout,
            use_layer_norm=True,
            use_residual=True
        )
        self.to(device)

    def build_subgraph(
            self,
            target_nodes: List[str],
            neighbor_dict: Dict[str, List[Tuple[str, str, str]]],
            node_features: Dict[str, np.ndarray],
            relation_to_id: Dict[str, int]
    ) -> Data:


        all_nodes = set()
        for node in target_nodes:
            if isinstance(node, str):
                all_nodes.add(node)
            else:
                print(f"Warning: target node {node} is not string, skipped")
        for tn in target_nodes:
            neighbors = neighbor_dict.get(tn, [])
            if isinstance(neighbors, set):
                neighbors = list(neighbors)
            for nb in neighbors:
                if isinstance(nb, (list, tuple)):
                    nb_id = nb[0]
                else:
                    nb_id = nb
                if isinstance(nb_id, str):
                    all_nodes.add(nb_id)
                else:
                    print(f"Warning: neighbor {nb_id} is not string, skipped")
        all_nodes = sorted(all_nodes)
        local_idx = {node: i for i, node in enumerate(all_nodes)}

        x_list = []
        for node in all_nodes:
            feat = node_features.get(node)
            if feat is None:
                feat = np.zeros(self.embedding_dim, dtype=np.float32)
            if isinstance(feat, set):
                print(f"Error: node {node} has set feature, using zero vector")
                feat = np.zeros(self.embedding_dim, dtype=np.float32)
            if torch.is_tensor(feat):
                feat = feat.cpu().numpy()
            if not isinstance(feat, np.ndarray):
                try:
                    feat = np.array(feat, dtype=np.float32)
                except:
                    feat = np.zeros(self.embedding_dim, dtype=np.float32)
            feat = feat.astype(np.float32)
            if feat.shape != (self.embedding_dim,):
                if feat.size == self.embedding_dim:
                    feat = feat.reshape(self.embedding_dim)
                else:
                    new_feat = np.zeros(self.embedding_dim, dtype=np.float32)
                    copy_len = min(feat.size, self.embedding_dim)
                    new_feat[:copy_len] = feat.flatten()[:copy_len]
                    feat = new_feat
            x_list.append(feat)

        x = torch.stack([torch.from_numpy(f) for f in x_list]).float().to(self.device)

        edge_src, edge_dst, edge_attrs = [], [], []
        for tn in target_nodes:
            if tn not in local_idx:
                continue
            src = local_idx[tn]
            for nb, rel, _ in neighbor_dict.get(tn, []):
                if isinstance(nb, (list, tuple)):
                    nb_id = nb[0]
                else:
                    nb_id = nb
                if nb_id not in local_idx:
                    continue
                dst = local_idx[nb_id]
                edge_src.append(src)
                edge_dst.append(dst)
                rel_id = relation_to_id.get(rel, 0)
                edge_attrs.append(rel_id)

        if not edge_src:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_attr = torch.empty((0,), dtype=torch.long, device=self.device)
        else:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=self.device)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long, device=self.device)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data


    def forward(
        self,
        batch_nodes: List[str],
        neighbor_dict: Dict[str, List[Tuple[str, str, str]]],
        node_features: Dict[str, np.ndarray],
        relation_to_id: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:

        subgraph = self.build_subgraph(batch_nodes, neighbor_dict, node_features, relation_to_id)
        out = self.gat_encoder(subgraph.x, subgraph.edge_index, subgraph.edge_attr)

        all_nodes = sorted(set(batch_nodes) | set([n for tn in batch_nodes for n,_,_ in neighbor_dict.get(tn, [])]))
        local_idx = {node: i for i, node in enumerate(all_nodes)}
        result = {node: out[local_idx[node]] for node in batch_nodes}
        return result