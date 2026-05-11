
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv

class Fusion(nn.Module):
    def __init__(self, args, hetero_data):
        super().__init__()
        self.d_node = getattr(args, 'emb_dim', 128)
        self.d_edge = self.d_node
        self.num_layers = getattr(args, 'num_layers', 1)
        self.device = torch.device(('cuda:%d' % args.gpu) if hasattr(args, 'gpu') and torch.cuda.is_available() else 'cpu')

        self.hetero_data = hetero_data.to(self.device)
        self.uk_edge_index = hetero_data.edge_index_dict[('user', 'u_k', 'knowledge')].to(self.device)
        self.ek_edge_index = hetero_data.edge_index_dict[('exercise', 'e_k', 'knowledge')].to(self.device)
        self.ku_edge_index = hetero_data.edge_index_dict[('knowledge', 'k_u', 'user')].to(self.device)
        self.ke_edge_index = hetero_data.edge_index_dict[('knowledge', 'k_e', 'exercise')].to(self.device)
        self.ue_edge_index = hetero_data.edge_index_dict[('user', 'u_e', 'exercise')].to(self.device)
        self.eu_edge_index = hetero_data.edge_index_dict[('exercise', 'e_u', 'user')].to(self.device)
        self.num_students = hetero_data['user'].num_nodes
        self.num_exercises = hetero_data['exercise'].num_nodes
        self.num_knowledge = hetero_data['knowledge'].num_nodes
        self.num_sc_edges = self.uk_edge_index.size(1)
        self.num_ce_edges = self.ek_edge_index.size(1)

        self.node_update_u = nn.Linear(self.d_node * 2, self.d_node)
        self.node_update_e = nn.Linear(self.d_node * 2, self.d_node)
        self.node_update_k = nn.Linear(self.d_node * 3, self.d_node)
        self.gate_update_u = nn.Linear(self.d_node * 2, self.d_node)
        self.gate_update_e = nn.Linear(self.d_node * 2, self.d_node)
        self.gate_update_k = nn.Linear(self.d_node * 3, self.d_node)
        self.ln_node_u = nn.LayerNorm(self.d_node)
        self.ln_node_e = nn.LayerNorm(self.d_node)
        self.ln_node_k = nn.LayerNorm(self.d_node)

        self.edge_update_sc = nn.Linear(self.d_node * 3, self.d_edge)
        self.edge_update_ce = nn.Linear(self.d_node * 3, self.d_edge)

        self.gate_sc_layer = nn.Sequential(nn.Linear(self.d_node * 3, self.d_edge))
        self.gate_ce_layer = nn.Sequential(nn.Linear(self.d_node * 3, self.d_edge))

        init_logit = 0.0
        init_noise = getattr(args, 'edge_init_noise', 0.05)
        self.edge_sc_logit = nn.Parameter(
            torch.randn(self.num_sc_edges, self.d_edge, device=self.device) * init_noise + init_logit
        )
        self.edge_ce_logit = nn.Parameter(
            torch.randn(self.num_ce_edges, self.d_edge, device=self.device) * init_noise + init_logit
        )

        self.edge_delta_clip = getattr(args, 'edge_delta_clip', 4.0)
        self.edge_logit_clip = getattr(args, 'edge_logit_clip', 3.0)

        self.gnn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.gnn_layers.append(HeteroConv({
                ('user', 'u_e', 'exercise'): GATConv(self.d_node, self.d_node, heads=1, concat=False, add_self_loops=False),
                ('exercise', 'e_k', 'knowledge'): GATConv(self.d_node, self.d_node, heads=1, concat=False, add_self_loops=False),
                ('exercise', 'e_u', 'user'): GATConv(self.d_node, self.d_node, heads=1, concat=False, add_self_loops=False),
                ('knowledge', 'k_e', 'exercise'): GATConv(self.d_node, self.d_node, heads=1, concat=False, add_self_loops=False)
            }, aggr='mean'))
        self.missing_sc_exp = getattr(args, 'missing_sc_exp', 'hybrid') 
        self.fill_only_q = getattr(args, 'fill_only_q', True)
        self.to(self.device)

    @staticmethod
    def scatter_sum(src, index, dim_size):
        if src.numel() == 0:
            return torch.zeros((dim_size, src.size(1)), device=src.device, dtype=src.dtype)
        out = torch.zeros((dim_size, src.size(1)), device=src.device, dtype=src.dtype)
        out.index_add_(0, index, src)
        return out
    @classmethod
    def scatter_mean(cls, src, index, dim_size):
        if src.numel() == 0:
            return torch.zeros((dim_size, src.size(1)), device=src.device, dtype=src.dtype)
        out = cls.scatter_sum(src, index, dim_size)
        cnt = torch.zeros((dim_size, 1), device=src.device, dtype=src.dtype)
        ones = torch.ones((index.size(0), 1), device=src.device, dtype=src.dtype)
        cnt.index_add_(0, index, ones)
        out = out / cnt.clamp_min(1.0)
        return out
    def forward(self, h_k, h_e, h_u, stu_id, exer_id):
        curr_edge_sc_logit = self.edge_sc_logit.clone()
        curr_edge_ce_logit = self.edge_ce_logit.clone()
        u_idx, k_idx_from_uk = self.uk_edge_index
        e_idx, k_idx_from_ek = self.ek_edge_index
        u_idx_ue, e_idx_ue = self.ue_edge_index
        e_idx_eu, u_idx_eu = self.eu_edge_index

        for layer in self.gnn_layers:

            h_dict = {'user': h_u, 'exercise': h_e, 'knowledge': h_k}
            h_update = layer(h_dict, self.hetero_data.edge_index_dict)
            h_u = h_u + h_update.get('user', torch.zeros_like(h_u))
            h_e = h_e + h_update.get('exercise', torch.zeros_like(h_e))
            h_k = h_k + h_update.get('knowledge', torch.zeros_like(h_k))
            
            old_sc = curr_edge_sc_logit
            old_ce = curr_edge_ce_logit
            edge_sc_prob = torch.sigmoid(old_sc)
            edge_ce_prob = torch.sigmoid(old_ce)

            delta_sc_raw = self.edge_update_sc(torch.cat([edge_sc_prob, h_u[u_idx], h_k[k_idx_from_uk]], dim=-1))
            # delta_sc = torch.clamp(delta_sc_raw, -self.edge_delta_clip, self.edge_delta_clip) 
            delta_sc = delta_sc_raw
            attn_sc = torch.sigmoid(torch.sum(h_u[u_idx] * h_k[k_idx_from_uk], dim=-1, keepdim=True))
            gate_sc = torch.sigmoid(self.gate_sc_layer(torch.cat([edge_sc_prob, h_u[u_idx], h_k[k_idx_from_uk]], dim=-1)))  
            new_sc = gate_sc * old_sc + (1.0 - gate_sc) * (delta_sc * attn_sc)

            delta_ce_raw = self.edge_update_ce(torch.cat([edge_ce_prob, h_k[k_idx_from_ek], h_e[e_idx]], dim=-1))
            # delta_ce = torch.clamp(delta_ce_raw, -self.edge_delta_clip, self.edge_delta_clip) 
            delta_ce = delta_ce_raw
            attn_ce = torch.sigmoid(torch.sum(h_k[k_idx_from_ek] * h_e[e_idx], dim=-1, keepdim=True))

            gate_ce = torch.sigmoid(self.gate_ce_layer(torch.cat([edge_ce_prob, h_k[k_idx_from_ek], h_e[e_idx]], dim=-1)))
            new_ce = gate_ce * old_ce + (1.0 - gate_ce) * (delta_ce * attn_ce)

            curr_edge_sc_logit = new_sc
            curr_edge_ce_logit = new_ce

            # if self.edge_logit_clip is not None:
            #     curr_edge_sc_logit = torch.clamp(curr_edge_sc_logit, -self.edge_logit_clip, self.edge_logit_clip)
            #     curr_edge_ce_logit = torch.clamp(curr_edge_ce_logit, -self.edge_logit_clip, self.edge_logit_clip)           


            edge_sc = torch.sigmoid(curr_edge_sc_logit)
            edge_ce = torch.sigmoid(curr_edge_ce_logit)


            new_h_u = self.ln_node_u(h_u + self.node_update_u(
                torch.cat([h_u, self.scatter_sum(edge_sc * h_k[k_idx_from_uk], u_idx, self.num_students)], dim=-1)))

            new_h_e = self.ln_node_e(h_e + self.node_update_e(
                torch.cat([h_e, self.scatter_sum(edge_ce * h_k[k_idx_from_ek], e_idx, self.num_exercises)], dim=-1)))

            new_h_k = self.ln_node_k(h_k + self.node_update_k(
                torch.cat([
                    h_k,
                    self.scatter_sum(edge_sc * h_u[u_idx], k_idx_from_uk, self.num_knowledge),
                    self.scatter_sum(edge_ce * h_e[e_idx], k_idx_from_ek, self.num_knowledge)
                ], dim=-1)))

            h_u = new_h_u
            h_e = new_h_e
            h_k = new_h_k


        final_edge_sc = torch.sigmoid(curr_edge_sc_logit)
        final_edge_ce = torch.sigmoid(curr_edge_ce_logit)

        B = stu_id.size(0)
        K = self.num_knowledge
        d = self.d_edge

        z_sc_batch = torch.zeros(B, K, d, device=h_k.device)
        z_ce_batch = torch.zeros(B, K, d, device=h_k.device)


        sc_seen = torch.zeros(B, K, dtype=torch.bool, device=h_k.device)

        if self.num_sc_edges > 0:
            mask_sc = stu_id.view(B, 1) == u_idx.view(1, -1)
            if mask_sc.any():
                b_idx_sc, edge_idx_sc = mask_sc.nonzero(as_tuple=True)
                k_idx_sc = k_idx_from_uk[edge_idx_sc]
                vals_sc = final_edge_sc[edge_idx_sc]
                flat_idx_sc = (
                    b_idx_sc.to(dtype=torch.long) * K + k_idx_sc.to(dtype=torch.long)
                ).to(device=h_k.device)

                z_sc_flat = z_sc_batch.view(B * K, d)
                z_sc_flat.index_add_(0, flat_idx_sc, vals_sc)

                sc_seen[b_idx_sc, k_idx_sc] = True

        if self.num_ce_edges > 0:
            mask_ce = exer_id.view(B, 1) == e_idx.view(1, -1)
            if mask_ce.any():
                b_idx_ce, edge_idx_ce = mask_ce.nonzero(as_tuple=True)
                k_idx_ce = k_idx_from_ek[edge_idx_ce]
                vals_ce = final_edge_ce[edge_idx_ce]

                flat_idx_ce = (
                    b_idx_ce.to(dtype=torch.long) * K + k_idx_ce.to(dtype=torch.long)
                ).to(device=h_k.device)

                z_ce_flat = z_ce_batch.view(B * K, d)
                z_ce_flat.index_add_(0, flat_idx_ce, vals_ce)

        if self.missing_sc_exp == 'hybrid' and self.num_sc_edges > 0:
            concept_prior = self.scatter_mean(final_edge_sc, k_idx_from_uk, self.num_knowledge)  # [K, d]

            student_prior = self.scatter_mean(final_edge_sc, u_idx, self.num_students)  # [num_students, d]

            if self.fill_only_q:
                q_mask = z_ce_batch.abs().sum(dim=-1) > 0   # [B, K]
            else:
                q_mask = torch.ones(B, K, dtype=torch.bool, device=h_k.device)

            missing_mask = (~sc_seen) & q_mask
            if missing_mask.any():
                b_miss, k_miss = missing_mask.nonzero(as_tuple=True)

                fill_vals = 0.5 * (
                    concept_prior[k_miss] + student_prior[stu_id[b_miss]]
                )

                z_sc_batch[b_miss, k_miss] = fill_vals

        return z_sc_batch, z_ce_batch, h_u, h_e, h_k