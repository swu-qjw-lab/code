import torch
import torch.nn as nn
from fusion import Fusion
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, emb_dim, num_expert=4,
                 t_min=0.3, t_max=5.0):
        super().__init__()
        self.num_expert = num_expert
        self.emb_dim = emb_dim
        self.experts = nn.ModuleList([
            nn.Linear(emb_dim, 1) for _ in range(num_expert)
        ])

        gate_dim = 3 * emb_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, num_expert)
        )
        self.log_t = nn.Parameter(torch.tensor(0.0)) 
        self.t_min = float(t_min)
        self.t_max = float(t_max)

    def forward(self, diff, stu_local, exer_local, kn_local, kn_r, m_batch, d_batch):
        B, K, D = diff.size()
        device = diff.device
        mask = kn_r > 0  
        stu_local_k = torch.zeros((B, K, D), device=device, dtype=diff.dtype)
        exer_local_k = torch.zeros((B, K, D), device=device, dtype=diff.dtype)
        kn_local_batch = torch.zeros((B, K, D), device=device, dtype=diff.dtype)
        b_idx, k_idx = mask.nonzero(as_tuple=True)
        if b_idx.numel() > 0:
            stu_local_k[b_idx, k_idx] = stu_local[b_idx]
            exer_local_k[b_idx, k_idx] = exer_local[b_idx]
            kn_local_batch[b_idx, k_idx] = kn_local[k_idx]
        gate_input = torch.cat([
            stu_local_k,
            exer_local_k,
            kn_local_batch
        ], dim=-1)
        logits = self.gate(gate_input)  
        t = torch.exp(self.log_t).clamp(self.t_min, self.t_max)
        gate_prob = torch.softmax(logits/t , dim=-1) 
        expert_outputs = [exp(diff) for exp in self.experts]        
        expert_stack = torch.stack(expert_outputs, dim=-1)      
        moe_out = torch.sum(gate_prob.unsqueeze(2) * expert_stack, dim=-1) 
        mask_f = (kn_r > 0).float()                
        denom = mask_f.sum().clamp_min(1.0)

        p = (gate_prob * mask_f.unsqueeze(-1)).sum(dim=(0,1)) / denom   
        lb_loss = ((p - 1.0/self.num_expert) ** 2).sum()
        return moe_out, gate_prob,lb_loss


class Net(nn.Module):
    def __init__(self, args, local_map):
        super(Net, self).__init__()
        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')

        self.exer_n = args.exer_n
        self.knowledge_n = args.knowledge_n
        self.student_n = args.student_n
        self.emb_dim = args.emb_dim 

        self.student_emb = nn.Embedding(self.student_n, self.emb_dim)
        self.knowledge_emb = nn.Embedding(self.knowledge_n, self.emb_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.emb_dim)

        self.k_index = torch.arange(self.knowledge_n).to(self.device)
        self.stu_index = torch.arange(self.student_n).to(self.device)
        self.exer_index = torch.arange(self.exer_n).to(self.device)

        self.FusionLayer = Fusion(args, local_map)
        self.moe = MoE(self.emb_dim, num_expert=2) #4
        self.prednet_full3 = nn.Linear(self.emb_dim, 1)

        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.normal_(param, mean=0.0, std=0.01) 
        self.to(self.device)

    def forward(self, stu_id, exer_id, kn_r, return_loss=False, return_analysis=False):
        h_u = self.student_emb(self.stu_index).to(self.device)
        h_k = self.knowledge_emb(self.k_index).to(self.device)
        h_e = self.exercise_emb(self.exer_index).to(self.device)

        m_batch, d_batch, h_u_new, h_e_new, h_k_new = self.FusionLayer(
            h_k, h_e, h_u, stu_id, exer_id
        )

        stu_local = h_u_new[stu_id]
        exer_local = h_e_new[exer_id]

        diff = m_batch - d_batch
        moe_out, gate_prob, lb_loss = self.moe(
            diff, stu_local, exer_local, h_k_new, kn_r, m_batch, d_batch
        )
        o = torch.sigmoid(moe_out)

        sum_out = torch.sum(o * kn_r.unsqueeze(2), dim=1)
        count_of_concept = torch.sum(kn_r, dim=1).unsqueeze(1)
        output = sum_out / count_of_concept

        if return_analysis:
            return output, m_batch, d_batch, gate_prob

        if return_loss:
            return output, lb_loss

        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.moe.experts.apply(clipper)


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
#model.py

