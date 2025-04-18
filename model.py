import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from torch_scatter import scatter_softmax, scatter_add
class FourierTimeEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(hidden_dim//2).abs() + 1e-8)
    def forward(self, ts):
        ts = ts.unsqueeze(-1)
        map_ts = ts * self.freq
        return torch.cat([torch.sin(map_ts), torch.cos(map_ts)], dim=-1)

class MSTODE(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.space_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.GELU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        self.time_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Softplus(),
            nn.Linear(hidden_dim//2, hidden_dim)
        )
        self.fusion_gate = nn.Linear(hidden_dim*2, 2)

    def forward(self, t, h):
        space_deriv = self.space_net(h)
        time_deriv = self.time_net(h)
        gates = torch.sigmoid(self.fusion_gate(torch.cat([space_deriv, time_deriv], dim=-1)))
        return gates[:,0:1]*space_deriv + gates[:,1:2]*time_deriv

class DRGCNLayer(nn.Module):
    def __init__(self, hidden_dim, num_relations, num_heads):
        super().__init__()
        self.relation_emb = nn.Embedding(num_relations, hidden_dim)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.time_coeff = nn.Parameter(torch.tensor(1.0))
        self.weight_gen = nn.Sequential(
            nn.Linear(hidden_dim*2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, graph, x, timestamps):
        src, dst = graph['edge_index']
        rel = graph['edge_type']
        edge_time = graph['edge_time']
        
        delta_t = (timestamps[dst] - edge_time)
        delta_t = delta_t / (self.time_coeff.abs() + 1e-9)
        time_mask = torch.sigmoid(delta_t).unsqueeze(-1)
        rel_emb = self.relation_emb(rel) * time_mask
        
        src_emb = x[src]
        weight_input = torch.cat([src_emb, rel_emb, time_mask], dim=-1)
        dynamic_weights = torch.sigmoid(self.weight_gen(weight_input))
        message = src_emb * rel_emb * dynamic_weights

        q = self.query(x[dst]).view(-1, self.num_heads, self.head_dim)
        k = self.key(message).view(-1, self.num_heads, self.head_dim)
        v = self.value(message).view(-1, self.num_heads, self.head_dim)
        
        attn_scores = (q * k).sum(dim=-1) / (self.head_dim ** 0.5)
        attn_weights = scatter_softmax(attn_scores, dst, dim=0)
        
        weighted_v = v * attn_weights.unsqueeze(-1)
        out = scatter_add(weighted_v, dst, dim=0, dim_size=x.size(0))
        
        return out.view(-1, self.num_heads * self.head_dim)

class CausalContrast(nn.Module):
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau  
    def forward(self, anchor_emb, positive_emb, negative_embs, anchor_ts, negative_ts):
        time_diff = torch.abs(anchor_ts - negative_ts)  
        time_weight = torch.exp(-time_diff / self.tau)  
        
        anchor = F.normalize(anchor_emb, dim=-1)
        positive = F.normalize(positive_emb, dim=-1)
        negatives = F.normalize(negative_embs, dim=-1)
        
        pos_sim = torch.sum(anchor * positive, dim=-1)  
        neg_sim = torch.matmul(anchor, negatives.T)      

        weighted_neg_sim = neg_sim * time_weight
        logits = torch.cat([pos_sim.unsqueeze(1), weighted_neg_sim], dim=1) / self.tau
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
        return F.cross_entropy(logits, labels)

class STAGEN(nn.Module):
    def __init__(self, num_ent, num_rel, hidden_dim, num_heads):
        super().__init__()
        self.ent_emb = nn.Embedding(num_ent, hidden_dim)
        self.rel_emb = nn.Embedding(num_rel, hidden_dim)
        self.drgcn = DRGCNLayer(hidden_dim, num_rel, num_heads)
        self.mst_ode = MSTODE(hidden_dim)
        self.ccl = CausalContrast(hidden_dim)
        self.time_enc = FourierTimeEncoder(hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_ent)
        )
    def forward(self, graph, query_time):
        x = self.ent_emb.weight
        rel = self.rel_emb.weight
        
        x = self.drgcn(graph, x, query_time)
        
        t_points = torch.linspace(0, 1, 2).to(x.device)
        h0 = torch.cat([x, rel], dim=0)
        h_ode = odeint(self.mst_ode, h0, t_points, method='dopri5')[-1]
        
        ent_emb = h_ode[:len(x)]
        rel_emb = h_ode[len(x):]
        
        return ent_emb, rel_emb