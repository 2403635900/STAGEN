import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from model import *
from util import *
from config import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class TemporalEvaluator:
    def __init__(self, model, train_events, test_events, config, entity_ts):
        self.model = model
        self.test_events = test_events
        self.config = config
        self.device = next(model.parameters()).device
        self.entity_ts = entity_ts
        self.negative_sampler = TemporalNegativeSampler(
            config.num_entities, entity_ts, 
            corruption='both', time_constraint='past'
        )
        self.time_system = TemporalKGSystem(config.num_entities, device='cuda')
        self.time_bins = self._quantile_split_time()
        self.dyn_graph = DynamicGraph()
        self.train_events = train_events
    def _quantile_split_time(self):
        ts = [e[3] for e in self.test_events]
        return np.quantile(
            ts, 
            np.linspace(0, 1, self.config.time_windows+1)
        )
    def evaluate(self):
        self.model.eval()
        all_ranks = []
        self.dyn_graph.add_events(self.train_events)
        with torch.no_grad():
            for win_idx in tqdm(range(len(self.time_bins)-1), desc="Evaluating"):
                window_end = self.time_bins[win_idx+1]
                window_events = [
                    e for e in self.test_events
                    if self.time_bins[win_idx] <= e[3] < window_end
                ]
                for batch in DataLoader(
                    window_events, 
                    batch_size=self.config.batch_size,
                    shuffle=False
                ):
                    h, r, t, ts = batch
                    h = h.to(self.device)
                    r = r.to(self.device)
                    t = t.to(self.device)
                    ts = ts.to(self.device)
                    g = self.dyn_graph.get_graph(self.device)
                    ent_emb, rel_emb = self.model(
                        g, 
                        self.time_system.node_timestamps
                    )
                    ranks = self.calculate_combined_ranks(
                        ent_emb, rel_emb, h, r, t, ts
                    )
                    all_ranks.extend(ranks.cpu().tolist())
                self.dyn_graph.add_events(window_events)
        
        return self.compute_metrics(torch.tensor(all_ranks))

    def calculate_combined_ranks(self, ent_emb, rel_emb, h, r, t, ts):
        batch_size = h.size(0)
        ranks = []
        for i in range(batch_size):
            tail_rank = self.predict_rank(
                ent_emb, rel_emb, 
                h[i], r[i], t[i], ts[i], 
                corrupt_tail=True
            )
            head_rank = self.predict_rank(
                ent_emb, rel_emb,
                t[i], r[i], h[i], ts[i],
                corrupt_tail=False
            )
            
            ranks.extend([tail_rank, head_rank])
        
        return torch.tensor(ranks)

    def predict_rank(self, ent_emb, rel_emb, h, r, t, ts, corrupt_tail=True):
        candidates = self.generate_candidates(h, r, t, ts, corrupt_tail)

        correct_entity = t if corrupt_tail else h
        if (candidates == correct_entity).sum() == 0:
            candidates = torch.cat([correct_entity.unsqueeze(0), candidates]).unique()
        
        if corrupt_tail:
            scores = self.score_triple(ent_emb[h], rel_emb[r], ent_emb[candidates], ts)
            target_idx = (candidates == t).nonzero(as_tuple=True)[0]
        else:
            scores = self.score_triple(ent_emb[candidates], rel_emb[r], ent_emb[t], ts)
            target_idx = (candidates == h).nonzero(as_tuple=True)[0]

        if target_idx.numel() == 0:
            return torch.tensor(float('inf'), device=self.device)

        sorted_indices = torch.argsort(scores, descending=True)
        ranks = (sorted_indices == target_idx).nonzero(as_tuple=True)[0] + 1
        return ranks.min().float()

    def generate_candidates(self, h, r, t, ts, corrupt_tail=True):

        time_mask = self.get_time_mask(t if corrupt_tail else h, ts)
        valid_entities = time_mask.nonzero(as_tuple=True)[0]

        num_available = len(valid_entities)
        num_neg = min(self.config.eval_neg_samples, num_available)
        if num_neg > 0:
            perm = torch.randperm(num_available, device=self.device)
            neg_entities = valid_entities[perm[:num_neg]]
        else:
            neg_entities = torch.tensor([], device=self.device)

        correct_entity = t if corrupt_tail else h
        correct_entity = correct_entity.unsqueeze(0)
        candidates = torch.cat([correct_entity, neg_entities]).unique()
    
        return candidates

    def get_time_mask(self, entity, ts):
        start_times = self.entity_ts[:, 0].to(self.device)
        return start_times <= ts

    def score_triple(self, h_emb, r_emb, t_candidates_emb, ts):
        time_emb = self.model.time_enc(ts.unsqueeze(0))
        return torch.sum(
            h_emb + r_emb + time_emb - t_candidates_emb,
            dim=-1
        )

    @staticmethod
    def compute_metrics(ranks):
        return {
            'mrr': torch.mean(1.0 / ranks).item(),
            'hits@1': (ranks <= 1).float().mean().item(),
            'hits@3': (ranks <= 3).float().mean().item(),
            'hits@10': (ranks <= 10).float().mean().item()
        }

