import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from model import *
import torch.optim as optim
import torch.optim.swa_utils as swa
from torch.optim.swa_utils import AveragedModel, SWALR
from util import *
from config import *
from evulate import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class TemporalTrainer:
    def __init__(self, model, train_events, test_events, config, entity_ts):
        self.model = model
        self.train_events = train_events
        self.config = config
        self.device = next(model.parameters()).device
        self.scaler = GradScaler(enabled=config.fp16)
        self.time_system = TemporalKGSystem(config.num_entities, device='cuda')
        self.dyn_graph = DynamicGraph()
        self.time_bins = self._quantile_split_time()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=2e-4, weight_decay=1e-5
        )
        self.swa_model = AveragedModel(model)
        self.swa_scheduler = SWALR(
            self.optimizer, swa_lr=0.8 * 2e-4
        )
        self.entity_ts = entity_ts
        self.simply_negative_sampler = TemporalNegativeSampler(config.num_entities, self.entity_ts)
        self.test_events = test_events
        self.tester = TemporalEvaluator(model, self.train_events, self.test_events, config, self.entity_ts)
    def _quantile_split_time(self):
        ts = [e[3] for e in self.train_events]
        return np.quantile(
            ts, 
            np.linspace(0, 1, self.config.time_windows+1)
        )

    def _create_sliding_loader(self, data):
        return DataLoader(
            SlidingDataset(data, self.config.batch_size),
            batch_size=None,
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()

        with tqdm(total=len(self.time_bins)-1, desc=f"Epoch {epoch}") as pbar:
            for win_idx in range(len(self.time_bins)-1):
                window_start = self.time_bins[win_idx]
                window_end = self.time_bins[win_idx+1]
                window_events = [
                    e for e in self.train_events
                    if window_start <= e[3] < window_end
                ]
                self.dyn_graph.add_events(window_events)
                current_graph = self.dyn_graph.get_graph(self.device)
                
                train_start = window_end
                train_end = train_start + self.config.time_horizon
                train_data = [
                    e for e in self.train_events
                    if train_start <= e[3] < train_end
                ]
                if not train_data:
                    pbar.update(1)
                    continue
                
                loader = self._create_sliding_loader(train_data)
                
                for batch_idx, batch in tqdm(enumerate(loader)):
                    h, r, t, ts = zip(*batch)
                    h = torch.LongTensor(h).to(self.device)
                    r = torch.LongTensor(r).to(self.device)
                    t = torch.LongTensor(t).to(self.device)
                    ts = torch.FloatTensor(ts).to(self.device)
                    self.time_system.update_timestamps((h, r, t, ts))
                    with autocast(device_type='cuda', dtype=torch.float16, enabled=self.config.fp16):
                        ent_emb, rel_emb = self.model(current_graph, self.time_system.node_timestamps)
                        loss = self.compute_loss(
                            ent_emb, rel_emb, h, r, t, ts
                        )
                        loss = loss / self.config.accum_steps
                        
                    self.scaler.scale(loss).backward()
                    
                    if (batch_idx+1) % self.config.accum_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.grad_clip
                        )

                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()

                        if epoch >= self.config.swa_start:
                            self.swa_model.update_parameters(self.model)
                            self.swa_scheduler.step()
                    
                    total_loss += loss.item() * self.config.accum_steps
                    pbar.set_postfix({'loss': total_loss/(batch_idx+1)})
                pbar.update(1)
        
        return total_loss / len(loader)

    def compute_loss(self, ent_emb, rel_emb, h, r, t, ts):
        pe = ent_emb[h]+rel_emb[r]+self.model.time_enc(ts)
        p_scores = self.model.decoder(pe)
        neg_samples = self.simply_negative_sampler.sample((h, r, t), ts)
        neg_h, neg_r, neg_t, neg_ts = neg_samples
        ne = ent_emb[neg_h]+rel_emb[neg_r]+self.model.time_enc(neg_ts)
        n_scores = self.model.decoder(ne)
        combined_scores = torch.cat([p_scores, n_scores], dim=1)
        main_loss = F.cross_entropy(combined_scores, torch.cat([t, neg_t], dim=1).long().to(device))
        contrast_loss = self.model.ccl(
            anchor_emb=ent_emb[h], 
            positive_emb=ent_emb[t], 
            negative_embs=ent_emb[neg_t], 
            anchor_ts=ts,
            negative_ts=neg_ts
        )
        return  0.7*main_loss + 0.3*contrast_loss
    def validate(self, use_swa=False):
        model = self.swa_model.module if use_swa else self.model
        with torch.no_grad():
            metrics = self.tester.evaluate()
        return metrics
    def save_best_model(self, path, metrics, best_metrics):
        if metrics['mrr'] > best_metrics['mrr']:
            torch.save(self.model.state_dict(), path)
            best_metrics = metrics
        return best_metrics
    def save_checkpoint(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'swa_model': self.swa_model.state_dict(),
            'dyn_graph': self.dyn_graph,
            'time_bins': self.time_bins
        }, path)
    
    @classmethod
    def from_checkpoint(cls, path, model, train_events, config):
        ckpt = torch.load(path)
        trainer = cls(model, train_events, config)
        trainer.model.load_state_dict(ckpt['model'])
        trainer.optimizer.load_state_dict(ckpt['optimizer'])
        trainer.swa_model.load_state_dict(ckpt['swa_model'])
        trainer.dyn_graph = ckpt['dyn_graph']
        trainer.time_bins = ckpt['time_bins']
        return trainer

