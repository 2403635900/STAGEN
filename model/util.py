import os
import torch
from torch.utils.data import Dataset
from torch_scatter import scatter_max  
from collections import defaultdict, namedtuple
def load_data(path):
    with open(os.path.join(path, "stat.txt")) as f:
        for line in f:
            line_split = line.split()
            num_e, num_r = int(line_split[0]), int(line_split[1])
    def read_events(file):
        events = []
        with open(os.path.join(path, file)) as f:
            for line in f:
                line_split = line.split()
                h = int(line_split[0])
                t = int(line_split[2])
                r = int(line_split[1])
                ts = int(line_split[3])
                events.append((int(h), int(r), int(t), int(ts)))
        return events
    train_events = read_events("train.txt")
    test_events = read_events("test.txt")
    return train_events, test_events, num_e, num_r

class SlidingDataset(Dataset):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.num_batches = len(data) // batch_size
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        return self.data[start:end]
    
class TemporalNegativeSampler:
    def __init__(self, num_entities, entity_ts, corruption='both', time_constraint='past'):
        self.num_entities = num_entities
        self.entity_ts = entity_ts  
        self.corruption = corruption
        self.time_constraint = time_constraint

    def sample(self, batch, ts):
        h, r, t = batch
        device = h.device
        batch_size = len(h)
        neg_h, neg_t = h.clone(), t.clone()

        entity_start = self.entity_ts[:, 0].expand(batch_size, -1)  
        entity_end = self.entity_ts[:, 1].expand(batch_size, -1)
        current_ts = ts.view(-1, 1).expand(-1, self.num_entities)   

        if self.time_constraint == 'past':
            valid_mask = (entity_start <= current_ts)  
        elif self.time_constraint == 'strict':
            valid_mask = (entity_start <= current_ts) & (current_ts < entity_end)
        else:
            raise ValueError(f"Unsupported time_constraint: {self.time_constraint}")

        mask = torch.rand(batch_size, device=device) < 0.5
        valid_mask = valid_mask.to(device)

        if self.corruption in ['t', 'both']:
            t_mask = mask & (self.corruption != 'h')
            for i in torch.where(t_mask)[0]:
                candidates = valid_mask[i].nonzero().view(-1)
                if len(candidates) > 0:
                    neg_t[i] = candidates[torch.randint(0, len(candidates), (1,))]

        if self.corruption in ['h', 'both']:
            h_mask = (~mask) & (self.corruption != 't')
            for i in torch.where(h_mask)[0]:
                candidates = valid_mask[i].nonzero().view(-1)
                if len(candidates) > 0:
                    neg_h[i] = candidates[torch.randint(0, len(candidates), (1,))]

        return (neg_h, r, neg_t, ts)
    
class DynamicGraph:
    def __init__(self):
        self.edge_index = []
        self.edge_type = []
        self.edge_time = []
        self._adj_cache = None
    
    def add_events(self, events):
        for h, r, t, ts in events:
            self.edge_index.append((h, t))
            self.edge_type.append(r)
            self.edge_time.append(ts)
        self._adj_cache = None
    
    def get_graph(self, device):
        if self._adj_cache is None:
            edges = torch.LongTensor(self.edge_index).T
            self._adj_cache = {
                'edge_index': edges.to(device),
                'edge_type': torch.LongTensor(self.edge_type).to(device),
                'edge_time': torch.FloatTensor(self.edge_time).to(device)
            }
        return self._adj_cache
    
class TemporalKGSystem:
    def __init__(self, num_entities, device='cpu'):
        self.num_entities = num_entities
        self.node_timestamps = torch.zeros(num_entities, device=device)
        self.node_update_count = torch.zeros(num_entities, dtype=torch.long, device=device)

    def update_timestamps(self, events):
        h, t, ts = events[0], events[2], events[3] 
        
        entities = torch.cat([h, t])
        timestamps = torch.cat([ts, ts])
        
        max_ts, _ = scatter_max(timestamps, entities, dim_size=self.num_entities)
        update_mask = max_ts > self.node_timestamps
        
        self.node_timestamps[update_mask] = max_ts[update_mask]
        self.node_update_count += torch.bincount(entities, minlength=self.num_entities)   
        
def build_entity_time_ranges(data, num_entities):
    entity_min_ts = torch.full((num_entities,), float('inf'))
    entity_max_ts = torch.full((num_entities,), float('-inf'))
    h, r, t, ts = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    for entity, timestamp in zip(h, ts):
        entity = entity.item()
        timestamp = timestamp.item()
        if timestamp < entity_min_ts[entity]:
            entity_min_ts[entity] = timestamp
        if timestamp > entity_max_ts[entity]:
            entity_max_ts[entity] = timestamp

    for entity, timestamp in zip(t, ts):
        entity = entity.item()
        timestamp = timestamp.item()
        if timestamp < entity_min_ts[entity]:
            entity_min_ts[entity] = timestamp
        if timestamp > entity_max_ts[entity]:
            entity_max_ts[entity] = timestamp

    global_min = ts.min().item()
    global_max = ts.max().item()
    entity_min_ts[entity_min_ts == float('inf')] = global_min
    entity_max_ts[entity_max_ts == float('-inf')] = global_max

    return torch.stack([entity_min_ts, entity_max_ts], dim=1)
