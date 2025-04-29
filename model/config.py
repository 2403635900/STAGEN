import torch
class TemporalConfig:
    def __init__(self, num_e, num_r):
        self.epochs = 100
        self.batch_size = 2048
        self.time_windows = 20      
        self.accum_steps = 4         
        self.swa_start = 75          
        self.tau = 0.1              
        self.grad_clip = 1.5         
        self.fp16 = True             
        self.time_horizon = 30    
        self.hidden_dim = 32        
        self.num_heads = 4           
        self.num_entities = num_e    
        self.num_relations = num_r   
        self.lr = 3e-4               
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval_neg_samples = 1000
        self.eval_every = 1
        self.save_path = "./best_model.pth"
