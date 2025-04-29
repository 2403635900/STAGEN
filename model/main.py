from util import *
from config import *
from model import *
from trainer import *

if __name__ == "__main__":
    train_events, test_events, num_e, num_r = load_data('./data/ICEWS14')
    entity_ts = build_entity_time_ranges(torch.tensor(train_events), num_entities=num_e).cuda()
    config = TemporalConfig(num_e, num_r)
    model = STAGEN(num_ent=config.num_entities, num_rel=config.num_relations, hidden_dim=config.hidden_dim, num_heads=config.num_heads).cuda()
    best_metrics = {'mrr': 0}
    trainer = TemporalTrainer(model, train_events, test_events, config, entity_ts)
    for epoch in range(config.epochs):
        avg_loss = trainer.train_epoch(epoch)
        if (epoch+1) % config.eval_every == 0:
            val_metrics = trainer.validate()
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | MRR: {val_metrics['mrr']:.4f} | h10: {val_metrics['hits@10']:.4f}")
            best_metrics = trainer.save_best_model(config.save_path, val_metrics, best_metrics)

        
    swa_metrics = trainer.validate(use_swa=True)
    print(f"SWA Model | MRR: {swa_metrics['mrr']:.4f}")
