import argparse
import math
import os
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch_geometric.seed import seed_everything
from torchmetrics.functional import auroc, average_precision
from tqdm import tqdm

from paphla.data.datasets import CustomCollate, MemMapDataset
from paphla.src.loss import LossComputer
from paphla.src.model import PAPHLA

# setup functions
def setup_optimizer_scheduler(model, finetune, lr, total_steps):
    if not finetune:
        optimizer = AdamW(
            model.parameters(), 
            lr=lr, weight_decay=1e-4,
            betas=(0.9, 0.95), eps=1e-8
        )
        scheduler = OneCycleLR(
            optimizer, max_lr=lr*2, total_steps=total_steps,
            div_factor=10.0, final_div_factor=10.0,
            pct_start=0.3, anneal_strategy='cos'
        )
    else:
        optim_groups = [
            {"params": model.head.parameters(),        "lr": 5e-4, "weight_decay": 0.01},
            {"params": model.RDBGPS.parameters(),      "lr": 1e-4, "weight_decay": 0.01},
            {"params": model.cross_attn.parameters(),  "lr": 1e-4, "weight_decay": 0.01},
            {"params": model.pep_encoder.parameters(), "lr": 1e-3, "weight_decay": 0.0},
            {"params": model.pep_proj.parameters(),    "lr": 1e-4, "weight_decay": 0.01},
            {"params": model.hla_proj.parameters(),    "lr": 1e-4, "weight_decay": 0.01},
            {"params": model.pma.parameters(),         "lr": 1e-4, "weight_decay": 0.01},
            {"params": model.mutual_proj.parameters(), "lr": 1e-4, "weight_decay": 0.01},
        ]
        optimizer = torch.optim.AdamW(optim_groups)

        def cosine_warmdown(step, total, warmup=0.05, min_ratio=0.1):
            w = int(total * warmup)
            if step < w:
                return (step+1)/max(1,w)
            t = (step - w) / max(1, total - w)
            return min_ratio + 0.5*(1-min_ratio)*(1 + math.cos(math.pi*t))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda s: cosine_warmdown(s, total_steps, 0.1, 0.3)
        )
    return optimizer, scheduler

def setup_loss_function(train_dataset, dro_config=None):
    if not dro_config:
        return torch.nn.BCEWithLogitsLoss()
    else:
        criterion = lambda z, y: torch.nn.functional.binary_cross_entropy_with_logits(
            z.view(-1), y.float().view(-1), reduction='none'
        )
        group_counts = train_dataset.group_counts().cpu().numpy()
        adj = np.sqrt(np.log(group_counts.max() / group_counts))
        return LossComputer(
            criterion, dataset=train_dataset, adj=adj, **dro_config
            # is_robust=True, alpha=0.15, 
            # normalize_loss=True,
            # btl=True, step_size=0.02,
        )

def compute_loss_and_collect_preds(model, batch, loss_fn, finetune):
    target = batch.y.float()
    
    if not finetune:
        preds, _ = model(batch)
        loss = loss_fn(preds, target)
        mask = torch.ones_like(target, dtype=torch.bool)
        canonical_loss = 0
    else:
        preds, canonical_loss = model(batch)
        mask = (batch.ptm_flag != 0)
        loss = loss_fn.loss(
            yhat=preds[mask], y=target[mask],
            group_idx=batch.ptm_flag[mask],
            is_training=True
        ) + canonical_loss
    
    with torch.no_grad():
        probs = torch.sigmoid(preds[mask]).detach().cpu()
        target_masked = target[mask].detach().cpu()
    
    return loss, probs, target_masked, canonical_loss

def save_checkpoint(model, model_name):
    torch.save(model.state_dict(), f'models/{model_name}.pt')

# Training
def train(train_path, val_path, model_config, lr=1e-4,
          batch_size=256, epochs=50, patience=5, model_name='exp0', seed=42, finetune=False, model_weight=None, dro=None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(seed)
    print(f"Using device: {device} | seed: {seed}")

    model = PAPHLA(model_config).to(device)

    if model_weight:
        model_path = os.path.join('models', f'{model_weight}.pt')
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)

    train_dataset = MemMapDataset(path=train_path)
    val_dataset = MemMapDataset(path=val_path)
    
    collate_fn = CustomCollate()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True,
                              prefetch_factor=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True,
                            prefetch_factor=4, collate_fn=collate_fn)
    
    steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
    total_steps = steps_per_epoch * epochs
    loss_fn = setup_loss_function(train_dataset, dro)
    optimizer, scheduler = setup_optimizer_scheduler(model, finetune, lr, total_steps)
    
    best_metric = -float("inf")
    count = 0
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler, finetune)
        print('-' * 15)
        val_metric, val_loss = eval_epoch(model, val_loader, loss_fn, device, finetune)

        if val_metric > best_metric:
            best_metric = val_metric
            count = 0
            print('Saving checkpoint')
            save_checkpoint(model, model_name)
        else:
            count += 1
            print(f"Patience: {count}/{patience}")
            if count >= patience:
                print("Early stopping.")
                break
        print('=' * 15)



def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, finetune):
    model.train()
    total_loss = 0
    num_samples = 0
    all_preds, all_target = [], []

    pbar = tqdm(data_loader, desc='Training', leave=True)

    for batch in pbar:
        batch = batch.to(device)
        
        loss, probs, targets, canonical_loss = compute_loss_and_collect_preds(
            model, batch, loss_fn, finetune
        )
        
        all_preds.append(probs)
        all_target.append(targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_samples += batch.y.size(0)
        total_loss += loss.item() * batch.y.size(0)
        
        if scheduler:
            scheduler.step()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'canon': f'{canonical_loss:.4f}' if finetune else '0.0'
        })

    all_preds = torch.cat(all_preds, dim=0)
    all_target = torch.cat(all_target, dim=0)
    metrics = eval_metrics(all_preds, all_target)
    avg_loss = total_loss / num_samples
    print(f'Train loss: {avg_loss:.4f} | AUROC: {metrics["auroc"]:.4f} | AUPRC: {metrics["auprc"]:.4f}')

def eval_epoch(model, data_loader, loss_fn, device, finetune=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_target = [], []
    all_group_idx = []

    pbar = tqdm(data_loader, desc='Predicting', leave=True)

    with torch.no_grad():
        for batch in pbar:
            batch  = batch.to(device)
            target = batch.y.float()
            if not finetune:
                preds, _ = model(batch)
                loss = loss_fn(preds, target)
                probs = torch.sigmoid(preds)
                all_preds.append(probs)
                all_target.append(target)
                batch_size = target.size(0)
                total_loss += loss.item() * batch_size
            else:
                preds, _ = model(batch)
                mask = (batch.ptm_flag != 0)
                probs = torch.sigmoid(preds[mask])
                all_preds.append(probs)
                all_target.append(target[mask])
                all_group_idx.append(batch.ptm_flag[mask].cpu())

    all_preds   = torch.cat(all_preds, dim=0)
    all_target = torch.cat(all_target, dim=0)

    if not finetune:
        metrics = eval_metrics(all_preds, all_target)
        avg_loss = total_loss / len(all_target)
        print(f'Val loss: {avg_loss:.4f} | AUROC: {metrics["auroc"]:.4f} | AUPRC: {metrics["auprc"]:.4f}')
        return metrics["auprc"], avg_loss
    else:
        all_group_idx = torch.cat(all_group_idx, dim=0)
        per_group = {}
        valid_scores = []
        
        for g in torch.unique(all_group_idx):
            g = int(g.item())
            if g == 0:
                continue
            m = (all_group_idx == g)
            y = all_target[m]
            p = all_preds[m]

            ap = eval_metrics(p, y)["auprc"]
            per_group[int(g)] = float(ap)
            valid_scores.append(ap.detach().cpu())

        macro = float(np.mean(valid_scores)) if len(valid_scores) > 0 else 0.0
        print(f'AUPRC: {macro:.4f}')
        return macro, None


# Evaluation
def confusion_matrix(preds: torch.Tensor, target: torch.Tensor):
    pred_label = (preds >= 0.5).long()
    target_bool = target.bool()
    pred_bool = pred_label.bool()

    tp = (pred_bool &  target_bool).sum()
    tn = (~pred_bool & ~target_bool).sum()
    fp = (pred_bool & ~target_bool).sum()
    fn = (~pred_bool &  target_bool).sum()

    return tn, fp, fn, tp

# def basic_metrics(tn, fp, fn, tp, eps=1e-8):
#     total = tp + tn + fp + fn

#     acc  = (tp + tn) / (total + eps)
#     prec = tp / (tp + fp + eps)
#     rec  = tp / (tp + fn + eps)
#     f1   = 2 * prec * rec / (prec + rec + eps)

#     return acc, prec, rec, f1

def ranking_metrics(preds: torch.Tensor, target: torch.Tensor):
    unique = torch.unique(target)
    if unique.numel() < 2:
        return torch.tensor(0.5), torch.tensor(0.5)

    auroc_val = auroc(preds, target.int(), task="binary")
    auprc_val = average_precision(preds, target.int(), task="binary")

    return auroc_val, auprc_val

def eval_metrics(preds: torch.Tensor, target: torch.Tensor):
    auroc_val, auprc_val = ranking_metrics(preds, target)
    # tn, fp, fn, tp = confusion_matrix(preds, target)
    # acc, prec, rec, f1 = basic_metrics(tn, fp, fn, tp)

    # metrics = {
    #     'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    #     'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
    #     'auroc': auroc_val, 'auprc': auprc_val, 'mcc': mcc_val
    # }
    
    # print(f'TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}')
    # print(f'Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}')
    # print(f'AUROC: {auroc_val:.4f} | AUPRC: {auprc_val:.4f} | MCC: {mcc_val:.4f}')

    metrics = {
        'auroc': auroc_val, 'auprc': auprc_val
    }

    return metrics


def main(): 
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the training config file')
    args = parser.parse_args()
    if not os.path.isabs(args.config):
        config_path = os.path.abspath(args.config)
    else:
        config_path = args.config
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract parameters from config
    train_path = config['train_data']
    val_path = config['val_data']
    model_config = SimpleNamespace(**config['model'])
    train_config = config['training']
    dro_config = config.get('dro', None)
    
    # Train
    train(
        train_path=train_path,
        val_path=val_path,
        model_config=model_config,
        dro=dro_config,
        **train_config
    )


if __name__ == "__main__":
    main()