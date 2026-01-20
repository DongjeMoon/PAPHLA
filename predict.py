import argparse
import os
import os.path as osp
from types import SimpleNamespace

import pandas as pd
from pathlib import Path
import torch
import yaml
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_geometric.seed import seed_everything

from paphla.data.datasets import CustomCollate, MemMapDataset
from paphla.src.model import PAPHLA


def evaluation(model, test_dataset, model_weight, seed):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(seed)
    print(f"Using device: {device} | seed: {seed}")
    collate_fn = CustomCollate()
    test_loader = DataLoader(test_dataset,
                             batch_size=128,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True,
                             persistent_workers=True,
                             prefetch_factor=4,
                             collate_fn=collate_fn
                            )
    model_path = osp.join('models', f'{model_weight}.pt')
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    all_preds, all_attn_weights = [], []

    pbar = tqdm(test_loader, desc='Predicting', leave=True)

    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device)
            preds, w = model.predict(batch)
            all_preds.append(preds.cpu())
            # all_attn_weights.append(w)
    
    all_preds  = torch.cat(all_preds, dim=0)
    # all_attn_weights  = torch.cat(all_attn_weights, dim=0)

    return all_preds, all_attn_weights

def output(path=None, model_config=None, model_weight="", save_dir="./", print_metrics=False, seed=42):
    root = Path(__file__).resolve().parent
        
    df = pd.read_csv(osp.join(root, "paphla", path))

    model = PAPHLA(model_config)
    data = MemMapDataset(path)
    preds, attn_weights = evaluation(model, data, model_weight, seed)
    df["score"] = preds.numpy()
    df["pred"]  = df["score"].apply(lambda x: 1 if x >= 0.5 else 0)
    df["label"] = df["label"].apply(lambda x: 1 if x >= 0.5 else 0)
    
    if print_metrics:
        print("AUROC:", roc_auc_score(df.label, df.score))
        print("AUPRC:", average_precision_score(df.label, df.score))
        print("Accuracy:", accuracy_score(df.label, df.pred))
        print("F1:", f1_score(df.label, df.pred))
        print("MCC:", matthews_corrcoef(df.label, df.pred))
    
    os.makedirs(save_dir, exist_ok=True)
    
    df.to_csv(f"{save_dir}/results.csv", index=False)
    # np.save(f"{save_dir}/attention_weights.npy", attn_weights.numpy())
    print(f"Results saved to {save_dir}")
    
    return df, attn_weights

def main(): 
    parser = argparse.ArgumentParser(description='Predicting')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the test config file')
    args = parser.parse_args()
    if not os.path.isabs(args.config):
        config_path = os.path.abspath(args.config)
    else:
        config_path = args.config
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    test_path = config['test_data']
    model_config = SimpleNamespace(**config['model'])
    test_config = config['test']

    results, attn_weights = output(
        path=test_path,
        model_config=model_config,
        **test_config
    )

    print(results.head())

if __name__ == "__main__":
    main()