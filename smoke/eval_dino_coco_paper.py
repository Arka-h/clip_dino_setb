"""Reproduce the official DINO-4scale-R50 (12ep) COCO val2017 detection AP, using the clean vendored
DINO code + the official checkpoint, and save the full 12-stat to outputs/dino_paper_verify/.

This is the base-detector paper-reproduction check (expected box AP ~0.49). Deterministic eval.
Usage: python smoke/eval_dino_coco_paper.py
"""
import os, sys, json, datetime
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch, numpy as np, random
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(0); np.random.seed(0); random.seed(0)

ROOT = "/home/rahul/arka/clip_dino_setb"
DINO = os.path.join(ROOT, "DINO_upstream")
sys.path.insert(0, DINO)
from main import get_args_parser, build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate
from torch.utils.data import DataLoader, SequentialSampler
import util.misc as utils

CKPT = os.path.join(ROOT, "provenance/dino_r50_4scale_coco_official.pth")
OUT  = os.path.join(ROOT, "outputs/dino_paper_verify")
CFG  = os.path.join(DINO, "config/DINO/DINO_4scale.py")
STAT_NAMES = ['AP','AP@50','AP@75','AP_s','AP_m','AP_l','AR@1','AR@10','AR@100','AR_s','AR_m','AR_l']

def build_args():
    args = get_args_parser().parse_args([
        '--config_file', CFG, '--coco_path', '/mnt/1tb/data/coco', '--output_dir', OUT, '--eval',
    ])
    for k, v in SLConfig.fromfile(CFG)._cfg_dict.to_dict().items():
        if not hasattr(args, k):
            setattr(args, k, v)
    args.device = 'cuda'; args.distributed = False; args.dataset_file = 'coco'
    args.num_workers = 4
    return args

def main():
    os.makedirs(OUT, exist_ok=True)
    args = build_args()
    model, criterion, postprocessors = build_model_main(args)
    model.to('cuda').eval()
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    miss, unexp = model.load_state_dict(ck['model'], strict=False)
    print(f"[load] missing={len(miss)} unexpected={len(unexp)} (expect 0/0)")

    ds_val = build_dataset(image_set='val', args=args)
    loader = DataLoader(ds_val, 2, sampler=SequentialSampler(ds_val), drop_last=False,
                        collate_fn=utils.collate_fn, num_workers=args.num_workers)
    base_ds = get_coco_api_from_dataset(ds_val)

    stats, _ = evaluate(model, criterion, postprocessors, loader, base_ds, 'cuda', OUT, args=args)
    coco12 = stats['coco_eval_bbox']
    metrics = dict(zip(STAT_NAMES, coco12))

    result = {
        "what": "DINO-4scale R50 (official 12ep ckpt) COCO val2017 detection reproduction",
        "checkpoint": CKPT, "config": "DINO_4scale.py", "code": "vendored DINO d84a491",
        "dataset": "/mnt/1tb/data/coco val2017 (5000 imgs)", "num_classes": 91,
        "deterministic": True, "load_missing": len(miss), "load_unexpected": len(unexp),
        "date": datetime.date.today().isoformat(),
        "expected_paper_box_AP": "~0.490 (DINO-4scale R50 12ep)",
        "coco_eval_bbox_12stat_raw": coco12,
        "metrics": metrics,
    }
    with open(os.path.join(OUT, "dino_coco_val_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    lines = ["# DINO-4scale R50 — COCO val2017 reproduction (base detector)\n",
             f"- checkpoint: official 12ep ({os.path.basename(CKPT)}); code: vendored DINO d84a491",
             f"- load: missing={len(miss)} unexpected={len(unexp)}; deterministic eval; val2017 5000 imgs",
             f"- expected paper box AP ~0.490\n", "| metric | value |", "|---|---|"]
    lines += [f"| {n} | {metrics[n]:.4f} |" for n in STAT_NAMES]
    lines.append(f"\n**box AP = {metrics['AP']:.4f}**")
    with open(os.path.join(OUT, "RESULT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved -> {OUT}/dino_coco_val_result.json + RESULT.md")

if __name__ == '__main__':
    main()
