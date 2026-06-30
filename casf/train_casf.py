"""DINO-CASF training entrypoint (Gate 2 build).

Ties together: deterministic build (analysis recipe, BEFORE build_model) -> DINO-CASF (num_classes=2,
casf=True) loaded shape-filtered from the official COCO ckpt -> AdamW 3-group optimizer + cos34 LambdaLR
-> casf engine train loop with gradient accumulation to effective batch 8 -> per-epoch checkpoint +
provenance bundle -> OW/CW seed-14 single-query pycocotools 12-stat eval + retention.

Train OW -> eval both OW (open_qd, Set-B) and CW (closed_qd, 56) from the SAME checkpoint (no separate
CW run). Dry run: --epochs 1 --data_frac 0.01 --microbatch 1 --max_iters 2.
"""
import os, sys, json, math, argparse, subprocess
# ---- determinism BEFORE any CUDA/build (analysis main.py:254-274 recipe) ----
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch, numpy as np, random
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DINO = os.path.join(ROOT, 'DINO_upstream')
sys.path.insert(0, DINO); sys.path.insert(0, ROOT)
from main import get_args_parser, build_model_main
from util.slconfig import SLConfig
from torch.utils.data import DataLoader
from casf.data import CocoSketchDINO, collate_fn_casf, build_seed14_binary_gt, SET_B
from casf.engine_casf import train_one_epoch_casf, evaluate_casf

CLIP_CKPT = "/home/rahul/arka/clip_ddetr_ow_repr/checkpoints/clip_model/vit_clip2.pth"
COCO_CKPT = os.path.join(ROOT, 'provenance/dino_r50_4scale_coco_official.pth')
COCO_PATH = '/mnt/1tb/data/coco'

# cos34 schedule (clip_ddetr_clean_run/util/scheduler.py): multiplier on each group's base_lr.
WARMUP, COS_END = 5, 34
START_MULT, PEAK_MULT, MIN_MULT = 0.10, 0.80, 0.075
def cos34_mult(epoch):
    if epoch < WARMUP:
        return START_MULT + (PEAK_MULT - START_MULT) * (epoch / WARMUP)
    if epoch < COS_END:
        p = (epoch - WARMUP) / (COS_END - WARMUP)
        return MIN_MULT + (PEAK_MULT - MIN_MULT) * 0.5 * (1.0 + math.cos(math.pi * p))
    return MIN_MULT


def build_args(device):
    cf = os.path.join(DINO, 'config/DINO/DINO_4scale.py')
    a = get_args_parser().parse_args(['--config_file', cf, '--coco_path', COCO_PATH, '--output_dir', '/tmp/_casf'])
    for k, v in SLConfig.fromfile(cf)._cfg_dict.to_dict().items():
        if k not in vars(a):
            setattr(a, k, v)
    a.device = device
    a.casf = True; a.casf_clip_ckpt = CLIP_CKPT; a.casf_freeze = 'partial'
    a.num_classes = 2; a.dn_labelbook_size = 2
    return a


def build_model_and_load(device):
    args = build_args(device)
    model, criterion, postprocessors = build_model_main(args)
    model.to(device)
    ck = torch.load(COCO_CKPT, map_location='cpu', weights_only=False)
    msd = model.state_dict()
    keep = {k: v for k, v in ck['model'].items() if k in msd and msd[k].shape == v.shape}
    dropped = [k for k in ck['model'] if k not in keep]
    mk, uk = model.load_state_dict(keep, strict=False)
    # box head (256-in) must have loaded; class heads (now 512-in / 2-way) must be among dropped/reinit
    box_loaded = any('bbox_embed' in k for k in keep)
    print(f"[load] kept={len(keep)} dropped={len(dropped)} missing(in-model-not-ckpt)={len(mk)} "
          f"unexpected={len(uk)} | bbox_embed loaded={box_loaded}")
    return model, criterion, postprocessors, args


def make_optimizer(model, lr=1e-4, lr_backbone=2e-5, lr_proj_mult=0.1, wd=1e-4):
    def match(n, kws): return any(k in n for k in kws)
    bb, proj = ['backbone.0'], ['reference_points', 'sampling_offsets']
    groups = [
        {'params': [p for n, p in model.named_parameters()
                    if not match(n, bb) and not match(n, proj) and p.requires_grad], 'lr': lr},
        {'params': [p for n, p in model.named_parameters()
                    if match(n, bb) and p.requires_grad], 'lr': lr_backbone},
        {'params': [p for n, p in model.named_parameters()
                    if match(n, proj) and p.requires_grad], 'lr': lr * lr_proj_mult},
    ]
    n = [len(g['params']) for g in groups]
    print(f"[optim] AdamW groups: main={n[0]}(lr{lr}) backbone={n[1]}(lr{lr_backbone}) proj={n[2]}(lr{lr*lr_proj_mult})")
    return torch.optim.AdamW(groups, lr=lr, weight_decay=wd)


def git_commit():
    try:
        return subprocess.check_output(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD']).decode().strip()
    except Exception:
        return ''


@torch.no_grad()
def run_eval(model, postprocessors, device, num_select, max_eval_imgs=None):
    out = {}
    for scheme, name in [('open', 'OW(open_qd/SetB)'), ('closed', 'CW(closed_qd/56)')]:
        ds = CocoSketchDINO('val', scheme, n_sketches=1)
        if max_eval_imgs:
            ds.ids = ds.ids[:max_eval_imgs]   # debug cap (dry run)
        gt = build_seed14_binary_gt(ds)
        ld = DataLoader(ds, batch_size=2, collate_fn=collate_fn_casf, num_workers=2)
        print(f"\n=== EVAL {name} ({len(ds)} imgs) ===")
        stats = evaluate_casf(model, postprocessors, ld, gt, device, num_select=num_select)
        out[scheme] = stats
        print(f"[{name}] 12-stat = {stats}")
    if out.get('open') and out.get('closed') and out['closed']['AP'] not in (0, 0.0):
        ret = out['open']['AP'] / out['closed']['AP']
        print(f"\n[retention] OW_AP/CW_AP = {out['open']['AP']:.4f} / {out['closed']['AP']:.4f} = {ret:.4f}")
    else:
        print("\n[retention] CW_AP==0 (untrained) -> retention undefined (path verified)")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_frac', type=float, default=1.0)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--microbatch', type=int, default=2)
    p.add_argument('--effective_batch', type=int, default=8)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--wandb_name', default='dino_casf_ow')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--resume', default='')
    p.add_argument('--device', default='cuda')
    p.add_argument('--max_iters', type=int, default=None, help='debug: cap train iters/epoch')
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--max_eval_imgs', type=int, default=None, help='debug: cap eval images')
    a = p.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    os.makedirs(a.output_dir, exist_ok=True)
    accum = max(1, a.effective_batch // a.microbatch)
    print(f"[cfg] microbatch={a.microbatch} accum={accum} -> effective={a.microbatch*accum} (target {a.effective_batch})")

    model, criterion, postprocessors, args = build_model_and_load(a.device)
    num_select = getattr(args, 'num_select', 300)

    if a.resume:
        sd = torch.load(a.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(sd['model']); print(f"[resume] {a.resume} (epoch {sd.get('epoch')})")

    if a.eval_only:
        run_eval(model, postprocessors, a.device, num_select, a.max_eval_imgs); return

    optimizer = make_optimizer(model)
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cos34_mult)
    base_main = sched.base_lrs[0]  # 1e-4; LambdaLR already applied mult to param_groups at construction
    print(f"[cos34] ep0 mult={cos34_mult(0):.3f} -> main lr={base_main*cos34_mult(0):.2e} "
          f"(expect 1.00e-05); ep5 mult={cos34_mult(5):.3f} (lr {base_main*cos34_mult(5):.2e}); "
          f"ep34 mult={cos34_mult(34):.4f} (lr {base_main*cos34_mult(34):.2e})")

    # provenance bundle (no overwrite)
    prov = os.path.join(a.output_dir, 'provenance.json')
    if not os.path.exists(prov):
        json.dump({'wandb_name': a.wandb_name, 'output_dir': a.output_dir, 'holdout_setB': sorted(SET_B),
                   'seed': a.seed, 'git_commit': git_commit(), 'data_frac': a.data_frac,
                   'effective_batch': a.microbatch*accum, 'schedule': 'cos34', 'epochs': a.epochs,
                   'design_log': 'design_log/DESIGN_LOG.md (D1 CDN,D2 LFT,D3 class,D6 enc-free)'},
                  open(prov, 'w'), indent=2)

    train_ds = CocoSketchDINO('train', 'open', data_frac=a.data_frac, n_sketches=1)
    train_ld = DataLoader(train_ds, batch_size=a.microbatch, shuffle=True,
                          collate_fn=collate_fn_casf, num_workers=a.num_workers, drop_last=True)
    print(f"[data] train open frac={a.data_frac}: {len(train_ds)} imgs, {len(train_ld)} micro-batches")

    for epoch in range(a.epochs):
        avg = train_one_epoch_casf(model, criterion, train_ld, optimizer, a.device, epoch,
                                   accum_steps=accum, max_norm=0.1, max_iters=a.max_iters)
        sched.step(epoch)  # absolute, once per epoch (clean_run main.py:505 semantics)
        ckpt = os.path.join(a.output_dir, 'checkpoint.pth')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'lr_scheduler': sched.state_dict(), 'epoch': epoch, 'args': vars(a)}, ckpt)
        print(f"[ep{epoch}] avg_loss={avg:.4f} lr={optimizer.param_groups[0]['lr']:.2e} -> saved {ckpt}")

    print("\n=== final eval (OW + CW) ===")
    run_eval(model, postprocessors, a.device, num_select, a.max_eval_imgs)


if __name__ == '__main__':
    main()
