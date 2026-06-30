"""DATA smoke test for the DINO-CASF dataset/engine adapter (task #6). No full training.
Checks: shapes, binary labels, holdout exclusion, seed-14 reproducibility, sketch normalization,
one end-to-end train step (forward+loss+backward), and a micro-batch memory probe."""
import os, sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch, numpy as np, random
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(0); np.random.seed(0); random.seed(0)

ROOT = "/home/rahul/arka/clip_dino_setb"
DINO = os.path.join(ROOT, "DINO_upstream")
sys.path.insert(0, ROOT); sys.path.insert(0, DINO)
from torch.utils.data import DataLoader
from casf.data import CocoSketchDINO, collate_fn_casf, build_seed14_binary_gt, SET_B, ALL_CATEGORIES

ok = True
def check(name, cond, extra=""):
    global ok; ok = ok and bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))

print(f"=== Set B ({len(SET_B)}): {SET_B}")
check("Set B == handover 14", sorted(SET_B) == sorted(
    ['backpack','bicycle','clock','couch','dog','elephant','knife','mouse','oven','pizza',
     'sandwich','skateboard','stop sign','train']))

# ---------------- TRAIN (open): binary + holdout exclusion ----------------
print("\n=== TRAIN (open) ===")
tr = CocoSketchDINO('train', 'open', data_frac=0.01)
setb_ids = {tr.class2id[c] for c in SET_B}
n_check = 5
all_bin, no_setb = True, True
for i in range(n_check):
    img, tgt, sk = tr[i]
    if i == 0:
        check("train image is CHW float", img.ndim == 3, f"shape={tuple(img.shape)}")
        check("train sketch [3,224,224]", tuple(sk.shape) == (3, 224, 224))
    all_bin &= bool((tgt['labels'] == 1).all())
    no_setb &= (tgt['sketch_cat'] not in setb_ids)
check("all train labels == 1 (binary)", all_bin)
check("no Set-B category selected in train", no_setb)

# ---------------- VAL open_qd / closed_qd + seed-14 reproducibility ----------------
print("\n=== VAL open_qd (Set B only) ===")
vo = CocoSketchDINO('val', 'open')
sel_o = []
for i in range(5):
    _, tgt, sk = vo[i]
    name = vo.id2class[tgt['sketch_cat']]
    sel_o.append((vo.ids[i], name))
    print(f"  img {vo.ids[i]}: seed-14 cat = {name}")
check("open_qd selected cats are all in Set B", all(n in SET_B for _, n in sel_o))
check("val sketch CLIP-normalized & finite", torch.isfinite(sk).all().item() and sk.min() < 0)

# reproducibility: rebuild dataset, same seed-14 picks
vo2 = CocoSketchDINO('val', 'open')
sel_o2 = [(vo2.ids[i], vo2.id2class[vo2[i][1]['sketch_cat']]) for i in range(5)]
check("seed-14 cat selection reproducible across constructions", sel_o == sel_o2, f"{sel_o2}")

print("\n=== VAL closed_qd (all 56) ===")
vc = CocoSketchDINO('val', 'closed')
sel_c = []
for i in range(5):
    _, tgt, _ = vc[i]
    sel_c.append(vc.id2class[tgt['sketch_cat']])
print(f"  closed seed-14 cats: {sel_c}")
check("closed_qd selected cats are within the 56", all(c in ALL_CATEGORIES for c in sel_c))

# seed-14 binary GT for pycocotools
gt = build_seed14_binary_gt(vo)
check("seed-14 binary GT built (cat id 1)", gt.dataset['categories'][0]['id'] == 1,
      f"{len(gt.dataset['images'])} imgs, {len(gt.dataset['annotations'])} anns")

# ---------------- ONE TRAIN STEP end-to-end (forward+loss+backward) ----------------
print("\n=== ONE TRAIN STEP (DINO-CASF, num_classes=2) ===")
from main import get_args_parser, build_model_main
from util.slconfig import SLConfig
CLIP_CKPT = "/home/rahul/arka/clip_ddetr_ow_repr/checkpoints/clip_model/vit_clip2.pth"
def make_args():
    cf = os.path.join(DINO, 'config/DINO/DINO_4scale.py')
    args = get_args_parser().parse_args(['--config_file', cf, '--coco_path', '/mnt/1tb/data/coco', '--output_dir', '/tmp/_casf'])
    for k, v in SLConfig.fromfile(cf)._cfg_dict.to_dict().items():
        if k not in vars(args): setattr(args, k, v)
    args.device = 'cuda'; args.casf = True; args.casf_clip_ckpt = CLIP_CKPT; args.casf_freeze = 'partial'
    args.num_classes = 2; args.dn_labelbook_size = 2
    return args
args = make_args()
model, criterion, _ = build_model_main(args); model.to('cuda'); model.train(); criterion.train()
ck = torch.load(os.path.join(ROOT, 'provenance/dino_r50_4scale_coco_official.pth'), map_location='cpu', weights_only=False)
msd = model.state_dict(); keep = {k: v for k, v in ck['model'].items() if k in msd and msd[k].shape == v.shape}
mk, uk = model.load_state_dict(keep, strict=False)
print(f"[load] kept={len(keep)} missing={len(mk)} unexpected={len(uk)} (num_classes=2 -> class heads reinit, expected)")

def to_dev(targets):
    out = []
    for t in targets:
        out.append({k: (v.to('cuda') if torch.is_tensor(v) else v) for k, v in t.items()})
    return out

loader = DataLoader(tr, batch_size=2, collate_fn=collate_fn_casf, num_workers=0)
samples, targets, sketches = next(iter(loader))
samples = samples.to('cuda'); sketches = sketches.to('cuda'); targets = to_dev(targets)
out = model(samples, targets=targets, sketches=sketches)
loss_dict = criterion(out, targets)
weight = criterion.weight_dict
total = sum(loss_dict[k] * weight[k] for k in loss_dict if k in weight)
total.backward()
finite = torch.isfinite(total).item()
check("one train step loss finite", finite, f"total_loss={total.item():.4f}")
print("  loss components:", {k: round(float(loss_dict[k]), 4) for k in list(loss_dict)[:6]})
# grad reached a CASF param?
g = model.transformer.decoder.refine_output[0].attn.w_q.weight.grad
check("grad reaches DePathBlock", g is not None and torch.isfinite(g).all())

# ---------------- MEMORY PROBE ----------------
print("\n=== MEMORY PROBE (micro-batch that fits) ===")
import math
max_bs = 0
for bs in (1, 2, 3, 4, 6):
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        ld = DataLoader(tr, batch_size=bs, collate_fn=collate_fn_casf, num_workers=0)
        s, t, sk = next(iter(ld)); s = s.to('cuda'); sk = sk.to('cuda'); t = to_dev(t)
        o = model(s, targets=t, sketches=sk)
        l = criterion(o, t); tot = sum(l[k] * weight[k] for k in l if k in weight); tot.backward()
        model.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  bs={bs}: OK, peak {peak:.1f} GB")
        max_bs = bs
    except RuntimeError as e:
        print(f"  bs={bs}: OOM/err ({str(e)[:60]})"); break
if max_bs:
    accum = math.ceil(8 / max_bs)
    print(f"  -> max micro-batch ~{max_bs}; grad-accum {accum} steps for effective batch {max_bs*accum} (target 8)")

print("\n" + ("ALL DATA-SMOKE CHECKS PASS" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
