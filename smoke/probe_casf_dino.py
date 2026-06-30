"""Gate-1-full + zero-init-invariant probe for DINO-CASF.
Builds full DINO-CASF (casf=True), loads the official COCO checkpoint (shape-filtered), runs a FIXED
image + FIXED sketch through it. Two uses:
  (a) run twice (separate processes) -> max abs diff must be 0.0 (Gate 1 full).
  (b) compare pred_boxes to base DINO run (smoke/out/run1.pt): with DePathBlock zero-init (identity) and
      BoxSketchAdapter gate=0 (identity), the box/localisation path is identity at init -> pred_boxes
      BIT-IDENTICAL to base DINO (sketch-independent). pred_logits differ (gated class head reinit).
Built with num_classes=91 so enc_out_class_embed loads pretrained -> identical query selection -> the
box-identity comparison is exact.

Usage: python probe_casf_dino.py --out run.pt [--train] [--compare_base smoke/out/run1.pt]
"""
import os, sys, argparse
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch, numpy as np, random
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(0); np.random.seed(0); random.seed(0)

ROOT = "/home/rahul/arka/clip_dino_setb"
DINO = os.path.join(ROOT, "DINO_upstream")
sys.path.insert(0, ROOT); sys.path.insert(0, DINO)
from main import get_args_parser, build_model_main
from util.slconfig import SLConfig
from util.misc import nested_tensor_from_tensor_list

CLIP_CKPT = "/home/rahul/arka/clip_ddetr_ow_repr/checkpoints/clip_model/vit_clip2.pth"

def make_args(config_file):
    args = get_args_parser().parse_args(
        ['--config_file', config_file, '--coco_path', '/mnt/1tb/data/coco', '--output_dir', '/tmp/_dino_probe'])
    cfg = SLConfig.fromfile(config_file)
    for k, v in cfg._cfg_dict.to_dict().items():
        if k not in vars(args):
            setattr(args, k, v)
    args.device = 'cuda'
    args.casf = True
    args.casf_clip_ckpt = CLIP_CKPT
    args.casf_freeze = 'partial'
    return args

def filter_ckpt(state, model):
    msd = model.state_dict()
    keep = {k: v for k, v in state.items() if k in msd and msd[k].shape == v.shape}
    return keep

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--train', action='store_true')
    p.add_argument('--compare_base', default=None)
    p.add_argument('--config_file', default=os.path.join(DINO, 'config/DINO/DINO_4scale.py'))
    p.add_argument('--ckpt', default=os.path.join(ROOT, 'provenance/dino_r50_4scale_coco_official.pth'))
    a = p.parse_args()

    args = make_args(a.config_file)
    model, criterion, postprocessors = build_model_main(args)
    model.to('cuda')
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    keep = filter_ckpt(ck['model'], model)
    missing, unexpected = model.load_state_dict(keep, strict=False)
    dropped = [k for k in ck['model'] if k not in keep]
    print(f"[load] kept={len(keep)} dropped(shape/absent)={len(dropped)} missing={len(missing)} unexpected={len(unexpected)}")
    print(f"[load] sample dropped: {[k for k in dropped][:4]}")
    # sanity: box head + enc_out_class_embed must have loaded (needed for box-identity invariant)
    assert all(not k.startswith('transformer.enc_out_class_embed') for k in missing), "enc_out_class_embed did NOT load!"

    g = torch.Generator().manual_seed(1234)
    img = torch.rand(3, 800, 800, generator=g)
    samples = nested_tensor_from_tensor_list([img]).to('cuda')
    gs = torch.Generator().manual_seed(4321)
    sketch = torch.rand(1, 3, 224, 224, generator=gs).to('cuda')

    if a.train:
        model.train()
        # one dummy target so CDN activates
        tgt = [{'labels': torch.tensor([1, 1], device='cuda'),
                'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]], device='cuda')}]
        out = model(samples, targets=tgt, sketches=sketch)
        print(f"[train] pred_logits {tuple(out['pred_logits'].shape)} pred_boxes {tuple(out['pred_boxes'].shape)} "
              f"dn_meta_keys={list(out['dn_meta'].keys()) if out['dn_meta'] else None}")
    else:
        model.eval()
        with torch.no_grad():
            out = model(samples, sketches=sketch)

    pl = out['pred_logits'].float().cpu(); pb = out['pred_boxes'].float().cpu()
    torch.save({'pred_logits': pl, 'pred_boxes': pb}, a.out)
    print(f"[probe] pred_logits {tuple(pl.shape)} sum={pl.sum().item():.6f}  "
          f"pred_boxes {tuple(pb.shape)} sum={pb.sum().item():.6f} -> {a.out}")

    if a.compare_base:
        base = torch.load(a.compare_base)
        db = (pb - base['pred_boxes']).abs().max().item()
        print(f"[zero-init invariant] max|Δ pred_boxes vs base DINO| = {db}")
        print("ZERO-INIT BOX INVARIANT PASS" if db == 0.0 else "ZERO-INIT BOX INVARIANT FAIL")

if __name__ == '__main__':
    main()
