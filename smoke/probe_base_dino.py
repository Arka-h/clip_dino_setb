"""Gate-1 base-DINO determinism probe.
Builds clean upstream DINO-4scale, loads the official COCO checkpoint, runs a FIXED input through it in
eval mode, and writes pred_logits/pred_boxes to --out. Run twice (two separate processes) and diff the
two outputs: max abs diff must be 0.0. Determinism flags are set BEFORE build_model (required).

Usage: python probe_base_dino.py --out /path/run1.pt
"""
import os, sys, argparse
# --- determinism BEFORE any CUDA/build work ---
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch, numpy as np, random
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

DINO = "/home/rahul/arka/clip_dino_setb/DINO_upstream"
sys.path.insert(0, DINO)
from main import get_args_parser, build_model_main
from util.slconfig import SLConfig
from util.misc import nested_tensor_from_tensor_list

def make_args(config_file):
    args = get_args_parser().parse_args([
        '--config_file', config_file,
        '--coco_path', '/mnt/1tb/data/coco',
        '--output_dir', '/tmp/_dino_probe',
    ])
    cfg = SLConfig.fromfile(config_file)
    cfg_dict = cfg._cfg_dict.to_dict()
    av = vars(args)
    for k, v in cfg_dict.items():
        if k not in av:
            setattr(args, k, v)
    # eval-time fields the build path may read
    args.device = 'cuda'
    return args

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--config_file', default=os.path.join(DINO, 'config/DINO/DINO_4scale.py'))
    p.add_argument('--ckpt', default='/home/rahul/arka/clip_dino_setb/provenance/dino_r50_4scale_coco_official.pth')
    a = p.parse_args()

    args = make_args(a.config_file)
    model, criterion, postprocessors = build_model_main(args)
    model.to('cuda').eval()
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(ck['model'], strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")

    # FIXED deterministic input: one 3x800x800 image (seed already fixed above)
    g = torch.Generator().manual_seed(1234)
    img = torch.rand(3, 800, 800, generator=g)
    samples = nested_tensor_from_tensor_list([img]).to('cuda')

    with torch.no_grad():
        out = model(samples)
    pl = out['pred_logits'].float().cpu()
    pb = out['pred_boxes'].float().cpu()
    torch.save({'pred_logits': pl, 'pred_boxes': pb}, a.out)
    print(f"[probe] pred_logits {tuple(pl.shape)} sum={pl.sum().item():.6f}  "
          f"pred_boxes {tuple(pb.shape)} sum={pb.sum().item():.6f} -> {a.out}")

if __name__ == '__main__':
    main()
