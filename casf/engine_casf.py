"""DINO-CASF train/eval engine helpers (task #6).
- train_one_epoch_casf: standard DINO loss loop + GRADIENT ACCUMULATION (effective batch = bs*accum_steps;
  the clean repo had no accum — single-GPU needs it to hit effective 8 without dropping micro-batch).
- evaluate_casf: forward with sketches -> class-agnostic (binary) detections -> pycocotools COCOeval vs the
  seed-14 binary GT -> full 12-stat.

EVAL SCORING DECISION (flag for Gate-2 verification): the objective is binary (num_classes=2, GT label 1 =
"object"). We score each query by the OBJECT-class probability sigmoid(pred_logits[...,1]) and assign
category_id=1, then keep top-`num_select`. This matches the binary GT (single category id 1) and the
class-agnostic protocol. To be confirmed against a trained checkpoint in Gate 2 (CW>OW, sane band)."""
import sys, math, os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'DINO_upstream'))
from util import misc as utils
from util.box_ops import box_cxcywh_to_xyxy


def _targets_to_device(targets, device):
    out = []
    for t in targets:
        out.append({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()})
    return out


def train_one_epoch_casf(model, criterion, data_loader, optimizer, device, epoch,
                         accum_steps=1, max_norm=0.1, print_every=50, max_iters=None):
    model.train(); criterion.train()
    weight = criterion.weight_dict
    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    for it, (samples, targets, sketches) in enumerate(data_loader):
        if max_iters is not None and it >= max_iters:
            break
        samples = samples.to(device); sketches = sketches.to(device)
        targets = _targets_to_device(targets, device)
        out = model(samples, targets=targets, sketches=sketches)
        loss_dict = criterion(out, targets)
        loss = sum(loss_dict[k] * weight[k] for k in loss_dict if k in weight)
        (loss / accum_steps).backward()                       # scale for accumulation
        running += loss.item()
        if (it + 1) % accum_steps == 0:
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if it % print_every == 0:
            print(f"  [ep{epoch} it{it}] loss={loss.item():.3f}", flush=True)
    return running / max(1, len(data_loader))


@torch.no_grad()
def evaluate_casf(model, postprocessors, data_loader, base_ds_gt, device, num_select=300):
    """base_ds_gt: pycocotools COCO object from build_seed14_binary_gt. Returns the 12-stat dict."""
    from pycocotools.cocoeval import COCOeval
    model.eval()
    results = []
    for samples, targets, sketches in data_loader:
        samples = samples.to(device); sketches = sketches.to(device)
        out = model(samples, sketches=sketches)
        logits = out['pred_logits']                            # [B,Q,2]
        boxes = out['pred_boxes']                              # [B,Q,4] cxcywh norm
        obj = logits[..., 1].sigmoid()                         # object-class score (see decision note)
        for b, t in enumerate(targets):
            orig_h, orig_w = [int(x) for x in t['orig_size'].tolist()]
            scores = obj[b]
            topk = min(num_select, scores.numel())
            sv, si = scores.topk(topk)
            bx = box_cxcywh_to_xyxy(boxes[b][si])
            scale = torch.tensor([orig_w, orig_h, orig_w, orig_h], device=device)
            bx = bx * scale
            image_id = int(t['image_id'])
            for s, box in zip(sv.tolist(), bx.tolist()):
                x0, y0, x1, y1 = box
                results.append({'image_id': image_id, 'category_id': 1,
                                'bbox': [x0, y0, x1 - x0, y1 - y0], 'score': s})
    if not results:
        print("[eval] no detections"); return None
    dt = base_ds_gt.loadRes(results)
    ev = COCOeval(base_ds_gt, dt, 'bbox')
    ev.evaluate(); ev.accumulate(); ev.summarize()
    names = ['AP', 'AP@50', 'AP@75', 'AP_s', 'AP_m', 'AP_l',
             'AR@1', 'AR@10', 'AR@100', 'AR_s', 'AR_m', 'AR_l']
    return dict(zip(names, ev.stats.tolist()))
