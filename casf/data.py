"""DINO-CASF dataset adapter: COCO + QuickDraw sketch queries, binary objective, OW/CW holdout (Set B),
seed-14 single-query eval. Data discipline ported from clip_ddetr_clean_run/datasets/coco.py; seed-14
eval protocol from clip_ddetr_analysis. Photo transforms + mask converter reused from vendored DINO.

A sample = (image, target, sketch). target is DINO-format (boxes cxcywh-normalized, labels all == 1).
"""
import os, sys, json, random, time
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                      # clip_dino_setb/
sys.path.insert(0, os.path.join(_ROOT, 'DINO_upstream'))
import datasets.transforms as T                     # DINO photo transforms
from datasets.coco import ConvertCocoPolysToMask, make_coco_transforms
# Seeded/nested/leak-free subset selection + versioned cache (torch-free, verbatim port).
from casf.subset_select import finalize_train_ids, cache_is_valid, write_train_cache

# 56 COCO∩QuickDraw categories, canonical order (clean_run CocoDetectionQD.ALL_CATEGORIES).
ALL_CATEGORIES = ['bicycle', 'car', 'airplane', 'bus', 'train', 'truck', 'traffic light', 'fire hydrant',
    'stop sign', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'suitcase', 'baseball bat', 'skateboard', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'banana', 'apple', 'sandwich', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'bed', 'toilet', 'laptop', 'mouse', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'book', 'clock', 'vase', 'scissors', 'toothbrush']
# Holdout Set B = every 4th (i%4==0) — verified == handover list.
SET_B = [ALL_CATEGORIES[i] for i in range(len(ALL_CATEGORIES)) if i % 4 == 0]

COCO_ROOT = '/mnt/1tb/data/coco'
SKETCH_ROOT = os.environ.get('SKETCH_HOME', '/mnt/1tb/data')
TEXT_EMB = '/home/rahul/arka/clip_ddetr_ow_repr/checkpoints/clip_model/text_embeddings.pkl'
ANN_OUT = os.path.join(_ROOT, 'annotations_casf')


def rasterize_stroke3(strokes, size=224, line_width=2, padding=10):
    """stroke-3 (dx,dy,pen) -> white-on-black PIL (CLIP convention). Verbatim from clean_run."""
    abs_coords = np.cumsum(strokes[:, :2], axis=0).astype(float)
    pen_states = strokes[:, 2]
    x, y = abs_coords[:, 0], abs_coords[:, 1]
    scale = (size - 2 * padding) / max(x.max() - x.min() or 1, y.max() - y.min() or 1)
    x = ((x - x.min()) * scale + padding).astype(int)
    y = ((y - y.min()) * scale + padding).astype(int)
    img = Image.new('RGB', (size, size), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    pts = []
    for i in range(len(strokes)):
        pts.append((int(x[i]), int(y[i])))
        if pen_states[i] == 1:
            if len(pts) >= 2:
                draw.line(pts, fill=(255, 255, 255), width=line_width)
            pts = []
    if len(pts) >= 2:
        draw.line(pts, fill=(255, 255, 255), width=line_width)
    return img


def build_filtered_json(ann_file, image_set, scheme, id2class, data_frac=1.0, subset_seed=14):
    """Port of clean_run subset construction. scheme in {'open','closed'}. Returns path to filtered json.
    Train(open): keep only seen-cat images, drop any image containing a Set-B object (leakage guard).
    Val(open): keep only Set-B cats. Val(closed): keep all 56.

    The training subset is built by the seeded/nested/leak-free finalize_train_ids
    (casf/subset_select.py): SEEN is subset without replacement and nested 0.25 ⊂ 0.50 ⊂ 1.00,
    and the FULL held-out (Set B) set is excluded at every fraction (no leakage). The output
    json is cached under a key that encodes image_set·scheme·frac·seed, plus a `.meta` sidecar
    stamped with SUBSET_LOGIC_VERSION; a cache is reused only when that key AND the logic
    version match (cache_is_valid), so a stale/differently-computed json can never be served.
    """
    os.makedirs(ANN_OUT, exist_ok=True)
    # Versioned cache key: seed in the filename, logic-version in the .meta sidecar.
    sketch_name = f'casf_{image_set}'
    out = os.path.join(ANN_OUT, f'casf_{image_set}_{scheme}_frac{data_frac:.2f}_seed{subset_seed}.json')
    if cache_is_valid(out, sketch_name, scheme, data_frac, subset_seed):
        return out
    with open(ann_file) as f:
        j = json.load(f)
    unseen = set(SET_B) if scheme == 'open' else set()
    annotate, seen_ids, unseen_ids = [], {}, {}
    for a in j['annotations']:
        name = id2class.get(a['category_id'])
        if name not in ALL_CATEGORIES:
            continue
        if image_set == 'train' or scheme == 'closed':
            annotate.append(a)
            (unseen_ids if name in unseen else seen_ids).setdefault(a['category_id'], []).append(a['image_id'])
        else:  # open val: only held-out cats
            if name in unseen:
                annotate.append(a)
                seen_ids.setdefault(a['category_id'], []).append(a['image_id'])
    if image_set == 'train':
        # Subset SEEN only (seeded, nested, replace=False); exclude the FULL unseen set at
        # every fraction. data_frac == 1.0 reduces to "all seen − full unseen" (no leak).
        final_image_ids, per_class = finalize_train_ids(seen_ids, unseen_ids, data_frac, subset_seed)
    else:  # val: eval pool is never subsampled
        seen = set().union(*seen_ids.values()) if seen_ids else set()
        un = set().union(*unseen_ids.values()) if unseen_ids else set()
        final_image_ids = seen - un             # leakage guard: drop images with any Set-B object
    j['annotations'] = annotate
    j['images'] = [im for im in j['images'] if im['id'] in final_image_ids]
    write_train_cache(out, j, sketch_name, scheme, data_frac, subset_seed)
    print(f"[casf] {image_set}/{scheme} frac={data_frac:.2f} seed={subset_seed}: "
          f"{len(j['images'])} imgs, {len(annotate)} anns -> {out}")
    return out


class CocoSketchDINO(torchvision.datasets.CocoDetection):
    def __init__(self, image_set, scheme, data_frac=1.0, n_sketches=1, return_masks=False, subset_seed=14):
        assert image_set in ('train', 'val') and scheme in ('open', 'closed')
        self.image_set, self.scheme, self.n_sketches = image_set, scheme, n_sketches
        self.subset_seed = subset_seed
        split_dir = 'train2017' if image_set == 'train' else 'val2017'
        img_folder = os.path.join(COCO_ROOT, split_dir)
        ann_file = os.path.join(COCO_ROOT, 'annotations', f'instances_{split_dir}.json')
        with open(ann_file) as f:
            cats = json.load(f)['categories']
        self.id2class = {c['id']: c['name'] for c in cats}
        self.class2id = {c['name']: c['id'] for c in cats}
        temp = build_filtered_json(ann_file, image_set, scheme, self.id2class, data_frac, subset_seed)
        super().__init__(img_folder, temp)
        self.ids = sorted(self.ids)
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self._transforms = make_coco_transforms(image_set if image_set == 'train' else 'val', args=None)
        import pickle
        with open(TEXT_EMB, 'rb') as f:
            self.text_embeddings = pickle.load(f)
        # sketch (CLIP-normalized)
        mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        self.tf_sketch = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
        self._setup_sketches()

    def _setup_sketches(self):
        qd = os.path.join(SKETCH_ROOT, 'quickdraw', 'sketchrnn')
        split = 'train' if self.image_set == 'train' else 'valid'
        self._qd = {}
        self.class2quick = {}
        for cat in ALL_CATEGORIES:
            pp, sp = os.path.join(qd, f'{cat}.{split}.ptr.npy'), os.path.join(qd, f'{cat}.{split}.strokes.npy')
            if not (os.path.exists(pp) and os.path.exists(sp)):
                print(f"[casf] WARN missing sketchrnn {cat!r} {split}")
                continue
            ptr = np.load(pp)
            self._qd[cat] = (ptr, np.load(sp, mmap_mode='r'))
            self.class2quick[cat] = [f'{cat}:{i}' for i in range(len(ptr) - 1)]

    def _render(self, ref):
        cat, i = ref.rsplit(':', 1)
        ptr, strokes = self._qd[cat]
        i = int(i)
        return self.tf_sketch(rasterize_stroke3(strokes[ptr[i]:ptr[i + 1]]))

    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        image_id = self.ids[idx]
        # Candidate categories from RAW anns (same set the seed-14 GT builder uses) so the dataset's
        # selected category and the pycocotools GT stay consistent (Gate-4). iscrowd is handled later
        # in scoring, not in selection — matches clip_ddetr_analysis _build_seed14_gt.
        present = sorted({a['category_id'] for a in anns})
        # seed-14 single-query eval (analysis protocol): fresh Random(14) for cat AND for sketch.
        if self.image_set != 'train':
            cat_rng = random.Random(14)
            sk_rng = random.Random(14)
        else:
            s = int(1000 * time.time()) ^ (idx * 2654435761 & 0xffffffff)
            cat_rng = random.Random(s); sk_rng = random.Random(s + 1)
        selected = cat_rng.choice(present)
        target = {'image_id': image_id, 'annotations': anns}
        img, target = self.prepare(img, target)
        keep = target['labels'] == selected
        for k in ('boxes', 'labels', 'area', 'iscrowd', 'masks'):
            if k in target:
                target[k] = target[k][keep]
        target['labels'] = torch.ones_like(target['labels'])        # binary: object == 1
        cat_name = self.id2class[selected]
        ref = sk_rng.choices(self.class2quick[cat_name], k=self.n_sketches)[0]
        sketch = self._render(ref)                                   # [3,224,224]
        old_boxes = target['boxes'].clone()
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        if self.image_set != 'train':
            target['boxes'] = old_boxes                             # eval: keep abs boxes (set by GT path)
        target['sketch_cat'] = selected                            # COCO id, for GT bookkeeping
        return img, target, sketch


def collate_fn_casf(batch):
    """Returns (NestedTensor images, list[target], sketch tensor [B,3,224,224])."""
    from util.misc import nested_tensor_from_tensor_list
    imgs, targets, sketches = zip(*batch)
    samples = nested_tensor_from_tensor_list(list(imgs))
    sketches = torch.stack(list(sketches), 0)
    return samples, list(targets), sketches


def build_seed14_binary_gt(dataset):
    """COCO GT object for pycocotools: per val image keep only the seed-14 selected cat's anns,
    relabeled to category_id=1 (binary). Mirrors clip_ddetr_analysis _build_seed14_gt."""
    from pycocotools.coco import COCO
    coco = dataset.coco
    out = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'object'}]}
    ann_id = 1
    for image_id in dataset.ids:
        anns = coco.imgToAnns.get(image_id, [])
        present = sorted({a['category_id'] for a in anns})
        if not present:
            continue
        selected = random.Random(14).choice(present)
        out['images'].append(coco.loadImgs(image_id)[0])
        for a in anns:
            if a['category_id'] == selected and a.get('iscrowd', 0) == 0 and a.get('area', 0) > 0:
                b = dict(a); b['category_id'] = 1; b['id'] = ann_id; ann_id += 1
                out['annotations'].append(b)
    gt = COCO()
    gt.dataset = out
    gt.createIndex()
    return gt
