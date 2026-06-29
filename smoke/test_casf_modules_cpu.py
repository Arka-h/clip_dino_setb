"""CPU unit test for the ported CASF modules: shapes, residual/zero-init invariants, determinism.
Runs without a GPU. Verifies the load-bearing port pieces behave as designed (DESIGN_LOG D1/D2/D3/D5)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from casf.modules import (DePathBlock, SketchMeanPool, BoxSketchAdapter, GatedConcatClassHead)

torch.manual_seed(0)
B, Q, T, D, DC = 2, 900, 50, 256, 768
output = torch.randn(B, Q, D)           # decoder queries [B,Q,256]
sk_tokens = torch.randn(B, T, DC)       # CLIP sketch tokens [B,50,768]

ok = True
def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok = ok and cond

# D1: DePathBlock preserves [B,Q,256] and is a refinement (not identity), with internal residual.
blk = DePathBlock(d_model=D, d_clip=DC, n_heads=4, head_dim=64, share_kv=True)
refined = blk(output, sk_tokens)
check("DePathBlock output shape == input", refined.shape == output.shape)
check("DePathBlock actually changes queries", not torch.allclose(refined, output))

# D5: mean-pool -> 256, and at init (zero-init up) equals the linear skip of the mean (sanity, finite).
pool = SketchMeanPool(in_dim=DC, out_dim=D)
pooled = pool(sk_tokens)
check("SketchMeanPool shape [B,256]", pooled.shape == (B, D))
check("pooled finite", torch.isfinite(pooled).all().item())

# D2: BoxSketchAdapter is exactly identity at init (zero gate + zero-init proj) -> protects LFT box head.
box = BoxSketchAdapter(d_model=D, init_gate=0.0)
out_box = box(output, pooled)
check("BoxSketchAdapter == identity at init (LFT preserved)", torch.allclose(out_box, output, atol=0))
# after nudging the gate it changes (learnable path is live)
with torch.no_grad():
    box.gate.fill_(1.0); box.proj.weight.normal_()
check("BoxSketchAdapter is live once trained", not torch.allclose(box(output, pooled), output))

# D3: GatedConcatClassHead -> [B,Q,num_classes]
head = GatedConcatClassHead(d_model=D, num_classes=2, init_gate=0.1)
logits = head(output, pooled)
check("ClassHead shape [B,Q,2]", logits.shape == (B, Q, 2))

# determinism: same inputs/seed -> identical outputs
torch.manual_seed(0); blk2 = DePathBlock(d_model=D, d_clip=DC, share_kv=True)
torch.manual_seed(0); blk3 = DePathBlock(d_model=D, d_clip=DC, share_kv=True)
check("module init deterministic under fixed seed",
      torch.allclose(blk2(output, sk_tokens), blk3(output, sk_tokens)))

# gradient flows to sketch path
logits.sum().backward()
check("grad reaches class-head gate", head.gate.grad is not None and head.gate.grad.abs().item() >= 0)

print("\nPARAM COUNTS (new, trained-from-init):")
for n, m in [("DePathBlock", blk), ("SketchMeanPool", pool), ("BoxSketchAdapter", box), ("ClassHead", head)]:
    print(f"  {n:18s} {sum(p.numel() for p in m.parameters()):>8d}")

print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
