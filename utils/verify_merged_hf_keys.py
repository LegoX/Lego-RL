"""Verify the merged HF checkpoint against the base model's key set.

Catches the fused/per-expert mismatch deterministically, on CPU, before any GPU
time is spent. Expected outcome: no unexpected keys, and the only missing keys
are the untrained mtp.* speculative head that veomni does not save.
"""
import json, os, sys, glob

if len(sys.argv) < 3:
    sys.exit("usage: verify_merged_hf_keys.py <merged_hf_dir> <base_hf_dir>")
MERGED, BASE = sys.argv[1], sys.argv[2]


def keys_and_shapes(d):
    """Read key -> shape from a safetensors dir without loading tensor data."""
    from safetensors import safe_open
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, framework="pt") as fh:
            for k in fh.keys():
                out[k] = tuple(fh.get_slice(k).get_shape())
    return out


b = keys_and_shapes(BASE)
m = keys_and_shapes(MERGED)
print(f"base   keys: {len(b)}")
print(f"merged keys: {len(m)}")

missing = sorted(set(b) - set(m))
unexpected = sorted(set(m) - set(b))
mismatch = sorted(k for k in (set(b) & set(m)) if b[k] != m[k])

def bucket(ks):
    out = {}
    for k in ks:
        tag = ("mtp" if ".mtp" in k or k.startswith("mtp") else
               "experts" if ".experts." in k else
               "visual" if ".visual." in k else "other")
        out.setdefault(tag, []).append(k)
    return out

print(f"\nMISSING    {len(missing)}")
for t, ks in bucket(missing).items():
    print(f"  [{t}] {len(ks)}  e.g. {ks[:2]}")
print(f"UNEXPECTED {len(unexpected)}")
for t, ks in bucket(unexpected).items():
    print(f"  [{t}] {len(ks)}  e.g. {ks[:2]}")
print(f"SHAPE MISMATCH {len(mismatch)}")
for k in mismatch[:8]:
    print(f"  {k}: base {b[k]} vs merged {m[k]}")

exp = [k for k in b if ".experts." in k]
gotexp = [k for k in m if ".experts." in k]
print(f"\nexpert keys: base {len(exp)}  merged {len(gotexp)}")
fused_base = sum(1 for k in b if k.endswith(("experts.gate_up_proj", "experts.down_proj")))
fused_m = sum(1 for k in m if k.endswith(("experts.gate_up_proj", "experts.down_proj")))
perexp_m = sum(1 for k in m if ".experts." in k and k.endswith(
    ("gate_proj.weight", "up_proj.weight", "down_proj.weight")))
print(f"fused-layout keys: base {fused_base}  merged {fused_m}")
print(f"per-expert-layout keys in merged: {perexp_m}  "
      f"(Qwen3.5: expected 768, the mtp.* head, which the base also stores per-expert)")

ok = (not unexpected) and (not mismatch) and all(
    ".mtp" in k or k.startswith("mtp") for k in missing)
print("\n" + ("PASS - safe to serve" if ok else
              "FAIL - do not serve; investigate above"))
sys.exit(0 if ok else 1)
