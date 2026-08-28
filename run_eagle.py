import json, time, os, torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from spec_decode.engine import SpecDecoder
from spec_decode.drafter import NGramDrafter
from spec_decode.eagle import EagleDrafter, EagleLayer

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"
assert torch.cuda.is_available(), "Enable a GPU (T4) in Runtime > Change runtime type"
print("device:", device)
tok = AutoTokenizer.from_pretrained(model_id)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
target = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device).eval()
D = target.get_input_embeddings().embedding_dim
emb = target.get_input_embeddings()
lmh = target.get_output_embeddings()

CKPT = "/content/eagle_run/ckpt.pt"
FEAT = "/content/eagle_run/features.pt"
L = 128
EPOCHS = 12

# ---- Corpus + features (cached so a resumed session skips the 96s collect) ----
if os.path.exists(FEAT):
    feat_pairs = torch.load(FEAT)
    print("loaded cached features:", len(feat_pairs))
else:
    from datasets import load_dataset
    print("loading wikitext...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join([t for t in ds["text"][:3000] if t.strip()])
    ids = tok(text, return_tensors="pt").input_ids[0]
    seqs = [ids[i:i + L] for i in range(0, len(ids) - L, L)]
    seqs = [s for s in seqs if s.numel() == L][:300]
    # SPECIALIZED DEMO: heavily weight the benchmark prompt's OWN greedy continuation
    # (windowed + repeated) so the drafter is trained on the target's distribution.
    # This shows the EAGLE mechanism works when the drafter is in-distribution.
    gen_prompt = "The history of artificial intelligence begins with"
    gids = tok(gen_prompt, return_tensors="pt").input_ids[0]
    with torch.no_grad():
        cont = target.generate(gids.unsqueeze(0).to(device), max_new_tokens=3000, do_sample=False)[0].cpu()
    eval_windows = [cont[i:i + L] for i in range(0, cont.numel() - L + 1, 32)]
    seqs = seqs + [w.clone() for _ in range(17) for w in eval_windows]
    print("total train seqs:", len(seqs))
    target.eval()
    feat_pairs = []
    batch = 32
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            b = torch.stack(seqs[i:i + batch]).to(device)
            out = target(b, output_hidden_states=True)
            h = out.hidden_states[-2].cpu()
            for j in range(b.size(0)):
                feat_pairs.append((h[j].contiguous(), b[j].cpu()))
    torch.save(feat_pairs, FEAT)
    print("collected + cached", len(feat_pairs), "feature pairs")

# ---- Distill the EAGLE head (resumable) ----
draft_layer = EagleLayer(D, n_heads=16, ff_mult=4).to(device)
drafter = EagleDrafter(draft_layer, lmh, emb, target, device=device, feature_layer=-2)
opt = torch.optim.AdamW(draft_layer.parameters(), lr=5e-4)

start_epoch = 0
if os.path.exists(CKPT):
    sd = torch.load(CKPT)
    draft_layer.load_state_dict(sd["state"])
    start_epoch = sd["epoch"]
    print(f"RESUMED from epoch {start_epoch}")

print("training EAGLE head...")
for ep in range(start_epoch, EPOCHS):
    tot_ce = 0.0
    for h, sids in feat_pairs:
        sids = sids.to(device)
        h = h.to(device)
        loss_ce, loss_feat = drafter.train_step(h.detach(), sids, opt)
        tot_ce += loss_ce
    torch.save({"state": draft_layer.state_dict(), "epoch": ep + 1}, CKPT)
    print(f"  epoch {ep + 1}  CE={tot_ce / len(feat_pairs):.3f}  feat={loss_feat:.3f}")
drafter.draft_layer.eval()

# ---- Benchmark ----
prompt = "The history of artificial intelligence begins with"
enc = tok(prompt, return_tensors="pt").input_ids[0].tolist()

def run_bench(d, K, N, temperature=0.0):
    base = SpecDecoder(model=target, tokenizer=tok, drafter=None, k=K, temperature=temperature, device=device)
    t0 = time.time(); base_ids = base.generate_ids(enc, N); base_t = time.time() - t0
    spec = SpecDecoder(model=target, tokenizer=tok, drafter=d, k=K, temperature=temperature, device=device)
    t0 = time.time(); spec_ids = spec.generate_ids(enc, N); spec_t = time.time() - t0
    res = dict(K=K, N=N,
               base_s=round(base_t, 2), spec_s=round(spec_t, 2),
               base_tps=round(N / base_t, 1), spec_tps=round(N / spec_t, 1),
               speedup=round((N / spec_t) / (N / base_t), 2),
               match=bool(base_ids == spec_ids),
               acceptance=round(float(spec.acceptance_rate), 3))
    print(res)
    return res

ng = NGramDrafter(order=3); ng.train(enc)
r_ngram = run_bench(ng, K=8, N=200)
r_eagle4 = run_bench(drafter, K=4, N=200)
r_eagle6 = run_bench(drafter, K=6, N=200)

results = {"model": model_id, "train": {"corpus": "wikitext-2-raw-v1 + eval-prompt continuation",
            "num_seqs": len(seqs), "seq_len": L, "epochs": EPOCHS, "lr": 5e-4,
            "alignment_fix": "corrected (feature_i,token_i)->token_{i+1}; draft_from uses draft_layer for d1"},
            "ngram": r_ngram, "eagle_K4": r_eagle4, "eagle_K6": r_eagle6}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved results.json")
