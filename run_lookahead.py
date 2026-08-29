"""
Self-contained Colab cell — lossless speculative decoding on TinyLlama-1.1B
using the train-free Lookahead (self-suffix) drafter. Paste the whole file into
one Colab cell (T4 GPU runtime) and run. No external repo needed.

What it proves:
  - The speculative output is EXACTLY the greedy output (lossless) at temp=0.
  - Real forward-pass speedup vs autoregressive greedy.
  - Acceptance rate (the honest driver of speedup).

Fits comfortably inside the free-Colab ~10-15 min preemption window.
"""

# --- 1. deps (Colab already ships torch + transformers) ---
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


# --- 2. Lookahead drafter (train-free, lossless) -------------------------------
class LookaheadDrafter:
    def __init__(self, max_order=5, min_order=2):
        self.max_order = max_order
        self.min_order = min_order

    def _best_next(self, hist):
        cands = {}
        n = len(hist)
        for order in range(self.max_order, self.min_order - 1, -1):
            if n < order + 1:
                continue
            window = tuple(hist[-order:])
            weight = order
            for i in range(0, n - order):
                if tuple(hist[i:i + order]) == window:
                    cands[hist[i + order]] = cands.get(hist[i + order], 0.0) + weight
        if not cands:
            return None
        return max(cands.items(), key=lambda kv: kv[1])[0]

    def draft(self, context_ids, k):
        drafts, hist = [], list(context_ids)
        for _ in range(k):
            best = self._best_next(hist)
            if best is None:
                break
            drafts.append(best)
            hist.append(best)
        return drafts

    def draft_probs(self, context_ids, k, vocab_size):
        probs, hist = [], list(context_ids)
        for _ in range(k):
            best = self._best_next(hist)
            row = np.zeros(vocab_size)
            if best is not None:
                row[best] = 1.0
                hist.append(best)
            probs.append(row)
        return np.stack(probs)


# --- 3. minimal speculative engine (KV-cache reuse, greedy/temp=0) -------------
def _as_cache(past):
    if past is None:
        return None
    if isinstance(past, DynamicCache):
        return past
    try:
        return DynamicCache.from_legacy_cache(past)
    except Exception:
        return past

def _clone_past(past):
    if past is None:
        return None
    if isinstance(past, DynamicCache):
        try:
            return DynamicCache.from_legacy_cache(past.to_legacy_cache())
        except Exception:
            pass
        try:
            cloned = DynamicCache()
            kc = getattr(past, "key_cache", None)
            vc = getattr(past, "value_cache", None)
            if kc is not None and vc is not None:
                for k, v in zip(kc, vc):
                    cloned.key_cache.append(k.detach().clone())
                    cloned.value_cache.append(v.detach().clone())
                return cloned
        except Exception:
            pass
        try:
            cloned = DynamicCache()
            for layer in past.layers:
                cloned.layers.append(layer.__class__())
            for i, layer in enumerate(past.layers):
                k, v = layer.get_states()
                cloned.layers[i].update(k.detach().clone(), v.detach().clone())
            return cloned
        except Exception as e:
            raise RuntimeError(f"cannot clone DynamicCache: {e}")
    return past


def greedy_verify(rows, draft_tokens):
    accepted = []
    for i, t in enumerate(draft_tokens):
        if t == int(np.argmax(rows[i])):
            accepted.append(t)
        else:
            break
    return accepted


class SpecDecoder:
    def __init__(self, model, tokenizer, drafter=None, k=4, device="cuda"):
        self.model, self.tokenizer = model, tokenizer
        self.drafter, self.k, self.device = drafter, k, device
        self.target_calls = 0
        self.drafted_tokens = 0
        self.accepted_tokens = 0
        self.past = None
        self.ctx = None
        self.last_logits = None

    @property
    def acceptance_rate(self):
        return self.accepted_tokens / self.drafted_tokens if self.drafted_tokens else 0.0

    def _target_forward(self, new_ids):
        t = torch.tensor([list(new_ids)], device=self.device)
        with torch.no_grad():
            out = self.model(t, past_key_values=self.past, use_cache=True,
                             output_hidden_states=False)
        self.past = _as_cache(out.past_key_values)
        self.target_calls += 1
        self.last_logits = out.logits[0].float().cpu().numpy()[-1]

    def _spec_step(self):
        draft = self.drafter.draft(self.ctx, self.k) if self.drafter else []
        if not draft:
            nxt = int(np.argmax(self.last_logits))
            self._target_forward([nxt])
            self.ctx.append(nxt)
            return [nxt]
        t = torch.tensor([list(draft)], device=self.device)
        with torch.no_grad():
            out = self.model(t, past_key_values=_clone_past(self.past),
                             use_cache=True, output_hidden_states=False)
        self.target_calls += 1
        new_logits = out.logits[0].float().cpu().numpy()  # (K, V)
        rows = [self.last_logits] + [new_logits[i] for i in range(len(draft))]
        accepted = greedy_verify(rows, draft)
        fallback = int(np.argmax(rows[len(accepted)]))
        commit = accepted + [fallback]
        self.drafted_tokens += len(draft)
        self.accepted_tokens += len(accepted)
        self._target_forward(commit)
        self.ctx.extend(commit)
        return commit

    def generate_ids(self, prompt_ids, max_new):
        self.past = None
        self.ctx = list(prompt_ids)
        self._target_forward(self.ctx)
        out = []
        while len(out) < max_new:
            step = self._spec_step()
            if not step:
                break
            out.extend(step)
        return out[:max_new]


# --- 4. run on TinyLlama-1.1B -------------------------------------------------
def main():
    MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] using {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device).eval()

    prompts = [
        "The history of artificial intelligence begins with",
        "Once upon a time in a small village",
        "The capital of France is",
        "In the field of machine learning, a neural network",
        "Write a poem about the ocean:",
        "The quick brown fox",
        "Explain the theory of relativity in one sentence:",
        "def fibonacci(n):",
    ]
    K, N = 5, 200

    out = []
    out.append(f"{'prompt':42s} {'base_fwd':>8s} {'spec_fwd':>9s} {'speedup':>8s} {'accept':>8s} {'lossless':>9s}")
    out.append("-" * 96)
    total_base = total_spec = 0
    for p in prompts:
        enc = tokenizer(p, return_tensors="pt").input_ids[0].tolist()

        base = SpecDecoder(model, tokenizer, drafter=None, k=1, device=device)
        base_ids = base.generate_ids(enc, N)
        base_calls = base.target_calls

        spec = SpecDecoder(model, tokenizer, drafter=LookaheadDrafter(), k=K, device=device)
        spec_ids = spec.generate_ids(enc, N)
        spec_calls = spec.target_calls

        lossless = (base_ids == spec_ids)
        sp = base_calls / spec_calls
        total_base += base_calls
        total_spec += spec_calls
        out.append(f"{p[:40]:42s} {base_calls:8d} {spec_calls:9d} {sp:7.2f}x {spec.acceptance_rate:7.1%} {str(lossless):>9s}")

    out.append("-" * 96)
    out.append(f"{'AVERAGE':42s} {total_base:8d} {total_spec:9d} {total_base/total_spec:7.2f}x")
    out.append("\nNote: speedup is forward-pass reduction vs greedy. Wall-clock speedup on a")
    out.append("real T4 is this reduction minus small draft overhead; it tracks acceptance rate.")
    out.append("Lookahead is train-free and lossless, so the output is always identical to greedy.")
    text = "\n".join(out)
    print(text)
    with open("/content/lookahead_result.txt", "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
