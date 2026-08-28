import random

import numpy as np
import torch

from .verify import rejection_verify, greedy_verify, softmax


def _clone_past(past):
    """Clone a KV cache (DynamicCache or legacy tuple) WITHOUT assuming its
    internal structure, so it works on every transformers version.

    IMPORTANT: a transformers DynamicCache is mutated IN PLACE by the model during
    a forward pass, so the verification forward MUST run on a true independent copy.
    `DynamicCache` has no `copy()` in many versions (e.g. 4.57.x, which uses the
    `self.layers` backend), so we rebuild via the legacy round-trip, which yields a
    fresh cache whose appends allocate new tensors (the original is untouched)."""
    if past is None:
        return None
    from transformers import DynamicCache
    if isinstance(past, DynamicCache):
        try:
            return DynamicCache.from_legacy_cache(past.to_legacy_cache())
        except Exception:
            # Ultimate fallback: deep-clone the underlying layer tensors.
            cloned = DynamicCache()
            for layer in past.layers:
                cloned.layers.append(layer.__class__())
            for i, layer in enumerate(past.layers):
                k, v = layer.get_states()
                cloned.layers[i].update(k.detach().clone(), v.detach().clone())
            return cloned
    if hasattr(past, "copy"):
        return past.copy()

    def _clone(x):
        if isinstance(x, torch.Tensor):
            return x.detach().clone()
        if isinstance(x, (tuple, list)):
            return type(x)(_clone(e) for e in x)
        return x

    return _clone(past)


def _as_cache(past):
    """Normalize a model-returned past_key_values to a transformers DynamicCache.
    Falls back to the raw object (legacy tuple) for transformers that still accept it."""
    if past is None:
        return None
    from transformers import DynamicCache
    if isinstance(past, DynamicCache):
        return past
    try:
        return DynamicCache.from_legacy_cache(past)
    except Exception:
        return past


class SpecDecoder:
    """Speculative decoder with KV-cache reuse.

    The target model's KV cache is kept across steps: each target forward only
    processes the *new* tokens (via `past_key_values`), which is what makes
    speculative decoding faster than autoregressive decoding in practice.
    """

    def __init__(self, model=None, tokenizer=None, drafter=None, k=4,
                 temperature=1.0, seed=0, target_fn=None, device="cpu",
                 feature_layer=-1):
        self.model = model
        self.tokenizer = tokenizer
        self.drafter = drafter
        self.k = k
        self.temperature = temperature
        self.rng = random.Random(seed)
        self.target_fn = target_fn
        self.device = device
        self.feature_layer = getattr(drafter, "feature_layer", feature_layer) if drafter else feature_layer
        self.target_calls = 0
        self.tokens_processed = 0
        self.drafted_tokens = 0
        self.accepted_tokens = 0
        self.past = None
        self.last_hidden = None
        self.last_logits = None
        self.ctx = None

    @property
    def acceptance_rate(self):
        """Fraction of drafted tokens accepted (the real speed metric)."""
        return self.accepted_tokens / self.drafted_tokens if self.drafted_tokens else 0.0

    def _target_forward(self, new_ids, want_hidden=False):
        """Return (logits, hidden) for the next token after `self.ctx + new_ids`.

        Supports two target backends (the canonical narrow interface
        `token_ids -> logits` from Leviathan et al. 2023):
          - a PyTorch `nn.Module` (GPU/real-LLM path, reuses KV cache), or
          - a callable `target_fn(context_ids) -> logits` (CPU/toy path).
        """
        if self.model is not None:
            dev = next(self.model.parameters()).device
            self.model.eval()
            t = torch.tensor([list(new_ids)], device=dev)
            with torch.no_grad():
                out = self.model(t, past_key_values=self.past, use_cache=True,
                                 output_hidden_states=want_hidden)
            self.past = _as_cache(out.past_key_values)
            self.tokens_processed += len(new_ids)
            self.target_calls += 1
            logits = out.logits[0].float().cpu().numpy()
            h = (out.hidden_states[self.feature_layer][0, -1].float().cpu().numpy()
                 if want_hidden else None)
            return logits, h
        elif self.target_fn is not None:
            logits = np.asarray(self.target_fn(self.ctx + list(new_ids)),
                                dtype=np.float64)[None, :]
            self.tokens_processed += len(new_ids)
            self.target_calls += 1
            return logits, None
        else:
            raise RuntimeError("SpecDecoder requires either `model` or `target_fn`.")

    def _prime(self, context_ids):
        self.past = None
        self.ctx = list(context_ids)
        logits, h = self._target_forward(self.ctx, want_hidden=True)
        self.last_hidden = h
        self.last_logits = logits[-1]

    def _greedy_step(self):
        last = self.last_logits / max(self.temperature, 1e-6)
        if self.temperature == 0:
            nxt = int(np.argmax(last))
        else:
            nxt = int(self.rng.choices(range(len(last)), weights=softmax(last))[0])
        logits, h = self._target_forward([nxt], want_hidden=True)
        self.last_hidden = h
        self.last_logits = logits[-1]
        self.ctx.append(nxt)
        return [nxt]

    def _spec_step(self):
        ctx = self.ctx
        if self.drafter is None:
            return self._greedy_step()

        draft_tokens = (self.drafter.draft_from(self.last_hidden, ctx[-1], self.k)
                        if hasattr(self.drafter, "draft_from") else self.drafter.draft(ctx, self.k))
        if not draft_tokens:
            return self._greedy_step()

        # Verify all drafted tokens in ONE target forward. The forward yields the
        # K target distributions p_2 .. p_{K+1}; together with self.last_logits
        # (= p_1, from the prime / previous commit) we get the K+1 distributions
        # Leviathan et al. Algorithm 1 needs. p_{K+1} is the bonus-token dist.
        if self.model is not None:
            dev = next(self.model.parameters()).device
            t = torch.tensor([list(draft_tokens)], device=dev)
            with torch.no_grad():
                out = self.model(t, past_key_values=_clone_past(self.past),
                                 use_cache=True, output_hidden_states=True)
            self.target_calls += 1
            self.tokens_processed += len(draft_tokens)
            new_logits = out.logits[0].float().cpu().numpy()  # (K, V)
        else:
            per_pos = []
            for i in range(len(draft_tokens)):
                ctx_i = self.ctx + list(draft_tokens[:i + 1])
                per_pos.append(np.asarray(self.target_fn(ctx_i), dtype=np.float64))
            new_logits = np.stack(per_pos)  # (K, V): target dist after context + draft[:i+1]
            self.target_calls += 1
            self.tokens_processed += len(draft_tokens)

        rows = [self.last_logits] + [new_logits[i] for i in range(len(draft_tokens))]
        p = softmax(np.stack(rows) / max(self.temperature, 1e-6))

        if hasattr(self.drafter, "draft_probs_from"):
            q = self.drafter.draft_probs_from(self.last_hidden, ctx[-1],
                                              len(draft_tokens), p.shape[-1])
        else:
            q = self.drafter.draft_probs(ctx, len(draft_tokens), p.shape[-1])

        if self.temperature == 0:
            # Greedy target: accept a draft token iff it matches the target's
            # argmax. This makes the speculative output exactly equal the greedy
            # target; the bonus token (p_{K+1} argmax) is emitted when all K pass.
            accepted = greedy_verify(rows, draft_tokens)
            fallback = int(np.argmax(p[len(accepted)]))
            commit = accepted + [fallback]
        else:
            accepted, fallback = rejection_verify(p, q, draft_tokens, self.rng)
            if fallback is not None:
                commit = accepted + [fallback]
            else:
                # All K accepted: sample a bonus token from p_{K+1} so the target
                # distribution is preserved exactly (Leviathan et al. 2023).
                bonus = int(self.rng.choices(range(p.shape[-1]),
                                             weights=p[len(accepted)])[0])
                commit = accepted + [bonus]
        self.drafted_tokens += len(draft_tokens)
        self.accepted_tokens += len(accepted)
        if commit:
            logits, h = self._target_forward(commit, want_hidden=True)
            self.last_hidden = h
            self.last_logits = logits[-1]
            self.ctx.extend(commit)
        return commit

    def generate_ids(self, prompt_ids, max_new_tokens):
        self._prime(prompt_ids)
        out = []
        while len(out) < max_new_tokens:
            step = self._spec_step()
            if not step:
                break
            out.extend(step)
        return out[:max_new_tokens]

    def generate(self, prompt, max_new_tokens=64):
        enc = self.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        ids = self.generate_ids(enc, max_new_tokens)
        return self.tokenizer.decode(ids), {"target_calls": self.target_calls,
                                            "tokens": self.tokens_processed}
