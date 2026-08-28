import numpy as np


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def greedy_verify(target_logits, draft_tokens):
    target_argmax = np.atleast_1d(np.argmax(target_logits, axis=-1))
    accepted = []
    for i, d in enumerate(draft_tokens):
        if d == int(target_argmax[i]):
            accepted.append(d)
        else:
            break
    return accepted


def rejection_verify(target_probs, draft_probs, draft_tokens, rng):
    accepted = []
    fallback = None
    for i, d in enumerate(draft_tokens):
        p = float(target_probs[i][d])
        q = float(draft_probs[i][d])
        if q <= 0:
            q = 1e-8
        if p >= q:
            accepted.append(d)
        else:
            r = rng.random()
            if r <= p / q:
                accepted.append(d)
            else:
                residual = np.clip(target_probs[i] - draft_probs[i], 0.0, None)
                s = residual.sum()
                if s <= 0:
                    fallback = int(np.argmax(target_probs[i]))
                else:
                    r = rng.random() * s
                    cum = 0.0
                    for j, val in enumerate(residual):
                        cum += val
                        if r < cum:
                            fallback = int(j)
                            break
                    else:
                        fallback = int(len(residual) - 1)
                break
    return accepted, fallback


def spec_generate_step(target_logits_fn, draft_tokens, draft_probs_fn, rng):
    target_logits = target_logits_fn(draft_tokens)
    target_probs = softmax(target_logits)
    draft_probs = draft_probs_fn(draft_tokens)
    accepted, fallback = rejection_verify(target_probs, draft_probs, draft_tokens, rng)
    if fallback is not None:
        return accepted + [fallback], len(accepted)
    target_argmax = int(np.argmax(target_logits[-1]))
    return accepted + [target_argmax], len(accepted)
