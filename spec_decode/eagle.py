import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EagleLayer(nn.Module):
    """EAGLE draft head: a single transformer decoder layer (Pre-LN, causal self-attention, FFN).

    Mirrors the official EAGLE design: a lightweight layer (a fraction of the target's
    size) that, conditioned on the target's second-to-top-layer feature and the current
    token embedding, predicts the next feature. The target's frozen LM head then turns
    that feature into a token distribution.
    """

    def __init__(self, D, n_heads=8, ff_mult=4):
        super().__init__()
        self.D = D
        self.ln1 = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(D)
        self.ff = nn.Sequential(
            nn.Linear(D, ff_mult * D), nn.GELU(), nn.Linear(ff_mult * D, D)
        )
        self.fuse = nn.Linear(2 * D, D)

    def forward(self, feat_seq, tok_seq):
        # feat_seq, tok_seq: (B, L, D). feature + token embedding are fused, then a
        # causal transformer layer produces the predicted next features.
        x = self.fuse(torch.cat([feat_seq, tok_seq], dim=-1))
        h = self.ln1(x)
        L = x.size(1)
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        h2 = self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        x = x + h2
        x = x + self.ff(self.ln2(x))
        return x


class EagleDrafter:
    """Feature-level (EAGLE) drafter.

    Implements the same `drafter` interface used by the engine:
    `draft` / `draft_probs` (context-based) and `draft_from` / `draft_probs_from`
    (given the target's last hidden state, no extra target calls).

    Correct EAGLE alignment (no off-by-one):
      train_step:  (feature_i, token_i) -> predicted feature_{i+1};  lm_head -> token_{i+1}
      draft_from:  pred_feature_{m} = draft_layer(feature_{m-1}, emb(token_{m-1}));  d1 = argmax(lm_head(pred_feature_m))
    """

    def __init__(self, draft_layer, lm_head, embeddings, target_model,
                 device="cpu", feature_layer=-2):
        self.draft_layer = draft_layer.to(device)
        self.lm_head = lm_head
        self.embeddings = embeddings
        self.target = target_model
        self.device = device
        self.feature_layer = feature_layer
        self.draft_layer.eval()

    def _last_hidden(self, context_ids):
        ids = torch.tensor([list(context_ids)], device=self.device)
        with torch.no_grad():
            out = self.target(ids, output_hidden_states=True)
            hs = out.hidden_states[self.feature_layer]
            return hs[0, -1]

    # ---- context-based (engine falls back to *_from when available) ----
    def draft(self, context_ids, k):
        h = self._last_hidden(context_ids)
        return self.draft_from(h, int(context_ids[-1]), k)

    def draft_probs(self, context_ids, k, vocab_size):
        h = self._last_hidden(context_ids)
        return self.draft_probs_from(h, int(context_ids[-1]), k, vocab_size)

    # ---- feature-based (no target re-forward) ----
    def draft_from(self, h_vec, last_tok, k):
        with torch.no_grad():
            # Strictly PER-POSITION drafting to match single-step training:
            # each step feeds ONLY (current feature, current token) as a length-1
            # sequence and predicts the next feature. The next step's token is the
            # just-drafted token (never the original last_tok, never a growing prefix).
            f = torch.tensor(np.asarray(h_vec, dtype=np.float32), device=self.device
                             ).unsqueeze(0).unsqueeze(0)                       # (1,1,D) feature_{m-1}
            t = self.embeddings(torch.tensor([[int(last_tok)]], device=self.device))  # (1,1,D) token_{m-1}
            drafts = []
            feat_seq = []
            for _ in range(k):
                pf = self.draft_layer(f, t)[0, 0]                              # (D) predicted feature_{next}
                d = int(torch.argmax(self.lm_head(pf), -1).item())
                drafts.append(d)
                feat_seq.append(pf)
                f = pf.unsqueeze(0).unsqueeze(0)                              # next feature = predicted (length-1)
                t = self.embeddings(torch.tensor([[d]], device=self.device))  # next token = just-drafted (length-1)
        return drafts

    def draft_probs_from(self, h_vec, last_tok, k, vocab_size):
        with torch.no_grad():
            # Same strictly per-position scheme as draft_from (for sampled drafting).
            f = torch.tensor(np.asarray(h_vec, dtype=np.float32), device=self.device
                             ).unsqueeze(0).unsqueeze(0)
            t = self.embeddings(torch.tensor([[int(last_tok)]], device=self.device))
            probs = []
            for _ in range(k):
                pf = self.draft_layer(f, t)[0, 0]
                lp = torch.softmax(self.lm_head(pf), -1).float()
                probs.append(lp)
                d = int(torch.argmax(lp, -1).item())
                f = pf.unsqueeze(0).unsqueeze(0)
                t = self.embeddings(torch.tensor([[d]], device=self.device))
            return torch.stack(probs, 0).cpu().numpy()

    # ---- training step: AUTOREGRESSIVE STUDENT-FORCING (matches inference exactly) ----
    # Earlier versions teacher-forced the TRUE feature/token at every step, so the head
    # only learned to draft d1 (which uses the true target feature). At inference d2/d3/...
    # are fed the head's OWN predicted feature and its OWN drafted (argmax) token, so they
    # collapsed -> acceptance stuck at ~25% despite 92% training accuracy. Here we train
    # with the head's own predictions as the next input (student forcing), so the recurrence
    # at training == recurrence at inference, and d2/d3/d4 become reliable.
    def train_step(self, target_hidden, ids, opt):
        # target_hidden: (L, D) second-to-top features; ids: (L,) tokens
        f_prev = target_hidden[0].detach().unsqueeze(0)   # (1,D) seed feature_0 (true, as at inference seed)
        t_prev = ids[0].unsqueeze(0)                        # (1,)   seed token_0   (true)
        pred_feats, pred_logits = [], []
        for i in range(target_hidden.shape[0] - 1):
            pf = self.draft_layer(f_prev.unsqueeze(0),
                                  self.embeddings(t_prev).unsqueeze(0))[0, 0]  # (D)
            lg = self.lm_head(pf)                           # (V,)
            pred_feats.append(pf)
            pred_logits.append(lg)
            f_prev = pf.detach()                            # NEXT input = PREDICTED feature (as at inference)
            t_prev = lg.argmax().unsqueeze(0)               # NEXT input = DRAFTED token    (as at inference)
        pred_feats = torch.stack(pred_feats, 0)             # (L-1, D)
        pred_logits = torch.stack(pred_logits, 0)           # (L-1, V)
        loss_feat = F.smooth_l1_loss(pred_feats, target_hidden[1:])
        loss_ce = F.cross_entropy(pred_logits, ids[1:])
        loss = loss_ce + 0.5 * loss_feat
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.draft_layer.parameters(), 1.0)
        opt.step()
        return float(loss_ce.item()), float(loss_feat.item())
