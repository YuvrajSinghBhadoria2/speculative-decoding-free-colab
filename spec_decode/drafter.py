from collections import defaultdict

import numpy as np


class NGramDrafter:
    def __init__(self, order=3):
        self.order = order
        self.table = defaultdict(lambda: defaultdict(int))

    def train(self, token_ids):
        for i in range(len(token_ids) - self.order):
            key = tuple(token_ids[i : i + self.order])
            nxt = token_ids[i + self.order]
            self.table[key][nxt] += 1

    def draft(self, context_ids, k):
        drafts = []
        for _ in range(k):
            key = tuple(context_ids[-self.order :])
            if key not in self.table:
                break
            best = max(self.table[key].items(), key=lambda kv: kv[1])[0]
            drafts.append(best)
            context_ids = context_ids + [best]
        return drafts

    def draft_probs(self, context_ids, k, vocab_size):
        probs = []
        for _ in range(k):
            key = tuple(context_ids[-self.order :])
            row = np.zeros(vocab_size)
            if key in self.table:
                total = sum(self.table[key].values())
                for tok, c in self.table[key].items():
                    row[tok] = c / total
            else:
                # Unseen context: uniform proposal so rejection sampling has a
                # well-defined q (all-zero q would force-accept and bias the
                # recovered target distribution).
                row[:] = 1.0 / vocab_size
            probs.append(row)
            best = int(np.argmax(row)) if row.sum() > 0 else 0
            context_ids = context_ids + [best]
        return np.stack(probs)
