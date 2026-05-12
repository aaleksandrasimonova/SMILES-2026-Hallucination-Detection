from __future__ import annotations

import torch
import torch.nn.functional as F


# SELECTED_LAYERS = [8, 12, 16, 20, 24]
SELECTED_LAYERS = [12, 14, 16, 18, 20]

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.bool()
    n_real = int(mask.sum().item())

    layers = [hidden_states[i][mask].float() for i in SELECTED_LAYERS]

    # Approximate response zone: final 40% of real tokens.
    n_tail = max(8, int(n_real * 0.40))

    pooled_features = []
    geometric_features = []
    late_means_by_layer = []

    for layer in layers:
        tail = layer[-n_tail:]

        chunk_size = max(1, n_tail // 3)
        early = tail[:chunk_size]
        middle = tail[chunk_size: 2 * chunk_size]
        late = tail[2 * chunk_size:]

        early_mean = early.mean(dim=0)
        middle_mean = middle.mean(dim=0)
        late_mean = late.mean(dim=0)
        late_means_by_layer.append(late_mean)

        pooled_features.extend([
            middle_mean,
            late_mean,
        ])

        # geometric_features.extend([
        #     F.cosine_similarity(early_mean.unsqueeze(0), late_mean.unsqueeze(0)).squeeze(),
        #     torch.norm(late_mean - early_mean, p=2),
        #     torch.norm(late_mean, p=2),
        #     tail.var(dim=0).mean(),
        # ])


        context = layer[:-n_tail]
        if context.size(0) == 0:
            context = layer[:1]

        context_mean = context.mean(dim=0)
        response_mean = tail.mean(dim=0)

        geometric_features.extend([
            F.cosine_similarity(early_mean.unsqueeze(0), late_mean.unsqueeze(0)).squeeze(),
            torch.norm(late_mean - early_mean, p=2),

            F.cosine_similarity(context_mean.unsqueeze(0), response_mean.unsqueeze(0)).squeeze(),
            torch.norm(response_mean - context_mean, p=2),
            torch.norm(response_mean, p=2) / (torch.norm(context_mean, p=2) + 1e-8),

            tail.var(dim=0).mean(),
            late.var(dim=0).mean(),

            torch.norm(late_mean, p=2),
        ])

    for i in range(len(late_means_by_layer) - 1):
      a = late_means_by_layer[i]
      b = late_means_by_layer[i + 1]

      geometric_features.extend([
          F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).squeeze(),
          torch.norm(b - a, p=2),
      ])

    pooled = torch.cat(pooled_features)
    geom = torch.stack(geometric_features)

    return torch.cat([pooled, geom])


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    return aggregate(hidden_states, attention_mask)