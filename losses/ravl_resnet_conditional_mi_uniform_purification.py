# from __future__ import annotations

# from dataclasses import dataclass
# import copy
# import math
# import os
# from pathlib import Path
# import random
# from typing import Dict, List, Optional, Tuple

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# import numpy as np
# from PIL import Image
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# @dataclass
# class RaVLDiscoveryResult:
#     best_k: int
#     best_silhouette: float
#     top_spurious_cluster: int
#     ranked_clusters: List[int]
#     influence_scores: Dict[int, float]
#     performance_gaps: Dict[int, float]
#     per_class_gaps: Dict[int, Dict[int, float]]
#     per_class_influence: Dict[int, Dict[int, float]]
#     num_images: int
#     num_regions: int

#     def summary(self) -> Dict[str, object]:
#         return {
#             "best_k": self.best_k,
#             "best_silhouette": self.best_silhouette,
#             "top_spurious_cluster": self.top_spurious_cluster,
#             "ranked_clusters": self.ranked_clusters,
#             "influence_scores": self.influence_scores,
#             "performance_gaps": self.performance_gaps,
#             "num_images": self.num_images,
#             "num_regions": self.num_regions,
#         }


# @dataclass
# class RaVLOutput:
#     loss_region: torch.Tensor
#     loss_R: torch.Tensor
#     loss_A: torch.Tensor
#     logits: torch.Tensor
#     raw_spurious_mask: torch.Tensor
#     spurious_mask: torch.Tensor
#     priority_relevant_mask: torch.Tensor
#     relevant_mask: torch.Tensor
#     cluster_ids: torch.Tensor
#     num_raw_spurious_regions: int
#     num_protected_regions: int
#     num_spurious_regions: int
#     num_relevant_regions: int
#     num_valid_images: int

#     # Backward compatibility with the previous uniform-neutralization version.
#     @property
#     def loss_nui(self) -> torch.Tensor:
#         return self.loss_region

#     def statistics(self) -> Dict[str, float]:
#         return {
#             "loss_region": float(self.loss_region.detach().item()),
#             "loss_R": float(self.loss_R.detach().item()),
#             "loss_A": float(self.loss_A.detach().item()),
#             "num_raw_spurious_regions": float(self.num_raw_spurious_regions),
#             "num_protected_regions": float(self.num_protected_regions),
#             "num_spurious_regions": float(self.num_spurious_regions),
#             "num_relevant_regions": float(self.num_relevant_regions),
#             "num_valid_images": float(self.num_valid_images),
#         }


# def _unwrap_model(model: nn.Module) -> nn.Module:
#     return model.module if hasattr(model, "module") else model


# def _unwrap_tensor(x, name: str) -> torch.Tensor:
#     if torch.is_tensor(x):
#         return x
#     if isinstance(x, (tuple, list)) and len(x) > 0 and torch.is_tensor(x[0]):
#         return x[0]
#     raise TypeError(
#         "{} must be Tensor or tuple/list whose first item is Tensor.".format(name)
#     )


# def _get_classifier(model: nn.Module) -> nn.Linear:
#     base = _unwrap_model(model)
#     candidates = []

#     if hasattr(base, "get_classifier"):
#         try:
#             candidates.append(base.get_classifier())
#         except Exception:
#             pass

#     for name in ("fc", "head", "classifier"):
#         if hasattr(base, name):
#             candidates.append(getattr(base, name))

#     for candidate in candidates:
#         if isinstance(candidate, nn.Linear):
#             return candidate
#         if isinstance(candidate, nn.Sequential):
#             for sub in reversed(candidate):
#                 if isinstance(sub, nn.Linear):
#                     return sub

#     raise AttributeError("Unable to locate final nn.Linear classifier.")


# def _capture_layer4_and_forward(
#     model: nn.Module,
#     inputs: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     base = _unwrap_model(model)

#     if not hasattr(base, "layer4"):
#         raise AttributeError(
#             "The model must be ResNet-like and contain model.layer4."
#         )

#     holder = {}

#     def hook(_module, _inputs, output):
#         holder["features"] = _unwrap_tensor(output, "layer4 output")

#     handle = base.layer4.register_forward_hook(hook)

#     try:
#         logits = _unwrap_tensor(model(inputs), "model(inputs)")
#     finally:
#         handle.remove()

#     if "features" not in holder:
#         raise RuntimeError("Failed to capture layer4 features.")

#     features = holder["features"]

#     if features.ndim != 4:
#         raise ValueError(
#             "layer4 must be [B,D,H,W], got {}".format(tuple(features.shape))
#         )

#     if logits.ndim != 2:
#         raise ValueError(
#             "logits must be [B,C], got {}".format(tuple(logits.shape))
#         )

#     return logits, features



# class RaVLResNet(object):
#     """
#     RaVL-style discovery + mitigation migrated from the NeurIPS 2024 RaVL
#     algorithm to a supervised ResNet classifier.

#     What is kept from the original RaVL:
#       Stage 1:
#         1) labeled reference/validation set
#         2) local candidate regions from correctly predicted reference samples
#         3) K-Medoids with cosine distance on the correct-only region pool
#         4) choose K by Silhouette score over [2|Y|, 5|Y|]
#         5) class-conditional MI I(S_k; Y_hat | Y=c)
#         6) harmful error-increase gating
#         7) class-balanced conditional nuisance score N_k
#         8) select the top-ranked nuisance cluster(s)

#       Stage 2:
#         1) keep the Stage-1 clustering model fixed
#         2) before nuisance partition, protect the rel_num regions most similar
#            to the ground-truth classifier direction w_{y_i}
#         3) protected regions have the highest priority and can never be nuisance
#         4) nuisance regions = nuisance-cluster regions minus protected regions
#         5) optimize the original bidirectional region-class losses L_R + L_A

#     Necessary modality substitutions:
#       - VLM text class embedding g(y) -> normalized frozen FC class direction w_y
#       - RoI candidate regions -> equal grid regions pooled from ResNet layer4
#       - paired-text assigned label y_hat -> supervised ground-truth class label y

#     Important:
#       - The clustering medoids and the Stage-1 assignment encoder remain fixed.
#       - The student/model training strategy is external to this class.
#       - This class introduces no learnable parameters.
#     """

#     def __init__(
#         self,
#         num_classes: int,
#         region_grid: int = 3,
#         temperature: float = 0.07,
#         lambda_cl: float = 0.80,
#         influence_threshold: float = 0.25,
#         num_spurious_clusters: int = 1,
#         rel_num: int = 4,
#         k_min_factor: int = 2,
#         k_max_factor: int = 3,
#         kmedoids_iterations: int = 30,
#         max_cluster_regions: Optional[int] = 20000,
#         silhouette_sample_size: int = 3000,
#         assignment_chunk_size: int = 8192,
#         random_seed: int = 0,
#     ) -> None:
#         if num_classes < 2:
#             raise ValueError("num_classes must be >= 2")
#         if region_grid < 1:
#             raise ValueError("region_grid must be >= 1")
#         if temperature <= 0:
#             raise ValueError("temperature must be > 0")
#         if not 0 <= lambda_cl <= 10:
#             raise ValueError("lambda_cl must be in [0,10]")
#         if influence_threshold < 0:
#             raise ValueError("influence_threshold must be >= 0")
#         if k_min_factor < 1 or k_max_factor < k_min_factor:
#             raise ValueError("invalid cluster-count factors")
#         if rel_num < 1:
#             raise ValueError("rel_num must be >= 1")

#         self.num_classes = int(num_classes)
#         self.region_grid = int(region_grid)
#         self.temperature = float(temperature)
#         self.lambda_cl = float(lambda_cl)
#         self.influence_threshold = float(influence_threshold)
#         self.num_spurious_clusters = int(num_spurious_clusters)
#         self.rel_num = int(rel_num)
#         self.k_min_factor = int(k_min_factor)
#         self.k_max_factor = int(k_max_factor)
#         self.kmedoids_iterations = int(kmedoids_iterations)
#         self.max_cluster_regions = max_cluster_regions
#         self.silhouette_sample_size = int(silhouette_sample_size)
#         self.assignment_chunk_size = int(assignment_chunk_size)
#         self.random_seed = int(random_seed)
#         self.eps = 1e-8

#         self.medoids_raw = None
#         self.medoids_norm = None
#         self.top_spurious_cluster = None
#         self.top_spurious_clusters = []
#         self.ranked_clusters = []
#         self.discovery_result = None

#         # Frozen copy of the Stage-1 model.
#         # It is used only for assigning Stage-2 regions to the fixed clusters.
#         self._assignment_model = None
#         self._assignment_device = None

#     # ------------------------------------------------------------
#     # Region construction
#     # ------------------------------------------------------------
#     def _regions_from_features(
#             self,
#             features: torch.Tensor,
#     ):
#         """
#         Directly use layer4 spatial tokens as regions.

#         Input:
#             features:
#                 [B,C,H,W]

#         Output:
#             regions:
#                 [B,H*W,C]

#         For ResNet50 layer4:
#             [B,2048,7,7]
#             ->
#             [B,49,2048]
#         """

#         B, C, H, W = features.shape
#         regions = (
#             features
#             .flatten(2)
#             .transpose(1, 2)
#         )
#         return regions

#     # ------------------------------------------------------------
#     # Cosine K-Medoids
#     # ------------------------------------------------------------
#     @torch.no_grad()
#     def _fit_kmedoids_cosine(
#         self,
#         x_raw: torch.Tensor,
#         k: int,
#         seed: int,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Alternating K-Medoids for cosine distance.

#         For fixed assignments and normalized samples, the exact medoid of a
#         cluster is the member maximizing similarity to the sum of cluster
#         members, so the update can be done without an O(n_cluster^2) matrix.

#         Returns:
#             medoid_ids: [k]
#             labels: [N]
#         """
#         n = int(x_raw.shape[0])

#         if k < 2 or k >= n:
#             raise ValueError("K-Medoids requires 2 <= k < N.")

#         x = F.normalize(
#             x_raw.float(),
#             p=2,
#             dim=1,
#             eps=self.eps,
#         )

#         gen = torch.Generator(device=x.device)
#         gen.manual_seed(int(seed))

#         first = int(
#             torch.randint(
#                 low=0,
#                 high=n,
#                 size=(1,),
#                 generator=gen,
#                 device=x.device,
#             ).item()
#         )

#         selected = [first]

#         # Farthest-point initialization in cosine distance.
#         min_dist = 1.0 - (x @ x[first:first + 1].t()).squeeze(1)

#         for _ in range(1, k):
#             idx = int(min_dist.argmax().item())
#             selected.append(idx)
#             dist = 1.0 - (x @ x[idx:idx + 1].t()).squeeze(1)
#             min_dist = torch.minimum(min_dist, dist)

#         medoid_ids = torch.tensor(
#             selected,
#             device=x.device,
#             dtype=torch.long,
#         )

#         old_labels = None

#         for _ in range(self.kmedoids_iterations):
#             medoids = x.index_select(0, medoid_ids)
#             sim = x @ medoids.t()
#             labels = sim.argmax(dim=1)

#             if old_labels is not None and torch.equal(labels, old_labels):
#                 break

#             old_labels = labels.clone()
#             new_ids = []

#             for cluster_id in range(k):
#                 members = labels.eq(cluster_id).nonzero(
#                     as_tuple=False
#                 ).squeeze(1)

#                 if members.numel() == 0:
#                     # Re-seed from the currently worst represented point.
#                     max_sim = sim.max(dim=1).values
#                     candidate = int(max_sim.argmin().item())
#                     new_ids.append(candidate)
#                     continue

#                 member_x = x.index_select(0, members)
#                 sum_direction = member_x.sum(dim=0)

#                 # Exact cosine-medoid criterion for the fixed cluster:
#                 # argmax_i sum_j cos(x_i, x_j)
#                 medoid_score = member_x @ sum_direction
#                 local_id = int(medoid_score.argmax().item())
#                 new_ids.append(int(members[local_id].item()))

#             new_ids_tensor = torch.tensor(
#                 new_ids,
#                 device=x.device,
#                 dtype=torch.long,
#             )

#             if torch.equal(new_ids_tensor, medoid_ids):
#                 medoid_ids = new_ids_tensor
#                 break

#             medoid_ids = new_ids_tensor

#         final_medoids = x.index_select(0, medoid_ids)
#         final_labels = (x @ final_medoids.t()).argmax(dim=1)

#         return medoid_ids, final_labels

#     @torch.no_grad()
#     def _silhouette_cosine(
#         self,
#         x_raw: torch.Tensor,
#         labels: torch.Tensor,
#         seed: int,
#     ) -> float:
#         """
#         Cosine Silhouette score.

#         If N > silhouette_sample_size, a deterministic random subset is used.
#         This is only an engineering memory cap; set silhouette_sample_size >= N
#         to recover the full score.
#         """
#         n = int(x_raw.shape[0])

#         if n < 3:
#             return -1.0

#         unique_labels = labels.unique()
#         if unique_labels.numel() < 2:
#             return -1.0

#         s = min(n, self.silhouette_sample_size)

#         if s < n:
#             gen = torch.Generator(device=x_raw.device)
#             gen.manual_seed(int(seed))
#             sample_ids = torch.randperm(
#                 n,
#                 generator=gen,
#                 device=x_raw.device,
#             )[:s]
#             x = x_raw.index_select(0, sample_ids)
#             y = labels.index_select(0, sample_ids)
#         else:
#             x = x_raw
#             y = labels

#         x = F.normalize(x.float(), p=2, dim=1, eps=self.eps)
#         dist = 1.0 - x @ x.t()
#         dist = dist.clamp_min(0.0)

#         sil = torch.zeros(
#             x.shape[0],
#             device=x.device,
#             dtype=x.dtype,
#         )

#         all_clusters = y.unique()

#         for i in range(x.shape[0]):
#             own = y[i]
#             own_mask = y.eq(own)
#             own_count = int(own_mask.sum().item())

#             if own_count <= 1:
#                 sil[i] = 0.0
#                 continue

#             a = dist[i, own_mask].sum() / float(own_count - 1)

#             b = None
#             for c in all_clusters:
#                 if int(c.item()) == int(own.item()):
#                     continue

#                 mask = y.eq(c)
#                 if not bool(mask.any().item()):
#                     continue

#                 mean_dist = dist[i, mask].mean()
#                 if b is None or mean_dist < b:
#                     b = mean_dist

#             if b is None:
#                 sil[i] = 0.0
#                 continue

#             denom = torch.maximum(a, b).clamp_min(self.eps)
#             sil[i] = (b - a) / denom

#         return float(sil.mean().item())

#     @torch.no_grad()
#     def _assign_to_medoids(
#         self,
#         regions: torch.Tensor,
#         medoids_norm: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         """
#         Assign [N,D] or [B,R,D] region embeddings to the fixed medoids
#         using cosine distance.
#         """
#         if medoids_norm is None:
#             if self.medoids_norm is None:
#                 raise RuntimeError("RaVL discovery has not been run.")
#             medoids_norm = self.medoids_norm

#         original_shape = regions.shape[:-1]
#         d = regions.shape[-1]
#         flat = regions.reshape(-1, d)

#         outputs = []
#         medoids_norm = medoids_norm.to(
#             device=flat.device,
#             dtype=flat.dtype,
#         )

#         for start in range(0, flat.shape[0], self.assignment_chunk_size):
#             end = min(start + self.assignment_chunk_size, flat.shape[0])
#             x = F.normalize(
#                 flat[start:end],
#                 p=2,
#                 dim=1,
#                 eps=self.eps,
#             )
#             outputs.append((x @ medoids_norm.t()).argmax(dim=1))

#         assigned = torch.cat(outputs, dim=0)
#         return assigned.view(*original_shape)

#     # ------------------------------------------------------------
#     # Stage 1: RaVL discovery
#     # ------------------------------------------------------------
#     @torch.no_grad()
#     def discover(
#         self,
#         model: nn.Module,
#         reference_loader,
#         device=None,
#         verbose: bool = True,
#         make_assignment_snapshot: bool = True,
#     ) -> RaVLDiscoveryResult:
#         """
#         Stage-1 discovery adapted to a supervised ResNet. K-Medoids
#         prototypes are constructed only from correctly predicted reference
#         samples; all reference samples are then assigned for nuisance scoring.

#         reference_loader must yield:
#             (inputs, labels)
#         or:
#             (inputs, labels, ...)
#         """
#         if device is None:
#             device = next(model.parameters()).device

#         was_training = model.training
#         model.eval()

#         classifier = _get_classifier(model)

#         all_regions = []
#         all_region_probs = []
#         all_labels = []
#         all_preds = []

#         try:
#             for batch in reference_loader:
#                 if not isinstance(batch, (tuple, list)) or len(batch) < 2:
#                     raise TypeError(
#                         "reference_loader must return (inputs, labels) or "
#                         "(inputs, labels, ...)."
#                     )

#                 inputs = batch[0]
#                 labels = batch[1]

#                 if isinstance(inputs, (tuple, list)):
#                     inputs = inputs[0]

#                 inputs = inputs.to(device, non_blocking=True)
#                 labels = labels.to(device, non_blocking=True).long()

#                 logits, features = _capture_layer4_and_forward(
#                     model,
#                     inputs,
#                 )

#                 if logits.shape[1] != self.num_classes:
#                     raise ValueError(
#                         "Expected {} classes, got {}.".format(
#                             self.num_classes,
#                             logits.shape[1],
#                         )
#                     )

#                 regions = self._regions_from_features(features)
#                 b, r, d = regions.shape

#                 if classifier.weight.shape[1] != d:
#                     raise ValueError(
#                         "Classifier input dim {} != region dim {}.".format(
#                             classifier.weight.shape[1],
#                             d,
#                         )
#                     )

#                 region_logits = classifier(
#                     regions.reshape(b * r, d)
#                 ).view(b, r, self.num_classes)

#                 region_probs = F.softmax(region_logits, dim=2)

#                 pred = logits.argmax(dim=1)

#                 all_regions.append(
#                     regions.detach().cpu().to(torch.float16)
#                 )
#                 all_region_probs.append(
#                     region_probs.detach().cpu().to(torch.float16)
#                 )
#                 all_labels.append(labels.detach().cpu())
#                 all_preds.append(pred.detach().cpu())

#         finally:
#             if was_training:
#                 model.train()
#             else:
#                 model.eval()

#         if len(all_regions) == 0:
#             raise RuntimeError("Reference loader is empty.")

#         regions_img = torch.cat(all_regions, dim=0).float()
#         region_probs_img = torch.cat(all_region_probs, dim=0).float()
#         labels_img = torch.cat(all_labels, dim=0).long()
#         preds_img = torch.cat(all_preds, dim=0).long()
#         correct_img = preds_img.eq(labels_img)

#         n_img, n_region, d = regions_img.shape
#         regions_flat = regions_img.reshape(n_img * n_region, d)

#         # --------------------------------------------------------
#         # Clustering pool: CORRECTLY PREDICTED samples only.
#         # --------------------------------------------------------
#         # Important design:
#         #   1) Correct samples are used only to CONSTRUCT the visual-pattern
#         #      dictionary (K-Medoids / medoids).
#         #   2) After the medoids are fixed, ALL reference samples (correct +
#         #      incorrect) are assigned to these medoids.
#         #   3) Conditional-MI nuisance scoring is still computed on ALL
#         #      reference samples, so incorrect samples remain essential for
#         #      identifying whether a pattern is harmful.
#         #
#         # This avoids allowing under-learned regions from misclassified samples
#         # to directly define the clustering prototypes, while retaining them in
#         # the subsequent nuisance-effect estimation.
#         total_regions = int(regions_flat.shape[0])

#         correct_image_ids = correct_img.nonzero(
#             as_tuple=False
#         ).squeeze(1)
#         num_correct_images = int(correct_image_ids.numel())

#         if num_correct_images == 0:
#             raise RuntimeError(
#                 "No correctly predicted sample exists in the reference set. "
#                 "Correct-only clustering cannot be constructed."
#             )

#         # [N_correct, R, D] -> [N_correct * R, D]
#         correct_regions_img = regions_img.index_select(
#             0,
#             correct_image_ids,
#         )
#         correct_regions_flat = correct_regions_img.reshape(
#             -1,
#             d,
#         )
#         total_correct_regions = int(correct_regions_flat.shape[0])

#         if total_correct_regions < 3:
#             raise RuntimeError(
#                 "Correct-only clustering requires at least 3 correct-region "
#                 "features, got {}.".format(total_correct_regions)
#             )

#         # Optional memory cap is applied ONLY to the correct-region clustering
#         # pool. The later all-sample assignment is unchanged.
#         if (
#             self.max_cluster_regions is not None
#             and total_correct_regions > int(self.max_cluster_regions)
#         ):
#             gen = torch.Generator()
#             gen.manual_seed(self.random_seed)
#             cluster_region_ids = torch.randperm(
#                 total_correct_regions,
#                 generator=gen,
#             )[: int(self.max_cluster_regions)]
#             cluster_pool_cpu = correct_regions_flat.index_select(
#                 0,
#                 cluster_region_ids,
#             )
#         else:
#             cluster_pool_cpu = correct_regions_flat

#         cluster_pool = cluster_pool_cpu.to(
#             device=device,
#             dtype=torch.float32,
#         )

#         n_pool = int(cluster_pool.shape[0])

#         k_min = self.num_classes * self.k_min_factor
#         k_max = self.num_classes * self.k_max_factor

#         k_min = max(2, min(k_min, n_pool - 1))
#         k_max = max(k_min, min(k_max, n_pool - 1))

#         if k_min >= n_pool:
#             raise RuntimeError(
#                 "Not enough candidate regions for RaVL clustering."
#             )

#         best_score = -float("inf")
#         best_k = None
#         best_medoid_ids = None

#         if verbose:
#             print("========== RaVL Stage 1: K-Medoids sweep ==========")
#             correct_rate = num_correct_images / float(max(n_img, 1))
#             print(
#                 "images={} | correct_images={} ({:.2%}) | regions/image={} | "
#                 "all_regions={} | correct_regions={} | cluster_pool={}".format(
#                     n_img,
#                     num_correct_images,
#                     correct_rate,
#                     n_region,
#                     total_regions,
#                     total_correct_regions,
#                     n_pool,
#                 )
#             )
#             # Helpful diagnostic under long-tailed data: clustering still uses
#             # all correctly predicted samples, but this print lets you inspect
#             # whether correct samples are strongly head-class dominated.
#             correct_per_class = []
#             for class_id in range(self.num_classes):
#                 class_correct = int(
#                     (correct_img & labels_img.eq(class_id)).sum().item()
#                 )
#                 correct_per_class.append(class_correct)
#             print("correct images per class = {}".format(correct_per_class))
#             print("K range: {} -> {}".format(k_min, k_max))

#         for k in range(k_min, k_max + 1):
#             medoid_ids, cluster_labels = self._fit_kmedoids_cosine(
#                 cluster_pool,
#                 k=k,
#                 seed=self.random_seed + k,
#             )

#             score = self._silhouette_cosine(
#                 cluster_pool,
#                 cluster_labels,
#                 seed=self.random_seed + 1000 + k,
#             )

#             if verbose:
#                 print(
#                     "K={:3d} | silhouette={:.6f}".format(
#                         k,
#                         score,
#                     )
#                 )

#             if score > best_score:
#                 best_score = score
#                 best_k = k
#                 best_medoid_ids = medoid_ids.detach().clone()

#         best_medoids_raw = cluster_pool.index_select(
#             0,
#             best_medoid_ids,
#         ).detach()

#         best_medoids_norm = F.normalize(
#             best_medoids_raw,
#             p=2,
#             dim=1,
#             eps=self.eps,
#         )

#         # Assign ALL reference regions using the selected fixed medoids.
#         assignments_flat = self._assign_to_medoids(
#             regions_flat.to(device=device, dtype=torch.float32),
#             medoids_norm=best_medoids_norm,
#         ).cpu()

#         assignments_img = assignments_flat.view(
#             n_img,
#             n_region,
#         )

#         # --------------------------------------------------------
#         # Conditional-MI nuisance cluster selection.
#         #
#         # Replace the original RaVL H_k / G_k selection only.
#         # Everything after cluster selection remains unchanged.
#         #
#         # For cluster k:
#         #   S_k = 1 if an image contains at least one region from k.
#         #
#         # For every ground-truth class y, estimate
#         #   I(S_k ; Y_hat | Y=y)
#         # and retain only the harmful direction
#         #   Delta_err = P(E=1 | S_k=1,Y=y)
#         #             - P(E=1 | S_k=0,Y=y).
#         #
#         # Per-class nuisance score:
#         #   N_{k,y} = balance * I(S_k;Y_hat|Y=y) * max(Delta_err, 0)
#         #
#         # Cluster score:
#         #   N_k = sum_y N_{k,y}
#         #
#         # The sum is class-balanced: each class contributes once rather
#         # than being weighted by its sample frequency, which is desirable
#         # for long-tailed recognition.
#         # --------------------------------------------------------

#         # Keep the original result-field names for backward compatibility:
#         #   influence_scores      -> normalized conditional nuisance score
#         #   performance_gaps      -> raw conditional nuisance score N_k
#         #   per_class_gaps        -> per-class error increase Delta_err
#         #   per_class_influence   -> per-class conditional MI
#         influence_scores = {}
#         performance_gaps = {}
#         per_class_gaps = {}
#         per_class_influence = {}

#         # Extra local dictionaries used only during selection / logging.
#         per_class_nuisance = {}
#         per_class_balance = {}

#         def _conditional_mi_binary_cluster(
#             cluster_present: torch.Tensor,
#             predicted_labels: torch.Tensor,
#         ) -> float:
#             """
#             Empirical mutual information I(S; Y_hat) in nats.

#             cluster_present: [N], bool, S in {0,1}
#             predicted_labels: [N], long, Y_hat in {0,...,C-1}

#             This function is called inside a fixed ground-truth class,
#             therefore it estimates I(S_k; Y_hat | Y=y).
#             """
#             n = int(cluster_present.numel())
#             if n <= 1:
#                 return 0.0

#             s = cluster_present.long()
#             y_hat = predicted_labels.long()

#             joint = torch.zeros(
#                 2,
#                 self.num_classes,
#                 dtype=torch.float64,
#             )

#             flat_index = s * self.num_classes + y_hat
#             counts = torch.bincount(
#                 flat_index,
#                 minlength=2 * self.num_classes,
#             ).double()
#             joint = counts.view(2, self.num_classes)

#             total = joint.sum()
#             if float(total.item()) <= 0.0:
#                 return 0.0

#             p_joint = joint / total
#             p_s = p_joint.sum(dim=1, keepdim=True)
#             p_yhat = p_joint.sum(dim=0, keepdim=True)
#             denom = p_s * p_yhat

#             valid = p_joint > 0
#             if not bool(valid.any().item()):
#                 return 0.0

#             mi = (
#                 p_joint[valid]
#                 * torch.log(
#                     p_joint[valid]
#                     / denom[valid].clamp_min(self.eps)
#                 )
#             ).sum()

#             return float(mi.item())

#         for cluster_id in range(best_k):
#             # S_k for every reference image.
#             present = assignments_img.eq(cluster_id).any(dim=1)

#             mi_by_class = {}
#             error_increase_by_class = {}
#             nuisance_by_class = {}
#             balance_by_class = {}

#             for y in range(self.num_classes):
#                 class_mask = labels_img.eq(y)

#                 # Need both S_k=1 and S_k=0 inside this class so that the
#                 # cluster-presence variable is actually comparable.
#                 in_mask = class_mask & present
#                 out_mask = class_mask & (~present)

#                 n_in = int(in_mask.sum().item())
#                 n_out = int(out_mask.sum().item())

#                 if n_in == 0 or n_out == 0:
#                     continue

#                 # ----------------------------------------------------
#                 # 1) Harmful direction: increase in error probability.
#                 # ----------------------------------------------------
#                 err_in = float(
#                     (~correct_img[in_mask]).float().mean().item()
#                 )
#                 err_out = float(
#                     (~correct_img[out_mask]).float().mean().item()
#                 )

#                 delta_err = err_in - err_out

#                 # Same presence/absence balancing idea as the old code,
#                 # but it now stabilizes the conditional-MI score rather
#                 # than defining G_k.
#                 balance = (
#                     2.0
#                     * min(n_in, n_out)
#                     / float(n_in + n_out)
#                 )

#                 # ----------------------------------------------------
#                 # 2) Conditional MI:
#                 #       I(S_k ; Y_hat | Y=y)
#                 # ----------------------------------------------------
#                 class_present = present[class_mask]
#                 class_preds = preds_img[class_mask]

#                 cmi_y = _conditional_mi_binary_cluster(
#                     cluster_present=class_present,
#                     predicted_labels=class_preds,
#                 )

#                 # ----------------------------------------------------
#                 # 3) Per-class nuisance score.
#                 # Only the harmful direction is kept.
#                 # ----------------------------------------------------
#                 harmful_delta = max(delta_err, 0.0)
#                 nuisance_y = balance * cmi_y * harmful_delta

#                 mi_by_class[y] = float(cmi_y)
#                 error_increase_by_class[y] = float(delta_err)
#                 nuisance_by_class[y] = float(nuisance_y)
#                 balance_by_class[y] = float(balance)

#             # Class-balanced aggregation. We intentionally do NOT multiply
#             # by the empirical class prior, otherwise head classes would
#             # dominate the cluster ranking in a long-tailed dataset.
#             nuisance_score = sum(nuisance_by_class.values())

#             performance_gaps[cluster_id] = float(nuisance_score)
#             per_class_gaps[cluster_id] = error_increase_by_class
#             per_class_influence[cluster_id] = mi_by_class
#             per_class_nuisance[cluster_id] = nuisance_by_class
#             per_class_balance[cluster_id] = balance_by_class

#         # Normalize only for thresholding so the existing constructor
#         # argument influence_threshold can be kept unchanged.
#         max_nuisance_score = max(
#             performance_gaps.values()
#         ) if len(performance_gaps) > 0 else 0.0

#         for cluster_id in range(best_k):
#             raw_score = performance_gaps.get(cluster_id, 0.0)
#             if max_nuisance_score > self.eps:
#                 normalized_score = raw_score / max_nuisance_score
#             else:
#                 normalized_score = 0.0
#             influence_scores[cluster_id] = float(normalized_score)

#         candidate_clusters = [
#             c
#             for c in range(best_k)
#             if performance_gaps.get(c, 0.0) > 0.0
#             and influence_scores.get(c, 0.0) >= self.influence_threshold
#         ]

#         # Rank by the raw conditional nuisance score N_k.
#         ranked_clusters = sorted(
#             candidate_clusters,
#             key=lambda c: performance_gaps.get(c, 0.0),
#             reverse=True,
#         )

#         if len(ranked_clusters) == 0:
#             raise RuntimeError(
#                 "Conditional-MI selection found no nuisance cluster with "
#                 "positive harmful score and normalized score >= {:.3f}. "
#                 "Try inspecting the reference split or lowering "
#                 "influence_threshold.".format(
#                     self.influence_threshold
#                 )
#             )

#         top_spurious_clusters = [int(c) for c in ranked_clusters[:self.num_spurious_clusters]]

#         # backward compatible: keep original single cluster variable
#         top_spurious_cluster = int(top_spurious_clusters[0])

#         # Keep the fixed Stage-1 clustering model.
#         self.medoids_raw = best_medoids_raw.detach().cpu()
#         self.medoids_norm = best_medoids_norm.detach().cpu()
#         self.top_spurious_cluster = top_spurious_cluster
#         self.top_spurious_clusters = top_spurious_clusters
#         self.ranked_clusters = list(ranked_clusters)

#         result = RaVLDiscoveryResult(
#             best_k=int(best_k),
#             best_silhouette=float(best_score),
#             top_spurious_cluster=top_spurious_cluster,
#             ranked_clusters=list(ranked_clusters),
#             influence_scores=influence_scores,
#             performance_gaps=performance_gaps,
#             per_class_gaps=per_class_gaps,
#             per_class_influence=per_class_influence,
#             num_images=int(n_img),
#             num_regions=int(total_regions),
#         )

#         self.discovery_result = result

#         if make_assignment_snapshot:
#             self.prepare_stage2_assignment_model(
#                 model=model,
#                 device=device,
#             )

#         if verbose:
#             print("========== RaVL Stage 1: Discovery Result ==========")
#             print(
#                 "best K={} | silhouette={:.6f}".format(
#                     best_k,
#                     best_score,
#                 )
#             )
#             print(
#                 "normalized nuisance-score threshold={:.3f}".format(
#                     self.influence_threshold
#                 )
#             )

#             for rank, cluster_id in enumerate(ranked_clusters[:10], 1):
#                 print(
#                     "Rank {:2d} | cluster {:3d} | N_norm={:.4f} | N_score={:.6f}".format(
#                         rank,
#                         cluster_id,
#                         influence_scores[cluster_id],
#                         performance_gaps[cluster_id],
#                     )
#                 )

#             print(
#                 "TOP spurious clusters = {}".format(
#                     top_spurious_clusters
#                 )
#             )
#             print("=====================================================")

#         return result

#     # ------------------------------------------------------------
#     # Fixed Stage-1 assignment model for Stage 2
#     # ------------------------------------------------------------
#     @torch.no_grad()
#     def prepare_stage2_assignment_model(
#         self,
#         model: nn.Module,
#         device=None,
#     ) -> None:
#         """
#         Freeze a snapshot of the Stage-1 encoder.

#         The paper first determines the spurious cluster, then uses the trained
#         clustering model to assign training regions to R^s / R^r before
#         mitigation. This frozen snapshot keeps region assignment in the
#         original Stage-1 feature space even while the student backbone changes.
#         """
#         base = _unwrap_model(model)

#         if device is None:
#             device = next(base.parameters()).device

#         frozen = copy.deepcopy(base).to(device)
#         frozen.eval()

#         for p in frozen.parameters():
#             p.requires_grad_(False)

#         # Bypass nn.Module registration because RaVLResNet is intentionally
#         # a parameter-free utility object.
#         self._assignment_model = frozen
#         self._assignment_device = device
    
#     @torch.no_grad()
#     def evaluate_classwise_nuisance_ratio(
#         self,
#         student_model: nn.Module,
#         data_loader,
#         device,
#         max_batches: Optional[int] = None,
#         verbose: bool = True,
#     ) -> Dict[str, np.ndarray]:
#         """
#         Evaluate class-wise nuisance patch statistics.

#         Current pipeline:
#             1. Current student features -> priority relevant Top-rel_num
#             2. Frozen Stage-1 model -> cluster assignment
#             3. Raw nuisance clusters
#             4. Remove priority relevant regions
#             5. Obtain FINAL nuisance regions

#         For each class c:

#             raw_nuisance_ratio[c]
#                 = # raw nuisance patches of class c
#                   / # all patches of class c

#             protected_ratio[c]
#                 = # raw nuisance patches protected by Top-rel_num
#                   / # all patches of class c

#             final_nuisance_ratio[c]
#                 = # final nuisance patches of class c
#                   / # all patches of class c

#         Also reports:

#             image_with_nuisance_ratio[c]
#                 = proportion of images in class c containing
#                   at least one final nuisance patch

#             mean_nuisance_ratio_per_image[c]
#                 = average final nuisance patch ratio over
#                   images belonging to class c

#         Returns:
#             Dictionary containing one numpy array per statistic,
#             with shape [num_classes].
#         """

#         # ============================================================
#         # 1. Prepare model
#         # ============================================================
#         base = _unwrap_model(student_model)
#         was_training = base.training
#         base.eval()

#         num_classes = int(self.num_classes)

#         # Number of images for each class
#         class_num_images = torch.zeros(
#             num_classes,
#             dtype=torch.long,
#         )

#         # Total number of spatial patches for each class
#         class_total_patches = torch.zeros(
#             num_classes,
#             dtype=torch.long,
#         )

#         # Raw cluster-based nuisance patches
#         class_raw_nuisance = torch.zeros(
#             num_classes,
#             dtype=torch.long,
#         )

#         # Raw nuisance patches protected by Top-rel_num
#         class_protected = torch.zeros(
#             num_classes,
#             dtype=torch.long,
#         )

#         # Final nuisance patches
#         class_final_nuisance = torch.zeros(
#             num_classes,
#             dtype=torch.long,
#         )

#         # Number of images containing >= 1 final nuisance patch
#         class_images_with_nuisance = torch.zeros(
#             num_classes,
#             dtype=torch.long,
#         )

#         # Sum of per-image nuisance ratios
#         class_sum_image_nuisance_ratio = torch.zeros(
#             num_classes,
#             dtype=torch.float64,
#         )

#         try:

#             # ========================================================
#             # 2. Iterate dataset
#             # ========================================================
#             for batch_idx, batch in enumerate(data_loader):

#                 if (
#                     max_batches is not None
#                     and batch_idx >= int(max_batches)
#                 ):
#                     break

#                 if (
#                     not isinstance(batch, (tuple, list))
#                     or len(batch) < 2
#                 ):
#                     raise TypeError(
#                         "data_loader must return "
#                         "(inputs, labels, ...)."
#                     )

#                 inputs = batch[0]
#                 labels = batch[1]

#                 if isinstance(inputs, (tuple, list)):
#                     inputs = inputs[0]

#                 inputs = inputs.to(
#                     device,
#                     non_blocking=True,
#                 )

#                 labels = labels.to(
#                     device,
#                     non_blocking=True,
#                 ).long()

#                 # ====================================================
#                 # 3. Current student representation
#                 # ====================================================
#                 _, student_features = (
#                     _capture_layer4_and_forward(
#                         student_model,
#                         inputs,
#                     )
#                 )

#                 student_regions = (
#                     self._regions_from_features(
#                         student_features
#                     )
#                 )

#                 # [B,R,D]
#                 b, r, _ = student_regions.shape

#                 classifier = _get_classifier(
#                     student_model
#                 )

#                 # ====================================================
#                 # 4. Top-rel_num priority relevant patches
#                 # ====================================================
#                 priority_relevant_mask, _ = (
#                     self._priority_relevant_mask(
#                         student_regions=student_regions,
#                         labels=labels,
#                         classifier=classifier,
#                     )
#                 )

#                 # ====================================================
#                 # 5. Frozen Stage-1 cluster assignment
#                 # ====================================================
#                 (
#                     cluster_ids,
#                     raw_spurious_mask,
#                     _,
#                 ) = self._stage2_region_partition(
#                     inputs
#                 )

#                 raw_spurious_mask = (
#                     raw_spurious_mask.to(
#                         device=student_regions.device,
#                         dtype=torch.bool,
#                     )
#                 )

#                 cluster_ids = cluster_ids.to(
#                     device=student_regions.device,
#                 )

#                 # Safety check
#                 if cluster_ids.shape != (b, r):
#                     raise ValueError(
#                         "Cluster assignment shape {} does not "
#                         "match student region shape [{},{}].".format(
#                             tuple(cluster_ids.shape),
#                             b,
#                             r,
#                         )
#                     )

#                 # ====================================================
#                 # 6. Priority protection
#                 # ====================================================
#                 (
#                     final_spurious_mask,
#                     _,
#                     protected_mask,
#                 ) = self._apply_priority_protection(
#                     raw_spurious_mask=raw_spurious_mask,
#                     priority_relevant_mask=priority_relevant_mask,
#                 )

#                 # ====================================================
#                 # 7. Statistics for each class
#                 # ====================================================
#                 for c in range(num_classes):

#                     class_mask = labels.eq(c)

#                     num_images_c = int(
#                         class_mask.sum().item()
#                     )

#                     if num_images_c == 0:
#                         continue

#                     # -----------------------------------------------
#                     # Masks belonging to class c
#                     #
#                     # [Nc,R]
#                     # -----------------------------------------------
#                     raw_c = raw_spurious_mask[
#                         class_mask
#                     ]

#                     protected_c = protected_mask[
#                         class_mask
#                     ]

#                     final_c = final_spurious_mask[
#                         class_mask
#                     ]

#                     # -----------------------------------------------
#                     # Image / patch counts
#                     # -----------------------------------------------
#                     class_num_images[c] += (
#                         num_images_c
#                     )

#                     class_total_patches[c] += (
#                         num_images_c * r
#                     )

#                     class_raw_nuisance[c] += int(
#                         raw_c.sum().item()
#                     )

#                     class_protected[c] += int(
#                         protected_c.sum().item()
#                     )

#                     class_final_nuisance[c] += int(
#                         final_c.sum().item()
#                     )

#                     # -----------------------------------------------
#                     # Does each image contain nuisance?
#                     # -----------------------------------------------
#                     image_has_nuisance = (
#                         final_c.any(dim=1)
#                     )

#                     class_images_with_nuisance[c] += int(
#                         image_has_nuisance.sum().item()
#                     )

#                     # -----------------------------------------------
#                     # Per-image nuisance ratio
#                     #
#                     # [Nc,R] -> [Nc]
#                     # -----------------------------------------------
#                     image_nuisance_ratio = (
#                         final_c.float()
#                         .mean(dim=1)
#                     )

#                     class_sum_image_nuisance_ratio[c] += (
#                         image_nuisance_ratio
#                         .double()
#                         .sum()
#                         .cpu()
#                     )

#         finally:

#             # ========================================================
#             # 8. Restore model state
#             # ========================================================
#             if was_training:
#                 base.train()
#             else:
#                 base.eval()

#         # ============================================================
#         # 9. Convert counts to ratios
#         # ============================================================
#         num_images_np = (
#             class_num_images
#             .numpy()
#             .astype(np.int64)
#         )

#         total_patches_np = (
#             class_total_patches
#             .numpy()
#             .astype(np.int64)
#         )

#         raw_nuisance_np = (
#             class_raw_nuisance
#             .numpy()
#             .astype(np.int64)
#         )

#         protected_np = (
#             class_protected
#             .numpy()
#             .astype(np.int64)
#         )

#         final_nuisance_np = (
#             class_final_nuisance
#             .numpy()
#             .astype(np.int64)
#         )

#         images_with_nuisance_np = (
#             class_images_with_nuisance
#             .numpy()
#             .astype(np.int64)
#         )

#         sum_image_ratio_np = (
#             class_sum_image_nuisance_ratio
#             .numpy()
#         )

#         # ------------------------------------------------------------
#         # Avoid divide-by-zero
#         # ------------------------------------------------------------
#         patch_denominator = np.maximum(
#             total_patches_np,
#             1,
#         )

#         image_denominator = np.maximum(
#             num_images_np,
#             1,
#         )

#         raw_nuisance_ratio = (
#             raw_nuisance_np
#             / patch_denominator
#         )

#         protected_ratio = (
#             protected_np
#             / patch_denominator
#         )

#         final_nuisance_ratio = (
#             final_nuisance_np
#             / patch_denominator
#         )

#         image_with_nuisance_ratio = (
#             images_with_nuisance_np
#             / image_denominator
#         )

#         mean_nuisance_ratio_per_image = (
#             sum_image_ratio_np
#             / image_denominator
#         )

#         # Classes absent from this loader -> NaN
#         empty_class = (
#             num_images_np == 0
#         )

#         raw_nuisance_ratio[
#             empty_class
#         ] = np.nan

#         protected_ratio[
#             empty_class
#         ] = np.nan

#         final_nuisance_ratio[
#             empty_class
#         ] = np.nan

#         image_with_nuisance_ratio[
#             empty_class
#         ] = np.nan

#         mean_nuisance_ratio_per_image[
#             empty_class
#         ] = np.nan

#         # ============================================================
#         # 10. Results
#         # ============================================================
#         results = {
#             "num_images": num_images_np,

#             "num_total_patches": total_patches_np,

#             "num_raw_nuisance_patches": raw_nuisance_np,

#             "num_protected_patches": protected_np,

#             "num_final_nuisance_patches": final_nuisance_np,

#             "raw_nuisance_ratio": raw_nuisance_ratio,

#             "protected_ratio": protected_ratio,

#             "final_nuisance_ratio": final_nuisance_ratio,

#             "image_with_nuisance_ratio": (
#                 image_with_nuisance_ratio
#             ),

#             "mean_nuisance_ratio_per_image": (
#                 mean_nuisance_ratio_per_image
#             ),
#         }

#         # ============================================================
#         # 11. Pretty print
#         # ============================================================
#         if verbose:

#             print(
#                 "\n"
#                 + "=" * 92
#             )

#             print(
#                 "Class-wise nuisance statistics "
#                 "(after Top-{} priority protection)".format(
#                     self.rel_num
#                 )
#             )

#             print(
#                 "=" * 92
#             )

#             print(
#                 "{:<7s} {:>8s} {:>10s} {:>10s} {:>10s} "
#                 "{:>10s} {:>10s}".format(
#                     "Class",
#                     "Images",
#                     "Raw(%)",
#                     "Protect(%)",
#                     "Final(%)",
#                     "ImgHit(%)",
#                     "FinalNum",
#                 )
#             )

#             print("-" * 92)

#             for c in range(num_classes):

#                 if num_images_np[c] == 0:

#                     print(
#                         "{:<7d} {:>8d} {:>10s} {:>10s} "
#                         "{:>10s} {:>10s} {:>10d}".format(
#                             c,
#                             0,
#                             "-",
#                             "-",
#                             "-",
#                             "-",
#                             0,
#                         )
#                     )

#                     continue

#                 print(
#                     "{:<7d} {:>8d} {:>10.2f} {:>10.2f} "
#                     "{:>10.2f} {:>10.2f} {:>10d}".format(
#                         c,
#                         num_images_np[c],

#                         100.0
#                         * raw_nuisance_ratio[c],

#                         100.0
#                         * protected_ratio[c],

#                         100.0
#                         * final_nuisance_ratio[c],

#                         100.0
#                         * image_with_nuisance_ratio[c],

#                         final_nuisance_np[c],
#                     )
#                 )

#             print("=" * 92)

#             valid_patch_total = int(
#                 total_patches_np.sum()
#             )

#             valid_final_total = int(
#                 final_nuisance_np.sum()
#             )

#             if valid_patch_total > 0:

#                 overall_ratio = (
#                     valid_final_total
#                     / valid_patch_total
#                 )

#             else:

#                 overall_ratio = 0.0

#             print(
#                 "Overall final nuisance ratio: "
#                 "{:.2f}% ({}/{})".format(
#                     100.0 * overall_ratio,
#                     valid_final_total,
#                     valid_patch_total,
#                 )
#             )

#             print("=" * 92 + "\n")

#         return results
    
    
#     # ------------------------------------------------------------
#     # Region assignment during Stage 2
#     # ------------------------------------------------------------
#     @torch.no_grad()
#     def _stage2_region_partition(
#         self,
#         inputs: torch.Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Raw cluster-based partition in the frozen Stage-1 feature space.

#         IMPORTANT:
#             The returned spurious_mask is the RAW nuisance-cluster mask.
#             It does not yet apply the rel_num priority-relevant protection.
#             The protection needs the current student representation and w_y.
#         """
#         if self._assignment_model is None:
#             raise RuntimeError(
#                 "No frozen Stage-1 assignment model. "
#                 "Run discover(..., make_assignment_snapshot=True) or call "
#                 "prepare_stage2_assignment_model(model)."
#             )

#         if self.medoids_norm is None or self.top_spurious_cluster is None:
#             raise RuntimeError("RaVL discovery has not been initialized.")

#         assignment_inputs = inputs.to(
#             self._assignment_device,
#             non_blocking=True,
#         )

#         _, assignment_features = _capture_layer4_and_forward(
#             self._assignment_model,
#             assignment_inputs,
#         )

#         assignment_regions = self._regions_from_features(
#             assignment_features
#         )

#         medoids_norm = self.medoids_norm.to(
#             device=assignment_regions.device,
#             dtype=assignment_regions.dtype,
#         )

#         cluster_ids = self._assign_to_medoids(
#             assignment_regions,
#             medoids_norm=medoids_norm,
#         )

#         raw_spurious_mask = torch.zeros_like(
#             cluster_ids,
#             dtype=torch.bool,
#         )
#         for c in self.top_spurious_clusters:
#             raw_spurious_mask |= cluster_ids.eq(int(c))

#         raw_relevant_mask = ~raw_spurious_mask
#         return cluster_ids, raw_spurious_mask, raw_relevant_mask

#     # ------------------------------------------------------------
#     # Highest-priority task-relevant regions
#     # ------------------------------------------------------------
#     def _priority_relevant_mask(
#         self,
#         student_regions: torch.Tensor,
#         labels: torch.Tensor,
#         classifier: nn.Linear,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Select rel_num regions with the largest cosine similarity to the
#         ground-truth class direction w_{y_i} for each image.

#         These regions have ABSOLUTE PRIORITY: once selected, they can never be
#         treated as nuisance even if their frozen cluster assignment belongs to
#         a discovered nuisance cluster.

#         Returns:
#             priority_mask: [B,R] bool
#             gt_similarity: [B,R] cosine similarity to w_{y_i}
#         """
#         b, r, d = student_regions.shape
#         if classifier.weight.shape != (self.num_classes, d):
#             raise ValueError(
#                 "Classifier weight shape {} incompatible with regions [*,{},{}].".format(
#                     tuple(classifier.weight.shape), r, d
#                 )
#             )
#         if labels.shape[0] != b:
#             raise ValueError("labels batch size must match student_regions.")

#         k = min(int(self.rel_num), int(r))
#         if k < 1:
#             raise RuntimeError("No spatial region is available for rel_num selection.")

#         region_n = F.normalize(
#             student_regions,
#             p=2,
#             dim=2,
#             eps=self.eps,
#         )
#         class_n = F.normalize(
#             classifier.weight.detach().to(
#                 device=student_regions.device,
#                 dtype=student_regions.dtype,
#             ),
#             p=2,
#             dim=1,
#             eps=self.eps,
#         )

#         gt_w = class_n.index_select(0, labels.long())  # [B,D]
#         gt_similarity = torch.einsum("brd,bd->br", region_n, gt_w)
#         top_idx = gt_similarity.topk(k=k, dim=1, largest=True, sorted=True).indices

#         priority_mask = torch.zeros(
#             (b, r),
#             dtype=torch.bool,
#             device=student_regions.device,
#         )
#         priority_mask.scatter_(1, top_idx, True)
#         return priority_mask, gt_similarity

#     def _apply_priority_protection(
#         self,
#         raw_spurious_mask: torch.Tensor,
#         priority_relevant_mask: torch.Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Priority rule:

#             final nuisance = raw nuisance AND NOT priority-relevant
#             final relevant = NOT final nuisance

#         Therefore, rel_num selected regions can never enter L_R/L_A as nuisance.
#         """
#         raw_spurious_mask = raw_spurious_mask.to(
#             device=priority_relevant_mask.device,
#             dtype=torch.bool,
#         )
#         priority_relevant_mask = priority_relevant_mask.bool()

#         protected_from_nuisance = raw_spurious_mask & priority_relevant_mask
#         spurious_mask = raw_spurious_mask & (~priority_relevant_mask)
#         relevant_mask = ~spurious_mask
#         return spurious_mask, relevant_mask, protected_from_nuisance

#     # ------------------------------------------------------------
#     # Original bidirectional RaVL-style L_R + L_A
#     # ------------------------------------------------------------
#     def _region_aware_loss(
#         self,
#         student_regions: torch.Tensor,
#         labels: torch.Tensor,
#         spurious_mask: torch.Tensor,
#         relevant_mask: torch.Tensor,
#         classifier: nn.Linear,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Compute the bidirectional region-class losses after rel_num protection.

#         L_R (region-to-class):
#             - positive: strongest relevant region of image i for GT class y_i
#             - class negatives: strongest relevant response of image i to labels
#               occurring in the mini-batch
#             - nuisance negatives: each nuisance patch's strongest response to
#               labels occurring in the mini-batch

#         L_A (class-to-region):
#             - fix GT class direction w_{y_i}
#             - positive: current image's strongest relevant response to w_{y_i}
#             - relevant negatives: other-class images' strongest relevant
#               response to w_{y_i}
#             - nuisance negatives: all nuisance patches' response to w_{y_i}

#         Classifier weights are detached, so L_R/L_A update the representation
#         rather than moving the class anchors.
#         """
#         b, r, d = student_regions.shape
#         if classifier.weight.shape != (self.num_classes, d):
#             raise ValueError(
#                 "Classifier weight shape {} incompatible with regions [*,{},{}].".format(
#                     tuple(classifier.weight.shape), r, d
#                 )
#             )

#         device = student_regions.device
#         labels = labels.to(device=device, dtype=torch.long)
#         spurious_mask = spurious_mask.to(device=device, dtype=torch.bool)
#         relevant_mask = relevant_mask.to(device=device, dtype=torch.bool)

#         region_n = F.normalize(
#             student_regions,
#             p=2,
#             dim=2,
#             eps=self.eps,
#         )
#         class_n = F.normalize(
#             classifier.weight.detach().to(
#                 device=device,
#                 dtype=student_regions.dtype,
#             ),
#             p=2,
#             dim=1,
#             eps=self.eps,
#         )

#         # [B,R,C]
#         sim = torch.einsum("brd,cd->brc", region_n, class_n)

#         # Relevant-region strongest response m_i(c).
#         neg_large = torch.finfo(sim.dtype).min
#         rel_sim = sim.masked_fill(~relevant_mask.unsqueeze(-1), neg_large)
#         valid_images = relevant_mask.any(dim=1)
#         rel_max_sim = rel_sim.max(dim=1).values  # [B,C]

#         # In this design rel_num>=1 normally guarantees valid_images=True,
#         # but keep a safe differentiable-zero fallback.
#         valid_ids = valid_images.nonzero(as_tuple=False).squeeze(1)
#         if valid_ids.numel() == 0:
#             zero = student_regions.sum() * 0.0
#             return zero, zero, zero, valid_images

#         # [Ns,C] nuisance responses. It is valid for Ns=0.
#         spur_sim = sim[spurious_mask]

#         losses_R = []
#         losses_A = []

#         # q(r_s): strongest nuisance response to labels occurring in batch.
#         if spur_sim.numel() > 0:
#             spur_max_over_batch_labels = (
#                 spur_sim.index_select(1, labels).max(dim=1).values
#                 / self.temperature
#             )
#         else:
#             spur_max_over_batch_labels = None

#         for i in valid_ids.tolist():
#             y_i = int(labels[i].item())
#             positive_log = rel_max_sim[i, y_i] / self.temperature

#             # ------------------------- L_R -------------------------
#             # Keep duplicate labels exactly as they occur in the batch.
#             label_terms_R = (
#                 rel_max_sim[i].index_select(0, labels)
#                 / self.temperature
#             )
#             denom_terms_R = [label_terms_R]
#             if spur_max_over_batch_labels is not None:
#                 denom_terms_R.append(spur_max_over_batch_labels)
#             denom_R = torch.cat(denom_terms_R, dim=0)
#             loss_R_i = -positive_log + torch.logsumexp(denom_R, dim=0)
#             losses_R.append(loss_R_i)

#             # ------------------------- L_A -------------------------
#             denom_terms_A = [positive_log.reshape(1)]

#             other_mask = valid_images & labels.ne(y_i)
#             other_ids = other_mask.nonzero(as_tuple=False).squeeze(1)
#             if other_ids.numel() > 0:
#                 other_rel_to_y = (
#                     rel_max_sim.index_select(0, other_ids)[:, y_i]
#                     / self.temperature
#                 )
#                 denom_terms_A.append(other_rel_to_y)

#             if spur_sim.numel() > 0:
#                 spur_to_y = spur_sim[:, y_i] / self.temperature
#                 denom_terms_A.append(spur_to_y)

#             denom_A = torch.cat(denom_terms_A, dim=0)
#             loss_A_i = -positive_log + torch.logsumexp(denom_A, dim=0)
#             losses_A.append(loss_A_i)

#         loss_R = torch.stack(losses_R).mean()
#         loss_A = torch.stack(losses_A).mean()
#         loss_region = loss_R + loss_A
#         return loss_region, loss_R, loss_A, valid_images

#     def _nuisance_class_orthogonal_loss(
#         self,
#         student_regions: torch.Tensor,
#         spurious_mask: torch.Tensor,
#         classifier: nn.Linear,
#     ) -> torch.Tensor:
#         """
#         Nuisance-Class Orthogonalization (NCO).

#         Only suppress FINAL nuisance regions.

#         For each nuisance region z^s, suppress its strongest absolute
#         similarity to all classifier directions:

#             L_NCO =
#                 mean_{z^s} [
#                     max_c cos^2(z^s, w_c)
#                 ]

#         Therefore:

#             cos(z^s, w_c) -> 0,  for all classes c.

#         The classifier weights are detached, so this loss only updates
#         the student representation and does not move classifier anchors.

#         Args:
#             student_regions:
#                 Tensor [B, R, D]

#             spurious_mask:
#                 Bool Tensor [B, R]

#                 IMPORTANT:
#                 This should be the FINAL nuisance mask after
#                 rel_num priority protection.

#             classifier:
#                 Final nn.Linear classifier.
#                 classifier.weight shape = [C, D]

#         Returns:
#             loss_nco:
#                 Scalar differentiable tensor.
#         """

#         # ============================================================
#         # 1. Shape check
#         # ============================================================
#         if student_regions.ndim != 3:
#             raise ValueError(
#                 "student_regions must be [B,R,D], got {}.".format(
#                     tuple(student_regions.shape)
#                 )
#             )

#         b, r, d = student_regions.shape

#         if classifier.weight.shape != (
#             self.num_classes,
#             d,
#         ):
#             raise ValueError(
#                 "Classifier weight shape {} incompatible with "
#                 "student_regions [B={}, R={}, D={}].".format(
#                     tuple(classifier.weight.shape),
#                     b,
#                     r,
#                     d,
#                 )
#             )

#         device = student_regions.device

#         spurious_mask = spurious_mask.to(
#             device=device,
#             dtype=torch.bool,
#         )

#         if spurious_mask.shape != (b, r):
#             raise ValueError(
#                 "spurious_mask must have shape [{},{}], got {}.".format(
#                     b,
#                     r,
#                     tuple(spurious_mask.shape),
#                 )
#             )

#         # ============================================================
#         # 2. Normalize student region features
#         #
#         # [B,R,D]
#         # ============================================================
#         region_n = F.normalize(
#             student_regions,
#             p=2,
#             dim=2,
#             eps=self.eps,
#         )

#         # ============================================================
#         # 3. Normalize classifier directions
#         #
#         # [C,D]
#         #
#         # detach():
#         # classifier directions are fixed anchors for this loss.
#         # Only student representation receives gradients.
#         # ============================================================
#         class_n = F.normalize(
#             classifier.weight.detach().to(
#                 device=device,
#                 dtype=student_regions.dtype,
#             ),
#             p=2,
#             dim=1,
#             eps=self.eps,
#         )

#         # ============================================================
#         # 4. Similarity between EVERY region and EVERY class
#         #
#         # region_n : [B,R,D]
#         # class_n  : [C,D]
#         #
#         # output:
#         #     similarity[b,r,c]
#         #
#         # [B,R,C]
#         # ============================================================
#         similarity = torch.einsum(
#             "brd,cd->brc",
#             region_n,
#             class_n,
#         )

#         # ============================================================
#         # 5. Select FINAL nuisance regions only
#         #
#         # [N_s,C]
#         # ============================================================
#         nuisance_similarity = similarity[
#             spurious_mask
#         ]

#         # ============================================================
#         # 6. No nuisance patch in this mini-batch
#         #
#         # Return differentiable zero.
#         # ============================================================
#         if nuisance_similarity.numel() == 0:
#             return (
#                 student_regions.sum()
#                 * 0.0
#             )

#         # ============================================================
#         # 7. Nuisance suppression over ALL class directions
#         #
#         # First:
#         #
#         #     cos^2(z_s, w_c)
#         #
#         # [N_s,C]
#         #
#         # Squaring is important:
#         #
#         #     +1 -> bad
#         #     -1 -> also bad
#         #      0 -> desired
#         # ============================================================
#         nuisance_similarity_sq = (
#             nuisance_similarity.pow(2)
#         )

#         # ============================================================
#         # 8. For each nuisance patch, find its strongest
#         #    class-direction response.
#         #
#         # max_c cos^2(z_s, w_c)
#         #
#         # [N_s]
#         #
#         # This avoids the problem that averaging over classes may hide
#         # one very large wrong-class response.
#         # ============================================================
#         strongest_class_similarity = (
#             nuisance_similarity_sq
#             .max(dim=1)
#             .values
#         )

#         # ============================================================
#         # 9. Final loss
#         #
#         # min:
#         #
#         #     max_c cos^2(z_s, w_c)
#         #
#         # therefore nuisance representation is pushed away from
#         # the entire task-discriminative classifier space.
#         # ============================================================
#         loss_nco = (
#             strongest_class_similarity.mean()
#         )

#         return loss_nco
    
#     # ------------------------------------------------------------
#     # Visualization helpers
#     # ------------------------------------------------------------
#     @staticmethod
#     def _vis_to_numpy_image(
#         image: torch.Tensor,
#         mean: Optional[List[float]] = None,
#         std: Optional[List[float]] = None,
#     ) -> np.ndarray:
#         """Convert [C,H,W] tensor to RGB numpy image in [0,1]."""
#         x = image.detach().cpu().float().clone()
#         if x.ndim != 3:
#             raise ValueError("image must be [C,H,W].")

#         if mean is not None and std is not None:
#             mean_t = torch.tensor(mean, dtype=x.dtype).view(-1, 1, 1)
#             std_t = torch.tensor(std, dtype=x.dtype).view(-1, 1, 1)
#             if mean_t.shape[0] == 1 and x.shape[0] == 3:
#                 mean_t = mean_t.repeat(3, 1, 1)
#                 std_t = std_t.repeat(3, 1, 1)
#             if mean_t.shape[0] != x.shape[0]:
#                 raise ValueError(
#                     "mean/std channel count {} does not match image channels {}.".format(
#                         mean_t.shape[0], x.shape[0]
#                     )
#                 )
#             x = x * std_t + mean_t
#         else:
#             # Medical images are often not in [0,1]. For display only, use
#             # sample-wise min-max scaling when needed.
#             xmin = float(x.min().item())
#             xmax = float(x.max().item())
#             if xmin < 0.0 or xmax > 1.0:
#                 x = (x - x.min()) / (x.max() - x.min() + 1e-8)

#         if x.shape[0] == 1:
#             x = x.repeat(3, 1, 1)
#         elif x.shape[0] >= 3:
#             x = x[:3]
#         else:
#             raise ValueError("Only 1-channel or >=3-channel images are supported.")

#         x = x.clamp(0.0, 1.0)
#         return x.permute(1, 2, 0).numpy()

#     @staticmethod
#     def _vis_upsample_map(
#         x: torch.Tensor,
#         size: Tuple[int, int],
#         mode: str,
#     ) -> torch.Tensor:
#         """Upsample a [H,W] map to image resolution."""
#         x = x[None, None].float()
#         if mode in ("bilinear", "bicubic"):
#             x = F.interpolate(x, size=size, mode=mode, align_corners=False)
#         else:
#             x = F.interpolate(x, size=size, mode=mode)
#         return x[0, 0]

#     @staticmethod
#     def _vis_draw_patch_boxes(
#         ax,
#         patch_mask: np.ndarray,
#         image_h: int,
#         image_w: int,
#         edgecolor: str = "white",
#         linewidth: float = 1.4,
#     ) -> None:
#         """Draw feature-grid patch boxes on an image axis."""
#         hf, wf = patch_mask.shape
#         for rr in range(hf):
#             for cc in range(wf):
#                 if not bool(patch_mask[rr, cc]):
#                     continue
#                 x0 = cc * image_w / wf
#                 x1 = (cc + 1) * image_w / wf
#                 y0 = rr * image_h / hf
#                 y1 = (rr + 1) * image_h / hf
#                 ax.add_patch(
#                     Rectangle(
#                         (x0, y0),
#                         x1 - x0,
#                         y1 - y0,
#                         fill=False,
#                         edgecolor=edgecolor,
#                         linewidth=linewidth,
#                     )
#                 )

#     @staticmethod
#     def _vis_crop_patch(
#         pil_img: Image.Image,
#         patch_index: int,
#         hf: int,
#         wf: int,
#     ) -> Image.Image:
#         """Crop one layer4 spatial token from the original image."""
#         image_w, image_h = pil_img.size
#         rr = int(patch_index) // wf
#         cc = int(patch_index) % wf
#         y0 = int(round(rr * image_h / hf))
#         y1 = int(round((rr + 1) * image_h / hf))
#         x0 = int(round(cc * image_w / wf))
#         x1 = int(round((cc + 1) * image_w / wf))
#         return pil_img.crop((x0, y0, x1, y1))

#     def _compute_standard_cam(
#         self,
#         features: torch.Tensor,
#         classifier: nn.Linear,
#         class_ids: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Compute standard CAM for a GAP + linear classifier.

#         features: [B,D,Hf,Wf]
#         class_ids: [B]
#         return: [B,Hf,Wf] normalized to [0,1]
#         """
#         if classifier.weight.shape[1] != features.shape[1]:
#             raise ValueError(
#                 "classifier dim {} != layer4 dim {}.".format(
#                     classifier.weight.shape[1], features.shape[1]
#                 )
#             )
#         weight = classifier.weight.detach().to(
#             device=features.device,
#             dtype=features.dtype,
#         )
#         selected = weight.index_select(0, class_ids.long())
#         cam = (features * selected[:, :, None, None]).sum(dim=1)
#         cam = F.relu(cam)
#         cam = cam - cam.amin(dim=(1, 2), keepdim=True)
#         cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + self.eps)
#         return cam

#     @torch.no_grad()
#     def visualize_nuisance_regions(
#         self,
#         student_model: nn.Module,
#         inputs: torch.Tensor,
#         labels: Optional[torch.Tensor] = None,
#         save_dir: str = "./nuisance_visualization",
#         file_prefix: str = "sample",
#         mean: Optional[List[float]] = None,
#         std: Optional[List[float]] = None,
#         cam_target: str = "gt",
#         cam_threshold: float = 0.35,
#         foreground_masks: Optional[torch.Tensor] = None,
#         foreground_patch_threshold: float = 0.25,
#         max_images: Optional[int] = 8,
#         save_patch_crops: bool = True,
#         max_patches_per_image: int = 32,
#     ) -> List[Dict[str, object]]:
#         """
#         High-resolution visualization of the FINAL nuisance regions after
#         rel_num priority-relevant protection.

#         The selection order is exactly the same as Stage 2 training:
#             1) select Top-rel_num regions most similar to w_{y_i};
#             2) obtain raw nuisance-cluster mask;
#             3) remove priority-relevant regions from the raw nuisance mask;
#             4) visualize only the remaining FINAL nuisance regions.

#         Panels:
#             Original
#             Cluster IDs
#             Priority Relevant (Top-rel_num)
#             Final Nuisance
#             CAM + Final Nuisance
#             optional Segmentation + Final Nuisance
#         """
#         if labels is None:
#             raise ValueError(
#                 "labels are required because priority-relevant regions are "
#                 "defined using the ground-truth class direction w_y."
#             )
#         if cam_target not in ("pred", "gt"):
#             raise ValueError("cam_target must be 'pred' or 'gt'.")
#         if not (0.0 <= cam_threshold <= 1.0):
#             raise ValueError("cam_threshold must be in [0,1].")
#         if not (0.0 <= foreground_patch_threshold <= 1.0):
#             raise ValueError("foreground_patch_threshold must be in [0,1].")

#         Path(save_dir).mkdir(parents=True, exist_ok=True)
#         base = _unwrap_model(student_model)
#         was_training = base.training
#         base.eval()

#         try:
#             device = next(base.parameters()).device
#             inputs = inputs.to(device, non_blocking=True)
#             labels = labels.to(device, non_blocking=True).long()

#             logits, features = _capture_layer4_and_forward(student_model, inputs)
#             student_regions = self._regions_from_features(features)
#             classifier = _get_classifier(student_model)

#             priority_mask, gt_similarity = self._priority_relevant_mask(
#                 student_regions=student_regions,
#                 labels=labels,
#                 classifier=classifier,
#             )
#             cluster_ids, raw_spurious_mask, _ = self._stage2_region_partition(inputs)
#             cluster_ids = cluster_ids.to(device)
#             raw_spurious_mask = raw_spurious_mask.to(device)
#             final_spurious_mask, relevant_mask, protected_mask = (
#                 self._apply_priority_protection(
#                     raw_spurious_mask=raw_spurious_mask,
#                     priority_relevant_mask=priority_mask,
#                 )
#             )

#             target_ids = labels if cam_target == "gt" else logits.argmax(dim=1)
#             cams = self._compute_standard_cam(
#                 features=features,
#                 classifier=classifier,
#                 class_ids=target_ids,
#             )

#             b, _, image_h, image_w = inputs.shape
#             _, _, hf, wf = features.shape
#             if int(cluster_ids.shape[1]) != hf * wf:
#                 raise ValueError(
#                     "cluster_ids regions={} but student layer4 grid={}x{}.".format(
#                         cluster_ids.shape[1], hf, wf
#                     )
#                 )

#             fg_patch_fraction = None
#             fg_masks_cpu = None
#             if foreground_masks is not None:
#                 fg = foreground_masks.detach().float()
#                 if fg.ndim == 3:
#                     fg = fg.unsqueeze(1)
#                 elif fg.ndim == 4 and fg.shape[1] != 1:
#                     fg = fg[:, :1]
#                 if fg.ndim != 4:
#                     raise ValueError(
#                         "foreground_masks must be [B,H,W] or [B,1,H,W]."
#                     )
#                 if fg.shape[0] != b:
#                     raise ValueError("foreground_masks batch size must match inputs.")
#                 fg = fg.to(device)
#                 fg_patch_fraction = F.adaptive_avg_pool2d(fg, (hf, wf))[:, 0]
#                 fg_masks_cpu = fg[:, 0].detach().cpu()

#             n_show = b if max_images is None else min(int(max_images), b)
#             summaries: List[Dict[str, object]] = []

#             for i in range(n_show):
#                 img_np = self._vis_to_numpy_image(inputs[i], mean=mean, std=std)
#                 cluster_grid = cluster_ids[i].view(hf, wf).detach().cpu()
#                 priority_grid = priority_mask[i].view(hf, wf).detach().cpu()
#                 raw_spur_grid = raw_spurious_mask[i].view(hf, wf).detach().cpu()
#                 spur_grid = final_spurious_mask[i].view(hf, wf).detach().cpu()
#                 protected_grid = protected_mask[i].view(hf, wf).detach().cpu()
#                 cam_grid = cams[i].detach().cpu()

#                 nuisance_cam = cam_grid[spur_grid]
#                 num_raw = int(raw_spur_grid.sum().item())
#                 num_priority = int(priority_grid.sum().item())
#                 num_protected = int(protected_grid.sum().item())
#                 num_nuisance = int(spur_grid.sum().item())
#                 num_total = int(hf * wf)
#                 nuisance_patch_ratio = num_nuisance / max(num_total, 1)

#                 if nuisance_cam.numel() > 0:
#                     low_cam_ratio = float(
#                         (nuisance_cam < cam_threshold).float().mean().item()
#                     )
#                     high_cam_ratio = float(
#                         (nuisance_cam >= cam_threshold).float().mean().item()
#                     )
#                     mean_cam = float(nuisance_cam.mean().item())
#                 else:
#                     low_cam_ratio = 0.0
#                     high_cam_ratio = 0.0
#                     mean_cam = 0.0

#                 priority_sim = gt_similarity[i][priority_mask[i]].detach().cpu()
#                 mean_priority_sim = (
#                     float(priority_sim.mean().item()) if priority_sim.numel() > 0 else 0.0
#                 )

#                 seg_bg_ratio = None
#                 seg_fg_ratio = None
#                 mean_fg_fraction = None
#                 if fg_patch_fraction is not None:
#                     nuisance_fg = fg_patch_fraction[i].detach().cpu()[spur_grid]
#                     if nuisance_fg.numel() > 0:
#                         seg_bg_ratio = float(
#                             (nuisance_fg < foreground_patch_threshold)
#                             .float().mean().item()
#                         )
#                         seg_fg_ratio = 1.0 - seg_bg_ratio
#                         mean_fg_fraction = float(nuisance_fg.mean().item())
#                     else:
#                         seg_bg_ratio = 0.0
#                         seg_fg_ratio = 0.0
#                         mean_fg_fraction = 0.0

#                 priority_up = self._vis_upsample_map(
#                     priority_grid.float(), (image_h, image_w), "nearest"
#                 ).numpy()
#                 spur_up = self._vis_upsample_map(
#                     spur_grid.float(), (image_h, image_w), "nearest"
#                 ).numpy()
#                 cam_up = self._vis_upsample_map(
#                     cam_grid, (image_h, image_w), "bilinear"
#                 ).numpy()
#                 cluster_up = self._vis_upsample_map(
#                     cluster_grid.float(), (image_h, image_w), "nearest"
#                 ).numpy()

#                 ncols = 6 if fg_masks_cpu is not None else 5
#                 fig, axes = plt.subplots(
#                     1, ncols, figsize=(5.0 * ncols, 5.2), squeeze=False
#                 )
#                 axes = axes[0]

#                 axes[0].imshow(img_np)
#                 axes[0].set_title("Original", fontsize=13)
#                 axes[0].axis("off")

#                 cluster_plot = axes[1].imshow(
#                     cluster_up, cmap="tab20", interpolation="nearest"
#                 )
#                 axes[1].set_title("Cluster IDs", fontsize=13)
#                 axes[1].axis("off")
#                 fig.colorbar(cluster_plot, ax=axes[1], fraction=0.046, pad=0.04)

#                 axes[2].imshow(img_np)
#                 axes[2].imshow(
#                     priority_up,
#                     cmap="Greens",
#                     alpha=0.36,
#                     vmin=0.0,
#                     vmax=1.0,
#                     interpolation="nearest",
#                 )
#                 self._vis_draw_patch_boxes(
#                     axes[2], priority_grid.numpy(), image_h, image_w,
#                     edgecolor="lime", linewidth=1.8,
#                 )
#                 axes[2].set_title(
#                     "Priority Relevant (Top-{})\nmean cos(z,w_y)={:.3f}".format(
#                         self.rel_num, mean_priority_sim
#                     ),
#                     fontsize=12,
#                 )
#                 axes[2].axis("off")

#                 axes[3].imshow(img_np)
#                 axes[3].imshow(
#                     spur_up,
#                     cmap="Reds",
#                     alpha=0.40,
#                     vmin=0.0,
#                     vmax=1.0,
#                     interpolation="nearest",
#                 )
#                 self._vis_draw_patch_boxes(
#                     axes[3], spur_grid.numpy(), image_h, image_w,
#                     edgecolor="white", linewidth=1.6,
#                 )
#                 axes[3].set_title(
#                     "Final Nuisance\nraw={} - protected={} = {} | ratio={:.3f}".format(
#                         num_raw, num_protected, num_nuisance, nuisance_patch_ratio
#                     ),
#                     fontsize=12,
#                 )
#                 axes[3].axis("off")

#                 axes[4].imshow(img_np)
#                 axes[4].imshow(
#                     cam_up, cmap="jet", alpha=0.42, vmin=0.0, vmax=1.0
#                 )
#                 self._vis_draw_patch_boxes(
#                     axes[4], spur_grid.numpy(), image_h, image_w,
#                     edgecolor="white", linewidth=1.6,
#                 )
#                 axes[4].set_title(
#                     "CAM + Final Nuisance\nlow={:.3f}, high={:.3f}, mean={:.3f}".format(
#                         low_cam_ratio, high_cam_ratio, mean_cam
#                     ),
#                     fontsize=12,
#                 )
#                 axes[4].axis("off")

#                 if fg_masks_cpu is not None:
#                     fg_up = F.interpolate(
#                         fg_masks_cpu[i][None, None].float(),
#                         size=(image_h, image_w),
#                         mode="nearest",
#                     )[0, 0].numpy()
#                     axes[5].imshow(img_np)
#                     axes[5].imshow(
#                         fg_up, cmap="Greens", alpha=0.30, vmin=0.0, vmax=1.0
#                     )
#                     self._vis_draw_patch_boxes(
#                         axes[5], spur_grid.numpy(), image_h, image_w,
#                         edgecolor="white", linewidth=1.6,
#                     )
#                     axes[5].set_title(
#                         "Segmentation + Final Nuisance\nsegBG={:.3f}, segFG={:.3f}".format(
#                             seg_bg_ratio, seg_fg_ratio
#                         ),
#                         fontsize=12,
#                     )
#                     axes[5].axis("off")

#                 pred_id = int(logits[i].argmax().item())
#                 gt_id = int(labels[i].item())
#                 fig.suptitle(
#                     "idx={} | gt={} | pred={} | rel_num={} | raw_nui={} | protected={} | final_nui={}".format(
#                         i, gt_id, pred_id, self.rel_num,
#                         num_raw, num_protected, num_nuisance,
#                     ),
#                     fontsize=14,
#                 )
#                 fig.tight_layout(rect=[0, 0, 1, 0.95])

#                 figure_path = os.path.join(
#                     save_dir, "{}_{:03d}.png".format(file_prefix, i)
#                 )
#                 fig.savefig(
#                     figure_path,
#                     dpi=600,
#                     bbox_inches="tight",
#                     pad_inches=0.04,
#                 )
#                 plt.close(fig)

#                 patch_paths: List[str] = []
#                 patch_dir = None
#                 if save_patch_crops:
#                     patch_dir = os.path.join(
#                         save_dir, "{}_{:03d}_final_nuisance_patches".format(file_prefix, i)
#                     )
#                     Path(patch_dir).mkdir(parents=True, exist_ok=True)
#                     pil_img = Image.fromarray((img_np * 255.0).astype(np.uint8))
#                     nuisance_indices = (
#                         final_spurious_mask[i]
#                         .nonzero(as_tuple=False)
#                         .squeeze(1)
#                         .detach().cpu().tolist()
#                     )[: int(max_patches_per_image)]

#                     for order, patch_index in enumerate(nuisance_indices):
#                         crop = self._vis_crop_patch(
#                             pil_img=pil_img,
#                             patch_index=int(patch_index),
#                             hf=hf,
#                             wf=wf,
#                         )
#                         cid = int(cluster_ids[i, patch_index].item())
#                         rr = int(patch_index) // wf
#                         cc = int(patch_index) % wf
#                         crop_path = os.path.join(
#                             patch_dir,
#                             "patch_{:02d}_cluster_{}_r{}_c{}.png".format(
#                                 order, cid, rr, cc
#                             ),
#                         )
#                         crop.save(crop_path, format="PNG", optimize=True)
#                         patch_paths.append(crop_path)

#                 summary = {
#                     "index": i,
#                     "gt": gt_id,
#                     "pred": pred_id,
#                     "rel_num": int(self.rel_num),
#                     "num_priority_relevant": num_priority,
#                     "num_raw_nuisance_regions": num_raw,
#                     "num_protected_from_nuisance": num_protected,
#                     "num_nuisance_regions": num_nuisance,
#                     "num_relevant_regions": int(relevant_mask[i].sum().item()),
#                     "nuisance_patch_ratio": float(nuisance_patch_ratio),
#                     "mean_priority_gt_similarity": mean_priority_sim,
#                     "nuisance_low_cam_ratio": low_cam_ratio,
#                     "nuisance_high_cam_ratio": high_cam_ratio,
#                     "nuisance_mean_cam": mean_cam,
#                     "figure_path": figure_path,
#                     "patch_dir": patch_dir,
#                     "patch_paths": patch_paths,
#                 }
#                 if seg_bg_ratio is not None:
#                     summary.update({
#                         "nuisance_seg_background_ratio": seg_bg_ratio,
#                         "nuisance_seg_foreground_ratio": seg_fg_ratio,
#                         "nuisance_mean_foreground_fraction": mean_fg_fraction,
#                     })
#                 summaries.append(summary)

#             return summaries
#         finally:
#             if was_training:
#                 base.train()
#             else:
#                 base.eval()

#     @torch.no_grad()
#     def visualize_nuisance_cluster_exemplars(
#         self,
#         data_loader,
#         device,
#         student_model: Optional[nn.Module] = None,
#         save_dir: str = "./nuisance_cluster_exemplars",
#         file_prefix: str = "cluster",
#         mean: Optional[List[float]] = None,
#         std: Optional[List[float]] = None,
#         target_clusters: Optional[List[int]] = None,
#         max_patches_per_cluster: int = 64,
#         max_batches: Optional[int] = 100,
#         columns: int = 8,
#     ) -> Dict[int, str]:
#         """
#         Save exemplar montages using FINAL nuisance patches only.

#         Patches protected by Top-rel_num similarity to w_y are explicitly
#         excluded even if they belong to a selected nuisance cluster.

#         student_model:
#             current model used to select priority-relevant patches. If None,
#             the frozen Stage-1 assignment model is used for both priority
#             selection and cluster assignment.
#         """
#         if self._assignment_model is None:
#             raise RuntimeError(
#                 "No frozen assignment model. Run discover(..., "
#                 "make_assignment_snapshot=True) first."
#             )
#         if target_clusters is None:
#             target_clusters = list(self.top_spurious_clusters)
#         target_clusters = [int(c) for c in target_clusters]
#         if len(target_clusters) == 0:
#             raise RuntimeError("No nuisance clusters are available.")

#         if student_model is None:
#             student_model = self._assignment_model
#         base = _unwrap_model(student_model)
#         student_was_training = base.training
#         base.eval()

#         Path(save_dir).mkdir(parents=True, exist_ok=True)
#         buckets = {c: [] for c in target_clusters}

#         try:
#             for batch_idx, batch in enumerate(data_loader):
#                 if max_batches is not None and batch_idx >= int(max_batches):
#                     break
#                 if not isinstance(batch, (tuple, list)) or len(batch) < 2:
#                     raise TypeError("data_loader must return (inputs, labels, ...).")

#                 inputs = batch[0]
#                 labels = batch[1]
#                 if isinstance(inputs, (tuple, list)):
#                     inputs = inputs[0]
#                 inputs = inputs.to(device, non_blocking=True)
#                 labels = labels.to(device, non_blocking=True).long()

#                 _, features = _capture_layer4_and_forward(student_model, inputs)
#                 student_regions = self._regions_from_features(features)
#                 classifier = _get_classifier(student_model)
#                 priority_mask, _ = self._priority_relevant_mask(
#                     student_regions, labels, classifier
#                 )

#                 cluster_ids, raw_spurious_mask, _ = self._stage2_region_partition(inputs)
#                 cluster_ids = cluster_ids.to(device)
#                 raw_spurious_mask = raw_spurious_mask.to(device)
#                 final_spurious_mask, _, _ = self._apply_priority_protection(
#                     raw_spurious_mask, priority_mask
#                 )

#                 b, _, hf, wf = features.shape
#                 if int(cluster_ids.shape[1]) != hf * wf:
#                     raise ValueError(
#                         "cluster_ids={} but student grid={}x{}.".format(
#                             cluster_ids.shape[1], hf, wf
#                         )
#                     )

#                 for i in range(b):
#                     img_np = self._vis_to_numpy_image(inputs[i], mean=mean, std=std)
#                     pil_img = Image.fromarray((img_np * 255.0).astype(np.uint8))
#                     ids_i = cluster_ids[i]
#                     final_i = final_spurious_mask[i]

#                     for cid in target_clusters:
#                         if len(buckets[cid]) >= int(max_patches_per_cluster):
#                             continue
#                         patch_indices = (
#                             (ids_i.eq(cid) & final_i)
#                             .nonzero(as_tuple=False)
#                             .squeeze(1)
#                             .detach().cpu().tolist()
#                         )
#                         for patch_index in patch_indices:
#                             if len(buckets[cid]) >= int(max_patches_per_cluster):
#                                 break
#                             crop = self._vis_crop_patch(
#                                 pil_img=pil_img,
#                                 patch_index=int(patch_index),
#                                 hf=hf,
#                                 wf=wf,
#                             )
#                             buckets[cid].append(crop.copy())

#                 if all(
#                     len(v) >= int(max_patches_per_cluster)
#                     for v in buckets.values()
#                 ):
#                     break
#         finally:
#             if student_was_training:
#                 base.train()
#             else:
#                 base.eval()

#         output_paths: Dict[int, str] = {}
#         for cid, crops in buckets.items():
#             if len(crops) == 0:
#                 continue
#             n = len(crops)
#             ncols = min(int(columns), n)
#             nrows = int(math.ceil(n / ncols))
#             fig, axes = plt.subplots(
#                 nrows, ncols,
#                 figsize=(2.3 * ncols, 2.3 * nrows),
#                 squeeze=False,
#             )
#             for ax in axes.ravel():
#                 ax.axis("off")
#             for j, crop in enumerate(crops):
#                 rr = j // ncols
#                 cc = j % ncols
#                 axes[rr, cc].imshow(crop)
#                 axes[rr, cc].axis("off")
#             fig.suptitle(
#                 "Final nuisance cluster {} exemplars after Top-{} protection (n={})".format(
#                     cid, self.rel_num, n
#                 ),
#                 fontsize=15,
#             )
#             fig.tight_layout(rect=[0, 0, 1, 0.97])
#             path = os.path.join(save_dir, "{}_{}.png".format(file_prefix, cid))
#             fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.04)
#             plt.close(fig)
#             output_paths[cid] = path
#         return output_paths

#     @torch.no_grad()
#     def evaluate_nuisance_cam_overlap(
#         self,
#         student_model: nn.Module,
#         data_loader,
#         device,
#         cam_target: str = "gt",
#         cam_threshold: float = 0.35,
#         max_batches: Optional[int] = None,
#     ) -> Dict[str, float]:
#         """Dataset-level CAM statistics for FINAL nuisance patches only."""
#         if cam_target not in ("pred", "gt"):
#             raise ValueError("cam_target must be 'pred' or 'gt'.")

#         base = _unwrap_model(student_model)
#         was_training = base.training
#         base.eval()
#         total_raw = 0
#         total_protected = 0
#         total_nuisance = 0
#         total_low_cam = 0
#         total_high_cam = 0
#         sum_cam = 0.0

#         try:
#             for batch_idx, batch in enumerate(data_loader):
#                 if max_batches is not None and batch_idx >= int(max_batches):
#                     break
#                 if not isinstance(batch, (tuple, list)) or len(batch) < 2:
#                     raise TypeError("data_loader must return (inputs, labels, ...).")

#                 inputs = batch[0]
#                 labels = batch[1]
#                 if isinstance(inputs, (tuple, list)):
#                     inputs = inputs[0]
#                 inputs = inputs.to(device, non_blocking=True)
#                 labels = labels.to(device, non_blocking=True).long()

#                 logits, features = _capture_layer4_and_forward(student_model, inputs)
#                 student_regions = self._regions_from_features(features)
#                 classifier = _get_classifier(student_model)
#                 priority_mask, _ = self._priority_relevant_mask(
#                     student_regions, labels, classifier
#                 )
#                 cluster_ids, raw_spurious_mask, _ = self._stage2_region_partition(inputs)
#                 raw_spurious_mask = raw_spurious_mask.to(device)
#                 final_spurious_mask, _, protected_mask = self._apply_priority_protection(
#                     raw_spurious_mask, priority_mask
#                 )

#                 target_ids = labels if cam_target == "gt" else logits.argmax(dim=1)
#                 cams = self._compute_standard_cam(features, classifier, target_ids)
#                 b, _, hf, wf = features.shape
#                 if int(cluster_ids.shape[1]) != hf * wf:
#                     raise ValueError(
#                         "Student feature grid and assignment region count do not match."
#                     )
#                 spur_grid = final_spurious_mask.view(b, hf, wf)
#                 nuisance_cam = cams[spur_grid]

#                 total_raw += int(raw_spurious_mask.sum().item())
#                 total_protected += int(protected_mask.sum().item())
#                 if nuisance_cam.numel() == 0:
#                     continue
#                 total_nuisance += int(nuisance_cam.numel())
#                 total_low_cam += int((nuisance_cam < cam_threshold).sum().item())
#                 total_high_cam += int((nuisance_cam >= cam_threshold).sum().item())
#                 sum_cam += float(nuisance_cam.sum().item())
#         finally:
#             if was_training:
#                 base.train()
#             else:
#                 base.eval()

#         if total_nuisance == 0:
#             return {
#                 "num_raw_nuisance_patches": float(total_raw),
#                 "num_protected_patches": float(total_protected),
#                 "num_final_nuisance_patches": 0.0,
#                 "low_cam_ratio": 0.0,
#                 "high_cam_ratio": 0.0,
#                 "mean_nuisance_cam": 0.0,
#             }
#         return {
#             "num_raw_nuisance_patches": float(total_raw),
#             "num_protected_patches": float(total_protected),
#             "num_final_nuisance_patches": float(total_nuisance),
#             "low_cam_ratio": total_low_cam / total_nuisance,
#             "high_cam_ratio": total_high_cam / total_nuisance,
#             "mean_nuisance_cam": sum_cam / total_nuisance,
#         }

#     @torch.no_grad()
#     def evaluate_nuisance_segmentation_overlap(
#         self,
#         data_loader,
#         device,
#         student_model: Optional[nn.Module] = None,
#         mask_index: int = 2,
#         foreground_patch_threshold: float = 0.25,
#         max_batches: Optional[int] = None,
#     ) -> Dict[str, float]:
#         """Segmentation overlap for FINAL nuisance patches after Top-rel_num protection."""
#         if not (0.0 <= foreground_patch_threshold <= 1.0):
#             raise ValueError("foreground_patch_threshold must be in [0,1].")
#         if student_model is None:
#             if self._assignment_model is None:
#                 raise RuntimeError("No model available for priority-region selection.")
#             student_model = self._assignment_model

#         base = _unwrap_model(student_model)
#         was_training = base.training
#         base.eval()
#         total_raw = 0
#         total_protected = 0
#         total_nuisance = 0
#         total_bg = 0
#         total_fg = 0
#         total_fg_fraction = 0.0

#         try:
#             for batch_idx, batch in enumerate(data_loader):
#                 if max_batches is not None and batch_idx >= int(max_batches):
#                     break
#                 if not isinstance(batch, (tuple, list)) or len(batch) <= max(mask_index, 1):
#                     raise TypeError(
#                         "data_loader must return (inputs, labels, ..., mask, ...)."
#                     )
#                 inputs = batch[0]
#                 labels = batch[1]
#                 masks = batch[mask_index]
#                 if isinstance(inputs, (tuple, list)):
#                     inputs = inputs[0]
#                 inputs = inputs.to(device, non_blocking=True)
#                 labels = labels.to(device, non_blocking=True).long()
#                 masks = masks.to(device, non_blocking=True).float()
#                 if masks.ndim == 3:
#                     masks = masks.unsqueeze(1)
#                 elif masks.ndim == 4 and masks.shape[1] != 1:
#                     masks = masks[:, :1]
#                 if masks.ndim != 4:
#                     raise ValueError("foreground masks must be [B,H,W] or [B,1,H,W].")

#                 _, features = _capture_layer4_and_forward(student_model, inputs)
#                 student_regions = self._regions_from_features(features)
#                 classifier = _get_classifier(student_model)
#                 priority_mask, _ = self._priority_relevant_mask(
#                     student_regions, labels, classifier
#                 )
#                 cluster_ids, raw_spurious_mask, _ = self._stage2_region_partition(inputs)
#                 raw_spurious_mask = raw_spurious_mask.to(device)
#                 final_spurious_mask, _, protected_mask = self._apply_priority_protection(
#                     raw_spurious_mask, priority_mask
#                 )

#                 b, _, hf, wf = features.shape
#                 if int(cluster_ids.shape[1]) != hf * wf:
#                     raise ValueError("Assignment region count does not match feature grid.")
#                 fg_fraction = F.adaptive_avg_pool2d(masks, (hf, wf))[:, 0]
#                 spur_grid = final_spurious_mask.view(b, hf, wf)
#                 nuisance_fg = fg_fraction[spur_grid]

#                 total_raw += int(raw_spurious_mask.sum().item())
#                 total_protected += int(protected_mask.sum().item())
#                 if nuisance_fg.numel() == 0:
#                     continue
#                 total_nuisance += int(nuisance_fg.numel())
#                 total_bg += int(
#                     (nuisance_fg < foreground_patch_threshold).sum().item()
#                 )
#                 total_fg += int(
#                     (nuisance_fg >= foreground_patch_threshold).sum().item()
#                 )
#                 total_fg_fraction += float(nuisance_fg.sum().item())
#         finally:
#             if was_training:
#                 base.train()
#             else:
#                 base.eval()

#         if total_nuisance == 0:
#             return {
#                 "num_raw_nuisance_patches": float(total_raw),
#                 "num_protected_patches": float(total_protected),
#                 "num_final_nuisance_patches": 0.0,
#                 "seg_background_ratio": 0.0,
#                 "seg_foreground_ratio": 0.0,
#                 "mean_foreground_fraction": 0.0,
#             }
#         return {
#             "num_raw_nuisance_patches": float(total_raw),
#             "num_protected_patches": float(total_protected),
#             "num_final_nuisance_patches": float(total_nuisance),
#             "seg_background_ratio": total_bg / total_nuisance,
#             "seg_foreground_ratio": total_fg / total_nuisance,
#             "mean_foreground_fraction": total_fg_fraction / total_nuisance,
#         }

#     # ------------------------------------------------------------
#     # Stage 2 public forward
#     # ------------------------------------------------------------
#     def forward(
#         self,
#         student_model: nn.Module,
#         inputs: torch.Tensor,
#         labels: torch.Tensor,
#     ) -> RaVLOutput:
#         """
#         Stage-2 forward with priority-relevant protection + L_R/L_A.

#         Order is intentionally fixed:
#             1) current student regions -> Top-rel_num regions most similar to w_y
#             2) frozen cluster assignment -> raw nuisance-cluster regions
#             3) remove protected Top-rel_num regions from raw nuisance mask
#             4) optimize L_R + L_A on the resulting relevant/nuisance partition
#         """
#         if self.top_spurious_cluster is None:
#             raise RuntimeError(
#                 "Run ravl.discover(model, reference_loader) before Stage 2."
#             )

#         logits, student_features = _capture_layer4_and_forward(
#             student_model,
#             inputs,
#         )
#         if logits.shape[1] != self.num_classes:
#             raise ValueError(
#                 "Expected {} classes, got {}.".format(
#                     self.num_classes,
#                     logits.shape[1],
#                 )
#             )

#         student_regions = self._regions_from_features(student_features)
#         classifier = _get_classifier(student_model)

#         # Highest-priority task-relevant regions are selected FIRST.
#         priority_relevant_mask, _ = self._priority_relevant_mask(
#             student_regions=student_regions,
#             labels=labels,
#             classifier=classifier,
#         )

#         # Then obtain raw nuisance-cluster membership from the frozen Stage-1 space.
#         cluster_ids, raw_spurious_mask, _ = self._stage2_region_partition(inputs)
#         raw_spurious_mask = raw_spurious_mask.to(priority_relevant_mask.device)
#         cluster_ids = cluster_ids.to(priority_relevant_mask.device)

#         # rel_num regions can never be nuisance.
#         spurious_mask, relevant_mask, protected_from_nuisance = (
#             self._apply_priority_protection(
#                 raw_spurious_mask=raw_spurious_mask,
#                 priority_relevant_mask=priority_relevant_mask,
#             )
#         )

#         # loss_region, loss_R, loss_A, valid_images = self._region_aware_loss(
#         #     student_regions=student_regions,
#         #     labels=labels,
#         #     spurious_mask=spurious_mask,
#         #     relevant_mask=relevant_mask,
#         #     classifier=classifier,
#         # )
        
#         loss_region = self._nuisance_class_orthogonal_loss(
#             student_regions=student_regions,
#             spurious_mask=spurious_mask,
#             classifier=classifier,
#         )

#         return RaVLOutput(
#             loss_region=loss_region,
#             loss_R=torch.tensor(0),
#             loss_A=torch.tensor(0),
#             logits=logits,
#             raw_spurious_mask=raw_spurious_mask.detach(),
#             spurious_mask=spurious_mask.detach(),
#             priority_relevant_mask=priority_relevant_mask.detach(),
#             relevant_mask=relevant_mask.detach(),
#             cluster_ids=cluster_ids.detach(),
#             num_raw_spurious_regions=int(raw_spurious_mask.sum().item()),
#             num_protected_regions=int(protected_from_nuisance.sum().item()),
#             num_spurious_regions=int(spurious_mask.sum().item()),
#             num_relevant_regions=int(relevant_mask.sum().item()),
#             num_valid_images=torch.tensor(0),
#         )

#     __call__ = forward

#     def combine_with_classification_loss(
#         self,
#         classification_loss: torch.Tensor,
#         output: RaVLOutput,
#         lambda_cl: Optional[float] = None,
#     ) -> torch.Tensor:
#         """
#         Keep the original public helper interface. The region regularizer is
#         the protected bidirectional loss L_R + L_A.
#         """
#         if lambda_cl is None:
#             lambda_cl = self.lambda_cl
#         return (1-float(lambda_cl))*classification_loss + float(lambda_cl)*output.loss_region

#     # ------------------------------------------------------------
#     # Persistence for the discovered clusters
#     # ------------------------------------------------------------
#     def save_discovery(self, path: str) -> None:
#         if self.medoids_raw is None or self.top_spurious_cluster is None:
#             raise RuntimeError("Nothing to save; run discover first.")

#         torch.save(
#             {
#                 "num_classes": self.num_classes,
#                 "region_grid": self.region_grid,
#                 "temperature": self.temperature,
#                 "lambda_cl": self.lambda_cl,
#                 "influence_threshold": self.influence_threshold,
#                 "medoids_raw": self.medoids_raw,
#                 "medoids_norm": self.medoids_norm,
#                 "top_spurious_cluster": self.top_spurious_cluster,
#             "top_spurious_clusters": self.top_spurious_clusters,
#             "num_spurious_clusters": self.num_spurious_clusters,
#                 "ranked_clusters": self.ranked_clusters,
#                 "discovery_result": self.discovery_result,
#             },
#             path,
#         )

#     def load_discovery(
#         self,
#         path: str,
#         model_for_assignment: Optional[nn.Module] = None,
#         device=None,
#     ) -> None:
#         state = torch.load(path, map_location="cpu")

#         if int(state["num_classes"]) != self.num_classes:
#             raise ValueError(
#                 "Saved num_classes={} but module num_classes={}.".format(
#                     state["num_classes"],
#                     self.num_classes,
#                 )
#             )

#         if int(state["region_grid"]) != self.region_grid:
#             raise ValueError(
#                 "Saved region_grid={} but module region_grid={}.".format(
#                     state["region_grid"],
#                     self.region_grid,
#                 )
#             )

#         self.medoids_raw = state["medoids_raw"].float()
#         self.medoids_norm = state["medoids_norm"].float()
#         self.top_spurious_cluster = int(
#             state["top_spurious_cluster"]
#         )
#         self.top_spurious_clusters = state.get(
#             "top_spurious_clusters",
#             [self.top_spurious_cluster]
#         )
#         self.rel_num = int(state.get("rel_num", self.rel_num))
#         self.ranked_clusters = list(state["ranked_clusters"])
#         self.discovery_result = state.get("discovery_result", None)

#         if model_for_assignment is not None:
#             self.prepare_stage2_assignment_model(
#                 model_for_assignment,
#                 device=device,
#             )


# # ================================================================
# # Tiny smoke test
# # ================================================================
# if __name__ == "__main__":
#     from torch.utils.data import DataLoader, TensorDataset

#     class TinyResNet(nn.Module):
#         def __init__(self, num_classes=3):
#             super().__init__()
#             self.stem = nn.Sequential(
#                 nn.Conv2d(3, 16, 3, padding=1),
#                 nn.ReLU(),
#                 nn.AdaptiveAvgPool2d((7, 7)),
#             )
#             self.layer4 = nn.Sequential(
#                 nn.Conv2d(16, 32, 3, padding=1),
#                 nn.ReLU(),
#             )
#             self.fc = nn.Linear(32, num_classes)

#         def forward(self, x):
#             x = self.stem(x)
#             x = self.layer4(x)
#             z = x.mean(dim=(2, 3))
#             return self.fc(z)

#     torch.manual_seed(7)
#     device = torch.device(
#         "cuda" if torch.cuda.is_available() else "cpu"
#     )

#     model = TinyResNet(num_classes=3).to(device)

#     ravl = RaVLResNet(
#         num_classes=3,
#         region_grid=2,
#         temperature=0.07,
#         rel_num=4,
#         influence_threshold=0.0,  # smoke-test only
#         k_min_factor=2,
#         k_max_factor=2,
#         max_cluster_regions=200,
#         silhouette_sample_size=200,
#         random_seed=7,
#     )

#     # Synthetic reference set.
#     ref_x = torch.randn(60, 3, 32, 32)
#     ref_y = torch.randint(0, 3, (60,))
#     ref_loader = DataLoader(
#         TensorDataset(ref_x, ref_y),
#         batch_size=12,
#         shuffle=False,
#     )

#     # Random model may still have G=0. We test the full clustering path but
#     # permit H=0 for this synthetic smoke test.
#     try:
#         discovery = ravl.discover(
#             model=model,
#             reference_loader=ref_loader,
#             device=device,
#             verbose=False,
#         )
#     except RuntimeError:
#         # For a random model, if all G/H degenerates, manually select the first
#         # discovered cluster is not possible if discovery did not commit state.
#         # Re-run is not necessary for syntax/gradient smoke testing; set a small
#         # fixed bank from reference features.
#         model.eval()
#         with torch.no_grad():
#             _, feat = _capture_layer4_and_forward(
#                 model,
#                 ref_x[:12].to(device),
#             )
#             regs = ravl._regions_from_features(feat)
#             flat = regs.reshape(-1, regs.shape[-1])
#             medoid_ids, _ = ravl._fit_kmedoids_cosine(
#                 flat,
#                 k=3,
#                 seed=7,
#             )
#             med = flat.index_select(0, medoid_ids)
#             ravl.medoids_raw = med.detach().cpu()
#             ravl.medoids_norm = F.normalize(
#                 med,
#                 dim=1,
#             ).detach().cpu()
#             ravl.top_spurious_cluster = 0
#             ravl.top_spurious_clusters = [0]
#             ravl.ranked_clusters = [0]
#             ravl.prepare_stage2_assignment_model(
#                 model,
#                 device=device,
#             )

#     model.train()

#     x = torch.randn(8, 3, 32, 32, device=device)
#     y = torch.randint(0, 3, (8,), device=device)

#     out = ravl(
#         student_model=model,
#         inputs=x,
#         labels=y,
#     )

#     loss_cls = F.cross_entropy(out.logits, y)
#     loss = ravl.combine_with_classification_loss(
#         loss_cls,
#         out,
#     )

#     optimizer = torch.optim.SGD(
#         model.parameters(),
#         lr=1e-3,
#     )
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

#     print("Priority-rel_num + L_R/L_A smoke test passed.")
#     print(out.statistics())



##################################################

# from __future__ import annotations

# from dataclasses import dataclass
# import copy
# import math
# import os
# from pathlib import Path
# import random
# from typing import Dict, List, Optional, Tuple

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# import numpy as np
# from PIL import Image
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# @dataclass
# class RaVLDiscoveryResult:
#     best_k: int
#     best_silhouette: float
#     top_spurious_cluster: int
#     ranked_clusters: List[int]
#     influence_scores: Dict[int, float]
#     performance_gaps: Dict[int, float]
#     per_class_gaps: Dict[int, Dict[int, float]]
#     per_class_influence: Dict[int, Dict[int, float]]
#     num_images: int
#     num_regions: int

#     def summary(self) -> Dict[str, object]:
#         return {
#             "best_k": self.best_k,
#             "best_silhouette": self.best_silhouette,
#             "top_spurious_cluster": self.top_spurious_cluster,
#             "ranked_clusters": self.ranked_clusters,
#             "influence_scores": self.influence_scores,
#             "performance_gaps": self.performance_gaps,
#             "num_images": self.num_images,
#             "num_regions": self.num_regions,
#         }


# @dataclass
# class RaVLOutput:
#     loss_region: torch.Tensor
#     loss_nui: torch.Tensor
#     logits: torch.Tensor
#     spurious_mask: torch.Tensor
#     relevant_mask: torch.Tensor
#     cluster_ids: torch.Tensor
#     num_spurious_regions: int
#     num_relevant_regions: int
#     num_valid_images: int

#     # Backward-compatible aliases. L_R/L_A are no longer used in Stage 2.
#     # Keeping these properties avoids breaking old logging code immediately.
#     @property
#     def loss_R(self) -> torch.Tensor:
#         return self.loss_nui

#     @property
#     def loss_A(self) -> torch.Tensor:
#         return self.loss_nui.new_zeros(())

#     def statistics(self) -> Dict[str, float]:
#         return {
#             "loss_region": float(self.loss_region.detach().item()),
#             "loss_nui": float(self.loss_nui.detach().item()),
#             # backward-compatible logging keys
#             "loss_R": float(self.loss_R.detach().item()),
#             "loss_A": float(self.loss_A.detach().item()),
#             "num_spurious_regions": float(self.num_spurious_regions),
#             "num_relevant_regions": float(self.num_relevant_regions),
#             "num_valid_images": float(self.num_valid_images),
#         }


# def _unwrap_model(model: nn.Module) -> nn.Module:
#     return model.module if hasattr(model, "module") else model


# def _unwrap_tensor(x, name: str) -> torch.Tensor:
#     if torch.is_tensor(x):
#         return x
#     if isinstance(x, (tuple, list)) and len(x) > 0 and torch.is_tensor(x[0]):
#         return x[0]
#     raise TypeError(
#         "{} must be Tensor or tuple/list whose first item is Tensor.".format(name)
#     )


# def _get_classifier(model: nn.Module) -> nn.Linear:
#     base = _unwrap_model(model)
#     candidates = []

#     if hasattr(base, "get_classifier"):
#         try:
#             candidates.append(base.get_classifier())
#         except Exception:
#             pass

#     for name in ("fc", "head", "classifier"):
#         if hasattr(base, name):
#             candidates.append(getattr(base, name))

#     for candidate in candidates:
#         if isinstance(candidate, nn.Linear):
#             return candidate
#         if isinstance(candidate, nn.Sequential):
#             for sub in reversed(candidate):
#                 if isinstance(sub, nn.Linear):
#                     return sub

#     raise AttributeError("Unable to locate final nn.Linear classifier.")


# def _capture_layer4_and_forward(
#     model: nn.Module,
#     inputs: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     base = _unwrap_model(model)

#     if not hasattr(base, "layer4"):
#         raise AttributeError(
#             "The model must be ResNet-like and contain model.layer4."
#         )

#     holder = {}

#     def hook(_module, _inputs, output):
#         holder["features"] = _unwrap_tensor(output, "layer4 output")

#     handle = base.layer4.register_forward_hook(hook)

#     try:
#         logits = _unwrap_tensor(model(inputs), "model(inputs)")
#     finally:
#         handle.remove()

#     if "features" not in holder:
#         raise RuntimeError("Failed to capture layer4 features.")

#     features = holder["features"]

#     if features.ndim != 4:
#         raise ValueError(
#             "layer4 must be [B,D,H,W], got {}".format(tuple(features.shape))
#         )

#     if logits.ndim != 2:
#         raise ValueError(
#             "logits must be [B,C], got {}".format(tuple(logits.shape))
#         )

#     return logits, features



# class RaVLResNet(object):
#     """
#     RaVL-style discovery + mitigation migrated from the NeurIPS 2024 RaVL
#     algorithm to a supervised ResNet classifier.

#     What is kept from the original RaVL:
#       Stage 1:
#         1) labeled reference/validation set
#         2) local candidate regions from correctly predicted reference samples
#         3) K-Medoids with cosine distance on the correct-only region pool
#         4) choose K by Silhouette score over [2|Y|, 5|Y|]
#         5) class-conditional MI I(S_k; Y_hat | Y=c)
#         6) harmful error-increase gating
#         7) class-balanced conditional nuisance score N_k
#         8) select the top-ranked nuisance cluster(s)

#       Stage 2:
#         1) keep the Stage-1 clustering model fixed
#         2) split regions into R_i^s and R_i^r
#         3) remove L_R and L_A
#         4) make nuisance-region class predictions maximally uncertain
#         5) minimize KL(q(Y|z_s) || Uniform(C))

#     Necessary modality substitutions:
#       - VLM text class embedding g(y) -> normalized frozen FC class direction w_y
#       - RoI candidate regions -> equal grid regions pooled from ResNet layer4
#       - paired-text assigned label y_hat -> supervised ground-truth class label y

#     Important:
#       - The clustering medoids and the Stage-1 assignment encoder remain fixed.
#       - The student/model training strategy is external to this class.
#       - This class introduces no learnable parameters.
#     """

#     def __init__(
#         self,
#         num_classes: int,
#         region_grid: int = 3,
#         temperature: float = 0.07,
#         lambda_cl: float = 0.80,
#         influence_threshold: float = 0.25,
#         num_spurious_clusters: int = 1,
#         k_min_factor: int = 2,
#         k_max_factor: int = 5,
#         kmedoids_iterations: int = 30,
#         max_cluster_regions: Optional[int] = 20000,
#         silhouette_sample_size: int = 3000,
#         assignment_chunk_size: int = 8192,
#         random_seed: int = 0,
#     ) -> None:
#         if num_classes < 2:
#             raise ValueError("num_classes must be >= 2")
#         if region_grid < 1:
#             raise ValueError("region_grid must be >= 1")
#         if temperature <= 0:
#             raise ValueError("temperature must be > 0")
#         if not 0 <= lambda_cl <= 1:
#             raise ValueError("lambda_cl must be in [0,1]")
#         if influence_threshold < 0:
#             raise ValueError("influence_threshold must be >= 0")
#         if k_min_factor < 1 or k_max_factor < k_min_factor:
#             raise ValueError("invalid cluster-count factors")

#         self.num_classes = int(num_classes)
#         self.region_grid = int(region_grid)
#         self.temperature = float(temperature)
#         self.lambda_cl = float(lambda_cl)
#         self.influence_threshold = float(influence_threshold)
#         self.num_spurious_clusters = int(num_spurious_clusters)
#         self.k_min_factor = int(k_min_factor)
#         self.k_max_factor = int(k_max_factor)
#         self.kmedoids_iterations = int(kmedoids_iterations)
#         self.max_cluster_regions = max_cluster_regions
#         self.silhouette_sample_size = int(silhouette_sample_size)
#         self.assignment_chunk_size = int(assignment_chunk_size)
#         self.random_seed = int(random_seed)
#         self.eps = 1e-8

#         self.medoids_raw = None
#         self.medoids_norm = None
#         self.top_spurious_cluster = None
#         self.top_spurious_clusters = []
#         self.ranked_clusters = []
#         self.discovery_result = None

#         # Frozen copy of the Stage-1 model.
#         # It is used only for assigning Stage-2 regions to the fixed clusters.
#         self._assignment_model = None
#         self._assignment_device = None

#     # ------------------------------------------------------------
#     # Region construction
#     # ------------------------------------------------------------
#     def _regions_from_features(
#             self,
#             features: torch.Tensor,
#     ):
#         """
#         Directly use layer4 spatial tokens as regions.

#         Input:
#             features:
#                 [B,C,H,W]

#         Output:
#             regions:
#                 [B,H*W,C]

#         For ResNet50 layer4:
#             [B,2048,7,7]
#             ->
#             [B,49,2048]
#         """

#         B, C, H, W = features.shape
#         regions = (
#             features
#             .flatten(2)
#             .transpose(1, 2)
#         )
#         return regions

#     # ------------------------------------------------------------
#     # Cosine K-Medoids
#     # ------------------------------------------------------------
#     @torch.no_grad()
#     def _fit_kmedoids_cosine(
#         self,
#         x_raw: torch.Tensor,
#         k: int,
#         seed: int,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Alternating K-Medoids for cosine distance.

#         For fixed assignments and normalized samples, the exact medoid of a
#         cluster is the member maximizing similarity to the sum of cluster
#         members, so the update can be done without an O(n_cluster^2) matrix.

#         Returns:
#             medoid_ids: [k]
#             labels: [N]
#         """
#         n = int(x_raw.shape[0])

#         if k < 2 or k >= n:
#             raise ValueError("K-Medoids requires 2 <= k < N.")

#         x = F.normalize(
#             x_raw.float(),
#             p=2,
#             dim=1,
#             eps=self.eps,
#         )

#         gen = torch.Generator(device=x.device)
#         gen.manual_seed(int(seed))

#         first = int(
#             torch.randint(
#                 low=0,
#                 high=n,
#                 size=(1,),
#                 generator=gen,
#                 device=x.device,
#             ).item()
#         )

#         selected = [first]

#         # Farthest-point initialization in cosine distance.
#         min_dist = 1.0 - (x @ x[first:first + 1].t()).squeeze(1)

#         for _ in range(1, k):
#             idx = int(min_dist.argmax().item())
#             selected.append(idx)
#             dist = 1.0 - (x @ x[idx:idx + 1].t()).squeeze(1)
#             min_dist = torch.minimum(min_dist, dist)

#         medoid_ids = torch.tensor(
#             selected,
#             device=x.device,
#             dtype=torch.long,
#         )

#         old_labels = None

#         for _ in range(self.kmedoids_iterations):
#             medoids = x.index_select(0, medoid_ids)
#             sim = x @ medoids.t()
#             labels = sim.argmax(dim=1)

#             if old_labels is not None and torch.equal(labels, old_labels):
#                 break

#             old_labels = labels.clone()
#             new_ids = []

#             for cluster_id in range(k):
#                 members = labels.eq(cluster_id).nonzero(
#                     as_tuple=False
#                 ).squeeze(1)

#                 if members.numel() == 0:
#                     # Re-seed from the currently worst represented point.
#                     max_sim = sim.max(dim=1).values
#                     candidate = int(max_sim.argmin().item())
#                     new_ids.append(candidate)
#                     continue

#                 member_x = x.index_select(0, members)
#                 sum_direction = member_x.sum(dim=0)

#                 # Exact cosine-medoid criterion for the fixed cluster:
#                 # argmax_i sum_j cos(x_i, x_j)
#                 medoid_score = member_x @ sum_direction
#                 local_id = int(medoid_score.argmax().item())
#                 new_ids.append(int(members[local_id].item()))

#             new_ids_tensor = torch.tensor(
#                 new_ids,
#                 device=x.device,
#                 dtype=torch.long,
#             )

#             if torch.equal(new_ids_tensor, medoid_ids):
#                 medoid_ids = new_ids_tensor
#                 break

#             medoid_ids = new_ids_tensor

#         final_medoids = x.index_select(0, medoid_ids)
#         final_labels = (x @ final_medoids.t()).argmax(dim=1)

#         return medoid_ids, final_labels

#     @torch.no_grad()
#     def _silhouette_cosine(
#         self,
#         x_raw: torch.Tensor,
#         labels: torch.Tensor,
#         seed: int,
#     ) -> float:
#         """
#         Cosine Silhouette score.

#         If N > silhouette_sample_size, a deterministic random subset is used.
#         This is only an engineering memory cap; set silhouette_sample_size >= N
#         to recover the full score.
#         """
#         n = int(x_raw.shape[0])

#         if n < 3:
#             return -1.0

#         unique_labels = labels.unique()
#         if unique_labels.numel() < 2:
#             return -1.0

#         s = min(n, self.silhouette_sample_size)

#         if s < n:
#             gen = torch.Generator(device=x_raw.device)
#             gen.manual_seed(int(seed))
#             sample_ids = torch.randperm(
#                 n,
#                 generator=gen,
#                 device=x_raw.device,
#             )[:s]
#             x = x_raw.index_select(0, sample_ids)
#             y = labels.index_select(0, sample_ids)
#         else:
#             x = x_raw
#             y = labels

#         x = F.normalize(x.float(), p=2, dim=1, eps=self.eps)
#         dist = 1.0 - x @ x.t()
#         dist = dist.clamp_min(0.0)

#         sil = torch.zeros(
#             x.shape[0],
#             device=x.device,
#             dtype=x.dtype,
#         )

#         all_clusters = y.unique()

#         for i in range(x.shape[0]):
#             own = y[i]
#             own_mask = y.eq(own)
#             own_count = int(own_mask.sum().item())

#             if own_count <= 1:
#                 sil[i] = 0.0
#                 continue

#             a = dist[i, own_mask].sum() / float(own_count - 1)

#             b = None
#             for c in all_clusters:
#                 if int(c.item()) == int(own.item()):
#                     continue

#                 mask = y.eq(c)
#                 if not bool(mask.any().item()):
#                     continue

#                 mean_dist = dist[i, mask].mean()
#                 if b is None or mean_dist < b:
#                     b = mean_dist

#             if b is None:
#                 sil[i] = 0.0
#                 continue

#             denom = torch.maximum(a, b).clamp_min(self.eps)
#             sil[i] = (b - a) / denom

#         return float(sil.mean().item())

#     @torch.no_grad()
#     def _assign_to_medoids(
#         self,
#         regions: torch.Tensor,
#         medoids_norm: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         """
#         Assign [N,D] or [B,R,D] region embeddings to the fixed medoids
#         using cosine distance.
#         """
#         if medoids_norm is None:
#             if self.medoids_norm is None:
#                 raise RuntimeError("RaVL discovery has not been run.")
#             medoids_norm = self.medoids_norm

#         original_shape = regions.shape[:-1]
#         d = regions.shape[-1]
#         flat = regions.reshape(-1, d)

#         outputs = []
#         medoids_norm = medoids_norm.to(
#             device=flat.device,
#             dtype=flat.dtype,
#         )

#         for start in range(0, flat.shape[0], self.assignment_chunk_size):
#             end = min(start + self.assignment_chunk_size, flat.shape[0])
#             x = F.normalize(
#                 flat[start:end],
#                 p=2,
#                 dim=1,
#                 eps=self.eps,
#             )
#             outputs.append((x @ medoids_norm.t()).argmax(dim=1))

#         assigned = torch.cat(outputs, dim=0)
#         return assigned.view(*original_shape)

#     # ------------------------------------------------------------
#     # Stage 1: RaVL discovery
#     # ------------------------------------------------------------
#     @torch.no_grad()
#     def discover(
#         self,
#         model: nn.Module,
#         reference_loader,
#         device=None,
#         verbose: bool = True,
#         make_assignment_snapshot: bool = True,
#     ) -> RaVLDiscoveryResult:
#         """
#         Stage-1 discovery adapted to a supervised ResNet. K-Medoids
#         prototypes are constructed only from correctly predicted reference
#         samples; all reference samples are then assigned for nuisance scoring.

#         reference_loader must yield:
#             (inputs, labels)
#         or:
#             (inputs, labels, ...)
#         """
#         if device is None:
#             device = next(model.parameters()).device

#         was_training = model.training
#         model.eval()

#         classifier = _get_classifier(model)

#         all_regions = []
#         all_region_probs = []
#         all_labels = []
#         all_preds = []

#         try:
#             for batch in reference_loader:
#                 if not isinstance(batch, (tuple, list)) or len(batch) < 2:
#                     raise TypeError(
#                         "reference_loader must return (inputs, labels) or "
#                         "(inputs, labels, ...)."
#                     )

#                 inputs = batch[0]
#                 labels = batch[1]

#                 if isinstance(inputs, (tuple, list)):
#                     inputs = inputs[0]

#                 inputs = inputs.to(device, non_blocking=True)
#                 labels = labels.to(device, non_blocking=True).long()

#                 logits, features = _capture_layer4_and_forward(
#                     model,
#                     inputs,
#                 )

#                 if logits.shape[1] != self.num_classes:
#                     raise ValueError(
#                         "Expected {} classes, got {}.".format(
#                             self.num_classes,
#                             logits.shape[1],
#                         )
#                     )

#                 regions = self._regions_from_features(features)
#                 b, r, d = regions.shape

#                 if classifier.weight.shape[1] != d:
#                     raise ValueError(
#                         "Classifier input dim {} != region dim {}.".format(
#                             classifier.weight.shape[1],
#                             d,
#                         )
#                     )

#                 region_logits = classifier(
#                     regions.reshape(b * r, d)
#                 ).view(b, r, self.num_classes)

#                 region_probs = F.softmax(region_logits, dim=2)

#                 pred = logits.argmax(dim=1)

#                 all_regions.append(
#                     regions.detach().cpu().to(torch.float16)
#                 )
#                 all_region_probs.append(
#                     region_probs.detach().cpu().to(torch.float16)
#                 )
#                 all_labels.append(labels.detach().cpu())
#                 all_preds.append(pred.detach().cpu())

#         finally:
#             if was_training:
#                 model.train()
#             else:
#                 model.eval()

#         if len(all_regions) == 0:
#             raise RuntimeError("Reference loader is empty.")

#         regions_img = torch.cat(all_regions, dim=0).float()
#         region_probs_img = torch.cat(all_region_probs, dim=0).float()
#         labels_img = torch.cat(all_labels, dim=0).long()
#         preds_img = torch.cat(all_preds, dim=0).long()
#         correct_img = preds_img.eq(labels_img)

#         n_img, n_region, d = regions_img.shape
#         regions_flat = regions_img.reshape(n_img * n_region, d)

#         # --------------------------------------------------------
#         # Clustering pool: CORRECTLY PREDICTED samples only.
#         # --------------------------------------------------------
#         # Important design:
#         #   1) Correct samples are used only to CONSTRUCT the visual-pattern
#         #      dictionary (K-Medoids / medoids).
#         #   2) After the medoids are fixed, ALL reference samples (correct +
#         #      incorrect) are assigned to these medoids.
#         #   3) Conditional-MI nuisance scoring is still computed on ALL
#         #      reference samples, so incorrect samples remain essential for
#         #      identifying whether a pattern is harmful.
#         #
#         # This avoids allowing under-learned regions from misclassified samples
#         # to directly define the clustering prototypes, while retaining them in
#         # the subsequent nuisance-effect estimation.
#         total_regions = int(regions_flat.shape[0])

#         correct_image_ids = correct_img.nonzero(
#             as_tuple=False
#         ).squeeze(1)
#         num_correct_images = int(correct_image_ids.numel())

#         if num_correct_images == 0:
#             raise RuntimeError(
#                 "No correctly predicted sample exists in the reference set. "
#                 "Correct-only clustering cannot be constructed."
#             )

#         # [N_correct, R, D] -> [N_correct * R, D]
#         correct_regions_img = regions_img.index_select(
#             0,
#             correct_image_ids,
#         )
#         correct_regions_flat = correct_regions_img.reshape(
#             -1,
#             d,
#         )
#         total_correct_regions = int(correct_regions_flat.shape[0])

#         if total_correct_regions < 3:
#             raise RuntimeError(
#                 "Correct-only clustering requires at least 3 correct-region "
#                 "features, got {}.".format(total_correct_regions)
#             )

#         # Optional memory cap is applied ONLY to the correct-region clustering
#         # pool. The later all-sample assignment is unchanged.
#         if (
#             self.max_cluster_regions is not None
#             and total_correct_regions > int(self.max_cluster_regions)
#         ):
#             gen = torch.Generator()
#             gen.manual_seed(self.random_seed)
#             cluster_region_ids = torch.randperm(
#                 total_correct_regions,
#                 generator=gen,
#             )[: int(self.max_cluster_regions)]
#             cluster_pool_cpu = correct_regions_flat.index_select(
#                 0,
#                 cluster_region_ids,
#             )
#         else:
#             cluster_pool_cpu = correct_regions_flat

#         cluster_pool = cluster_pool_cpu.to(
#             device=device,
#             dtype=torch.float32,
#         )

#         n_pool = int(cluster_pool.shape[0])

#         k_min = self.num_classes * self.k_min_factor
#         k_max = self.num_classes * self.k_max_factor

#         k_min = max(2, min(k_min, n_pool - 1))
#         k_max = max(k_min, min(k_max, n_pool - 1))

#         if k_min >= n_pool:
#             raise RuntimeError(
#                 "Not enough candidate regions for RaVL clustering."
#             )

#         best_score = -float("inf")
#         best_k = None
#         best_medoid_ids = None

#         if verbose:
#             print("========== RaVL Stage 1: K-Medoids sweep ==========")
#             correct_rate = num_correct_images / float(max(n_img, 1))
#             print(
#                 "images={} | correct_images={} ({:.2%}) | regions/image={} | "
#                 "all_regions={} | correct_regions={} | cluster_pool={}".format(
#                     n_img,
#                     num_correct_images,
#                     correct_rate,
#                     n_region,
#                     total_regions,
#                     total_correct_regions,
#                     n_pool,
#                 )
#             )
#             # Helpful diagnostic under long-tailed data: clustering still uses
#             # all correctly predicted samples, but this print lets you inspect
#             # whether correct samples are strongly head-class dominated.
#             correct_per_class = []
#             for class_id in range(self.num_classes):
#                 class_correct = int(
#                     (correct_img & labels_img.eq(class_id)).sum().item()
#                 )
#                 correct_per_class.append(class_correct)
#             print("correct images per class = {}".format(correct_per_class))
#             print("K range: {} -> {}".format(k_min, k_max))

#         for k in range(k_min, k_max + 1):
#             medoid_ids, cluster_labels = self._fit_kmedoids_cosine(
#                 cluster_pool,
#                 k=k,
#                 seed=self.random_seed + k,
#             )

#             score = self._silhouette_cosine(
#                 cluster_pool,
#                 cluster_labels,
#                 seed=self.random_seed + 1000 + k,
#             )

#             if verbose:
#                 print(
#                     "K={:3d} | silhouette={:.6f}".format(
#                         k,
#                         score,
#                     )
#                 )

#             if score > best_score:
#                 best_score = score
#                 best_k = k
#                 best_medoid_ids = medoid_ids.detach().clone()

#         best_medoids_raw = cluster_pool.index_select(
#             0,
#             best_medoid_ids,
#         ).detach()

#         best_medoids_norm = F.normalize(
#             best_medoids_raw,
#             p=2,
#             dim=1,
#             eps=self.eps,
#         )

#         # Assign ALL reference regions using the selected fixed medoids.
#         assignments_flat = self._assign_to_medoids(
#             regions_flat.to(device=device, dtype=torch.float32),
#             medoids_norm=best_medoids_norm,
#         ).cpu()

#         assignments_img = assignments_flat.view(
#             n_img,
#             n_region,
#         )

#         # --------------------------------------------------------
#         # Conditional-MI nuisance cluster selection.
#         #
#         # Replace the original RaVL H_k / G_k selection only.
#         # Everything after cluster selection remains unchanged.
#         #
#         # For cluster k:
#         #   S_k = 1 if an image contains at least one region from k.
#         #
#         # For every ground-truth class y, estimate
#         #   I(S_k ; Y_hat | Y=y)
#         # and retain only the harmful direction
#         #   Delta_err = P(E=1 | S_k=1,Y=y)
#         #             - P(E=1 | S_k=0,Y=y).
#         #
#         # Per-class nuisance score:
#         #   N_{k,y} = balance * I(S_k;Y_hat|Y=y) * max(Delta_err, 0)
#         #
#         # Cluster score:
#         #   N_k = sum_y N_{k,y}
#         #
#         # The sum is class-balanced: each class contributes once rather
#         # than being weighted by its sample frequency, which is desirable
#         # for long-tailed recognition.
#         # --------------------------------------------------------

#         # Keep the original result-field names for backward compatibility:
#         #   influence_scores      -> normalized conditional nuisance score
#         #   performance_gaps      -> raw conditional nuisance score N_k
#         #   per_class_gaps        -> per-class error increase Delta_err
#         #   per_class_influence   -> per-class conditional MI
#         influence_scores = {}
#         performance_gaps = {}
#         per_class_gaps = {}
#         per_class_influence = {}

#         # Extra local dictionaries used only during selection / logging.
#         per_class_nuisance = {}
#         per_class_balance = {}

#         def _conditional_mi_binary_cluster(
#             cluster_present: torch.Tensor,
#             predicted_labels: torch.Tensor,
#         ) -> float:
#             """
#             Empirical mutual information I(S; Y_hat) in nats.

#             cluster_present: [N], bool, S in {0,1}
#             predicted_labels: [N], long, Y_hat in {0,...,C-1}

#             This function is called inside a fixed ground-truth class,
#             therefore it estimates I(S_k; Y_hat | Y=y).
#             """
#             n = int(cluster_present.numel())
#             if n <= 1:
#                 return 0.0

#             s = cluster_present.long()
#             y_hat = predicted_labels.long()

#             joint = torch.zeros(
#                 2,
#                 self.num_classes,
#                 dtype=torch.float64,
#             )

#             flat_index = s * self.num_classes + y_hat
#             counts = torch.bincount(
#                 flat_index,
#                 minlength=2 * self.num_classes,
#             ).double()
#             joint = counts.view(2, self.num_classes)

#             total = joint.sum()
#             if float(total.item()) <= 0.0:
#                 return 0.0

#             p_joint = joint / total
#             p_s = p_joint.sum(dim=1, keepdim=True)
#             p_yhat = p_joint.sum(dim=0, keepdim=True)
#             denom = p_s * p_yhat

#             valid = p_joint > 0
#             if not bool(valid.any().item()):
#                 return 0.0

#             mi = (
#                 p_joint[valid]
#                 * torch.log(
#                     p_joint[valid]
#                     / denom[valid].clamp_min(self.eps)
#                 )
#             ).sum()

#             return float(mi.item())

#         for cluster_id in range(best_k):
#             # S_k for every reference image.
#             present = assignments_img.eq(cluster_id).any(dim=1)

#             mi_by_class = {}
#             error_increase_by_class = {}
#             nuisance_by_class = {}
#             balance_by_class = {}

#             for y in range(self.num_classes):
#                 class_mask = labels_img.eq(y)

#                 # Need both S_k=1 and S_k=0 inside this class so that the
#                 # cluster-presence variable is actually comparable.
#                 in_mask = class_mask & present
#                 out_mask = class_mask & (~present)

#                 n_in = int(in_mask.sum().item())
#                 n_out = int(out_mask.sum().item())

#                 if n_in == 0 or n_out == 0:
#                     continue

#                 # ----------------------------------------------------
#                 # 1) Harmful direction: increase in error probability.
#                 # ----------------------------------------------------
#                 err_in = float(
#                     (~correct_img[in_mask]).float().mean().item()
#                 )
#                 err_out = float(
#                     (~correct_img[out_mask]).float().mean().item()
#                 )

#                 delta_err = err_in - err_out

#                 # Same presence/absence balancing idea as the old code,
#                 # but it now stabilizes the conditional-MI score rather
#                 # than defining G_k.
#                 balance = (
#                     2.0
#                     * min(n_in, n_out)
#                     / float(n_in + n_out)
#                 )

#                 # ----------------------------------------------------
#                 # 2) Conditional MI:
#                 #       I(S_k ; Y_hat | Y=y)
#                 # ----------------------------------------------------
#                 class_present = present[class_mask]
#                 class_preds = preds_img[class_mask]

#                 cmi_y = _conditional_mi_binary_cluster(
#                     cluster_present=class_present,
#                     predicted_labels=class_preds,
#                 )

#                 # ----------------------------------------------------
#                 # 3) Per-class nuisance score.
#                 # Only the harmful direction is kept.
#                 # ----------------------------------------------------
#                 harmful_delta = max(delta_err, 0.0)
#                 nuisance_y = balance * cmi_y * harmful_delta

#                 mi_by_class[y] = float(cmi_y)
#                 error_increase_by_class[y] = float(delta_err)
#                 nuisance_by_class[y] = float(nuisance_y)
#                 balance_by_class[y] = float(balance)

#             # Class-balanced aggregation. We intentionally do NOT multiply
#             # by the empirical class prior, otherwise head classes would
#             # dominate the cluster ranking in a long-tailed dataset.
#             nuisance_score = sum(nuisance_by_class.values())

#             performance_gaps[cluster_id] = float(nuisance_score)
#             per_class_gaps[cluster_id] = error_increase_by_class
#             per_class_influence[cluster_id] = mi_by_class
#             per_class_nuisance[cluster_id] = nuisance_by_class
#             per_class_balance[cluster_id] = balance_by_class

#         # Normalize only for thresholding so the existing constructor
#         # argument influence_threshold can be kept unchanged.
#         max_nuisance_score = max(
#             performance_gaps.values()
#         ) if len(performance_gaps) > 0 else 0.0

#         for cluster_id in range(best_k):
#             raw_score = performance_gaps.get(cluster_id, 0.0)
#             if max_nuisance_score > self.eps:
#                 normalized_score = raw_score / max_nuisance_score
#             else:
#                 normalized_score = 0.0
#             influence_scores[cluster_id] = float(normalized_score)

#         candidate_clusters = [
#             c
#             for c in range(best_k)
#             if performance_gaps.get(c, 0.0) > 0.0
#             and influence_scores.get(c, 0.0) >= self.influence_threshold
#         ]

#         # Rank by the raw conditional nuisance score N_k.
#         ranked_clusters = sorted(
#             candidate_clusters,
#             key=lambda c: performance_gaps.get(c, 0.0),
#             reverse=True,
#         )

#         if len(ranked_clusters) == 0:
#             raise RuntimeError(
#                 "Conditional-MI selection found no nuisance cluster with "
#                 "positive harmful score and normalized score >= {:.3f}. "
#                 "Try inspecting the reference split or lowering "
#                 "influence_threshold.".format(
#                     self.influence_threshold
#                 )
#             )

#         top_spurious_clusters = [int(c) for c in ranked_clusters[:self.num_spurious_clusters]]

#         # backward compatible: keep original single cluster variable
#         top_spurious_cluster = int(top_spurious_clusters[0])

#         # Keep the fixed Stage-1 clustering model.
#         self.medoids_raw = best_medoids_raw.detach().cpu()
#         self.medoids_norm = best_medoids_norm.detach().cpu()
#         self.top_spurious_cluster = top_spurious_cluster
#         self.top_spurious_clusters = top_spurious_clusters
#         self.ranked_clusters = list(ranked_clusters)

#         result = RaVLDiscoveryResult(
#             best_k=int(best_k),
#             best_silhouette=float(best_score),
#             top_spurious_cluster=top_spurious_cluster,
#             ranked_clusters=list(ranked_clusters),
#             influence_scores=influence_scores,
#             performance_gaps=performance_gaps,
#             per_class_gaps=per_class_gaps,
#             per_class_influence=per_class_influence,
#             num_images=int(n_img),
#             num_regions=int(total_regions),
#         )

#         self.discovery_result = result

#         if make_assignment_snapshot:
#             self.prepare_stage2_assignment_model(
#                 model=model,
#                 device=device,
#             )

#         if verbose:
#             print("========== RaVL Stage 1: Discovery Result ==========")
#             print(
#                 "best K={} | silhouette={:.6f}".format(
#                     best_k,
#                     best_score,
#                 )
#             )
#             print(
#                 "normalized nuisance-score threshold={:.3f}".format(
#                     self.influence_threshold
#                 )
#             )

#             for rank, cluster_id in enumerate(ranked_clusters[:10], 1):
#                 print(
#                     "Rank {:2d} | cluster {:3d} | N_norm={:.4f} | N_score={:.6f}".format(
#                         rank,
#                         cluster_id,
#                         influence_scores[cluster_id],
#                         performance_gaps[cluster_id],
#                     )
#                 )

#             print(
#                 "TOP spurious clusters = {}".format(
#                     top_spurious_clusters
#                 )
#             )
#             print("=====================================================")

#         return result

#     # ------------------------------------------------------------
#     # Fixed Stage-1 assignment model for Stage 2
#     # ------------------------------------------------------------
#     @torch.no_grad()
#     def prepare_stage2_assignment_model(
#         self,
#         model: nn.Module,
#         device=None,
#     ) -> None:
#         """
#         Freeze a snapshot of the Stage-1 encoder.

#         The paper first determines the spurious cluster, then uses the trained
#         clustering model to assign training regions to R^s / R^r before
#         mitigation. This frozen snapshot keeps region assignment in the
#         original Stage-1 feature space even while the student backbone changes.
#         """
#         base = _unwrap_model(model)

#         if device is None:
#             device = next(base.parameters()).device

#         frozen = copy.deepcopy(base).to(device)
#         frozen.eval()

#         for p in frozen.parameters():
#             p.requires_grad_(False)

#         # Bypass nn.Module registration because RaVLResNet is intentionally
#         # a parameter-free utility object.
#         self._assignment_model = frozen
#         self._assignment_device = device

#     # ------------------------------------------------------------
#     # Region assignment during Stage 2
#     # ------------------------------------------------------------
#     @torch.no_grad()
#     def _stage2_region_partition(
#         self,
#         inputs: torch.Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         if self._assignment_model is None:
#             raise RuntimeError(
#                 "No frozen Stage-1 assignment model. "
#                 "Run discover(..., make_assignment_snapshot=True) or call "
#                 "prepare_stage2_assignment_model(model)."
#             )

#         if self.medoids_norm is None or self.top_spurious_cluster is None:
#             raise RuntimeError("RaVL discovery has not been initialized.")

#         assignment_inputs = inputs.to(
#             self._assignment_device,
#             non_blocking=True,
#         )

#         _, assignment_features = _capture_layer4_and_forward(
#             self._assignment_model,
#             assignment_inputs,
#         )

#         assignment_regions = self._regions_from_features(
#             assignment_features
#         )

#         medoids_norm = self.medoids_norm.to(
#             device=assignment_regions.device,
#             dtype=assignment_regions.dtype,
#         )

#         cluster_ids = self._assign_to_medoids(
#             assignment_regions,
#             medoids_norm=medoids_norm,
#         )

#         spurious_mask = torch.zeros_like(
#             cluster_ids, dtype=torch.bool
#         )

#         for c in self.top_spurious_clusters:
#             spurious_mask |= cluster_ids.eq(int(c))
#         relevant_mask = ~spurious_mask

#         return (
#             cluster_ids,
#             spurious_mask,
#             relevant_mask,
#         )

#     # ------------------------------------------------------------
#     # Nuisance class-neutralization loss
#     # ------------------------------------------------------------
#     def _region_aware_loss(
#         self,
#         student_regions: torch.Tensor,
#         labels: torch.Tensor,
#         spurious_mask: torch.Tensor,
#         relevant_mask: torch.Tensor,
#         classifier: nn.Linear,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Replace the original RaVL L_R + L_A by a direct nuisance
#         class-neutralization objective.

#         For every discovered nuisance region z_s, compute its class posterior
#         using cosine similarity to the DETACHED classifier directions:

#             q(c | z_s) = softmax(cos(z_s, w_c) / tau).

#         Then minimize

#             KL(q(Y | z_s) || U_C)
#             = log(C) - H(q(Y | z_s)),

#         where U_C is the uniform distribution over C task classes.
#         Therefore, minimizing this loss is exactly equivalent to maximizing
#         the class-prediction entropy of nuisance regions. The classifier
#         weights are detached so the backbone must neutralize nuisance-region
#         representations rather than moving the class anchors.

#         Returns:
#             loss_nui: scalar nuisance neutralization loss
#             valid_images: [B] bool, True if an image contains >=1 nuisance region
#         """
#         b, r, d = student_regions.shape

#         if classifier.weight.shape != (self.num_classes, d):
#             raise ValueError(
#                 "Classifier weight shape {} incompatible with regions [*,{},{}].".format(
#                     tuple(classifier.weight.shape),
#                     r,
#                     d,
#                 )
#             )

#         device = student_regions.device
#         spurious_mask = spurious_mask.to(device=device, dtype=torch.bool)
#         relevant_mask = relevant_mask.to(device=device, dtype=torch.bool)

#         # labels are intentionally not used by this loss. Once a region has
#         # been discovered as nuisance, it is required to be neutral with
#         # respect to ALL task classes rather than only the image ground truth.
#         _ = labels
#         _ = relevant_mask

#         # Which images contribute at least one nuisance patch.
#         valid_images = spurious_mask.any(dim=1)

#         # No nuisance patch in the current batch -> differentiable zero.
#         if not bool(spurious_mask.any().item()):
#             zero = student_regions.sum() * 0.0
#             return zero, valid_images

#         # Normalize region features and freeze the FC class directions.
#         region_n = F.normalize(
#             student_regions,
#             p=2,
#             dim=2,
#             eps=self.eps,
#         )

#         class_n = F.normalize(
#             classifier.weight.detach().to(
#                 device=device,
#                 dtype=student_regions.dtype,
#             ),
#             p=2,
#             dim=1,
#             eps=self.eps,
#         )

#         # [B,R,C], cosine similarity to all class directions.
#         sim = torch.einsum(
#             "brd,cd->brc",
#             region_n,
#             class_n,
#         )

#         # Only discovered nuisance regions participate in the purification.
#         # [N_s, C]
#         nuisance_logits = sim[spurious_mask] / self.temperature

#         # q(Y | z_s), written in log-space for numerical stability.
#         log_q = F.log_softmax(nuisance_logits, dim=1)
#         q = log_q.exp()

#         # KL(q || U_C) = sum_c q_c log(q_c / (1/C))
#         #               = log(C) - H(q).
#         log_uniform = -math.log(float(self.num_classes))
#         kl_to_uniform = (
#             q * (log_q - log_uniform)
#         ).sum(dim=1)

#         # Patch-level mean: every nuisance patch contributes equally.
#         loss_nui = kl_to_uniform.mean()

#         return loss_nui, valid_images


#     # ------------------------------------------------------------
#     # Visualization helpers
#     # ------------------------------------------------------------
#     @staticmethod
#     def _vis_to_numpy_image(
#         image: torch.Tensor,
#         mean: Optional[List[float]] = None,
#         std: Optional[List[float]] = None,
#     ) -> np.ndarray:
#         """Convert [C,H,W] tensor to RGB numpy image in [0,1]."""
#         x = image.detach().cpu().float().clone()
#         if x.ndim != 3:
#             raise ValueError("image must be [C,H,W].")

#         if mean is not None and std is not None:
#             mean_t = torch.tensor(mean, dtype=x.dtype).view(-1, 1, 1)
#             std_t = torch.tensor(std, dtype=x.dtype).view(-1, 1, 1)
#             if mean_t.shape[0] == 1 and x.shape[0] == 3:
#                 mean_t = mean_t.repeat(3, 1, 1)
#                 std_t = std_t.repeat(3, 1, 1)
#             if mean_t.shape[0] != x.shape[0]:
#                 raise ValueError(
#                     "mean/std channel count {} does not match image channels {}.".format(
#                         mean_t.shape[0], x.shape[0]
#                     )
#                 )
#             x = x * std_t + mean_t
#         else:
#             # Medical images are often not in [0,1]. For display only, use
#             # sample-wise min-max scaling when needed.
#             xmin = float(x.min().item())
#             xmax = float(x.max().item())
#             if xmin < 0.0 or xmax > 1.0:
#                 x = (x - x.min()) / (x.max() - x.min() + 1e-8)

#         if x.shape[0] == 1:
#             x = x.repeat(3, 1, 1)
#         elif x.shape[0] >= 3:
#             x = x[:3]
#         else:
#             raise ValueError("Only 1-channel or >=3-channel images are supported.")

#         x = x.clamp(0.0, 1.0)
#         return x.permute(1, 2, 0).numpy()

#     @staticmethod
#     def _vis_upsample_map(
#         x: torch.Tensor,
#         size: Tuple[int, int],
#         mode: str,
#     ) -> torch.Tensor:
#         """Upsample a [H,W] map to image resolution."""
#         x = x[None, None].float()
#         if mode in ("bilinear", "bicubic"):
#             x = F.interpolate(x, size=size, mode=mode, align_corners=False)
#         else:
#             x = F.interpolate(x, size=size, mode=mode)
#         return x[0, 0]

#     @staticmethod
#     def _vis_draw_patch_boxes(
#         ax,
#         patch_mask: np.ndarray,
#         image_h: int,
#         image_w: int,
#         edgecolor: str = "white",
#         linewidth: float = 1.4,
#     ) -> None:
#         """Draw feature-grid patch boxes on an image axis."""
#         hf, wf = patch_mask.shape
#         for rr in range(hf):
#             for cc in range(wf):
#                 if not bool(patch_mask[rr, cc]):
#                     continue
#                 x0 = cc * image_w / wf
#                 x1 = (cc + 1) * image_w / wf
#                 y0 = rr * image_h / hf
#                 y1 = (rr + 1) * image_h / hf
#                 ax.add_patch(
#                     Rectangle(
#                         (x0, y0),
#                         x1 - x0,
#                         y1 - y0,
#                         fill=False,
#                         edgecolor=edgecolor,
#                         linewidth=linewidth,
#                     )
#                 )

#     @staticmethod
#     def _vis_crop_patch(
#         pil_img: Image.Image,
#         patch_index: int,
#         hf: int,
#         wf: int,
#     ) -> Image.Image:
#         """Crop one layer4 spatial token from the original image."""
#         image_w, image_h = pil_img.size
#         rr = int(patch_index) // wf
#         cc = int(patch_index) % wf
#         y0 = int(round(rr * image_h / hf))
#         y1 = int(round((rr + 1) * image_h / hf))
#         x0 = int(round(cc * image_w / wf))
#         x1 = int(round((cc + 1) * image_w / wf))
#         return pil_img.crop((x0, y0, x1, y1))

#     def _compute_standard_cam(
#         self,
#         features: torch.Tensor,
#         classifier: nn.Linear,
#         class_ids: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Compute standard CAM for a GAP + linear classifier.

#         features: [B,D,Hf,Wf]
#         class_ids: [B]
#         return: [B,Hf,Wf] normalized to [0,1]
#         """
#         if classifier.weight.shape[1] != features.shape[1]:
#             raise ValueError(
#                 "classifier dim {} != layer4 dim {}.".format(
#                     classifier.weight.shape[1], features.shape[1]
#                 )
#             )
#         weight = classifier.weight.detach().to(
#             device=features.device,
#             dtype=features.dtype,
#         )
#         selected = weight.index_select(0, class_ids.long())
#         cam = (features * selected[:, :, None, None]).sum(dim=1)
#         cam = F.relu(cam)
#         cam = cam - cam.amin(dim=(1, 2), keepdim=True)
#         cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + self.eps)
#         return cam

#     @torch.no_grad()
#     def visualize_nuisance_regions(
#         self,
#         student_model: nn.Module,
#         inputs: torch.Tensor,
#         labels: Optional[torch.Tensor] = None,
#         save_dir: str = "./nuisance_visualization",
#         file_prefix: str = "sample",
#         mean: Optional[List[float]] = None,
#         std: Optional[List[float]] = None,
#         cam_target: str = "gt",
#         cam_threshold: float = 0.35,
#         foreground_masks: Optional[torch.Tensor] = None,
#         foreground_patch_threshold: float = 0.25,
#         max_images: Optional[int] = 8,
#         save_patch_crops: bool = True,
#         max_patches_per_image: int = 32,
#     ) -> List[Dict[str, object]]:
#         """
#         Visualize where discovered nuisance patches lie in each image.

#         Panels:
#             Original | Cluster IDs | Nuisance Patches | CAM + Nuisance
#             + optional Segmentation + Nuisance when foreground_masks is given.

#         Quantitative outputs per image:
#             nuisance_patch_ratio:
#                 fraction of layer4 tokens selected as nuisance.
#             nuisance_low_cam_ratio:
#                 fraction of nuisance tokens whose normalized CAM < cam_threshold.
#                 This is a model-based background proxy, NOT an anatomical proof.
#             nuisance_mean_cam:
#                 mean CAM response over nuisance tokens.
#             nuisance_seg_background_ratio (optional):
#                 fraction of nuisance tokens whose foreground-mask occupancy is
#                 < foreground_patch_threshold. This is stronger evidence of
#                 anatomical background than CAM when a reliable segmentation mask exists.
#         """
#         if cam_target not in ("pred", "gt"):
#             raise ValueError("cam_target must be 'pred' or 'gt'.")
#         if cam_target == "gt" and labels is None:
#             raise ValueError("labels are required when cam_target='gt'.")
#         if not (0.0 <= cam_threshold <= 1.0):
#             raise ValueError("cam_threshold must be in [0,1].")
#         if not (0.0 <= foreground_patch_threshold <= 1.0):
#             raise ValueError("foreground_patch_threshold must be in [0,1].")

#         Path(save_dir).mkdir(parents=True, exist_ok=True)
#         base = _unwrap_model(student_model)
#         was_training = base.training
#         base.eval()

#         try:
#             device = next(base.parameters()).device
#             inputs = inputs.to(device, non_blocking=True)
#             if labels is not None:
#                 labels = labels.to(device, non_blocking=True).long()

#             logits, features = _capture_layer4_and_forward(student_model, inputs)
#             cluster_ids, spurious_mask, relevant_mask = self._stage2_region_partition(inputs)
#             classifier = _get_classifier(student_model)

#             if cam_target == "gt":
#                 target_ids = labels
#             else:
#                 target_ids = logits.argmax(dim=1)

#             cams = self._compute_standard_cam(
#                 features=features,
#                 classifier=classifier,
#                 class_ids=target_ids,
#             )

#             b, _, image_h, image_w = inputs.shape
#             _, _, hf, wf = features.shape
#             if int(cluster_ids.shape[1]) != hf * wf:
#                 raise ValueError(
#                     "cluster_ids regions={} but student layer4 grid={}x{}.".format(
#                         cluster_ids.shape[1], hf, wf
#                     )
#                 )

#             fg_patch_fraction = None
#             fg_masks_cpu = None
#             if foreground_masks is not None:
#                 fg = foreground_masks.detach().float()
#                 if fg.ndim == 3:
#                     fg = fg.unsqueeze(1)
#                 elif fg.ndim == 4 and fg.shape[1] != 1:
#                     fg = fg[:, :1]
#                 if fg.ndim != 4:
#                     raise ValueError(
#                         "foreground_masks must be [B,H,W] or [B,1,H,W]."
#                     )
#                 if fg.shape[0] != b:
#                     raise ValueError("foreground_masks batch size must match inputs.")
#                 fg = fg.to(device)
#                 # Average foreground occupancy inside each feature-grid token.
#                 fg_patch_fraction = F.adaptive_avg_pool2d(fg, (hf, wf))[:, 0]
#                 fg_masks_cpu = fg[:, 0].detach().cpu()

#             if max_images is None:
#                 n_show = b
#             else:
#                 n_show = min(int(max_images), b)

#             summaries: List[Dict[str, object]] = []

#             for i in range(n_show):
#                 img_np = self._vis_to_numpy_image(inputs[i], mean=mean, std=std)
#                 cluster_grid = cluster_ids[i].view(hf, wf).detach().cpu()
#                 spur_grid = spurious_mask[i].view(hf, wf).detach().cpu()
#                 cam_grid = cams[i].detach().cpu()

#                 nuisance_cam = cam_grid[spur_grid]
#                 num_nuisance = int(spur_grid.sum().item())
#                 num_total = int(hf * wf)
#                 nuisance_patch_ratio = num_nuisance / max(num_total, 1)

#                 if nuisance_cam.numel() > 0:
#                     low_cam_ratio = float(
#                         (nuisance_cam < cam_threshold).float().mean().item()
#                     )
#                     high_cam_ratio = float(
#                         (nuisance_cam >= cam_threshold).float().mean().item()
#                     )
#                     mean_cam = float(nuisance_cam.mean().item())
#                 else:
#                     low_cam_ratio = 0.0
#                     high_cam_ratio = 0.0
#                     mean_cam = 0.0

#                 seg_bg_ratio = None
#                 seg_fg_ratio = None
#                 mean_fg_fraction = None
#                 if fg_patch_fraction is not None:
#                     nuisance_fg = fg_patch_fraction[i].detach().cpu()[spur_grid]
#                     if nuisance_fg.numel() > 0:
#                         seg_bg_ratio = float(
#                             (nuisance_fg < foreground_patch_threshold)
#                             .float()
#                             .mean()
#                             .item()
#                         )
#                         seg_fg_ratio = 1.0 - seg_bg_ratio
#                         mean_fg_fraction = float(nuisance_fg.mean().item())
#                     else:
#                         seg_bg_ratio = 0.0
#                         seg_fg_ratio = 0.0
#                         mean_fg_fraction = 0.0

#                 spur_up = self._vis_upsample_map(
#                     spur_grid.float(), (image_h, image_w), "nearest"
#                 ).numpy()
#                 cam_up = self._vis_upsample_map(
#                     cam_grid, (image_h, image_w), "bilinear"
#                 ).numpy()
#                 cluster_up = self._vis_upsample_map(
#                     cluster_grid.float(), (image_h, image_w), "nearest"
#                 ).numpy()

#                 ncols = 5 if fg_masks_cpu is not None else 4
#                 fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.8))
#                 axes = np.atleast_1d(axes)

#                 axes[0].imshow(img_np)
#                 axes[0].set_title("Original")
#                 axes[0].axis("off")

#                 cluster_plot = axes[1].imshow(
#                     cluster_up,
#                     cmap="tab20",
#                     interpolation="nearest",
#                 )
#                 axes[1].set_title("Cluster IDs")
#                 axes[1].axis("off")
#                 fig.colorbar(cluster_plot, ax=axes[1], fraction=0.046, pad=0.04)

#                 axes[2].imshow(img_np)
#                 axes[2].imshow(
#                     spur_up,
#                     cmap="Reds",
#                     alpha=0.42,
#                     vmin=0.0,
#                     vmax=1.0,
#                     interpolation="nearest",
#                 )
#                 self._vis_draw_patch_boxes(
#                     axes[2],
#                     spur_grid.numpy(),
#                     image_h=image_h,
#                     image_w=image_w,
#                 )
#                 axes[2].set_title(
#                     "Nuisance Patches\npatch ratio={:.3f}".format(
#                         nuisance_patch_ratio
#                     )
#                 )
#                 axes[2].axis("off")

#                 axes[3].imshow(img_np)
#                 axes[3].imshow(
#                     cam_up,
#                     cmap="jet",
#                     alpha=0.42,
#                     vmin=0.0,
#                     vmax=1.0,
#                 )
#                 self._vis_draw_patch_boxes(
#                     axes[3],
#                     spur_grid.numpy(),
#                     image_h=image_h,
#                     image_w=image_w,
#                 )
#                 axes[3].set_title(
#                     "CAM + Nuisance\nlowCAM={:.3f}, highCAM={:.3f}, mean={:.3f}".format(
#                         low_cam_ratio,
#                         high_cam_ratio,
#                         mean_cam,
#                     )
#                 )
#                 axes[3].axis("off")

#                 if fg_masks_cpu is not None:
#                     fg_up = F.interpolate(
#                         fg_masks_cpu[i][None, None].float(),
#                         size=(image_h, image_w),
#                         mode="nearest",
#                     )[0, 0].numpy()
#                     axes[4].imshow(img_np)
#                     axes[4].imshow(
#                         fg_up,
#                         cmap="Greens",
#                         alpha=0.30,
#                         vmin=0.0,
#                         vmax=1.0,
#                     )
#                     self._vis_draw_patch_boxes(
#                         axes[4],
#                         spur_grid.numpy(),
#                         image_h=image_h,
#                         image_w=image_w,
#                     )
#                     axes[4].set_title(
#                         "Segmentation + Nuisance\nsegBG={:.3f}, segFG={:.3f}".format(
#                             seg_bg_ratio,
#                             seg_fg_ratio,
#                         )
#                     )
#                     axes[4].axis("off")

#                 pred_id = int(logits[i].argmax().item())
#                 gt_id = int(labels[i].item()) if labels is not None else -1
#                 fig.suptitle(
#                     "idx={} | gt={} | pred={} | CAM target={} | nuisance={}".format(
#                         i,
#                         gt_id,
#                         pred_id,
#                         int(target_ids[i].item()),
#                         num_nuisance,
#                     ),
#                     fontsize=12,
#                 )
#                 fig.tight_layout()

#                 figure_path = os.path.join(
#                     save_dir,
#                     "{}_{:03d}.png".format(file_prefix, i),
#                 )
#                 fig.savefig(figure_path, dpi=400, bbox_inches="tight", pad_inches=0.05)
#                 plt.close(fig)

#                 patch_paths: List[str] = []
#                 patch_dir = None
#                 if save_patch_crops:
#                     patch_dir = os.path.join(
#                         save_dir,
#                         "{}_{:03d}_patches".format(file_prefix, i),
#                     )
#                     Path(patch_dir).mkdir(parents=True, exist_ok=True)
#                     pil_img = Image.fromarray((img_np * 255.0).astype(np.uint8))
#                     nuisance_indices = (
#                         spurious_mask[i]
#                         .nonzero(as_tuple=False)
#                         .squeeze(1)
#                         .detach()
#                         .cpu()
#                         .tolist()
#                     )[: int(max_patches_per_image)]

#                     for order, patch_index in enumerate(nuisance_indices):
#                         crop = self._vis_crop_patch(
#                             pil_img=pil_img,
#                             patch_index=int(patch_index),
#                             hf=hf,
#                             wf=wf,
#                         )
#                         cid = int(cluster_ids[i, patch_index].item())
#                         rr = int(patch_index) // wf
#                         cc = int(patch_index) % wf
#                         crop_path = os.path.join(
#                             patch_dir,
#                             "patch_{:02d}_cluster_{}_r{}_c{}.png".format(
#                                 order, cid, rr, cc
#                             ),
#                         )
#                         crop.save(crop_path, format="PNG", optimize=True)
#                         patch_paths.append(crop_path)

#                 summary = {
#                     "index": i,
#                     "gt": gt_id,
#                     "pred": pred_id,
#                     "num_nuisance_regions": num_nuisance,
#                     "num_relevant_regions": int(relevant_mask[i].sum().item()),
#                     "nuisance_patch_ratio": float(nuisance_patch_ratio),
#                     "nuisance_low_cam_ratio": low_cam_ratio,
#                     "nuisance_high_cam_ratio": high_cam_ratio,
#                     "nuisance_mean_cam": mean_cam,
#                     "figure_path": figure_path,
#                     "patch_dir": patch_dir,
#                     "patch_paths": patch_paths,
#                 }
#                 if seg_bg_ratio is not None:
#                     summary.update(
#                         {
#                             "nuisance_seg_background_ratio": seg_bg_ratio,
#                             "nuisance_seg_foreground_ratio": seg_fg_ratio,
#                             "nuisance_mean_foreground_fraction": mean_fg_fraction,
#                         }
#                     )
#                 summaries.append(summary)

#             return summaries

#         finally:
#             if was_training:
#                 base.train()
#             else:
#                 base.eval()

#     @torch.no_grad()
#     def visualize_nuisance_cluster_exemplars(
#         self,
#         data_loader,
#         device,
#         save_dir: str = "./nuisance_cluster_exemplars",
#         file_prefix: str = "cluster",
#         mean: Optional[List[float]] = None,
#         std: Optional[List[float]] = None,
#         target_clusters: Optional[List[int]] = None,
#         max_patches_per_cluster: int = 64,
#         max_batches: Optional[int] = 100,
#         columns: int = 8,
#     ) -> Dict[int, str]:
#         """
#         Collect original-image crops assigned to each selected nuisance cluster
#         and save one montage per cluster. This is the most direct qualitative
#         check of whether a nuisance cluster mainly captures background, border,
#         artifacts, irrelevant anatomy, hair, markers, etc.
#         """
#         if self._assignment_model is None:
#             raise RuntimeError(
#                 "No frozen assignment model. Run discover(..., "
#                 "make_assignment_snapshot=True) first."
#             )
#         if target_clusters is None:
#             target_clusters = list(self.top_spurious_clusters)
#         target_clusters = [int(c) for c in target_clusters]
#         if len(target_clusters) == 0:
#             raise RuntimeError("No nuisance clusters are available.")

#         Path(save_dir).mkdir(parents=True, exist_ok=True)
#         buckets = {c: [] for c in target_clusters}
#         assignment_model = self._assignment_model
#         assignment_device = self._assignment_device
#         was_training = assignment_model.training
#         assignment_model.eval()

#         try:
#             for batch_idx, batch in enumerate(data_loader):
#                 if max_batches is not None and batch_idx >= int(max_batches):
#                     break
#                 if not isinstance(batch, (tuple, list)) or len(batch) < 1:
#                     raise TypeError("data_loader must return (inputs, labels, ...).")

#                 inputs = batch[0]
#                 if isinstance(inputs, (tuple, list)):
#                     inputs = inputs[0]
#                 inputs = inputs.to(device, non_blocking=True)

#                 cluster_ids, _, _ = self._stage2_region_partition(inputs)
#                 assignment_inputs = inputs.to(assignment_device, non_blocking=True)
#                 _, assignment_features = _capture_layer4_and_forward(
#                     assignment_model, assignment_inputs
#                 )
#                 b, _, hf, wf = assignment_features.shape
#                 if int(cluster_ids.shape[1]) != hf * wf:
#                     raise ValueError(
#                         "cluster_ids={} but assignment grid={}x{}.".format(
#                             cluster_ids.shape[1], hf, wf
#                         )
#                     )

#                 for i in range(b):
#                     img_np = self._vis_to_numpy_image(inputs[i], mean=mean, std=std)
#                     pil_img = Image.fromarray((img_np * 255.0).astype(np.uint8))
#                     ids_i = cluster_ids[i].detach().cpu()

#                     for cid in target_clusters:
#                         if len(buckets[cid]) >= int(max_patches_per_cluster):
#                             continue
#                         patch_indices = (
#                             ids_i.eq(cid)
#                             .nonzero(as_tuple=False)
#                             .squeeze(1)
#                             .tolist()
#                         )
#                         for patch_index in patch_indices:
#                             if len(buckets[cid]) >= int(max_patches_per_cluster):
#                                 break
#                             crop = self._vis_crop_patch(
#                                 pil_img=pil_img,
#                                 patch_index=int(patch_index),
#                                 hf=hf,
#                                 wf=wf,
#                             )
#                             buckets[cid].append(crop.copy())

#                 if all(
#                     len(v) >= int(max_patches_per_cluster)
#                     for v in buckets.values()
#                 ):
#                     break
#         finally:
#             if was_training:
#                 assignment_model.train()
#             else:
#                 assignment_model.eval()

#         output_paths: Dict[int, str] = {}
#         for cid, crops in buckets.items():
#             if len(crops) == 0:
#                 continue
#             n = len(crops)
#             ncols = min(int(columns), n)
#             nrows = int(math.ceil(n / ncols))
#             fig, axes = plt.subplots(
#                 nrows,
#                 ncols,
#                 figsize=(2.0 * ncols, 2.0 * nrows),
#                 squeeze=False,
#             )
#             for ax in axes.ravel():
#                 ax.axis("off")
#             for j, crop in enumerate(crops):
#                 rr = j // ncols
#                 cc = j % ncols
#                 axes[rr, cc].imshow(crop)
#                 axes[rr, cc].axis("off")
#             fig.suptitle(
#                 "Nuisance cluster {} exemplars (n={})".format(cid, n),
#                 fontsize=14,
#             )
#             fig.tight_layout()
#             path = os.path.join(
#                 save_dir,
#                 "{}_{}.png".format(file_prefix, cid),
#             )
#             fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.05)
#             plt.close(fig)
#             output_paths[cid] = path
#         return output_paths

#     @torch.no_grad()
#     def evaluate_nuisance_cam_overlap(
#         self,
#         student_model: nn.Module,
#         data_loader,
#         device,
#         cam_target: str = "gt",
#         cam_threshold: float = 0.35,
#         max_batches: Optional[int] = None,
#     ) -> Dict[str, float]:
#         """
#         Dataset-level CAM localization statistics for nuisance patches.

#         low_cam_ratio close to 1 means most nuisance tokens fall in regions that
#         have low class-specific CAM response. This supports a background/shortcut
#         interpretation, but it is not an anatomical ground-truth measurement.
#         """
#         if cam_target not in ("pred", "gt"):
#             raise ValueError("cam_target must be 'pred' or 'gt'.")

#         base = _unwrap_model(student_model)
#         was_training = base.training
#         base.eval()
#         total_nuisance = 0
#         total_low_cam = 0
#         total_high_cam = 0
#         sum_cam = 0.0

#         try:
#             for batch_idx, batch in enumerate(data_loader):
#                 if max_batches is not None and batch_idx >= int(max_batches):
#                     break
#                 if not isinstance(batch, (tuple, list)) or len(batch) < 2:
#                     raise TypeError("data_loader must return (inputs, labels, ...).")

#                 inputs = batch[0]
#                 labels = batch[1]
#                 if isinstance(inputs, (tuple, list)):
#                     inputs = inputs[0]
#                 inputs = inputs.to(device, non_blocking=True)
#                 labels = labels.to(device, non_blocking=True).long()

#                 logits, features = _capture_layer4_and_forward(student_model, inputs)
#                 cluster_ids, spurious_mask, _ = self._stage2_region_partition(inputs)
#                 classifier = _get_classifier(student_model)
#                 target_ids = labels if cam_target == "gt" else logits.argmax(dim=1)
#                 cams = self._compute_standard_cam(features, classifier, target_ids)

#                 b, _, hf, wf = features.shape
#                 if int(cluster_ids.shape[1]) != hf * wf:
#                     raise ValueError(
#                         "Student feature grid and assignment region count do not match."
#                     )
#                 spur_grid = spurious_mask.view(b, hf, wf)
#                 nuisance_cam = cams[spur_grid]
#                 if nuisance_cam.numel() == 0:
#                     continue

#                 total_nuisance += int(nuisance_cam.numel())
#                 total_low_cam += int((nuisance_cam < cam_threshold).sum().item())
#                 total_high_cam += int((nuisance_cam >= cam_threshold).sum().item())
#                 sum_cam += float(nuisance_cam.sum().item())
#         finally:
#             if was_training:
#                 base.train()
#             else:
#                 base.eval()

#         if total_nuisance == 0:
#             return {
#                 "num_nuisance_patches": 0.0,
#                 "low_cam_ratio": 0.0,
#                 "high_cam_ratio": 0.0,
#                 "mean_nuisance_cam": 0.0,
#             }

#         return {
#             "num_nuisance_patches": float(total_nuisance),
#             "low_cam_ratio": total_low_cam / total_nuisance,
#             "high_cam_ratio": total_high_cam / total_nuisance,
#             "mean_nuisance_cam": sum_cam / total_nuisance,
#         }

#     @torch.no_grad()
#     def evaluate_nuisance_segmentation_overlap(
#         self,
#         data_loader,
#         device,
#         mask_index: int = 2,
#         foreground_patch_threshold: float = 0.25,
#         max_batches: Optional[int] = None,
#     ) -> Dict[str, float]:
#         """
#         Dataset-level anatomical-background validation using segmentation masks.

#         Expected loader output:
#             (inputs, labels, ..., foreground_mask, ...)
#         and mask_index selects the mask field.

#         A nuisance token is counted as background when the mean foreground-mask
#         occupancy inside that layer4 token is below foreground_patch_threshold.
#         """
#         if not (0.0 <= foreground_patch_threshold <= 1.0):
#             raise ValueError("foreground_patch_threshold must be in [0,1].")

#         total_nuisance = 0
#         total_bg = 0
#         total_fg = 0
#         total_fg_fraction = 0.0

#         for batch_idx, batch in enumerate(data_loader):
#             if max_batches is not None and batch_idx >= int(max_batches):
#                 break
#             if not isinstance(batch, (tuple, list)) or len(batch) <= mask_index:
#                 raise TypeError(
#                     "data_loader must contain foreground masks at batch[{}].".format(
#                         mask_index
#                     )
#                 )
#             inputs = batch[0]
#             masks = batch[mask_index]
#             if isinstance(inputs, (tuple, list)):
#                 inputs = inputs[0]
#             inputs = inputs.to(device, non_blocking=True)
#             masks = masks.to(device, non_blocking=True).float()
#             if masks.ndim == 3:
#                 masks = masks.unsqueeze(1)
#             elif masks.ndim == 4 and masks.shape[1] != 1:
#                 masks = masks[:, :1]
#             if masks.ndim != 4:
#                 raise ValueError("foreground masks must be [B,H,W] or [B,1,H,W].")

#             # Use the frozen assignment representation to obtain exact grid size.
#             assignment_inputs = inputs.to(self._assignment_device, non_blocking=True)
#             _, assignment_features = _capture_layer4_and_forward(
#                 self._assignment_model,
#                 assignment_inputs,
#             )
#             b, _, hf, wf = assignment_features.shape
#             cluster_ids, spurious_mask, _ = self._stage2_region_partition(inputs)
#             if int(cluster_ids.shape[1]) != hf * wf:
#                 raise ValueError("Assignment region count does not match feature grid.")

#             fg_fraction = F.adaptive_avg_pool2d(masks, (hf, wf))[:, 0]
#             spur_grid = spurious_mask.view(b, hf, wf)
#             nuisance_fg = fg_fraction[spur_grid]
#             if nuisance_fg.numel() == 0:
#                 continue

#             total_nuisance += int(nuisance_fg.numel())
#             total_bg += int(
#                 (nuisance_fg < foreground_patch_threshold).sum().item()
#             )
#             total_fg += int(
#                 (nuisance_fg >= foreground_patch_threshold).sum().item()
#             )
#             total_fg_fraction += float(nuisance_fg.sum().item())

#         if total_nuisance == 0:
#             return {
#                 "num_nuisance_patches": 0.0,
#                 "seg_background_ratio": 0.0,
#                 "seg_foreground_ratio": 0.0,
#                 "mean_foreground_fraction": 0.0,
#             }

#         return {
#             "num_nuisance_patches": float(total_nuisance),
#             "seg_background_ratio": total_bg / total_nuisance,
#             "seg_foreground_ratio": total_fg / total_nuisance,
#             "mean_foreground_fraction": total_fg_fraction / total_nuisance,
#         }

#     # ------------------------------------------------------------
#     # Stage 2 public forward
#     # ------------------------------------------------------------
#     def forward(
#         self,
#         student_model: nn.Module,
#         inputs: torch.Tensor,
#         labels: torch.Tensor,
#     ) -> RaVLOutput:
#         """
#         Compute Stage-2 nuisance class-neutralization loss.

#         This function DOES NOT update the model and DOES NOT include the user's
#         ordinary classification loss.
#         """
#         if self.top_spurious_cluster is None:
#             raise RuntimeError(
#                 "Run ravl.discover(model, reference_loader) before Stage 2."
#             )

#         logits, student_features = _capture_layer4_and_forward(
#             student_model,
#             inputs,
#         )

#         if logits.shape[1] != self.num_classes:
#             raise ValueError(
#                 "Expected {} classes, got {}.".format(
#                     self.num_classes,
#                     logits.shape[1],
#                 )
#             )

#         student_regions = self._regions_from_features(
#             student_features
#         )

#         cluster_ids, spurious_mask, relevant_mask = (
#             self._stage2_region_partition(inputs)
#         )

#         classifier = _get_classifier(student_model)

#         loss_nui, valid_images = self._region_aware_loss(
#             student_regions=student_regions,
#             labels=labels,
#             spurious_mask=spurious_mask,
#             relevant_mask=relevant_mask,
#             classifier=classifier,
#         )

#         return RaVLOutput(
#             loss_region=loss_nui,
#             loss_nui=loss_nui,
#             logits=logits,
#             spurious_mask=spurious_mask.detach(),
#             relevant_mask=relevant_mask.detach(),
#             cluster_ids=cluster_ids.detach(),
#             num_spurious_regions=int(spurious_mask.sum().item()),
#             num_relevant_regions=int(relevant_mask.sum().item()),
#             num_valid_images=int(valid_images.sum().item()),
#         )

#     __call__ = forward

#     def combine_with_classification_loss(
#         self,
#         classification_loss: torch.Tensor,
#         output: RaVLOutput,
#         lambda_cl: Optional[float] = None,
#     ) -> torch.Tensor:
#         """
#         Keep the original public helper interface. The returned regularizer
#         is now the nuisance class-neutralization loss instead of L_R + L_A.
#         """
#         if lambda_cl is None:
#             lambda_cl = self.lambda_cl
#         return classification_loss +  float(lambda_cl)*output.loss_region

#     # ------------------------------------------------------------
#     # Persistence for the discovered clusters
#     # ------------------------------------------------------------
#     def save_discovery(self, path: str) -> None:
#         if self.medoids_raw is None or self.top_spurious_cluster is None:
#             raise RuntimeError("Nothing to save; run discover first.")

#         torch.save(
#             {
#                 "num_classes": self.num_classes,
#                 "region_grid": self.region_grid,
#                 "temperature": self.temperature,
#                 "lambda_cl": self.lambda_cl,
#                 "influence_threshold": self.influence_threshold,
#                 "medoids_raw": self.medoids_raw,
#                 "medoids_norm": self.medoids_norm,
#                 "top_spurious_cluster": self.top_spurious_cluster,
#             "top_spurious_clusters": self.top_spurious_clusters,
#             "num_spurious_clusters": self.num_spurious_clusters,
#                 "ranked_clusters": self.ranked_clusters,
#                 "discovery_result": self.discovery_result,
#             },
#             path,
#         )

#     def load_discovery(
#         self,
#         path: str,
#         model_for_assignment: Optional[nn.Module] = None,
#         device=None,
#     ) -> None:
#         state = torch.load(path, map_location="cpu")

#         if int(state["num_classes"]) != self.num_classes:
#             raise ValueError(
#                 "Saved num_classes={} but module num_classes={}.".format(
#                     state["num_classes"],
#                     self.num_classes,
#                 )
#             )

#         if int(state["region_grid"]) != self.region_grid:
#             raise ValueError(
#                 "Saved region_grid={} but module region_grid={}.".format(
#                     state["region_grid"],
#                     self.region_grid,
#                 )
#             )

#         self.medoids_raw = state["medoids_raw"].float()
#         self.medoids_norm = state["medoids_norm"].float()
#         self.top_spurious_cluster = int(
#             state["top_spurious_cluster"]
#         )
#         self.top_spurious_clusters = state.get(
#             "top_spurious_clusters",
#             [self.top_spurious_cluster]
#         )
#         self.ranked_clusters = list(state["ranked_clusters"])
#         self.discovery_result = state.get("discovery_result", None)

#         if model_for_assignment is not None:
#             self.prepare_stage2_assignment_model(
#                 model_for_assignment,
#                 device=device,
#             )


# # ================================================================
# # Tiny smoke test
# # ================================================================
# if __name__ == "__main__":
#     from torch.utils.data import DataLoader, TensorDataset

#     class TinyResNet(nn.Module):
#         def __init__(self, num_classes=3):
#             super().__init__()
#             self.stem = nn.Sequential(
#                 nn.Conv2d(3, 16, 3, padding=1),
#                 nn.ReLU(),
#                 nn.AdaptiveAvgPool2d((7, 7)),
#             )
#             self.layer4 = nn.Sequential(
#                 nn.Conv2d(16, 32, 3, padding=1),
#                 nn.ReLU(),
#             )
#             self.fc = nn.Linear(32, num_classes)

#         def forward(self, x):
#             x = self.stem(x)
#             x = self.layer4(x)
#             z = x.mean(dim=(2, 3))
#             return self.fc(z)

#     torch.manual_seed(7)
#     device = torch.device(
#         "cuda" if torch.cuda.is_available() else "cpu"
#     )

#     model = TinyResNet(num_classes=3).to(device)

#     ravl = RaVLResNet(
#         num_classes=3,
#         region_grid=2,
#         temperature=0.07,
#         influence_threshold=0.0,  # smoke-test only
#         k_min_factor=2,
#         k_max_factor=2,
#         max_cluster_regions=200,
#         silhouette_sample_size=200,
#         random_seed=7,
#     )

#     # Synthetic reference set.
#     ref_x = torch.randn(60, 3, 32, 32)
#     ref_y = torch.randint(0, 3, (60,))
#     ref_loader = DataLoader(
#         TensorDataset(ref_x, ref_y),
#         batch_size=12,
#         shuffle=False,
#     )

#     # Random model may still have G=0. We test the full clustering path but
#     # permit H=0 for this synthetic smoke test.
#     try:
#         discovery = ravl.discover(
#             model=model,
#             reference_loader=ref_loader,
#             device=device,
#             verbose=False,
#         )
#     except RuntimeError:
#         # For a random model, if all G/H degenerates, manually select the first
#         # discovered cluster is not possible if discovery did not commit state.
#         # Re-run is not necessary for syntax/gradient smoke testing; set a small
#         # fixed bank from reference features.
#         model.eval()
#         with torch.no_grad():
#             _, feat = _capture_layer4_and_forward(
#                 model,
#                 ref_x[:12].to(device),
#             )
#             regs = ravl._regions_from_features(feat)
#             flat = regs.reshape(-1, regs.shape[-1])
#             medoid_ids, _ = ravl._fit_kmedoids_cosine(
#                 flat,
#                 k=3,
#                 seed=7,
#             )
#             med = flat.index_select(0, medoid_ids)
#             ravl.medoids_raw = med.detach().cpu()
#             ravl.medoids_norm = F.normalize(
#                 med,
#                 dim=1,
#             ).detach().cpu()
#             ravl.top_spurious_cluster = 0
#             ravl.top_spurious_clusters = [0]
#             ravl.ranked_clusters = [0]
#             ravl.prepare_stage2_assignment_model(
#                 model,
#                 device=device,
#             )

#     model.train()

#     x = torch.randn(8, 3, 32, 32, device=device)
#     y = torch.randint(0, 3, (8,), device=device)

#     out = ravl(
#         student_model=model,
#         inputs=x,
#         labels=y,
#     )

#     loss_cls = F.cross_entropy(out.logits, y)
#     loss = ravl.combine_with_classification_loss(
#         loss_cls,
#         out,
#     )

#     optimizer = torch.optim.SGD(
#         model.parameters(),
#         lr=1e-3,
#     )
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

#     print("Conditional-MI + uniform nuisance purification smoke test passed.")
#     print(out.statistics())








# from __future__ import annotations

# from dataclasses import dataclass
# import copy
# import math
# import os
# from pathlib import Path
# import random
# from typing import Dict, List, Optional, Tuple

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# import numpy as np
# from PIL import Image
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# @dataclass
# class RaVLDiscoveryResult:
#     best_k: int
#     best_silhouette: float
#     top_spurious_cluster: int
#     ranked_clusters: List[int]
#     influence_scores: Dict[int, float]
#     performance_gaps: Dict[int, float]
#     per_class_gaps: Dict[int, Dict[int, float]]
#     per_class_influence: Dict[int, Dict[int, float]]
#     num_images: int
#     num_regions: int

#     def summary(self) -> Dict[str, object]:
#         return {
#             "best_k": self.best_k,
#             "best_silhouette": self.best_silhouette,
#             "top_spurious_cluster": self.top_spurious_cluster,
#             "ranked_clusters": self.ranked_clusters,
#             "influence_scores": self.influence_scores,
#             "performance_gaps": self.performance_gaps,
#             "num_images": self.num_images,
#             "num_regions": self.num_regions,
#         }


# @dataclass
# class RaVLOutput:
#     loss_region: torch.Tensor
#     loss_nui: torch.Tensor
#     logits: torch.Tensor
#     spurious_mask: torch.Tensor
#     relevant_mask: torch.Tensor
#     cluster_ids: torch.Tensor
#     num_spurious_regions: int
#     num_relevant_regions: int
#     num_valid_images: int

#     # Backward-compatible aliases. L_R/L_A are no longer used in Stage 2.
#     # Keeping these properties avoids breaking old logging code immediately.
#     @property
#     def loss_R(self) -> torch.Tensor:
#         return self.loss_nui

#     @property
#     def loss_A(self) -> torch.Tensor:
#         return self.loss_nui.new_zeros(())

#     def statistics(self) -> Dict[str, float]:
#         return {
#             "loss_region": float(self.loss_region.detach().item()),
#             "loss_nui": float(self.loss_nui.detach().item()),
#             # backward-compatible logging keys
#             "loss_R": float(self.loss_R.detach().item()),
#             "loss_A": float(self.loss_A.detach().item()),
#             "num_spurious_regions": float(self.num_spurious_regions),
#             "num_relevant_regions": float(self.num_relevant_regions),
#             "num_valid_images": float(self.num_valid_images),
#         }


# def _unwrap_model(model: nn.Module) -> nn.Module:
#     return model.module if hasattr(model, "module") else model


# def _unwrap_tensor(x, name: str) -> torch.Tensor:
#     if torch.is_tensor(x):
#         return x
#     if isinstance(x, (tuple, list)) and len(x) > 0 and torch.is_tensor(x[0]):
#         return x[0]
#     raise TypeError(
#         "{} must be Tensor or tuple/list whose first item is Tensor.".format(name)
#     )


# def _get_classifier(model: nn.Module) -> nn.Linear:
#     base = _unwrap_model(model)
#     candidates = []

#     if hasattr(base, "get_classifier"):
#         try:
#             candidates.append(base.get_classifier())
#         except Exception:
#             pass

#     for name in ("fc", "head", "classifier"):
#         if hasattr(base, name):
#             candidates.append(getattr(base, name))

#     for candidate in candidates:
#         if isinstance(candidate, nn.Linear):
#             return candidate
#         if isinstance(candidate, nn.Sequential):
#             for sub in reversed(candidate):
#                 if isinstance(sub, nn.Linear):
#                     return sub

#     raise AttributeError("Unable to locate final nn.Linear classifier.")


# def _capture_layer4_and_forward(
#     model: nn.Module,
#     inputs: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     base = _unwrap_model(model)

#     if not hasattr(base, "layer4"):
#         raise AttributeError(
#             "The model must be ResNet-like and contain model.layer4."
#         )

#     holder = {}

#     def hook(_module, _inputs, output):
#         holder["features"] = _unwrap_tensor(output, "layer4 output")

#     handle = base.layer4.register_forward_hook(hook)

#     try:
#         logits = _unwrap_tensor(model(inputs), "model(inputs)")
#     finally:
#         handle.remove()

#     if "features" not in holder:
#         raise RuntimeError("Failed to capture layer4 features.")

#     features = holder["features"]

#     if features.ndim != 4:
#         raise ValueError(
#             "layer4 must be [B,D,H,W], got {}".format(tuple(features.shape))
#         )

#     if logits.ndim != 2:
#         raise ValueError(
#             "logits must be [B,C], got {}".format(tuple(logits.shape))
#         )

#     return logits, features



# class RaVLResNet(object):
#     """
#     RaVL-style discovery + mitigation migrated from the NeurIPS 2024 RaVL
#     algorithm to a supervised ResNet classifier.

#     What is kept from the original RaVL:
#       Stage 1:
#         1) labeled reference/validation set
#         2) local candidate regions from correctly predicted reference samples
#         3) K-Medoids with cosine distance on the correct-only region pool
#         4) choose K by Silhouette score over [2|Y|, 5|Y|]
#         5) class-conditional MI I(S_k; Y_hat | Y=c)
#         6) harmful error-increase gating
#         7) class-balanced conditional nuisance score N_k
#         8) select the top-ranked nuisance cluster(s)

#       Stage 2:
#         1) keep the Stage-1 clustering model fixed
#         2) split regions into R_i^s and R_i^r
#         3) remove L_R and L_A
#         4) make nuisance-region class predictions maximally uncertain
#         5) minimize KL(q(Y|z_s) || Uniform(C))

#     Necessary modality substitutions:
#       - VLM text class embedding g(y) -> normalized frozen FC class direction w_y
#       - RoI candidate regions -> equal grid regions pooled from ResNet layer4
#       - paired-text assigned label y_hat -> supervised ground-truth class label y

#     Important:
#       - The clustering medoids and the Stage-1 assignment encoder remain fixed.
#       - The student/model training strategy is external to this class.
#       - This class introduces no learnable parameters.
#     """

#     def __init__(
#         self,
#         num_classes: int,
#         region_grid: int = 3,
#         temperature: float = 0.07,
#         lambda_cl: float = 0.80,
#         influence_threshold: float = 0.25,
#         num_spurious_clusters: int = 1,
#         k_min_factor: int = 2,
#         k_max_factor: int = 5,
#         kmedoids_iterations: int = 30,
#         max_cluster_regions: Optional[int] = 20000,
#         silhouette_sample_size: int = 3000,
#         assignment_chunk_size: int = 8192,
#         random_seed: int = 0,
#     ) -> None:
#         if num_classes < 2:
#             raise ValueError("num_classes must be >= 2")
#         if region_grid < 1:
#             raise ValueError("region_grid must be >= 1")
#         if temperature <= 0:
#             raise ValueError("temperature must be > 0")
#         if not 0 <= lambda_cl <= 1:
#             raise ValueError("lambda_cl must be in [0,1]")
#         if influence_threshold < 0:
#             raise ValueError("influence_threshold must be >= 0")
#         if k_min_factor < 1 or k_max_factor < k_min_factor:
#             raise ValueError("invalid cluster-count factors")

#         self.num_classes = int(num_classes)
#         self.region_grid = int(region_grid)
#         self.temperature = float(temperature)
#         self.lambda_cl = float(lambda_cl)
#         self.influence_threshold = float(influence_threshold)
#         self.num_spurious_clusters = int(num_spurious_clusters)
#         self.k_min_factor = int(k_min_factor)
#         self.k_max_factor = int(k_max_factor)
#         self.kmedoids_iterations = int(kmedoids_iterations)
#         self.max_cluster_regions = max_cluster_regions
#         self.silhouette_sample_size = int(silhouette_sample_size)
#         self.assignment_chunk_size = int(assignment_chunk_size)
#         self.random_seed = int(random_seed)
#         self.eps = 1e-8

#         self.medoids_raw = None
#         self.medoids_norm = None
#         self.top_spurious_cluster = None
#         self.top_spurious_clusters = []
#         self.ranked_clusters = []
#         self.discovery_result = None

#         # Frozen copy of the Stage-1 model.
#         # It is used only for assigning Stage-2 regions to the fixed clusters.
#         self._assignment_model = None
#         self._assignment_device = None

#     # ------------------------------------------------------------
#     # Region construction
#     # ------------------------------------------------------------
#     def _regions_from_features(
#             self,
#             features: torch.Tensor,
#     ):
#         """
#         Directly use layer4 spatial tokens as regions.

#         Input:
#             features:
#                 [B,C,H,W]

#         Output:
#             regions:
#                 [B,H*W,C]

#         For ResNet50 layer4:
#             [B,2048,7,7]
#             ->
#             [B,49,2048]
#         """

#         B, C, H, W = features.shape
#         regions = (
#             features
#             .flatten(2)
#             .transpose(1, 2)
#         )
#         return regions

#     # ------------------------------------------------------------
#     # Cosine K-Medoids
#     # ------------------------------------------------------------
#     @torch.no_grad()
#     def _fit_kmedoids_cosine(
#         self,
#         x_raw: torch.Tensor,
#         k: int,
#         seed: int,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Alternating K-Medoids for cosine distance.

#         For fixed assignments and normalized samples, the exact medoid of a
#         cluster is the member maximizing similarity to the sum of cluster
#         members, so the update can be done without an O(n_cluster^2) matrix.

#         Returns:
#             medoid_ids: [k]
#             labels: [N]
#         """
#         n = int(x_raw.shape[0])

#         if k < 2 or k >= n:
#             raise ValueError("K-Medoids requires 2 <= k < N.")

#         x = F.normalize(
#             x_raw.float(),
#             p=2,
#             dim=1,
#             eps=self.eps,
#         )

#         gen = torch.Generator(device=x.device)
#         gen.manual_seed(int(seed))

#         first = int(
#             torch.randint(
#                 low=0,
#                 high=n,
#                 size=(1,),
#                 generator=gen,
#                 device=x.device,
#             ).item()
#         )

#         selected = [first]

#         # Farthest-point initialization in cosine distance.
#         min_dist = 1.0 - (x @ x[first:first + 1].t()).squeeze(1)

#         for _ in range(1, k):
#             idx = int(min_dist.argmax().item())
#             selected.append(idx)
#             dist = 1.0 - (x @ x[idx:idx + 1].t()).squeeze(1)
#             min_dist = torch.minimum(min_dist, dist)

#         medoid_ids = torch.tensor(
#             selected,
#             device=x.device,
#             dtype=torch.long,
#         )

#         old_labels = None

#         for _ in range(self.kmedoids_iterations):
#             medoids = x.index_select(0, medoid_ids)
#             sim = x @ medoids.t()
#             labels = sim.argmax(dim=1)

#             if old_labels is not None and torch.equal(labels, old_labels):
#                 break

#             old_labels = labels.clone()
#             new_ids = []

#             for cluster_id in range(k):
#                 members = labels.eq(cluster_id).nonzero(
#                     as_tuple=False
#                 ).squeeze(1)

#                 if members.numel() == 0:
#                     # Re-seed from the currently worst represented point.
#                     max_sim = sim.max(dim=1).values
#                     candidate = int(max_sim.argmin().item())
#                     new_ids.append(candidate)
#                     continue

#                 member_x = x.index_select(0, members)
#                 sum_direction = member_x.sum(dim=0)

#                 # Exact cosine-medoid criterion for the fixed cluster:
#                 # argmax_i sum_j cos(x_i, x_j)
#                 medoid_score = member_x @ sum_direction
#                 local_id = int(medoid_score.argmax().item())
#                 new_ids.append(int(members[local_id].item()))

#             new_ids_tensor = torch.tensor(
#                 new_ids,
#                 device=x.device,
#                 dtype=torch.long,
#             )

#             if torch.equal(new_ids_tensor, medoid_ids):
#                 medoid_ids = new_ids_tensor
#                 break

#             medoid_ids = new_ids_tensor

#         final_medoids = x.index_select(0, medoid_ids)
#         final_labels = (x @ final_medoids.t()).argmax(dim=1)

#         return medoid_ids, final_labels

#     @torch.no_grad()
#     def _silhouette_cosine(
#         self,
#         x_raw: torch.Tensor,
#         labels: torch.Tensor,
#         seed: int,
#     ) -> float:
#         """
#         Cosine Silhouette score.

#         If N > silhouette_sample_size, a deterministic random subset is used.
#         This is only an engineering memory cap; set silhouette_sample_size >= N
#         to recover the full score.
#         """
#         n = int(x_raw.shape[0])

#         if n < 3:
#             return -1.0

#         unique_labels = labels.unique()
#         if unique_labels.numel() < 2:
#             return -1.0

#         s = min(n, self.silhouette_sample_size)

#         if s < n:
#             gen = torch.Generator(device=x_raw.device)
#             gen.manual_seed(int(seed))
#             sample_ids = torch.randperm(
#                 n,
#                 generator=gen,
#                 device=x_raw.device,
#             )[:s]
#             x = x_raw.index_select(0, sample_ids)
#             y = labels.index_select(0, sample_ids)
#         else:
#             x = x_raw
#             y = labels

#         x = F.normalize(x.float(), p=2, dim=1, eps=self.eps)
#         dist = 1.0 - x @ x.t()
#         dist = dist.clamp_min(0.0)

#         sil = torch.zeros(
#             x.shape[0],
#             device=x.device,
#             dtype=x.dtype,
#         )

#         all_clusters = y.unique()

#         for i in range(x.shape[0]):
#             own = y[i]
#             own_mask = y.eq(own)
#             own_count = int(own_mask.sum().item())

#             if own_count <= 1:
#                 sil[i] = 0.0
#                 continue

#             a = dist[i, own_mask].sum() / float(own_count - 1)

#             b = None
#             for c in all_clusters:
#                 if int(c.item()) == int(own.item()):
#                     continue

#                 mask = y.eq(c)
#                 if not bool(mask.any().item()):
#                     continue

#                 mean_dist = dist[i, mask].mean()
#                 if b is None or mean_dist < b:
#                     b = mean_dist

#             if b is None:
#                 sil[i] = 0.0
#                 continue

#             denom = torch.maximum(a, b).clamp_min(self.eps)
#             sil[i] = (b - a) / denom

#         return float(sil.mean().item())

#     @torch.no_grad()
#     def _assign_to_medoids(
#         self,
#         regions: torch.Tensor,
#         medoids_norm: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         """
#         Assign [N,D] or [B,R,D] region embeddings to the fixed medoids
#         using cosine distance.
#         """
#         if medoids_norm is None:
#             if self.medoids_norm is None:
#                 raise RuntimeError("RaVL discovery has not been run.")
#             medoids_norm = self.medoids_norm

#         original_shape = regions.shape[:-1]
#         d = regions.shape[-1]
#         flat = regions.reshape(-1, d)

#         outputs = []
#         medoids_norm = medoids_norm.to(
#             device=flat.device,
#             dtype=flat.dtype,
#         )

#         for start in range(0, flat.shape[0], self.assignment_chunk_size):
#             end = min(start + self.assignment_chunk_size, flat.shape[0])
#             x = F.normalize(
#                 flat[start:end],
#                 p=2,
#                 dim=1,
#                 eps=self.eps,
#             )
#             outputs.append((x @ medoids_norm.t()).argmax(dim=1))

#         assigned = torch.cat(outputs, dim=0)
#         return assigned.view(*original_shape)

#     # ------------------------------------------------------------
#     # Stage 1: RaVL discovery
#     # ------------------------------------------------------------
#     @torch.no_grad()
#     def discover(
#         self,
#         model: nn.Module,
#         reference_loader,
#         device=None,
#         verbose: bool = True,
#         make_assignment_snapshot: bool = True,
#     ) -> RaVLDiscoveryResult:
#         """
#         Stage-1 discovery adapted to a supervised ResNet. K-Medoids
#         prototypes are constructed only from correctly predicted reference
#         samples; all reference samples are then assigned for nuisance scoring.

#         reference_loader must yield:
#             (inputs, labels)
#         or:
#             (inputs, labels, ...)
#         """
#         if device is None:
#             device = next(model.parameters()).device

#         was_training = model.training
#         model.eval()

#         classifier = _get_classifier(model)

#         all_regions = []
#         all_region_probs = []
#         all_labels = []
#         all_preds = []

#         try:
#             for batch in reference_loader:
#                 if not isinstance(batch, (tuple, list)) or len(batch) < 2:
#                     raise TypeError(
#                         "reference_loader must return (inputs, labels) or "
#                         "(inputs, labels, ...)."
#                     )

#                 inputs = batch[0]
#                 labels = batch[1]

#                 if isinstance(inputs, (tuple, list)):
#                     inputs = inputs[0]

#                 inputs = inputs.to(device, non_blocking=True)
#                 labels = labels.to(device, non_blocking=True).long()

#                 logits, features = _capture_layer4_and_forward(
#                     model,
#                     inputs,
#                 )

#                 if logits.shape[1] != self.num_classes:
#                     raise ValueError(
#                         "Expected {} classes, got {}.".format(
#                             self.num_classes,
#                             logits.shape[1],
#                         )
#                     )

#                 regions = self._regions_from_features(features)
#                 b, r, d = regions.shape

#                 if classifier.weight.shape[1] != d:
#                     raise ValueError(
#                         "Classifier input dim {} != region dim {}.".format(
#                             classifier.weight.shape[1],
#                             d,
#                         )
#                     )

#                 region_logits = classifier(
#                     regions.reshape(b * r, d)
#                 ).view(b, r, self.num_classes)

#                 region_probs = F.softmax(region_logits, dim=2)

#                 pred = logits.argmax(dim=1)

#                 all_regions.append(
#                     regions.detach().cpu().to(torch.float16)
#                 )
#                 all_region_probs.append(
#                     region_probs.detach().cpu().to(torch.float16)
#                 )
#                 all_labels.append(labels.detach().cpu())
#                 all_preds.append(pred.detach().cpu())

#         finally:
#             if was_training:
#                 model.train()
#             else:
#                 model.eval()

#         if len(all_regions) == 0:
#             raise RuntimeError("Reference loader is empty.")

#         regions_img = torch.cat(all_regions, dim=0).float()
#         region_probs_img = torch.cat(all_region_probs, dim=0).float()
#         labels_img = torch.cat(all_labels, dim=0).long()
#         preds_img = torch.cat(all_preds, dim=0).long()
#         correct_img = preds_img.eq(labels_img)

#         n_img, n_region, d = regions_img.shape
#         regions_flat = regions_img.reshape(n_img * n_region, d)

#         # --------------------------------------------------------
#         # Clustering pool: CORRECTLY PREDICTED samples only.
#         # --------------------------------------------------------
#         # Important design:
#         #   1) Correct samples are used only to CONSTRUCT the visual-pattern
#         #      dictionary (K-Medoids / medoids).
#         #   2) After the medoids are fixed, ALL reference samples (correct +
#         #      incorrect) are assigned to these medoids.
#         #   3) Conditional-MI nuisance scoring is still computed on ALL
#         #      reference samples, so incorrect samples remain essential for
#         #      identifying whether a pattern is harmful.
#         #
#         # This avoids allowing under-learned regions from misclassified samples
#         # to directly define the clustering prototypes, while retaining them in
#         # the subsequent nuisance-effect estimation.
#         total_regions = int(regions_flat.shape[0])

#         correct_image_ids = correct_img.nonzero(
#             as_tuple=False
#         ).squeeze(1)
#         num_correct_images = int(correct_image_ids.numel())

#         if num_correct_images == 0:
#             raise RuntimeError(
#                 "No correctly predicted sample exists in the reference set. "
#                 "Correct-only clustering cannot be constructed."
#             )

#         # [N_correct, R, D] -> [N_correct * R, D]
#         correct_regions_img = regions_img.index_select(
#             0,
#             correct_image_ids,
#         )
#         correct_regions_flat = correct_regions_img.reshape(
#             -1,
#             d,
#         )
#         total_correct_regions = int(correct_regions_flat.shape[0])

#         if total_correct_regions < 3:
#             raise RuntimeError(
#                 "Correct-only clustering requires at least 3 correct-region "
#                 "features, got {}.".format(total_correct_regions)
#             )

#         # Optional memory cap is applied ONLY to the correct-region clustering
#         # pool. The later all-sample assignment is unchanged.
#         if (
#             self.max_cluster_regions is not None
#             and total_correct_regions > int(self.max_cluster_regions)
#         ):
#             gen = torch.Generator()
#             gen.manual_seed(self.random_seed)
#             cluster_region_ids = torch.randperm(
#                 total_correct_regions,
#                 generator=gen,
#             )[: int(self.max_cluster_regions)]
#             cluster_pool_cpu = correct_regions_flat.index_select(
#                 0,
#                 cluster_region_ids,
#             )
#         else:
#             cluster_pool_cpu = correct_regions_flat

#         cluster_pool = cluster_pool_cpu.to(
#             device=device,
#             dtype=torch.float32,
#         )

#         n_pool = int(cluster_pool.shape[0])

#         k_min = self.num_classes * self.k_min_factor
#         k_max = self.num_classes * self.k_max_factor

#         k_min = max(2, min(k_min, n_pool - 1))
#         k_max = max(k_min, min(k_max, n_pool - 1))

#         if k_min >= n_pool:
#             raise RuntimeError(
#                 "Not enough candidate regions for RaVL clustering."
#             )

#         best_score = -float("inf")
#         best_k = None
#         best_medoid_ids = None

#         if verbose:
#             print("========== RaVL Stage 1: K-Medoids sweep ==========")
#             correct_rate = num_correct_images / float(max(n_img, 1))
#             print(
#                 "images={} | correct_images={} ({:.2%}) | regions/image={} | "
#                 "all_regions={} | correct_regions={} | cluster_pool={}".format(
#                     n_img,
#                     num_correct_images,
#                     correct_rate,
#                     n_region,
#                     total_regions,
#                     total_correct_regions,
#                     n_pool,
#                 )
#             )
#             # Helpful diagnostic under long-tailed data: clustering still uses
#             # all correctly predicted samples, but this print lets you inspect
#             # whether correct samples are strongly head-class dominated.
#             correct_per_class = []
#             for class_id in range(self.num_classes):
#                 class_correct = int(
#                     (correct_img & labels_img.eq(class_id)).sum().item()
#                 )
#                 correct_per_class.append(class_correct)
#             print("correct images per class = {}".format(correct_per_class))
#             print("K range: {} -> {}".format(k_min, k_max))

#         for k in range(k_min, k_max + 1):
#             medoid_ids, cluster_labels = self._fit_kmedoids_cosine(
#                 cluster_pool,
#                 k=k,
#                 seed=self.random_seed + k,
#             )

#             score = self._silhouette_cosine(
#                 cluster_pool,
#                 cluster_labels,
#                 seed=self.random_seed + 1000 + k,
#             )

#             if verbose:
#                 print(
#                     "K={:3d} | silhouette={:.6f}".format(
#                         k,
#                         score,
#                     )
#                 )

#             if score > best_score:
#                 best_score = score
#                 best_k = k
#                 best_medoid_ids = medoid_ids.detach().clone()

#         best_medoids_raw = cluster_pool.index_select(
#             0,
#             best_medoid_ids,
#         ).detach()

#         best_medoids_norm = F.normalize(
#             best_medoids_raw,
#             p=2,
#             dim=1,
#             eps=self.eps,
#         )

#         # Assign ALL reference regions using the selected fixed medoids.
#         assignments_flat = self._assign_to_medoids(
#             regions_flat.to(device=device, dtype=torch.float32),
#             medoids_norm=best_medoids_norm,
#         ).cpu()

#         assignments_img = assignments_flat.view(
#             n_img,
#             n_region,
#         )

#         # --------------------------------------------------------
#         # Conditional-MI nuisance cluster selection.
#         #
#         # Replace the original RaVL H_k / G_k selection only.
#         # Everything after cluster selection remains unchanged.
#         #
#         # For cluster k:
#         #   S_k = 1 if an image contains at least one region from k.
#         #
#         # For every ground-truth class y, estimate
#         #   I(S_k ; Y_hat | Y=y)
#         # and retain only the harmful direction
#         #   Delta_err = P(E=1 | S_k=1,Y=y)
#         #             - P(E=1 | S_k=0,Y=y).
#         #
#         # Per-class nuisance score:
#         #   N_{k,y} = balance * I(S_k;Y_hat|Y=y) * max(Delta_err, 0)
#         #
#         # Cluster score:
#         #   N_k = sum_y N_{k,y}
#         #
#         # The sum is class-balanced: each class contributes once rather
#         # than being weighted by its sample frequency, which is desirable
#         # for long-tailed recognition.
#         # --------------------------------------------------------

#         # Keep the original result-field names for backward compatibility:
#         #   influence_scores      -> normalized conditional nuisance score
#         #   performance_gaps      -> raw conditional nuisance score N_k
#         #   per_class_gaps        -> per-class error increase Delta_err
#         #   per_class_influence   -> per-class conditional MI
#         influence_scores = {}
#         performance_gaps = {}
#         per_class_gaps = {}
#         per_class_influence = {}

#         # Extra local dictionaries used only during selection / logging.
#         per_class_nuisance = {}
#         per_class_balance = {}

#         def _conditional_mi_binary_cluster(
#             cluster_present: torch.Tensor,
#             predicted_labels: torch.Tensor,
#         ) -> float:
#             """
#             Empirical mutual information I(S; Y_hat) in nats.

#             cluster_present: [N], bool, S in {0,1}
#             predicted_labels: [N], long, Y_hat in {0,...,C-1}

#             This function is called inside a fixed ground-truth class,
#             therefore it estimates I(S_k; Y_hat | Y=y).
#             """
#             n = int(cluster_present.numel())
#             if n <= 1:
#                 return 0.0

#             s = cluster_present.long()
#             y_hat = predicted_labels.long()

#             joint = torch.zeros(
#                 2,
#                 self.num_classes,
#                 dtype=torch.float64,
#             )

#             flat_index = s * self.num_classes + y_hat
#             counts = torch.bincount(
#                 flat_index,
#                 minlength=2 * self.num_classes,
#             ).double()
#             joint = counts.view(2, self.num_classes)

#             total = joint.sum()
#             if float(total.item()) <= 0.0:
#                 return 0.0

#             p_joint = joint / total
#             p_s = p_joint.sum(dim=1, keepdim=True)
#             p_yhat = p_joint.sum(dim=0, keepdim=True)
#             denom = p_s * p_yhat

#             valid = p_joint > 0
#             if not bool(valid.any().item()):
#                 return 0.0

#             mi = (
#                 p_joint[valid]
#                 * torch.log(
#                     p_joint[valid]
#                     / denom[valid].clamp_min(self.eps)
#                 )
#             ).sum()

#             return float(mi.item())

#         for cluster_id in range(best_k):
#             # S_k for every reference image.
#             present = assignments_img.eq(cluster_id).any(dim=1)

#             mi_by_class = {}
#             error_increase_by_class = {}
#             nuisance_by_class = {}
#             balance_by_class = {}

#             for y in range(self.num_classes):
#                 class_mask = labels_img.eq(y)

#                 # Need both S_k=1 and S_k=0 inside this class so that the
#                 # cluster-presence variable is actually comparable.
#                 in_mask = class_mask & present
#                 out_mask = class_mask & (~present)

#                 n_in = int(in_mask.sum().item())
#                 n_out = int(out_mask.sum().item())

#                 if n_in == 0 or n_out == 0:
#                     continue

#                 # ----------------------------------------------------
#                 # 1) Harmful direction: increase in error probability.
#                 # ----------------------------------------------------
#                 err_in = float(
#                     (~correct_img[in_mask]).float().mean().item()
#                 )
#                 err_out = float(
#                     (~correct_img[out_mask]).float().mean().item()
#                 )

#                 delta_err = err_in - err_out

#                 # Same presence/absence balancing idea as the old code,
#                 # but it now stabilizes the conditional-MI score rather
#                 # than defining G_k.
#                 balance = (
#                     2.0
#                     * min(n_in, n_out)
#                     / float(n_in + n_out)
#                 )

#                 # ----------------------------------------------------
#                 # 2) Conditional MI:
#                 #       I(S_k ; Y_hat | Y=y)
#                 # ----------------------------------------------------
#                 class_present = present[class_mask]
#                 class_preds = preds_img[class_mask]

#                 cmi_y = _conditional_mi_binary_cluster(
#                     cluster_present=class_present,
#                     predicted_labels=class_preds,
#                 )

#                 # ----------------------------------------------------
#                 # 3) Per-class nuisance score.
#                 # Only the harmful direction is kept.
#                 # ----------------------------------------------------
#                 harmful_delta = max(delta_err, 0.0)
#                 nuisance_y = balance * cmi_y * harmful_delta

#                 mi_by_class[y] = float(cmi_y)
#                 error_increase_by_class[y] = float(delta_err)
#                 nuisance_by_class[y] = float(nuisance_y)
#                 balance_by_class[y] = float(balance)

#             # Class-balanced aggregation. We intentionally do NOT multiply
#             # by the empirical class prior, otherwise head classes would
#             # dominate the cluster ranking in a long-tailed dataset.
#             nuisance_score = sum(nuisance_by_class.values())

#             performance_gaps[cluster_id] = float(nuisance_score)
#             per_class_gaps[cluster_id] = error_increase_by_class
#             per_class_influence[cluster_id] = mi_by_class
#             per_class_nuisance[cluster_id] = nuisance_by_class
#             per_class_balance[cluster_id] = balance_by_class

#         # Normalize only for thresholding so the existing constructor
#         # argument influence_threshold can be kept unchanged.
#         max_nuisance_score = max(
#             performance_gaps.values()
#         ) if len(performance_gaps) > 0 else 0.0

#         for cluster_id in range(best_k):
#             raw_score = performance_gaps.get(cluster_id, 0.0)
#             if max_nuisance_score > self.eps:
#                 normalized_score = raw_score / max_nuisance_score
#             else:
#                 normalized_score = 0.0
#             influence_scores[cluster_id] = float(normalized_score)

#         candidate_clusters = [
#             c
#             for c in range(best_k)
#             if performance_gaps.get(c, 0.0) > 0.0
#             and influence_scores.get(c, 0.0) >= self.influence_threshold
#         ]

#         # Rank by the raw conditional nuisance score N_k.
#         ranked_clusters = sorted(
#             candidate_clusters,
#             key=lambda c: performance_gaps.get(c, 0.0),
#             reverse=True,
#         )

#         if len(ranked_clusters) == 0:
#             raise RuntimeError(
#                 "Conditional-MI selection found no nuisance cluster with "
#                 "positive harmful score and normalized score >= {:.3f}. "
#                 "Try inspecting the reference split or lowering "
#                 "influence_threshold.".format(
#                     self.influence_threshold
#                 )
#             )

#         top_spurious_clusters = [int(c) for c in ranked_clusters[:self.num_spurious_clusters]]

#         # backward compatible: keep original single cluster variable
#         top_spurious_cluster = int(top_spurious_clusters[0])

#         # Keep the fixed Stage-1 clustering model.
#         self.medoids_raw = best_medoids_raw.detach().cpu()
#         self.medoids_norm = best_medoids_norm.detach().cpu()
#         self.top_spurious_cluster = top_spurious_cluster
#         self.top_spurious_clusters = top_spurious_clusters
#         self.ranked_clusters = list(ranked_clusters)

#         result = RaVLDiscoveryResult(
#             best_k=int(best_k),
#             best_silhouette=float(best_score),
#             top_spurious_cluster=top_spurious_cluster,
#             ranked_clusters=list(ranked_clusters),
#             influence_scores=influence_scores,
#             performance_gaps=performance_gaps,
#             per_class_gaps=per_class_gaps,
#             per_class_influence=per_class_influence,
#             num_images=int(n_img),
#             num_regions=int(total_regions),
#         )

#         self.discovery_result = result

#         if make_assignment_snapshot:
#             self.prepare_stage2_assignment_model(
#                 model=model,
#                 device=device,
#             )

#         if verbose:
#             print("========== RaVL Stage 1: Discovery Result ==========")
#             print(
#                 "best K={} | silhouette={:.6f}".format(
#                     best_k,
#                     best_score,
#                 )
#             )
#             print(
#                 "normalized nuisance-score threshold={:.3f}".format(
#                     self.influence_threshold
#                 )
#             )

#             for rank, cluster_id in enumerate(ranked_clusters[:10], 1):
#                 print(
#                     "Rank {:2d} | cluster {:3d} | N_norm={:.4f} | N_score={:.6f}".format(
#                         rank,
#                         cluster_id,
#                         influence_scores[cluster_id],
#                         performance_gaps[cluster_id],
#                     )
#                 )

#             print(
#                 "TOP spurious clusters = {}".format(
#                     top_spurious_clusters
#                 )
#             )
#             print("=====================================================")

#         return result

#     # ------------------------------------------------------------
#     # Fixed Stage-1 assignment model for Stage 2
#     # ------------------------------------------------------------
#     @torch.no_grad()
#     def prepare_stage2_assignment_model(
#         self,
#         model: nn.Module,
#         device=None,
#     ) -> None:
#         """
#         Freeze a snapshot of the Stage-1 encoder.

#         The paper first determines the spurious cluster, then uses the trained
#         clustering model to assign training regions to R^s / R^r before
#         mitigation. This frozen snapshot keeps region assignment in the
#         original Stage-1 feature space even while the student backbone changes.
#         """
#         base = _unwrap_model(model)

#         if device is None:
#             device = next(base.parameters()).device

#         frozen = copy.deepcopy(base).to(device)
#         frozen.eval()

#         for p in frozen.parameters():
#             p.requires_grad_(False)

#         # Bypass nn.Module registration because RaVLResNet is intentionally
#         # a parameter-free utility object.
#         self._assignment_model = frozen
#         self._assignment_device = device

#     # ------------------------------------------------------------
#     # Region assignment during Stage 2
#     # ------------------------------------------------------------
#     @torch.no_grad()
#     def _stage2_region_partition(
#         self,
#         inputs: torch.Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         if self._assignment_model is None:
#             raise RuntimeError(
#                 "No frozen Stage-1 assignment model. "
#                 "Run discover(..., make_assignment_snapshot=True) or call "
#                 "prepare_stage2_assignment_model(model)."
#             )

#         if self.medoids_norm is None or self.top_spurious_cluster is None:
#             raise RuntimeError("RaVL discovery has not been initialized.")

#         assignment_inputs = inputs.to(
#             self._assignment_device,
#             non_blocking=True,
#         )

#         _, assignment_features = _capture_layer4_and_forward(
#             self._assignment_model,
#             assignment_inputs,
#         )

#         assignment_regions = self._regions_from_features(
#             assignment_features
#         )

#         medoids_norm = self.medoids_norm.to(
#             device=assignment_regions.device,
#             dtype=assignment_regions.dtype,
#         )

#         cluster_ids = self._assign_to_medoids(
#             assignment_regions,
#             medoids_norm=medoids_norm,
#         )

#         spurious_mask = torch.zeros_like(
#             cluster_ids, dtype=torch.bool
#         )

#         for c in self.top_spurious_clusters:
#             spurious_mask |= cluster_ids.eq(int(c))
#         relevant_mask = ~spurious_mask

#         return (
#             cluster_ids,
#             spurious_mask,
#             relevant_mask,
#         )

#     # ------------------------------------------------------------
#     # Nuisance class-neutralization loss
#     # ------------------------------------------------------------
#     def _region_aware_loss(
#         self,
#         student_regions: torch.Tensor,
#         labels: torch.Tensor,
#         spurious_mask: torch.Tensor,
#         relevant_mask: torch.Tensor,
#         classifier: nn.Linear,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Replace the original RaVL L_R + L_A by a direct nuisance
#         class-neutralization objective.

#         For every discovered nuisance region z_s, compute its class posterior
#         using cosine similarity to the DETACHED classifier directions:

#             q(c | z_s) = softmax(cos(z_s, w_c) / tau).

#         Then minimize

#             KL(q(Y | z_s) || U_C)
#             = log(C) - H(q(Y | z_s)),

#         where U_C is the uniform distribution over C task classes.
#         Therefore, minimizing this loss is exactly equivalent to maximizing
#         the class-prediction entropy of nuisance regions. The classifier
#         weights are detached so the backbone must neutralize nuisance-region
#         representations rather than moving the class anchors.

#         Returns:
#             loss_nui: scalar nuisance neutralization loss
#             valid_images: [B] bool, True if an image contains >=1 nuisance region
#         """
#         b, r, d = student_regions.shape

#         if classifier.weight.shape != (self.num_classes, d):
#             raise ValueError(
#                 "Classifier weight shape {} incompatible with regions [*,{},{}].".format(
#                     tuple(classifier.weight.shape),
#                     r,
#                     d,
#                 )
#             )

#         device = student_regions.device
#         spurious_mask = spurious_mask.to(device=device, dtype=torch.bool)
#         relevant_mask = relevant_mask.to(device=device, dtype=torch.bool)

#         # labels are intentionally not used by this loss. Once a region has
#         # been discovered as nuisance, it is required to be neutral with
#         # respect to ALL task classes rather than only the image ground truth.
#         _ = labels
#         _ = relevant_mask

#         # Which images contribute at least one nuisance patch.
#         valid_images = spurious_mask.any(dim=1)

#         # No nuisance patch in the current batch -> differentiable zero.
#         if not bool(spurious_mask.any().item()):
#             zero = student_regions.sum() * 0.0
#             return zero, valid_images

#         # Normalize region features and freeze the FC class directions.
#         region_n = F.normalize(
#             student_regions,
#             p=2,
#             dim=2,
#             eps=self.eps,
#         )

#         class_n = F.normalize(
#             classifier.weight.detach().to(
#                 device=device,
#                 dtype=student_regions.dtype,
#             ),
#             p=2,
#             dim=1,
#             eps=self.eps,
#         )

#         # [B,R,C], cosine similarity to all class directions.
#         sim = torch.einsum(
#             "brd,cd->brc",
#             region_n,
#             class_n,
#         )

#         # Only discovered nuisance regions participate in the purification.
#         # [N_s, C]
#         nuisance_logits = sim[spurious_mask] / self.temperature

#         # q(Y | z_s), written in log-space for numerical stability.
#         log_q = F.log_softmax(nuisance_logits, dim=1)
#         q = log_q.exp()

#         # KL(q || U_C) = sum_c q_c log(q_c / (1/C))
#         #               = log(C) - H(q).
#         log_uniform = -math.log(float(self.num_classes))
#         kl_to_uniform = (
#             q * (log_q - log_uniform)
#         ).sum(dim=1)

#         # Patch-level mean: every nuisance patch contributes equally.
#         loss_nui = kl_to_uniform.mean()

#         return loss_nui, valid_images


#     # ------------------------------------------------------------
#     # Visualization helpers
#     # ------------------------------------------------------------
#     @staticmethod
#     def _vis_to_numpy_image(
#         image: torch.Tensor,
#         mean: Optional[List[float]] = None,
#         std: Optional[List[float]] = None,
#     ) -> np.ndarray:
#         """Convert [C,H,W] tensor to RGB numpy image in [0,1]."""
#         x = image.detach().cpu().float().clone()
#         if x.ndim != 3:
#             raise ValueError("image must be [C,H,W].")

#         if mean is not None and std is not None:
#             mean_t = torch.tensor(mean, dtype=x.dtype).view(-1, 1, 1)
#             std_t = torch.tensor(std, dtype=x.dtype).view(-1, 1, 1)
#             if mean_t.shape[0] == 1 and x.shape[0] == 3:
#                 mean_t = mean_t.repeat(3, 1, 1)
#                 std_t = std_t.repeat(3, 1, 1)
#             if mean_t.shape[0] != x.shape[0]:
#                 raise ValueError(
#                     "mean/std channel count {} does not match image channels {}.".format(
#                         mean_t.shape[0], x.shape[0]
#                     )
#                 )
#             x = x * std_t + mean_t
#         else:
#             # Medical images are often not in [0,1]. For display only, use
#             # sample-wise min-max scaling when needed.
#             xmin = float(x.min().item())
#             xmax = float(x.max().item())
#             if xmin < 0.0 or xmax > 1.0:
#                 x = (x - x.min()) / (x.max() - x.min() + 1e-8)

#         if x.shape[0] == 1:
#             x = x.repeat(3, 1, 1)
#         elif x.shape[0] >= 3:
#             x = x[:3]
#         else:
#             raise ValueError("Only 1-channel or >=3-channel images are supported.")

#         x = x.clamp(0.0, 1.0)
#         return x.permute(1, 2, 0).numpy()

#     @staticmethod
#     def _vis_upsample_map(
#         x: torch.Tensor,
#         size: Tuple[int, int],
#         mode: str,
#     ) -> torch.Tensor:
#         """Upsample a [H,W] map to image resolution."""
#         x = x[None, None].float()
#         if mode in ("bilinear", "bicubic"):
#             x = F.interpolate(x, size=size, mode=mode, align_corners=False)
#         else:
#             x = F.interpolate(x, size=size, mode=mode)
#         return x[0, 0]

#     @staticmethod
#     def _vis_draw_patch_boxes(
#         ax,
#         patch_mask: np.ndarray,
#         image_h: int,
#         image_w: int,
#         edgecolor: str = "white",
#         linewidth: float = 1.4,
#     ) -> None:
#         """Draw feature-grid patch boxes on an image axis."""
#         hf, wf = patch_mask.shape
#         for rr in range(hf):
#             for cc in range(wf):
#                 if not bool(patch_mask[rr, cc]):
#                     continue
#                 x0 = cc * image_w / wf
#                 x1 = (cc + 1) * image_w / wf
#                 y0 = rr * image_h / hf
#                 y1 = (rr + 1) * image_h / hf
#                 ax.add_patch(
#                     Rectangle(
#                         (x0, y0),
#                         x1 - x0,
#                         y1 - y0,
#                         fill=False,
#                         edgecolor=edgecolor,
#                         linewidth=linewidth,
#                     )
#                 )

#     @staticmethod
#     def _vis_crop_patch(
#         pil_img: Image.Image,
#         patch_index: int,
#         hf: int,
#         wf: int,
#     ) -> Image.Image:
#         """Crop one layer4 spatial token from the original image."""
#         image_w, image_h = pil_img.size
#         rr = int(patch_index) // wf
#         cc = int(patch_index) % wf
#         y0 = int(round(rr * image_h / hf))
#         y1 = int(round((rr + 1) * image_h / hf))
#         x0 = int(round(cc * image_w / wf))
#         x1 = int(round((cc + 1) * image_w / wf))
#         return pil_img.crop((x0, y0, x1, y1))

#     def _compute_standard_cam(
#         self,
#         features: torch.Tensor,
#         classifier: nn.Linear,
#         class_ids: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Compute standard CAM for a GAP + linear classifier.

#         features: [B,D,Hf,Wf]
#         class_ids: [B]
#         return: [B,Hf,Wf] normalized to [0,1]
#         """
#         if classifier.weight.shape[1] != features.shape[1]:
#             raise ValueError(
#                 "classifier dim {} != layer4 dim {}.".format(
#                     classifier.weight.shape[1], features.shape[1]
#                 )
#             )
#         weight = classifier.weight.detach().to(
#             device=features.device,
#             dtype=features.dtype,
#         )
#         selected = weight.index_select(0, class_ids.long())
#         cam = (features * selected[:, :, None, None]).sum(dim=1)
#         cam = F.relu(cam)
#         cam = cam - cam.amin(dim=(1, 2), keepdim=True)
#         cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + self.eps)
#         return cam

#     @torch.no_grad()
#     def visualize_nuisance_regions(
#         self,
#         student_model: nn.Module,
#         inputs: torch.Tensor,
#         labels: Optional[torch.Tensor] = None,
#         save_dir: str = "./nuisance_visualization",
#         file_prefix: str = "sample",
#         mean: Optional[List[float]] = None,
#         std: Optional[List[float]] = None,
#         cam_target: str = "gt",
#         cam_threshold: float = 0.35,
#         foreground_masks: Optional[torch.Tensor] = None,
#         foreground_patch_threshold: float = 0.25,
#         max_images: Optional[int] = 8,
#         save_patch_crops: bool = True,
#         max_patches_per_image: int = 32,
#     ) -> List[Dict[str, object]]:
#         """
#         Visualize where discovered nuisance patches lie in each image.

#         Panels:
#             Original | Cluster IDs | Nuisance Patches | CAM + Nuisance
#             + optional Segmentation + Nuisance when foreground_masks is given.

#         Quantitative outputs per image:
#             nuisance_patch_ratio:
#                 fraction of layer4 tokens selected as nuisance.
#             nuisance_low_cam_ratio:
#                 fraction of nuisance tokens whose normalized CAM < cam_threshold.
#                 This is a model-based background proxy, NOT an anatomical proof.
#             nuisance_mean_cam:
#                 mean CAM response over nuisance tokens.
#             nuisance_seg_background_ratio (optional):
#                 fraction of nuisance tokens whose foreground-mask occupancy is
#                 < foreground_patch_threshold. This is stronger evidence of
#                 anatomical background than CAM when a reliable segmentation mask exists.
#         """
#         if cam_target not in ("pred", "gt"):
#             raise ValueError("cam_target must be 'pred' or 'gt'.")
#         if cam_target == "gt" and labels is None:
#             raise ValueError("labels are required when cam_target='gt'.")
#         if not (0.0 <= cam_threshold <= 1.0):
#             raise ValueError("cam_threshold must be in [0,1].")
#         if not (0.0 <= foreground_patch_threshold <= 1.0):
#             raise ValueError("foreground_patch_threshold must be in [0,1].")

#         Path(save_dir).mkdir(parents=True, exist_ok=True)
#         base = _unwrap_model(student_model)
#         was_training = base.training
#         base.eval()

#         try:
#             device = next(base.parameters()).device
#             inputs = inputs.to(device, non_blocking=True)
#             if labels is not None:
#                 labels = labels.to(device, non_blocking=True).long()

#             logits, features = _capture_layer4_and_forward(student_model, inputs)
#             cluster_ids, spurious_mask, relevant_mask = self._stage2_region_partition(inputs)
#             classifier = _get_classifier(student_model)

#             if cam_target == "gt":
#                 target_ids = labels
#             else:
#                 target_ids = logits.argmax(dim=1)

#             cams = self._compute_standard_cam(
#                 features=features,
#                 classifier=classifier,
#                 class_ids=target_ids,
#             )

#             b, _, image_h, image_w = inputs.shape
#             _, _, hf, wf = features.shape
#             if int(cluster_ids.shape[1]) != hf * wf:
#                 raise ValueError(
#                     "cluster_ids regions={} but student layer4 grid={}x{}.".format(
#                         cluster_ids.shape[1], hf, wf
#                     )
#                 )

#             fg_patch_fraction = None
#             fg_masks_cpu = None
#             if foreground_masks is not None:
#                 fg = foreground_masks.detach().float()
#                 if fg.ndim == 3:
#                     fg = fg.unsqueeze(1)
#                 elif fg.ndim == 4 and fg.shape[1] != 1:
#                     fg = fg[:, :1]
#                 if fg.ndim != 4:
#                     raise ValueError(
#                         "foreground_masks must be [B,H,W] or [B,1,H,W]."
#                     )
#                 if fg.shape[0] != b:
#                     raise ValueError("foreground_masks batch size must match inputs.")
#                 fg = fg.to(device)
#                 # Average foreground occupancy inside each feature-grid token.
#                 fg_patch_fraction = F.adaptive_avg_pool2d(fg, (hf, wf))[:, 0]
#                 fg_masks_cpu = fg[:, 0].detach().cpu()

#             if max_images is None:
#                 n_show = b
#             else:
#                 n_show = min(int(max_images), b)

#             summaries: List[Dict[str, object]] = []

#             for i in range(n_show):
#                 img_np = self._vis_to_numpy_image(inputs[i], mean=mean, std=std)
#                 cluster_grid = cluster_ids[i].view(hf, wf).detach().cpu()
#                 spur_grid = spurious_mask[i].view(hf, wf).detach().cpu()
#                 cam_grid = cams[i].detach().cpu()

#                 nuisance_cam = cam_grid[spur_grid]
#                 num_nuisance = int(spur_grid.sum().item())
#                 num_total = int(hf * wf)
#                 nuisance_patch_ratio = num_nuisance / max(num_total, 1)

#                 if nuisance_cam.numel() > 0:
#                     low_cam_ratio = float(
#                         (nuisance_cam < cam_threshold).float().mean().item()
#                     )
#                     high_cam_ratio = float(
#                         (nuisance_cam >= cam_threshold).float().mean().item()
#                     )
#                     mean_cam = float(nuisance_cam.mean().item())
#                 else:
#                     low_cam_ratio = 0.0
#                     high_cam_ratio = 0.0
#                     mean_cam = 0.0

#                 seg_bg_ratio = None
#                 seg_fg_ratio = None
#                 mean_fg_fraction = None
#                 if fg_patch_fraction is not None:
#                     nuisance_fg = fg_patch_fraction[i].detach().cpu()[spur_grid]
#                     if nuisance_fg.numel() > 0:
#                         seg_bg_ratio = float(
#                             (nuisance_fg < foreground_patch_threshold)
#                             .float()
#                             .mean()
#                             .item()
#                         )
#                         seg_fg_ratio = 1.0 - seg_bg_ratio
#                         mean_fg_fraction = float(nuisance_fg.mean().item())
#                     else:
#                         seg_bg_ratio = 0.0
#                         seg_fg_ratio = 0.0
#                         mean_fg_fraction = 0.0

#                 spur_up = self._vis_upsample_map(
#                     spur_grid.float(), (image_h, image_w), "nearest"
#                 ).numpy()
#                 cam_up = self._vis_upsample_map(
#                     cam_grid, (image_h, image_w), "bilinear"
#                 ).numpy()
#                 cluster_up = self._vis_upsample_map(
#                     cluster_grid.float(), (image_h, image_w), "nearest"
#                 ).numpy()

#                 ncols = 5 if fg_masks_cpu is not None else 4
#                 fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.8))
#                 axes = np.atleast_1d(axes)

#                 axes[0].imshow(img_np)
#                 axes[0].set_title("Original")
#                 axes[0].axis("off")

#                 cluster_plot = axes[1].imshow(
#                     cluster_up,
#                     cmap="tab20",
#                     interpolation="nearest",
#                 )
#                 axes[1].set_title("Cluster IDs")
#                 axes[1].axis("off")
#                 fig.colorbar(cluster_plot, ax=axes[1], fraction=0.046, pad=0.04)

#                 axes[2].imshow(img_np)
#                 axes[2].imshow(
#                     spur_up,
#                     cmap="Reds",
#                     alpha=0.42,
#                     vmin=0.0,
#                     vmax=1.0,
#                     interpolation="nearest",
#                 )
#                 self._vis_draw_patch_boxes(
#                     axes[2],
#                     spur_grid.numpy(),
#                     image_h=image_h,
#                     image_w=image_w,
#                 )
#                 axes[2].set_title(
#                     "Nuisance Patches\npatch ratio={:.3f}".format(
#                         nuisance_patch_ratio
#                     )
#                 )
#                 axes[2].axis("off")

#                 axes[3].imshow(img_np)
#                 axes[3].imshow(
#                     cam_up,
#                     cmap="jet",
#                     alpha=0.42,
#                     vmin=0.0,
#                     vmax=1.0,
#                 )
#                 self._vis_draw_patch_boxes(
#                     axes[3],
#                     spur_grid.numpy(),
#                     image_h=image_h,
#                     image_w=image_w,
#                 )
#                 axes[3].set_title(
#                     "CAM + Nuisance\nlowCAM={:.3f}, highCAM={:.3f}, mean={:.3f}".format(
#                         low_cam_ratio,
#                         high_cam_ratio,
#                         mean_cam,
#                     )
#                 )
#                 axes[3].axis("off")

#                 if fg_masks_cpu is not None:
#                     fg_up = F.interpolate(
#                         fg_masks_cpu[i][None, None].float(),
#                         size=(image_h, image_w),
#                         mode="nearest",
#                     )[0, 0].numpy()
#                     axes[4].imshow(img_np)
#                     axes[4].imshow(
#                         fg_up,
#                         cmap="Greens",
#                         alpha=0.30,
#                         vmin=0.0,
#                         vmax=1.0,
#                     )
#                     self._vis_draw_patch_boxes(
#                         axes[4],
#                         spur_grid.numpy(),
#                         image_h=image_h,
#                         image_w=image_w,
#                     )
#                     axes[4].set_title(
#                         "Segmentation + Nuisance\nsegBG={:.3f}, segFG={:.3f}".format(
#                             seg_bg_ratio,
#                             seg_fg_ratio,
#                         )
#                     )
#                     axes[4].axis("off")

#                 pred_id = int(logits[i].argmax().item())
#                 gt_id = int(labels[i].item()) if labels is not None else -1
#                 fig.suptitle(
#                     "idx={} | gt={} | pred={} | CAM target={} | nuisance={}".format(
#                         i,
#                         gt_id,
#                         pred_id,
#                         int(target_ids[i].item()),
#                         num_nuisance,
#                     ),
#                     fontsize=12,
#                 )
#                 fig.tight_layout()

#                 figure_path = os.path.join(
#                     save_dir,
#                     "{}_{:03d}.png".format(file_prefix, i),
#                 )
#                 fig.savefig(figure_path, dpi=400, bbox_inches="tight", pad_inches=0.05)
#                 plt.close(fig)

#                 patch_paths: List[str] = []
#                 patch_dir = None
#                 if save_patch_crops:
#                     patch_dir = os.path.join(
#                         save_dir,
#                         "{}_{:03d}_patches".format(file_prefix, i),
#                     )
#                     Path(patch_dir).mkdir(parents=True, exist_ok=True)
#                     pil_img = Image.fromarray((img_np * 255.0).astype(np.uint8))
#                     nuisance_indices = (
#                         spurious_mask[i]
#                         .nonzero(as_tuple=False)
#                         .squeeze(1)
#                         .detach()
#                         .cpu()
#                         .tolist()
#                     )[: int(max_patches_per_image)]

#                     for order, patch_index in enumerate(nuisance_indices):
#                         crop = self._vis_crop_patch(
#                             pil_img=pil_img,
#                             patch_index=int(patch_index),
#                             hf=hf,
#                             wf=wf,
#                         )
#                         cid = int(cluster_ids[i, patch_index].item())
#                         rr = int(patch_index) // wf
#                         cc = int(patch_index) % wf
#                         crop_path = os.path.join(
#                             patch_dir,
#                             "patch_{:02d}_cluster_{}_r{}_c{}.png".format(
#                                 order, cid, rr, cc
#                             ),
#                         )
#                         crop.save(crop_path, format="PNG", optimize=True)
#                         patch_paths.append(crop_path)

#                 summary = {
#                     "index": i,
#                     "gt": gt_id,
#                     "pred": pred_id,
#                     "num_nuisance_regions": num_nuisance,
#                     "num_relevant_regions": int(relevant_mask[i].sum().item()),
#                     "nuisance_patch_ratio": float(nuisance_patch_ratio),
#                     "nuisance_low_cam_ratio": low_cam_ratio,
#                     "nuisance_high_cam_ratio": high_cam_ratio,
#                     "nuisance_mean_cam": mean_cam,
#                     "figure_path": figure_path,
#                     "patch_dir": patch_dir,
#                     "patch_paths": patch_paths,
#                 }
#                 if seg_bg_ratio is not None:
#                     summary.update(
#                         {
#                             "nuisance_seg_background_ratio": seg_bg_ratio,
#                             "nuisance_seg_foreground_ratio": seg_fg_ratio,
#                             "nuisance_mean_foreground_fraction": mean_fg_fraction,
#                         }
#                     )
#                 summaries.append(summary)

#             return summaries

#         finally:
#             if was_training:
#                 base.train()
#             else:
#                 base.eval()

#     @torch.no_grad()
#     def visualize_nuisance_cluster_exemplars(
#         self,
#         data_loader,
#         device,
#         save_dir: str = "./nuisance_cluster_exemplars",
#         file_prefix: str = "cluster",
#         mean: Optional[List[float]] = None,
#         std: Optional[List[float]] = None,
#         target_clusters: Optional[List[int]] = None,
#         max_patches_per_cluster: int = 64,
#         max_batches: Optional[int] = 100,
#         columns: int = 8,
#     ) -> Dict[int, str]:
#         """
#         Collect original-image crops assigned to each selected nuisance cluster
#         and save one montage per cluster. This is the most direct qualitative
#         check of whether a nuisance cluster mainly captures background, border,
#         artifacts, irrelevant anatomy, hair, markers, etc.
#         """
#         if self._assignment_model is None:
#             raise RuntimeError(
#                 "No frozen assignment model. Run discover(..., "
#                 "make_assignment_snapshot=True) first."
#             )
#         if target_clusters is None:
#             target_clusters = list(self.top_spurious_clusters)
#         target_clusters = [int(c) for c in target_clusters]
#         if len(target_clusters) == 0:
#             raise RuntimeError("No nuisance clusters are available.")

#         Path(save_dir).mkdir(parents=True, exist_ok=True)
#         buckets = {c: [] for c in target_clusters}
#         assignment_model = self._assignment_model
#         assignment_device = self._assignment_device
#         was_training = assignment_model.training
#         assignment_model.eval()

#         try:
#             for batch_idx, batch in enumerate(data_loader):
#                 if max_batches is not None and batch_idx >= int(max_batches):
#                     break
#                 if not isinstance(batch, (tuple, list)) or len(batch) < 1:
#                     raise TypeError("data_loader must return (inputs, labels, ...).")

#                 inputs = batch[0]
#                 if isinstance(inputs, (tuple, list)):
#                     inputs = inputs[0]
#                 inputs = inputs.to(device, non_blocking=True)

#                 cluster_ids, _, _ = self._stage2_region_partition(inputs)
#                 assignment_inputs = inputs.to(assignment_device, non_blocking=True)
#                 _, assignment_features = _capture_layer4_and_forward(
#                     assignment_model, assignment_inputs
#                 )
#                 b, _, hf, wf = assignment_features.shape
#                 if int(cluster_ids.shape[1]) != hf * wf:
#                     raise ValueError(
#                         "cluster_ids={} but assignment grid={}x{}.".format(
#                             cluster_ids.shape[1], hf, wf
#                         )
#                     )

#                 for i in range(b):
#                     img_np = self._vis_to_numpy_image(inputs[i], mean=mean, std=std)
#                     pil_img = Image.fromarray((img_np * 255.0).astype(np.uint8))
#                     ids_i = cluster_ids[i].detach().cpu()

#                     for cid in target_clusters:
#                         if len(buckets[cid]) >= int(max_patches_per_cluster):
#                             continue
#                         patch_indices = (
#                             ids_i.eq(cid)
#                             .nonzero(as_tuple=False)
#                             .squeeze(1)
#                             .tolist()
#                         )
#                         for patch_index in patch_indices:
#                             if len(buckets[cid]) >= int(max_patches_per_cluster):
#                                 break
#                             crop = self._vis_crop_patch(
#                                 pil_img=pil_img,
#                                 patch_index=int(patch_index),
#                                 hf=hf,
#                                 wf=wf,
#                             )
#                             buckets[cid].append(crop.copy())

#                 if all(
#                     len(v) >= int(max_patches_per_cluster)
#                     for v in buckets.values()
#                 ):
#                     break
#         finally:
#             if was_training:
#                 assignment_model.train()
#             else:
#                 assignment_model.eval()

#         output_paths: Dict[int, str] = {}
#         for cid, crops in buckets.items():
#             if len(crops) == 0:
#                 continue
#             n = len(crops)
#             ncols = min(int(columns), n)
#             nrows = int(math.ceil(n / ncols))
#             fig, axes = plt.subplots(
#                 nrows,
#                 ncols,
#                 figsize=(2.0 * ncols, 2.0 * nrows),
#                 squeeze=False,
#             )
#             for ax in axes.ravel():
#                 ax.axis("off")
#             for j, crop in enumerate(crops):
#                 rr = j // ncols
#                 cc = j % ncols
#                 axes[rr, cc].imshow(crop)
#                 axes[rr, cc].axis("off")
#             fig.suptitle(
#                 "Nuisance cluster {} exemplars (n={})".format(cid, n),
#                 fontsize=14,
#             )
#             fig.tight_layout()
#             path = os.path.join(
#                 save_dir,
#                 "{}_{}.png".format(file_prefix, cid),
#             )
#             fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.05)
#             plt.close(fig)
#             output_paths[cid] = path
#         return output_paths

#     @torch.no_grad()
#     def evaluate_nuisance_cam_overlap(
#         self,
#         student_model: nn.Module,
#         data_loader,
#         device,
#         cam_target: str = "gt",
#         cam_threshold: float = 0.35,
#         max_batches: Optional[int] = None,
#     ) -> Dict[str, float]:
#         """
#         Dataset-level CAM localization statistics for nuisance patches.

#         low_cam_ratio close to 1 means most nuisance tokens fall in regions that
#         have low class-specific CAM response. This supports a background/shortcut
#         interpretation, but it is not an anatomical ground-truth measurement.
#         """
#         if cam_target not in ("pred", "gt"):
#             raise ValueError("cam_target must be 'pred' or 'gt'.")

#         base = _unwrap_model(student_model)
#         was_training = base.training
#         base.eval()
#         total_nuisance = 0
#         total_low_cam = 0
#         total_high_cam = 0
#         sum_cam = 0.0

#         try:
#             for batch_idx, batch in enumerate(data_loader):
#                 if max_batches is not None and batch_idx >= int(max_batches):
#                     break
#                 if not isinstance(batch, (tuple, list)) or len(batch) < 2:
#                     raise TypeError("data_loader must return (inputs, labels, ...).")

#                 inputs = batch[0]
#                 labels = batch[1]
#                 if isinstance(inputs, (tuple, list)):
#                     inputs = inputs[0]
#                 inputs = inputs.to(device, non_blocking=True)
#                 labels = labels.to(device, non_blocking=True).long()

#                 logits, features = _capture_layer4_and_forward(student_model, inputs)
#                 cluster_ids, spurious_mask, _ = self._stage2_region_partition(inputs)
#                 classifier = _get_classifier(student_model)
#                 target_ids = labels if cam_target == "gt" else logits.argmax(dim=1)
#                 cams = self._compute_standard_cam(features, classifier, target_ids)

#                 b, _, hf, wf = features.shape
#                 if int(cluster_ids.shape[1]) != hf * wf:
#                     raise ValueError(
#                         "Student feature grid and assignment region count do not match."
#                     )
#                 spur_grid = spurious_mask.view(b, hf, wf)
#                 nuisance_cam = cams[spur_grid]
#                 if nuisance_cam.numel() == 0:
#                     continue

#                 total_nuisance += int(nuisance_cam.numel())
#                 total_low_cam += int((nuisance_cam < cam_threshold).sum().item())
#                 total_high_cam += int((nuisance_cam >= cam_threshold).sum().item())
#                 sum_cam += float(nuisance_cam.sum().item())
#         finally:
#             if was_training:
#                 base.train()
#             else:
#                 base.eval()

#         if total_nuisance == 0:
#             return {
#                 "num_nuisance_patches": 0.0,
#                 "low_cam_ratio": 0.0,
#                 "high_cam_ratio": 0.0,
#                 "mean_nuisance_cam": 0.0,
#             }

#         return {
#             "num_nuisance_patches": float(total_nuisance),
#             "low_cam_ratio": total_low_cam / total_nuisance,
#             "high_cam_ratio": total_high_cam / total_nuisance,
#             "mean_nuisance_cam": sum_cam / total_nuisance,
#         }

#     @torch.no_grad()
#     def evaluate_nuisance_segmentation_overlap(
#         self,
#         data_loader,
#         device,
#         mask_index: int = 2,
#         foreground_patch_threshold: float = 0.25,
#         max_batches: Optional[int] = None,
#     ) -> Dict[str, float]:
#         """
#         Dataset-level anatomical-background validation using segmentation masks.

#         Expected loader output:
#             (inputs, labels, ..., foreground_mask, ...)
#         and mask_index selects the mask field.

#         A nuisance token is counted as background when the mean foreground-mask
#         occupancy inside that layer4 token is below foreground_patch_threshold.
#         """
#         if not (0.0 <= foreground_patch_threshold <= 1.0):
#             raise ValueError("foreground_patch_threshold must be in [0,1].")

#         total_nuisance = 0
#         total_bg = 0
#         total_fg = 0
#         total_fg_fraction = 0.0

#         for batch_idx, batch in enumerate(data_loader):
#             if max_batches is not None and batch_idx >= int(max_batches):
#                 break
#             if not isinstance(batch, (tuple, list)) or len(batch) <= mask_index:
#                 raise TypeError(
#                     "data_loader must contain foreground masks at batch[{}].".format(
#                         mask_index
#                     )
#                 )
#             inputs = batch[0]
#             masks = batch[mask_index]
#             if isinstance(inputs, (tuple, list)):
#                 inputs = inputs[0]
#             inputs = inputs.to(device, non_blocking=True)
#             masks = masks.to(device, non_blocking=True).float()
#             if masks.ndim == 3:
#                 masks = masks.unsqueeze(1)
#             elif masks.ndim == 4 and masks.shape[1] != 1:
#                 masks = masks[:, :1]
#             if masks.ndim != 4:
#                 raise ValueError("foreground masks must be [B,H,W] or [B,1,H,W].")

#             # Use the frozen assignment representation to obtain exact grid size.
#             assignment_inputs = inputs.to(self._assignment_device, non_blocking=True)
#             _, assignment_features = _capture_layer4_and_forward(
#                 self._assignment_model,
#                 assignment_inputs,
#             )
#             b, _, hf, wf = assignment_features.shape
#             cluster_ids, spurious_mask, _ = self._stage2_region_partition(inputs)
#             if int(cluster_ids.shape[1]) != hf * wf:
#                 raise ValueError("Assignment region count does not match feature grid.")

#             fg_fraction = F.adaptive_avg_pool2d(masks, (hf, wf))[:, 0]
#             spur_grid = spurious_mask.view(b, hf, wf)
#             nuisance_fg = fg_fraction[spur_grid]
#             if nuisance_fg.numel() == 0:
#                 continue

#             total_nuisance += int(nuisance_fg.numel())
#             total_bg += int(
#                 (nuisance_fg < foreground_patch_threshold).sum().item()
#             )
#             total_fg += int(
#                 (nuisance_fg >= foreground_patch_threshold).sum().item()
#             )
#             total_fg_fraction += float(nuisance_fg.sum().item())

#         if total_nuisance == 0:
#             return {
#                 "num_nuisance_patches": 0.0,
#                 "seg_background_ratio": 0.0,
#                 "seg_foreground_ratio": 0.0,
#                 "mean_foreground_fraction": 0.0,
#             }

#         return {
#             "num_nuisance_patches": float(total_nuisance),
#             "seg_background_ratio": total_bg / total_nuisance,
#             "seg_foreground_ratio": total_fg / total_nuisance,
#             "mean_foreground_fraction": total_fg_fraction / total_nuisance,
#         }

#     # ------------------------------------------------------------
#     # Stage 2 public forward
#     # ------------------------------------------------------------
#     def forward(
#         self,
#         student_model: nn.Module,
#         inputs: torch.Tensor,
#         labels: torch.Tensor,
#     ) -> RaVLOutput:
#         """
#         Compute Stage-2 nuisance class-neutralization loss.

#         This function DOES NOT update the model and DOES NOT include the user's
#         ordinary classification loss.
#         """
#         if self.top_spurious_cluster is None:
#             raise RuntimeError(
#                 "Run ravl.discover(model, reference_loader) before Stage 2."
#             )

#         logits, student_features = _capture_layer4_and_forward(
#             student_model,
#             inputs,
#         )

#         if logits.shape[1] != self.num_classes:
#             raise ValueError(
#                 "Expected {} classes, got {}.".format(
#                     self.num_classes,
#                     logits.shape[1],
#                 )
#             )

#         student_regions = self._regions_from_features(
#             student_features
#         )

#         cluster_ids, spurious_mask, relevant_mask = (
#             self._stage2_region_partition(inputs)
#         )

#         classifier = _get_classifier(student_model)

#         loss_nui, valid_images = self._region_aware_loss(
#             student_regions=student_regions,
#             labels=labels,
#             spurious_mask=spurious_mask,
#             relevant_mask=relevant_mask,
#             classifier=classifier,
#         )

#         return RaVLOutput(
#             loss_region=loss_nui,
#             loss_nui=loss_nui,
#             logits=logits,
#             spurious_mask=spurious_mask.detach(),
#             relevant_mask=relevant_mask.detach(),
#             cluster_ids=cluster_ids.detach(),
#             num_spurious_regions=int(spurious_mask.sum().item()),
#             num_relevant_regions=int(relevant_mask.sum().item()),
#             num_valid_images=int(valid_images.sum().item()),
#         )

#     __call__ = forward

#     def combine_with_classification_loss(
#         self,
#         classification_loss: torch.Tensor,
#         output: RaVLOutput,
#         lambda_cl: Optional[float] = None,
#     ) -> torch.Tensor:
#         """
#         Keep the original public helper interface. The returned regularizer
#         is now the nuisance class-neutralization loss instead of L_R + L_A.
#         """
#         if lambda_cl is None:
#             lambda_cl = self.lambda_cl
#         return classification_loss +  float(lambda_cl)*output.loss_region

#     # ------------------------------------------------------------
#     # Persistence for the discovered clusters
#     # ------------------------------------------------------------
#     def save_discovery(self, path: str) -> None:
#         if self.medoids_raw is None or self.top_spurious_cluster is None:
#             raise RuntimeError("Nothing to save; run discover first.")

#         torch.save(
#             {
#                 "num_classes": self.num_classes,
#                 "region_grid": self.region_grid,
#                 "temperature": self.temperature,
#                 "lambda_cl": self.lambda_cl,
#                 "influence_threshold": self.influence_threshold,
#                 "medoids_raw": self.medoids_raw,
#                 "medoids_norm": self.medoids_norm,
#                 "top_spurious_cluster": self.top_spurious_cluster,
#             "top_spurious_clusters": self.top_spurious_clusters,
#             "num_spurious_clusters": self.num_spurious_clusters,
#                 "ranked_clusters": self.ranked_clusters,
#                 "discovery_result": self.discovery_result,
#             },
#             path,
#         )

#     def load_discovery(
#         self,
#         path: str,
#         model_for_assignment: Optional[nn.Module] = None,
#         device=None,
#     ) -> None:
#         state = torch.load(path, map_location="cpu")

#         if int(state["num_classes"]) != self.num_classes:
#             raise ValueError(
#                 "Saved num_classes={} but module num_classes={}.".format(
#                     state["num_classes"],
#                     self.num_classes,
#                 )
#             )

#         if int(state["region_grid"]) != self.region_grid:
#             raise ValueError(
#                 "Saved region_grid={} but module region_grid={}.".format(
#                     state["region_grid"],
#                     self.region_grid,
#                 )
#             )

#         self.medoids_raw = state["medoids_raw"].float()
#         self.medoids_norm = state["medoids_norm"].float()
#         self.top_spurious_cluster = int(
#             state["top_spurious_cluster"]
#         )
#         self.top_spurious_clusters = state.get(
#             "top_spurious_clusters",
#             [self.top_spurious_cluster]
#         )
#         self.ranked_clusters = list(state["ranked_clusters"])
#         self.discovery_result = state.get("discovery_result", None)

#         if model_for_assignment is not None:
#             self.prepare_stage2_assignment_model(
#                 model_for_assignment,
#                 device=device,
#             )


# # ================================================================
# # Tiny smoke test
# # ================================================================
# if __name__ == "__main__":
#     from torch.utils.data import DataLoader, TensorDataset

#     class TinyResNet(nn.Module):
#         def __init__(self, num_classes=3):
#             super().__init__()
#             self.stem = nn.Sequential(
#                 nn.Conv2d(3, 16, 3, padding=1),
#                 nn.ReLU(),
#                 nn.AdaptiveAvgPool2d((7, 7)),
#             )
#             self.layer4 = nn.Sequential(
#                 nn.Conv2d(16, 32, 3, padding=1),
#                 nn.ReLU(),
#             )
#             self.fc = nn.Linear(32, num_classes)

#         def forward(self, x):
#             x = self.stem(x)
#             x = self.layer4(x)
#             z = x.mean(dim=(2, 3))
#             return self.fc(z)

#     torch.manual_seed(7)
#     device = torch.device(
#         "cuda" if torch.cuda.is_available() else "cpu"
#     )

#     model = TinyResNet(num_classes=3).to(device)

#     ravl = RaVLResNet(
#         num_classes=3,
#         region_grid=2,
#         temperature=0.07,
#         influence_threshold=0.0,  # smoke-test only
#         k_min_factor=2,
#         k_max_factor=2,
#         max_cluster_regions=200,
#         silhouette_sample_size=200,
#         random_seed=7,
#     )

#     # Synthetic reference set.
#     ref_x = torch.randn(60, 3, 32, 32)
#     ref_y = torch.randint(0, 3, (60,))
#     ref_loader = DataLoader(
#         TensorDataset(ref_x, ref_y),
#         batch_size=12,
#         shuffle=False,
#     )

#     # Random model may still have G=0. We test the full clustering path but
#     # permit H=0 for this synthetic smoke test.
#     try:
#         discovery = ravl.discover(
#             model=model,
#             reference_loader=ref_loader,
#             device=device,
#             verbose=False,
#         )
#     except RuntimeError:
#         # For a random model, if all G/H degenerates, manually select the first
#         # discovered cluster is not possible if discovery did not commit state.
#         # Re-run is not necessary for syntax/gradient smoke testing; set a small
#         # fixed bank from reference features.
#         model.eval()
#         with torch.no_grad():
#             _, feat = _capture_layer4_and_forward(
#                 model,
#                 ref_x[:12].to(device),
#             )
#             regs = ravl._regions_from_features(feat)
#             flat = regs.reshape(-1, regs.shape[-1])
#             medoid_ids, _ = ravl._fit_kmedoids_cosine(
#                 flat,
#                 k=3,
#                 seed=7,
#             )
#             med = flat.index_select(0, medoid_ids)
#             ravl.medoids_raw = med.detach().cpu()
#             ravl.medoids_norm = F.normalize(
#                 med,
#                 dim=1,
#             ).detach().cpu()
#             ravl.top_spurious_cluster = 0
#             ravl.top_spurious_clusters = [0]
#             ravl.ranked_clusters = [0]
#             ravl.prepare_stage2_assignment_model(
#                 model,
#                 device=device,
#             )

#     model.train()

#     x = torch.randn(8, 3, 32, 32, device=device)
#     y = torch.randint(0, 3, (8,), device=device)

#     out = ravl(
#         student_model=model,
#         inputs=x,
#         labels=y,
#     )

#     loss_cls = F.cross_entropy(out.logits, y)
#     loss = ravl.combine_with_classification_loss(
#         loss_cls,
#         out,
#     )

#     optimizer = torch.optim.SGD(
#         model.parameters(),
#         lr=1e-3,
#     )
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

#     print("Conditional-MI + uniform nuisance purification smoke test passed.")
#     print(out.statistics())





# MITIGATION_VERSION: global_topk_wrong_class_positive_suppression
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import os
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RaVLDiscoveryResult:
    best_k: int
    best_silhouette: float
    top_spurious_cluster: int
    ranked_clusters: List[int]
    influence_scores: Dict[int, float]
    performance_gaps: Dict[int, float]
    per_class_gaps: Dict[int, Dict[int, float]]
    per_class_influence: Dict[int, Dict[int, float]]
    num_images: int
    num_regions: int

    def summary(self) -> Dict[str, object]:
        return {
            "best_k": self.best_k,
            "best_silhouette": self.best_silhouette,
            "top_spurious_cluster": self.top_spurious_cluster,
            "ranked_clusters": self.ranked_clusters,
            "influence_scores": self.influence_scores,
            "performance_gaps": self.performance_gaps,
            "num_images": self.num_images,
            "num_regions": self.num_regions,
        }


@dataclass
class RaVLOutput:
    loss_region: torch.Tensor
    loss_R: torch.Tensor
    loss_A: torch.Tensor
    logits: torch.Tensor
    raw_spurious_mask: torch.Tensor
    spurious_mask: torch.Tensor
    priority_relevant_mask: torch.Tensor
    relevant_mask: torch.Tensor
    cluster_ids: torch.Tensor
    num_raw_spurious_regions: int
    num_protected_regions: int
    num_spurious_regions: int
    num_relevant_regions: int
    num_valid_images: int

    # Backward compatibility with the previous uniform-neutralization version.
    @property
    def loss_nui(self) -> torch.Tensor:
        return self.loss_region

    def statistics(self) -> Dict[str, float]:
        return {
            "loss_region": float(self.loss_region.detach().item()),
            "loss_R": float(self.loss_R.detach().item()),
            "loss_A": float(self.loss_A.detach().item()),
            "num_raw_spurious_regions": float(self.num_raw_spurious_regions),
            "num_protected_regions": float(self.num_protected_regions),
            "num_spurious_regions": float(self.num_spurious_regions),
            "num_relevant_regions": float(self.num_relevant_regions),
            "num_valid_images": float(self.num_valid_images),
        }


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _unwrap_tensor(x, name: str) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)) and len(x) > 0 and torch.is_tensor(x[0]):
        return x[0]
    raise TypeError(
        "{} must be Tensor or tuple/list whose first item is Tensor.".format(name)
    )


def _get_classifier(model: nn.Module) -> nn.Linear:
    base = _unwrap_model(model)
    candidates = []

    if hasattr(base, "get_classifier"):
        try:
            candidates.append(base.get_classifier())
        except Exception:
            pass

    for name in ("fc", "head", "classifier"):
        if hasattr(base, name):
            candidates.append(getattr(base, name))

    for candidate in candidates:
        if isinstance(candidate, nn.Linear):
            return candidate
        if isinstance(candidate, nn.Sequential):
            for sub in reversed(candidate):
                if isinstance(sub, nn.Linear):
                    return sub

    raise AttributeError("Unable to locate final nn.Linear classifier.")


def _capture_layer4_and_forward(
    model: nn.Module,
    inputs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    base = _unwrap_model(model)

    if not hasattr(base, "layer4"):
        raise AttributeError(
            "The model must be ResNet-like and contain model.layer4."
        )

    holder = {}

    def hook(_module, _inputs, output):
        holder["features"] = _unwrap_tensor(output, "layer4 output")

    handle = base.layer4.register_forward_hook(hook)

    try:
        logits = _unwrap_tensor(model(inputs), "model(inputs)")
    finally:
        handle.remove()

    if "features" not in holder:
        raise RuntimeError("Failed to capture layer4 features.")

    features = holder["features"]

    if features.ndim != 4:
        raise ValueError(
            "layer4 must be [B,D,H,W], got {}".format(tuple(features.shape))
        )

    if logits.ndim != 2:
        raise ValueError(
            "logits must be [B,C], got {}".format(tuple(logits.shape))
        )

    return logits, features



class RaVLResNet(object):
    """
    RaVL-style discovery + mitigation migrated from the NeurIPS 2024 RaVL
    algorithm to a supervised ResNet classifier.

    What is kept from the original RaVL:
      Stage 1:
        1) labeled reference/validation set
        2) local candidate regions from correctly predicted reference samples
        3) K-Medoids with cosine distance on the correct-only region pool
        4) choose K by Silhouette score over [2|Y|, 5|Y|]
        5) class-conditional MI I(S_k; Y_hat | Y=c)
        6) harmful error-increase gating
        7) class-balanced conditional nuisance score N_k
        8) select the top-ranked nuisance cluster(s)

      Stage 2:
        1) keep the Stage-1 clustering model fixed
        2) before nuisance partition, protect the rel_num regions most similar
           to the ground-truth classifier direction w_{y_i}
        3) protected regions have the highest priority and can never be nuisance
        4) nuisance regions = GLOBAL nuisance-cluster regions minus protected regions
        5) preserve any ground-truth evidence carried by nuisance patches
        6) suppress ONLY positive evidence toward wrong class directions

    Necessary modality substitutions:
      - VLM text class embedding g(y) -> normalized frozen FC class direction w_y
      - RoI candidate regions -> equal grid regions pooled from ResNet layer4
      - paired-text assigned label y_hat -> supervised ground-truth class label y

    Important:
      - The clustering medoids and the Stage-1 assignment encoder remain fixed.
      - The student/model training strategy is external to this class.
      - This class introduces no learnable parameters.
    """

    def __init__(
        self,
        num_classes: int,
        region_grid: int = 3,
        temperature: float = 0.07,
        lambda_cl: float = 0.80,
        influence_threshold: float = 0.25,
        num_spurious_clusters: int = 1,
        rel_num: int = 4,
        k_min_factor: int = 2,
        k_max_factor: int = 3,
        kmedoids_iterations: int = 30,
        max_cluster_regions: Optional[int] = 20000,
        silhouette_sample_size: int = 3000,
        assignment_chunk_size: int = 8192,
        random_seed: int = 0,
    ) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if region_grid < 1:
            raise ValueError("region_grid must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not 0 <= lambda_cl <= 10:
            raise ValueError("lambda_cl must be in [0,10]")
        if influence_threshold < 0:
            raise ValueError("influence_threshold must be >= 0")
        if k_min_factor < 1 or k_max_factor < k_min_factor:
            raise ValueError("invalid cluster-count factors")
        if rel_num < 1:
            raise ValueError("rel_num must be >= 1")

        self.num_classes = int(num_classes)
        self.region_grid = int(region_grid)
        self.temperature = float(temperature)
        self.lambda_cl = float(lambda_cl)
        self.influence_threshold = float(influence_threshold)
        self.num_spurious_clusters = int(num_spurious_clusters)
        self.rel_num = int(rel_num)
        self.k_min_factor = int(k_min_factor)
        self.k_max_factor = int(k_max_factor)
        self.kmedoids_iterations = int(kmedoids_iterations)
        self.max_cluster_regions = max_cluster_regions
        self.silhouette_sample_size = int(silhouette_sample_size)
        self.assignment_chunk_size = int(assignment_chunk_size)
        self.random_seed = int(random_seed)
        self.eps = 1e-8

        self.medoids_raw = None
        self.medoids_norm = None
        self.top_spurious_cluster = None
        self.top_spurious_clusters = []
        self.ranked_clusters = []
        self.discovery_result = None

        # Frozen copy of the Stage-1 model.
        # It is used only for assigning Stage-2 regions to the fixed clusters.
        self._assignment_model = None
        self._assignment_device = None

    # ------------------------------------------------------------
    # Region construction
    # ------------------------------------------------------------
    def _regions_from_features(
            self,
            features: torch.Tensor,
    ):
        """
        Directly use layer4 spatial tokens as regions.

        Input:
            features:
                [B,C,H,W]

        Output:
            regions:
                [B,H*W,C]

        For ResNet50 layer4:
            [B,2048,7,7]
            ->
            [B,49,2048]
        """

        B, C, H, W = features.shape
        regions = (
            features
            .flatten(2)
            .transpose(1, 2)
        )
        return regions

    # ------------------------------------------------------------
    # Cosine K-Medoids
    # ------------------------------------------------------------
    @torch.no_grad()
    def _fit_kmedoids_cosine(
        self,
        x_raw: torch.Tensor,
        k: int,
        seed: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Alternating K-Medoids for cosine distance.

        For fixed assignments and normalized samples, the exact medoid of a
        cluster is the member maximizing similarity to the sum of cluster
        members, so the update can be done without an O(n_cluster^2) matrix.

        Returns:
            medoid_ids: [k]
            labels: [N]
        """
        n = int(x_raw.shape[0])

        if k < 2 or k >= n:
            raise ValueError("K-Medoids requires 2 <= k < N.")

        x = F.normalize(
            x_raw.float(),
            p=2,
            dim=1,
            eps=self.eps,
        )

        gen = torch.Generator(device=x.device)
        gen.manual_seed(int(seed))

        first = int(
            torch.randint(
                low=0,
                high=n,
                size=(1,),
                generator=gen,
                device=x.device,
            ).item()
        )

        selected = [first]

        # Farthest-point initialization in cosine distance.
        min_dist = 1.0 - (x @ x[first:first + 1].t()).squeeze(1)

        for _ in range(1, k):
            idx = int(min_dist.argmax().item())
            selected.append(idx)
            dist = 1.0 - (x @ x[idx:idx + 1].t()).squeeze(1)
            min_dist = torch.minimum(min_dist, dist)

        medoid_ids = torch.tensor(
            selected,
            device=x.device,
            dtype=torch.long,
        )

        old_labels = None

        for _ in range(self.kmedoids_iterations):
            medoids = x.index_select(0, medoid_ids)
            sim = x @ medoids.t()
            labels = sim.argmax(dim=1)

            if old_labels is not None and torch.equal(labels, old_labels):
                break

            old_labels = labels.clone()
            new_ids = []

            for cluster_id in range(k):
                members = labels.eq(cluster_id).nonzero(
                    as_tuple=False
                ).squeeze(1)

                if members.numel() == 0:
                    # Re-seed from the currently worst represented point.
                    max_sim = sim.max(dim=1).values
                    candidate = int(max_sim.argmin().item())
                    new_ids.append(candidate)
                    continue

                member_x = x.index_select(0, members)
                sum_direction = member_x.sum(dim=0)

                # Exact cosine-medoid criterion for the fixed cluster:
                # argmax_i sum_j cos(x_i, x_j)
                medoid_score = member_x @ sum_direction
                local_id = int(medoid_score.argmax().item())
                new_ids.append(int(members[local_id].item()))

            new_ids_tensor = torch.tensor(
                new_ids,
                device=x.device,
                dtype=torch.long,
            )

            if torch.equal(new_ids_tensor, medoid_ids):
                medoid_ids = new_ids_tensor
                break

            medoid_ids = new_ids_tensor

        final_medoids = x.index_select(0, medoid_ids)
        final_labels = (x @ final_medoids.t()).argmax(dim=1)

        return medoid_ids, final_labels

    @torch.no_grad()
    def _silhouette_cosine(
        self,
        x_raw: torch.Tensor,
        labels: torch.Tensor,
        seed: int,
    ) -> float:
        """
        Cosine Silhouette score.

        If N > silhouette_sample_size, a deterministic random subset is used.
        This is only an engineering memory cap; set silhouette_sample_size >= N
        to recover the full score.
        """
        n = int(x_raw.shape[0])

        if n < 3:
            return -1.0

        unique_labels = labels.unique()
        if unique_labels.numel() < 2:
            return -1.0

        s = min(n, self.silhouette_sample_size)

        if s < n:
            gen = torch.Generator(device=x_raw.device)
            gen.manual_seed(int(seed))
            sample_ids = torch.randperm(
                n,
                generator=gen,
                device=x_raw.device,
            )[:s]
            x = x_raw.index_select(0, sample_ids)
            y = labels.index_select(0, sample_ids)
        else:
            x = x_raw
            y = labels

        x = F.normalize(x.float(), p=2, dim=1, eps=self.eps)
        dist = 1.0 - x @ x.t()
        dist = dist.clamp_min(0.0)

        sil = torch.zeros(
            x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )

        all_clusters = y.unique()

        for i in range(x.shape[0]):
            own = y[i]
            own_mask = y.eq(own)
            own_count = int(own_mask.sum().item())

            if own_count <= 1:
                sil[i] = 0.0
                continue

            a = dist[i, own_mask].sum() / float(own_count - 1)

            b = None
            for c in all_clusters:
                if int(c.item()) == int(own.item()):
                    continue

                mask = y.eq(c)
                if not bool(mask.any().item()):
                    continue

                mean_dist = dist[i, mask].mean()
                if b is None or mean_dist < b:
                    b = mean_dist

            if b is None:
                sil[i] = 0.0
                continue

            denom = torch.maximum(a, b).clamp_min(self.eps)
            sil[i] = (b - a) / denom

        return float(sil.mean().item())

    @torch.no_grad()
    def _assign_to_medoids(
        self,
        regions: torch.Tensor,
        medoids_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Assign [N,D] or [B,R,D] region embeddings to the fixed medoids
        using cosine distance.
        """
        if medoids_norm is None:
            if self.medoids_norm is None:
                raise RuntimeError("RaVL discovery has not been run.")
            medoids_norm = self.medoids_norm

        original_shape = regions.shape[:-1]
        d = regions.shape[-1]
        flat = regions.reshape(-1, d)

        outputs = []
        medoids_norm = medoids_norm.to(
            device=flat.device,
            dtype=flat.dtype,
        )

        for start in range(0, flat.shape[0], self.assignment_chunk_size):
            end = min(start + self.assignment_chunk_size, flat.shape[0])
            x = F.normalize(
                flat[start:end],
                p=2,
                dim=1,
                eps=self.eps,
            )
            outputs.append((x @ medoids_norm.t()).argmax(dim=1))

        assigned = torch.cat(outputs, dim=0)
        return assigned.view(*original_shape)

    # ------------------------------------------------------------
    # Stage 1: RaVL discovery
    # ------------------------------------------------------------
    @torch.no_grad()
    def discover(
        self,
        model: nn.Module,
        reference_loader,
        device=None,
        verbose: bool = True,
        make_assignment_snapshot: bool = True,
    ) -> RaVLDiscoveryResult:
        """
        Stage-1 discovery adapted to a supervised ResNet. K-Medoids
        prototypes are constructed only from correctly predicted reference
        samples; all reference samples are then assigned for nuisance scoring.

        reference_loader must yield:
            (inputs, labels)
        or:
            (inputs, labels, ...)
        """
        if device is None:
            device = next(model.parameters()).device

        was_training = model.training
        model.eval()

        classifier = _get_classifier(model)

        all_regions = []
        all_region_probs = []
        all_labels = []
        all_preds = []

        try:
            for batch in reference_loader:
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise TypeError(
                        "reference_loader must return (inputs, labels) or "
                        "(inputs, labels, ...)."
                    )

                inputs = batch[0]
                labels = batch[1]

                if isinstance(inputs, (tuple, list)):
                    inputs = inputs[0]

                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()

                logits, features = _capture_layer4_and_forward(
                    model,
                    inputs,
                )

                if logits.shape[1] != self.num_classes:
                    raise ValueError(
                        "Expected {} classes, got {}.".format(
                            self.num_classes,
                            logits.shape[1],
                        )
                    )

                regions = self._regions_from_features(features)
                b, r, d = regions.shape

                if classifier.weight.shape[1] != d:
                    raise ValueError(
                        "Classifier input dim {} != region dim {}.".format(
                            classifier.weight.shape[1],
                            d,
                        )
                    )

                region_logits = classifier(
                    regions.reshape(b * r, d)
                ).view(b, r, self.num_classes)

                region_probs = F.softmax(region_logits, dim=2)

                pred = logits.argmax(dim=1)

                all_regions.append(
                    regions.detach().cpu().to(torch.float16)
                )
                all_region_probs.append(
                    region_probs.detach().cpu().to(torch.float16)
                )
                all_labels.append(labels.detach().cpu())
                all_preds.append(pred.detach().cpu())

        finally:
            if was_training:
                model.train()
            else:
                model.eval()

        if len(all_regions) == 0:
            raise RuntimeError("Reference loader is empty.")

        regions_img = torch.cat(all_regions, dim=0).float()
        region_probs_img = torch.cat(all_region_probs, dim=0).float()
        labels_img = torch.cat(all_labels, dim=0).long()
        preds_img = torch.cat(all_preds, dim=0).long()
        correct_img = preds_img.eq(labels_img)

        n_img, n_region, d = regions_img.shape
        regions_flat = regions_img.reshape(n_img * n_region, d)

        # --------------------------------------------------------
        # Clustering pool: CORRECTLY PREDICTED samples only.
        # --------------------------------------------------------
        # Important design:
        #   1) Correct samples are used only to CONSTRUCT the visual-pattern
        #      dictionary (K-Medoids / medoids).
        #   2) After the medoids are fixed, ALL reference samples (correct +
        #      incorrect) are assigned to these medoids.
        #   3) Conditional-MI nuisance scoring is still computed on ALL
        #      reference samples, so incorrect samples remain essential for
        #      identifying whether a pattern is harmful.
        #
        # This avoids allowing under-learned regions from misclassified samples
        # to directly define the clustering prototypes, while retaining them in
        # the subsequent nuisance-effect estimation.
        total_regions = int(regions_flat.shape[0])

        correct_image_ids = correct_img.nonzero(
            as_tuple=False
        ).squeeze(1)
        num_correct_images = int(correct_image_ids.numel())

        if num_correct_images == 0:
            raise RuntimeError(
                "No correctly predicted sample exists in the reference set. "
                "Correct-only clustering cannot be constructed."
            )

        # [N_correct, R, D] -> [N_correct * R, D]
        correct_regions_img = regions_img.index_select(
            0,
            correct_image_ids,
        )
        correct_regions_flat = correct_regions_img.reshape(
            -1,
            d,
        )
        total_correct_regions = int(correct_regions_flat.shape[0])

        if total_correct_regions < 3:
            raise RuntimeError(
                "Correct-only clustering requires at least 3 correct-region "
                "features, got {}.".format(total_correct_regions)
            )

        # Optional memory cap is applied ONLY to the correct-region clustering
        # pool. The later all-sample assignment is unchanged.
        if (
            self.max_cluster_regions is not None
            and total_correct_regions > int(self.max_cluster_regions)
        ):
            gen = torch.Generator()
            gen.manual_seed(self.random_seed)
            cluster_region_ids = torch.randperm(
                total_correct_regions,
                generator=gen,
            )[: int(self.max_cluster_regions)]
            cluster_pool_cpu = correct_regions_flat.index_select(
                0,
                cluster_region_ids,
            )
        else:
            cluster_pool_cpu = correct_regions_flat

        cluster_pool = cluster_pool_cpu.to(
            device=device,
            dtype=torch.float32,
        )

        n_pool = int(cluster_pool.shape[0])

        k_min = self.num_classes * self.k_min_factor
        k_max = self.num_classes * self.k_max_factor

        k_min = max(2, min(k_min, n_pool - 1))
        k_max = max(k_min, min(k_max, n_pool - 1))

        if k_min >= n_pool:
            raise RuntimeError(
                "Not enough candidate regions for RaVL clustering."
            )

        best_score = -float("inf")
        best_k = None
        best_medoid_ids = None

        if verbose:
            print("========== RaVL Stage 1: K-Medoids sweep ==========")
            correct_rate = num_correct_images / float(max(n_img, 1))
            print(
                "images={} | correct_images={} ({:.2%}) | regions/image={} | "
                "all_regions={} | correct_regions={} | cluster_pool={}".format(
                    n_img,
                    num_correct_images,
                    correct_rate,
                    n_region,
                    total_regions,
                    total_correct_regions,
                    n_pool,
                )
            )
            # Helpful diagnostic under long-tailed data: clustering still uses
            # all correctly predicted samples, but this print lets you inspect
            # whether correct samples are strongly head-class dominated.
            correct_per_class = []
            for class_id in range(self.num_classes):
                class_correct = int(
                    (correct_img & labels_img.eq(class_id)).sum().item()
                )
                correct_per_class.append(class_correct)
            print("correct images per class = {}".format(correct_per_class))
            print("K range: {} -> {}".format(k_min, k_max))

        for k in range(k_min, k_max + 1):
            medoid_ids, cluster_labels = self._fit_kmedoids_cosine(
                cluster_pool,
                k=k,
                seed=self.random_seed + k,
            )

            score = self._silhouette_cosine(
                cluster_pool,
                cluster_labels,
                seed=self.random_seed + 1000 + k,
            )

            if verbose:
                print(
                    "K={:3d} | silhouette={:.6f}".format(
                        k,
                        score,
                    )
                )

            if score > best_score:
                best_score = score
                best_k = k
                best_medoid_ids = medoid_ids.detach().clone()

        best_medoids_raw = cluster_pool.index_select(
            0,
            best_medoid_ids,
        ).detach()

        best_medoids_norm = F.normalize(
            best_medoids_raw,
            p=2,
            dim=1,
            eps=self.eps,
        )

        # Assign ALL reference regions using the selected fixed medoids.
        assignments_flat = self._assign_to_medoids(
            regions_flat.to(device=device, dtype=torch.float32),
            medoids_norm=best_medoids_norm,
        ).cpu()

        assignments_img = assignments_flat.view(
            n_img,
            n_region,
        )

        # --------------------------------------------------------
        # Conditional-MI nuisance cluster selection.
        #
        # Replace the original RaVL H_k / G_k selection only.
        # Everything after cluster selection remains unchanged.
        #
        # For cluster k:
        #   S_k = 1 if an image contains at least one region from k.
        #
        # For every ground-truth class y, estimate
        #   I(S_k ; Y_hat | Y=y)
        # and retain only the harmful direction
        #   Delta_err = P(E=1 | S_k=1,Y=y)
        #             - P(E=1 | S_k=0,Y=y).
        #
        # Per-class nuisance score:
        #   N_{k,y} = balance * I(S_k;Y_hat|Y=y) * max(Delta_err, 0)
        #
        # Cluster score:
        #   N_k = sum_y N_{k,y}
        #
        # The sum is class-balanced: each class contributes once rather
        # than being weighted by its sample frequency, which is desirable
        # for long-tailed recognition.
        # --------------------------------------------------------

        # Keep the original result-field names for backward compatibility:
        #   influence_scores      -> normalized conditional nuisance score
        #   performance_gaps      -> raw conditional nuisance score N_k
        #   per_class_gaps        -> per-class error increase Delta_err
        #   per_class_influence   -> per-class conditional MI
        influence_scores = {}
        performance_gaps = {}
        per_class_gaps = {}
        per_class_influence = {}

        # Extra local dictionaries used only during selection / logging.
        per_class_nuisance = {}
        per_class_balance = {}

        def _conditional_mi_binary_cluster(
            cluster_present: torch.Tensor,
            predicted_labels: torch.Tensor,
        ) -> float:
            """
            Empirical mutual information I(S; Y_hat) in nats.

            cluster_present: [N], bool, S in {0,1}
            predicted_labels: [N], long, Y_hat in {0,...,C-1}

            This function is called inside a fixed ground-truth class,
            therefore it estimates I(S_k; Y_hat | Y=y).
            """
            n = int(cluster_present.numel())
            if n <= 1:
                return 0.0

            s = cluster_present.long()
            y_hat = predicted_labels.long()

            joint = torch.zeros(
                2,
                self.num_classes,
                dtype=torch.float64,
            )

            flat_index = s * self.num_classes + y_hat
            counts = torch.bincount(
                flat_index,
                minlength=2 * self.num_classes,
            ).double()
            joint = counts.view(2, self.num_classes)

            total = joint.sum()
            if float(total.item()) <= 0.0:
                return 0.0

            p_joint = joint / total
            p_s = p_joint.sum(dim=1, keepdim=True)
            p_yhat = p_joint.sum(dim=0, keepdim=True)
            denom = p_s * p_yhat

            valid = p_joint > 0
            if not bool(valid.any().item()):
                return 0.0

            mi = (
                p_joint[valid]
                * torch.log(
                    p_joint[valid]
                    / denom[valid].clamp_min(self.eps)
                )
            ).sum()

            return float(mi.item())

        for cluster_id in range(best_k):
            # S_k for every reference image.
            present = assignments_img.eq(cluster_id).any(dim=1)

            mi_by_class = {}
            error_increase_by_class = {}
            nuisance_by_class = {}
            balance_by_class = {}

            for y in range(self.num_classes):
                class_mask = labels_img.eq(y)

                # Need both S_k=1 and S_k=0 inside this class so that the
                # cluster-presence variable is actually comparable.
                in_mask = class_mask & present
                out_mask = class_mask & (~present)

                n_in = int(in_mask.sum().item())
                n_out = int(out_mask.sum().item())

                if n_in == 0 or n_out == 0:
                    continue

                # ----------------------------------------------------
                # 1) Harmful direction: increase in error probability.
                # ----------------------------------------------------
                err_in = float(
                    (~correct_img[in_mask]).float().mean().item()
                )
                err_out = float(
                    (~correct_img[out_mask]).float().mean().item()
                )

                delta_err = err_in - err_out

                # Same presence/absence balancing idea as the old code,
                # but it now stabilizes the conditional-MI score rather
                # than defining G_k.
                balance = (
                    2.0
                    * min(n_in, n_out)
                    / float(n_in + n_out)
                )

                # ----------------------------------------------------
                # 2) Conditional MI:
                #       I(S_k ; Y_hat | Y=y)
                # ----------------------------------------------------
                class_present = present[class_mask]
                class_preds = preds_img[class_mask]

                cmi_y = _conditional_mi_binary_cluster(
                    cluster_present=class_present,
                    predicted_labels=class_preds,
                )

                # ----------------------------------------------------
                # 3) Per-class nuisance score.
                # Only the harmful direction is kept.
                # ----------------------------------------------------
                harmful_delta = max(delta_err, 0.0)
                nuisance_y = balance * cmi_y * harmful_delta

                mi_by_class[y] = float(cmi_y)
                error_increase_by_class[y] = float(delta_err)
                nuisance_by_class[y] = float(nuisance_y)
                balance_by_class[y] = float(balance)

            # Class-balanced aggregation. We intentionally do NOT multiply
            # by the empirical class prior, otherwise head classes would
            # dominate the cluster ranking in a long-tailed dataset.
            nuisance_score = sum(nuisance_by_class.values())

            performance_gaps[cluster_id] = float(nuisance_score)
            per_class_gaps[cluster_id] = error_increase_by_class
            per_class_influence[cluster_id] = mi_by_class
            per_class_nuisance[cluster_id] = nuisance_by_class
            per_class_balance[cluster_id] = balance_by_class

        # Normalize only for thresholding so the existing constructor
        # argument influence_threshold can be kept unchanged.
        max_nuisance_score = max(
            performance_gaps.values()
        ) if len(performance_gaps) > 0 else 0.0

        for cluster_id in range(best_k):
            raw_score = performance_gaps.get(cluster_id, 0.0)
            if max_nuisance_score > self.eps:
                normalized_score = raw_score / max_nuisance_score
            else:
                normalized_score = 0.0
            influence_scores[cluster_id] = float(normalized_score)

        candidate_clusters = [
            c
            for c in range(best_k)
            if performance_gaps.get(c, 0.0) > 0.0
            and influence_scores.get(c, 0.0) >= self.influence_threshold
        ]

        # Rank by the raw conditional nuisance score N_k.
        ranked_clusters = sorted(
            candidate_clusters,
            key=lambda c: performance_gaps.get(c, 0.0),
            reverse=True,
        )

        if len(ranked_clusters) == 0:
            raise RuntimeError(
                "Conditional-MI selection found no nuisance cluster with "
                "positive harmful score and normalized score >= {:.3f}. "
                "Try inspecting the reference split or lowering "
                "influence_threshold.".format(
                    self.influence_threshold
                )
            )

        top_spurious_clusters = [int(c) for c in ranked_clusters[:self.num_spurious_clusters]]

        # backward compatible: keep original single cluster variable
        top_spurious_cluster = int(top_spurious_clusters[0])

        # Keep the fixed Stage-1 clustering model.
        self.medoids_raw = best_medoids_raw.detach().cpu()
        self.medoids_norm = best_medoids_norm.detach().cpu()
        self.top_spurious_cluster = top_spurious_cluster
        self.top_spurious_clusters = top_spurious_clusters
        self.ranked_clusters = list(ranked_clusters)

        result = RaVLDiscoveryResult(
            best_k=int(best_k),
            best_silhouette=float(best_score),
            top_spurious_cluster=top_spurious_cluster,
            ranked_clusters=list(ranked_clusters),
            influence_scores=influence_scores,
            performance_gaps=performance_gaps,
            per_class_gaps=per_class_gaps,
            per_class_influence=per_class_influence,
            num_images=int(n_img),
            num_regions=int(total_regions),
        )

        self.discovery_result = result

        if make_assignment_snapshot:
            self.prepare_stage2_assignment_model(
                model=model,
                device=device,
            )

        if verbose:
            print("========== RaVL Stage 1: Discovery Result ==========")
            print(
                "best K={} | silhouette={:.6f}".format(
                    best_k,
                    best_score,
                )
            )
            print(
                "normalized nuisance-score threshold={:.3f}".format(
                    self.influence_threshold
                )
            )

            for rank, cluster_id in enumerate(ranked_clusters[:10], 1):
                print(
                    "Rank {:2d} | cluster {:3d} | N_norm={:.4f} | N_score={:.6f}".format(
                        rank,
                        cluster_id,
                        influence_scores[cluster_id],
                        performance_gaps[cluster_id],
                    )
                )

            print(
                "TOP spurious clusters = {}".format(
                    top_spurious_clusters
                )
            )
            print("=====================================================")

        return result

    # ------------------------------------------------------------
    # Fixed Stage-1 assignment model for Stage 2
    # ------------------------------------------------------------
    @torch.no_grad()
    def prepare_stage2_assignment_model(
        self,
        model: nn.Module,
        device=None,
    ) -> None:
        """
        Freeze a snapshot of the Stage-1 encoder.

        The paper first determines the spurious cluster, then uses the trained
        clustering model to assign training regions to R^s / R^r before
        mitigation. This frozen snapshot keeps region assignment in the
        original Stage-1 feature space even while the student backbone changes.
        """
        base = _unwrap_model(model)

        if device is None:
            device = next(base.parameters()).device

        frozen = copy.deepcopy(base).to(device)
        frozen.eval()

        for p in frozen.parameters():
            p.requires_grad_(False)

        # Bypass nn.Module registration because RaVLResNet is intentionally
        # a parameter-free utility object.
        self._assignment_model = frozen
        self._assignment_device = device
    
    @torch.no_grad()
    def evaluate_classwise_nuisance_ratio(
        self,
        student_model: nn.Module,
        data_loader,
        device,
        max_batches: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate class-wise nuisance patch statistics.

        Current pipeline:
            1. Current student features -> priority relevant Top-rel_num
            2. Frozen Stage-1 model -> cluster assignment
            3. Raw nuisance clusters
            4. Remove priority relevant regions
            5. Obtain FINAL nuisance regions

        For each class c:

            raw_nuisance_ratio[c]
                = # raw nuisance patches of class c
                  / # all patches of class c

            protected_ratio[c]
                = # raw nuisance patches protected by Top-rel_num
                  / # all patches of class c

            final_nuisance_ratio[c]
                = # final nuisance patches of class c
                  / # all patches of class c

        Also reports:

            image_with_nuisance_ratio[c]
                = proportion of images in class c containing
                  at least one final nuisance patch

            mean_nuisance_ratio_per_image[c]
                = average final nuisance patch ratio over
                  images belonging to class c

        Returns:
            Dictionary containing one numpy array per statistic,
            with shape [num_classes].
        """

        # ============================================================
        # 1. Prepare model
        # ============================================================
        base = _unwrap_model(student_model)
        was_training = base.training
        base.eval()

        num_classes = int(self.num_classes)

        # Number of images for each class
        class_num_images = torch.zeros(
            num_classes,
            dtype=torch.long,
        )

        # Total number of spatial patches for each class
        class_total_patches = torch.zeros(
            num_classes,
            dtype=torch.long,
        )

        # Raw cluster-based nuisance patches
        class_raw_nuisance = torch.zeros(
            num_classes,
            dtype=torch.long,
        )

        # Raw nuisance patches protected by Top-rel_num
        class_protected = torch.zeros(
            num_classes,
            dtype=torch.long,
        )

        # Final nuisance patches
        class_final_nuisance = torch.zeros(
            num_classes,
            dtype=torch.long,
        )

        # Number of images containing >= 1 final nuisance patch
        class_images_with_nuisance = torch.zeros(
            num_classes,
            dtype=torch.long,
        )

        # Sum of per-image nuisance ratios
        class_sum_image_nuisance_ratio = torch.zeros(
            num_classes,
            dtype=torch.float64,
        )

        try:

            # ========================================================
            # 2. Iterate dataset
            # ========================================================
            for batch_idx, batch in enumerate(data_loader):

                if (
                    max_batches is not None
                    and batch_idx >= int(max_batches)
                ):
                    break

                if (
                    not isinstance(batch, (tuple, list))
                    or len(batch) < 2
                ):
                    raise TypeError(
                        "data_loader must return "
                        "(inputs, labels, ...)."
                    )

                inputs = batch[0]
                labels = batch[1]

                if isinstance(inputs, (tuple, list)):
                    inputs = inputs[0]

                inputs = inputs.to(
                    device,
                    non_blocking=True,
                )

                labels = labels.to(
                    device,
                    non_blocking=True,
                ).long()

                # ====================================================
                # 3. Current student representation
                # ====================================================
                _, student_features = (
                    _capture_layer4_and_forward(
                        student_model,
                        inputs,
                    )
                )

                student_regions = (
                    self._regions_from_features(
                        student_features
                    )
                )

                # [B,R,D]
                b, r, _ = student_regions.shape

                classifier = _get_classifier(
                    student_model
                )

                # ====================================================
                # 4. Top-rel_num priority relevant patches
                # ====================================================
                priority_relevant_mask, _ = (
                    self._priority_relevant_mask(
                        student_regions=student_regions,
                        labels=labels,
                        classifier=classifier,
                    )
                )

                # ====================================================
                # 5. Frozen Stage-1 cluster assignment
                # ====================================================
                (
                    cluster_ids,
                    raw_spurious_mask,
                    _,
                ) = self._stage2_region_partition(
                    inputs
                )

                raw_spurious_mask = (
                    raw_spurious_mask.to(
                        device=student_regions.device,
                        dtype=torch.bool,
                    )
                )

                cluster_ids = cluster_ids.to(
                    device=student_regions.device,
                )

                # Safety check
                if cluster_ids.shape != (b, r):
                    raise ValueError(
                        "Cluster assignment shape {} does not "
                        "match student region shape [{},{}].".format(
                            tuple(cluster_ids.shape),
                            b,
                            r,
                        )
                    )

                # ====================================================
                # 6. Priority protection
                # ====================================================
                (
                    final_spurious_mask,
                    _,
                    protected_mask,
                ) = self._apply_priority_protection(
                    raw_spurious_mask=raw_spurious_mask,
                    priority_relevant_mask=priority_relevant_mask,
                )

                # ====================================================
                # 7. Statistics for each class
                # ====================================================
                for c in range(num_classes):

                    class_mask = labels.eq(c)

                    num_images_c = int(
                        class_mask.sum().item()
                    )

                    if num_images_c == 0:
                        continue

                    # -----------------------------------------------
                    # Masks belonging to class c
                    #
                    # [Nc,R]
                    # -----------------------------------------------
                    raw_c = raw_spurious_mask[
                        class_mask
                    ]

                    protected_c = protected_mask[
                        class_mask
                    ]

                    final_c = final_spurious_mask[
                        class_mask
                    ]

                    # -----------------------------------------------
                    # Image / patch counts
                    # -----------------------------------------------
                    class_num_images[c] += (
                        num_images_c
                    )

                    class_total_patches[c] += (
                        num_images_c * r
                    )

                    class_raw_nuisance[c] += int(
                        raw_c.sum().item()
                    )

                    class_protected[c] += int(
                        protected_c.sum().item()
                    )

                    class_final_nuisance[c] += int(
                        final_c.sum().item()
                    )

                    # -----------------------------------------------
                    # Does each image contain nuisance?
                    # -----------------------------------------------
                    image_has_nuisance = (
                        final_c.any(dim=1)
                    )

                    class_images_with_nuisance[c] += int(
                        image_has_nuisance.sum().item()
                    )

                    # -----------------------------------------------
                    # Per-image nuisance ratio
                    #
                    # [Nc,R] -> [Nc]
                    # -----------------------------------------------
                    image_nuisance_ratio = (
                        final_c.float()
                        .mean(dim=1)
                    )

                    class_sum_image_nuisance_ratio[c] += (
                        image_nuisance_ratio
                        .double()
                        .sum()
                        .cpu()
                    )

        finally:

            # ========================================================
            # 8. Restore model state
            # ========================================================
            if was_training:
                base.train()
            else:
                base.eval()

        # ============================================================
        # 9. Convert counts to ratios
        # ============================================================
        num_images_np = (
            class_num_images
            .numpy()
            .astype(np.int64)
        )

        total_patches_np = (
            class_total_patches
            .numpy()
            .astype(np.int64)
        )

        raw_nuisance_np = (
            class_raw_nuisance
            .numpy()
            .astype(np.int64)
        )

        protected_np = (
            class_protected
            .numpy()
            .astype(np.int64)
        )

        final_nuisance_np = (
            class_final_nuisance
            .numpy()
            .astype(np.int64)
        )

        images_with_nuisance_np = (
            class_images_with_nuisance
            .numpy()
            .astype(np.int64)
        )

        sum_image_ratio_np = (
            class_sum_image_nuisance_ratio
            .numpy()
        )

        # ------------------------------------------------------------
        # Avoid divide-by-zero
        # ------------------------------------------------------------
        patch_denominator = np.maximum(
            total_patches_np,
            1,
        )

        image_denominator = np.maximum(
            num_images_np,
            1,
        )

        raw_nuisance_ratio = (
            raw_nuisance_np
            / patch_denominator
        )

        protected_ratio = (
            protected_np
            / patch_denominator
        )

        final_nuisance_ratio = (
            final_nuisance_np
            / patch_denominator
        )

        image_with_nuisance_ratio = (
            images_with_nuisance_np
            / image_denominator
        )

        mean_nuisance_ratio_per_image = (
            sum_image_ratio_np
            / image_denominator
        )

        # Classes absent from this loader -> NaN
        empty_class = (
            num_images_np == 0
        )

        raw_nuisance_ratio[
            empty_class
        ] = np.nan

        protected_ratio[
            empty_class
        ] = np.nan

        final_nuisance_ratio[
            empty_class
        ] = np.nan

        image_with_nuisance_ratio[
            empty_class
        ] = np.nan

        mean_nuisance_ratio_per_image[
            empty_class
        ] = np.nan

        # ============================================================
        # 10. Results
        # ============================================================
        results = {
            "num_images": num_images_np,

            "num_total_patches": total_patches_np,

            "num_raw_nuisance_patches": raw_nuisance_np,

            "num_protected_patches": protected_np,

            "num_final_nuisance_patches": final_nuisance_np,

            "raw_nuisance_ratio": raw_nuisance_ratio,

            "protected_ratio": protected_ratio,

            "final_nuisance_ratio": final_nuisance_ratio,

            "image_with_nuisance_ratio": (
                image_with_nuisance_ratio
            ),

            "mean_nuisance_ratio_per_image": (
                mean_nuisance_ratio_per_image
            ),
        }

        # ============================================================
        # 11. Pretty print
        # ============================================================
        if verbose:

            print(
                "\n"
                + "=" * 92
            )

            print(
                "Class-wise nuisance statistics "
                "(after Top-{} priority protection)".format(
                    self.rel_num
                )
            )

            print(
                "=" * 92
            )

            print(
                "{:<7s} {:>8s} {:>10s} {:>10s} {:>10s} "
                "{:>10s} {:>10s}".format(
                    "Class",
                    "Images",
                    "Raw(%)",
                    "Protect(%)",
                    "Final(%)",
                    "ImgHit(%)",
                    "FinalNum",
                )
            )

            print("-" * 92)

            for c in range(num_classes):

                if num_images_np[c] == 0:

                    print(
                        "{:<7d} {:>8d} {:>10s} {:>10s} "
                        "{:>10s} {:>10s} {:>10d}".format(
                            c,
                            0,
                            "-",
                            "-",
                            "-",
                            "-",
                            0,
                        )
                    )

                    continue

                print(
                    "{:<7d} {:>8d} {:>10.2f} {:>10.2f} "
                    "{:>10.2f} {:>10.2f} {:>10d}".format(
                        c,
                        num_images_np[c],

                        100.0
                        * raw_nuisance_ratio[c],

                        100.0
                        * protected_ratio[c],

                        100.0
                        * final_nuisance_ratio[c],

                        100.0
                        * image_with_nuisance_ratio[c],

                        final_nuisance_np[c],
                    )
                )

            print("=" * 92)

            valid_patch_total = int(
                total_patches_np.sum()
            )

            valid_final_total = int(
                final_nuisance_np.sum()
            )

            if valid_patch_total > 0:

                overall_ratio = (
                    valid_final_total
                    / valid_patch_total
                )

            else:

                overall_ratio = 0.0

            print(
                "Overall final nuisance ratio: "
                "{:.2f}% ({}/{})".format(
                    100.0 * overall_ratio,
                    valid_final_total,
                    valid_patch_total,
                )
            )

            print("=" * 92 + "\n")

        return results
    
    
    # ------------------------------------------------------------
    # Region assignment during Stage 2
    # ------------------------------------------------------------
    @torch.no_grad()
    def _stage2_region_partition(
        self,
        inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Raw cluster-based partition in the frozen Stage-1 feature space.

        IMPORTANT:
            The returned spurious_mask is the RAW nuisance-cluster mask.
            It does not yet apply the rel_num priority-relevant protection.
            The protection needs the current student representation and w_y.
        """
        if self._assignment_model is None:
            raise RuntimeError(
                "No frozen Stage-1 assignment model. "
                "Run discover(..., make_assignment_snapshot=True) or call "
                "prepare_stage2_assignment_model(model)."
            )

        if self.medoids_norm is None or self.top_spurious_cluster is None:
            raise RuntimeError("RaVL discovery has not been initialized.")

        assignment_inputs = inputs.to(
            self._assignment_device,
            non_blocking=True,
        )

        _, assignment_features = _capture_layer4_and_forward(
            self._assignment_model,
            assignment_inputs,
        )

        assignment_regions = self._regions_from_features(
            assignment_features
        )

        medoids_norm = self.medoids_norm.to(
            device=assignment_regions.device,
            dtype=assignment_regions.dtype,
        )

        cluster_ids = self._assign_to_medoids(
            assignment_regions,
            medoids_norm=medoids_norm,
        )

        raw_spurious_mask = torch.zeros_like(
            cluster_ids,
            dtype=torch.bool,
        )
        for c in self.top_spurious_clusters:
            raw_spurious_mask |= cluster_ids.eq(int(c))

        raw_relevant_mask = ~raw_spurious_mask
        return cluster_ids, raw_spurious_mask, raw_relevant_mask

    # ------------------------------------------------------------
    # Highest-priority task-relevant regions
    # ------------------------------------------------------------
    def _priority_relevant_mask(
        self,
        student_regions: torch.Tensor,
        labels: torch.Tensor,
        classifier: nn.Linear,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select rel_num regions with the largest cosine similarity to the
        ground-truth class direction w_{y_i} for each image.

        These regions have ABSOLUTE PRIORITY: once selected, they can never be
        treated as nuisance even if their frozen cluster assignment belongs to
        a discovered nuisance cluster.

        Returns:
            priority_mask: [B,R] bool
            gt_similarity: [B,R] cosine similarity to w_{y_i}
        """
        b, r, d = student_regions.shape
        if classifier.weight.shape != (self.num_classes, d):
            raise ValueError(
                "Classifier weight shape {} incompatible with regions [*,{},{}].".format(
                    tuple(classifier.weight.shape), r, d
                )
            )
        if labels.shape[0] != b:
            raise ValueError("labels batch size must match student_regions.")

        k = min(int(self.rel_num), int(r))
        if k < 1:
            raise RuntimeError("No spatial region is available for rel_num selection.")

        region_n = F.normalize(
            student_regions,
            p=2,
            dim=2,
            eps=self.eps,
        )
        class_n = F.normalize(
            classifier.weight.detach().to(
                device=student_regions.device,
                dtype=student_regions.dtype,
            ),
            p=2,
            dim=1,
            eps=self.eps,
        )

        gt_w = class_n.index_select(0, labels.long())  # [B,D]
        gt_similarity = torch.einsum("brd,bd->br", region_n, gt_w)
        top_idx = gt_similarity.topk(k=k, dim=1, largest=True, sorted=True).indices

        priority_mask = torch.zeros(
            (b, r),
            dtype=torch.bool,
            device=student_regions.device,
        )
        priority_mask.scatter_(1, top_idx, True)
        return priority_mask, gt_similarity

    def _apply_priority_protection(
        self,
        raw_spurious_mask: torch.Tensor,
        priority_relevant_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Priority rule:

            final nuisance = raw nuisance AND NOT priority-relevant
            final relevant = NOT final nuisance

        Therefore, rel_num selected regions can never enter L_R/L_A as nuisance.
        """
        raw_spurious_mask = raw_spurious_mask.to(
            device=priority_relevant_mask.device,
            dtype=torch.bool,
        )
        priority_relevant_mask = priority_relevant_mask.bool()

        protected_from_nuisance = raw_spurious_mask & priority_relevant_mask
        spurious_mask = raw_spurious_mask & (~priority_relevant_mask)
        relevant_mask = ~spurious_mask
        return spurious_mask, relevant_mask, protected_from_nuisance

    # ------------------------------------------------------------
    # Original bidirectional RaVL-style L_R + L_A
    # ------------------------------------------------------------
    def _region_aware_loss(
        self,
        student_regions: torch.Tensor,
        labels: torch.Tensor,
        spurious_mask: torch.Tensor,
        relevant_mask: torch.Tensor,
        classifier: nn.Linear,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the bidirectional region-class losses after rel_num protection.

        L_R (region-to-class):
            - positive: strongest relevant region of image i for GT class y_i
            - class negatives: strongest relevant response of image i to labels
              occurring in the mini-batch
            - nuisance negatives: each nuisance patch's strongest response to
              labels occurring in the mini-batch

        L_A (class-to-region):
            - fix GT class direction w_{y_i}
            - positive: current image's strongest relevant response to w_{y_i}
            - relevant negatives: other-class images' strongest relevant
              response to w_{y_i}
            - nuisance negatives: all nuisance patches' response to w_{y_i}

        Classifier weights are detached, so L_R/L_A update the representation
        rather than moving the class anchors.
        """
        b, r, d = student_regions.shape
        if classifier.weight.shape != (self.num_classes, d):
            raise ValueError(
                "Classifier weight shape {} incompatible with regions [*,{},{}].".format(
                    tuple(classifier.weight.shape), r, d
                )
            )

        device = student_regions.device
        labels = labels.to(device=device, dtype=torch.long)
        spurious_mask = spurious_mask.to(device=device, dtype=torch.bool)
        relevant_mask = relevant_mask.to(device=device, dtype=torch.bool)

        region_n = F.normalize(
            student_regions,
            p=2,
            dim=2,
            eps=self.eps,
        )
        class_n = F.normalize(
            classifier.weight.detach().to(
                device=device,
                dtype=student_regions.dtype,
            ),
            p=2,
            dim=1,
            eps=self.eps,
        )

        # [B,R,C]
        sim = torch.einsum("brd,cd->brc", region_n, class_n)

        # Relevant-region strongest response m_i(c).
        neg_large = torch.finfo(sim.dtype).min
        rel_sim = sim.masked_fill(~relevant_mask.unsqueeze(-1), neg_large)
        valid_images = relevant_mask.any(dim=1)
        rel_max_sim = rel_sim.max(dim=1).values  # [B,C]

        # In this design rel_num>=1 normally guarantees valid_images=True,
        # but keep a safe differentiable-zero fallback.
        valid_ids = valid_images.nonzero(as_tuple=False).squeeze(1)
        if valid_ids.numel() == 0:
            zero = student_regions.sum() * 0.0
            return zero, zero, zero, valid_images

        # [Ns,C] nuisance responses. It is valid for Ns=0.
        spur_sim = sim[spurious_mask]

        losses_R = []
        losses_A = []

        # q(r_s): strongest nuisance response to labels occurring in batch.
        if spur_sim.numel() > 0:
            spur_max_over_batch_labels = (
                spur_sim.index_select(1, labels).max(dim=1).values
                / self.temperature
            )
        else:
            spur_max_over_batch_labels = None

        for i in valid_ids.tolist():
            y_i = int(labels[i].item())
            positive_log = rel_max_sim[i, y_i] / self.temperature

            # ------------------------- L_R -------------------------
            # Keep duplicate labels exactly as they occur in the batch.
            label_terms_R = (
                rel_max_sim[i].index_select(0, labels)
                / self.temperature
            )
            denom_terms_R = [label_terms_R]
            if spur_max_over_batch_labels is not None:
                denom_terms_R.append(spur_max_over_batch_labels)
            denom_R = torch.cat(denom_terms_R, dim=0)
            loss_R_i = -positive_log + torch.logsumexp(denom_R, dim=0)
            losses_R.append(loss_R_i)

            # ------------------------- L_A -------------------------
            denom_terms_A = [positive_log.reshape(1)]

            other_mask = valid_images & labels.ne(y_i)
            other_ids = other_mask.nonzero(as_tuple=False).squeeze(1)
            if other_ids.numel() > 0:
                other_rel_to_y = (
                    rel_max_sim.index_select(0, other_ids)[:, y_i]
                    / self.temperature
                )
                denom_terms_A.append(other_rel_to_y)

            if spur_sim.numel() > 0:
                spur_to_y = spur_sim[:, y_i] / self.temperature
                denom_terms_A.append(spur_to_y)

            denom_A = torch.cat(denom_terms_A, dim=0)
            loss_A_i = -positive_log + torch.logsumexp(denom_A, dim=0)
            losses_A.append(loss_A_i)

        loss_R = torch.stack(losses_R).mean()
        loss_A = torch.stack(losses_A).mean()
        loss_region = loss_R + loss_A
        return loss_region, loss_R, loss_A, valid_images

    def _nuisance_class_orthogonal_loss(
        self,
        student_regions: torch.Tensor,
        spurious_mask: torch.Tensor,
        classifier: nn.Linear,
    ) -> torch.Tensor:
        """
        Nuisance-Class Orthogonalization (NCO).

        Only suppress FINAL nuisance regions.

        For each nuisance region z^s, suppress its strongest absolute
        similarity to all classifier directions:

            L_NCO =
                mean_{z^s} [
                    max_c cos^2(z^s, w_c)
                ]

        Therefore:

            cos(z^s, w_c) -> 0,  for all classes c.

        The classifier weights are detached, so this loss only updates
        the student representation and does not move classifier anchors.

        Args:
            student_regions:
                Tensor [B, R, D]

            spurious_mask:
                Bool Tensor [B, R]

                IMPORTANT:
                This should be the FINAL nuisance mask after
                rel_num priority protection.

            classifier:
                Final nn.Linear classifier.
                classifier.weight shape = [C, D]

        Returns:
            loss_nco:
                Scalar differentiable tensor.
        """

        # ============================================================
        # 1. Shape check
        # ============================================================
        if student_regions.ndim != 3:
            raise ValueError(
                "student_regions must be [B,R,D], got {}.".format(
                    tuple(student_regions.shape)
                )
            )

        b, r, d = student_regions.shape

        if classifier.weight.shape != (
            self.num_classes,
            d,
        ):
            raise ValueError(
                "Classifier weight shape {} incompatible with "
                "student_regions [B={}, R={}, D={}].".format(
                    tuple(classifier.weight.shape),
                    b,
                    r,
                    d,
                )
            )

        device = student_regions.device

        spurious_mask = spurious_mask.to(
            device=device,
            dtype=torch.bool,
        )

        if spurious_mask.shape != (b, r):
            raise ValueError(
                "spurious_mask must have shape [{},{}], got {}.".format(
                    b,
                    r,
                    tuple(spurious_mask.shape),
                )
            )

        # ============================================================
        # 2. Normalize student region features
        #
        # [B,R,D]
        # ============================================================
        region_n = F.normalize(
            student_regions,
            p=2,
            dim=2,
            eps=self.eps,
        )

        # ============================================================
        # 3. Normalize classifier directions
        #
        # [C,D]
        #
        # detach():
        # classifier directions are fixed anchors for this loss.
        # Only student representation receives gradients.
        # ============================================================
        class_n = F.normalize(
            classifier.weight.detach().to(
                device=device,
                dtype=student_regions.dtype,
            ),
            p=2,
            dim=1,
            eps=self.eps,
        )

        # ============================================================
        # 4. Similarity between EVERY region and EVERY class
        #
        # region_n : [B,R,D]
        # class_n  : [C,D]
        #
        # output:
        #     similarity[b,r,c]
        #
        # [B,R,C]
        # ============================================================
        similarity = torch.einsum(
            "brd,cd->brc",
            region_n,
            class_n,
        )

        # ============================================================
        # 5. Select FINAL nuisance regions only
        #
        # [N_s,C]
        # ============================================================
        nuisance_similarity = similarity[
            spurious_mask
        ]

        # ============================================================
        # 6. No nuisance patch in this mini-batch
        #
        # Return differentiable zero.
        # ============================================================
        if nuisance_similarity.numel() == 0:
            return (
                student_regions.sum()
                * 0.0
            )

        # ============================================================
        # 7. Nuisance suppression over ALL class directions
        #
        # First:
        #
        #     cos^2(z_s, w_c)
        #
        # [N_s,C]
        #
        # Squaring is important:
        #
        #     +1 -> bad
        #     -1 -> also bad
        #      0 -> desired
        # ============================================================
        nuisance_similarity_sq = (
            nuisance_similarity.pow(2)
        )

        # ============================================================
        # 8. For each nuisance patch, find its strongest
        #    class-direction response.
        #
        # max_c cos^2(z_s, w_c)
        #
        # [N_s]
        #
        # This avoids the problem that averaging over classes may hide
        # one very large wrong-class response.
        # ============================================================
        strongest_class_similarity = (
            nuisance_similarity_sq
            .max(dim=1)
            .values
        )

        # ============================================================
        # 9. Final loss
        #
        # min:
        #
        #     max_c cos^2(z_s, w_c)
        #
        # therefore nuisance representation is pushed away from
        # the entire task-discriminative classifier space.
        # ============================================================
        loss_nco = (
            strongest_class_similarity.mean()
        )

        return loss_nco

    def _nuisance_wrong_class_suppression_loss(
        self,
        student_regions: torch.Tensor,
        labels: torch.Tensor,
        spurious_mask: torch.Tensor,
        classifier: nn.Linear,
    ) -> torch.Tensor:
        """
        Wrong-Class Positive Evidence Suppression (WPS).

        Only FINAL nuisance regions participate.  Unlike all-class NCO, this
        objective does NOT force nuisance regions to be orthogonal to the whole
        classifier space and does NOT penalize their similarity to the ground-
        truth direction w_{y_i}.

        For nuisance region z^s_{i,m}, define cosine similarities to all class
        directions w_c.  We suppress only the strongest POSITIVE response among
        WRONG classes:

            L_WPS = mean_{i,m: S_{i,m}=1}
                    [ max_{c != y_i} ReLU(cos(z^s_{i,m}, w_c)) ]^2.

        Therefore:
            - positive GT evidence is untouched by this auxiliary loss;
            - negative wrong-class cosine is untouched because it is not positive
              misleading evidence;
            - if all wrong-class cosine similarities are <= 0, this patch receives
              zero auxiliary loss;
            - classifier weights are detached, so only the student representation
              is updated by WPS.

        Args:
            student_regions:
                Tensor [B,R,D].

            labels:
                Tensor [B], ground-truth class labels.

            spurious_mask:
                Bool Tensor [B,R].  This must be the FINAL nuisance mask after
                rel_num priority protection.

            classifier:
                Final nn.Linear classifier with weight [C,D].

        Returns:
            Scalar differentiable WPS loss.
        """
        if student_regions.ndim != 3:
            raise ValueError(
                "student_regions must be [B,R,D], got {}.".format(
                    tuple(student_regions.shape)
                )
            )

        b, r, d = student_regions.shape
        device = student_regions.device

        if classifier.weight.shape != (self.num_classes, d):
            raise ValueError(
                "Classifier weight shape {} incompatible with "
                "student_regions [B={}, R={}, D={}].".format(
                    tuple(classifier.weight.shape), b, r, d
                )
            )

        labels = labels.to(
            device=device,
            dtype=torch.long,
        )
        if labels.shape != (b,):
            raise ValueError(
                "labels must have shape [{}], got {}.".format(
                    b, tuple(labels.shape)
                )
            )
        if bool((labels < 0).any().item()) or bool(
            (labels >= self.num_classes).any().item()
        ):
            raise ValueError("labels contain class ids outside [0, C-1].")

        spurious_mask = spurious_mask.to(
            device=device,
            dtype=torch.bool,
        )
        if spurious_mask.shape != (b, r):
            raise ValueError(
                "spurious_mask must have shape [{},{}], got {}.".format(
                    b, r, tuple(spurious_mask.shape)
                )
            )

        # No final nuisance patch in this mini-batch -> differentiable zero.
        if not bool(spurious_mask.any().item()):
            return student_regions.sum() * 0.0

        # Normalize student regions and DETACH classifier anchors.
        region_n = F.normalize(
            student_regions,
            p=2,
            dim=2,
            eps=self.eps,
        )
        class_n = F.normalize(
            classifier.weight.detach().to(
                device=device,
                dtype=student_regions.dtype,
            ),
            p=2,
            dim=1,
            eps=self.eps,
        )

        # [B,R,C]: cosine similarity between each patch and every class direction.
        similarity = torch.einsum(
            "brd,cd->brc",
            region_n,
            class_n,
        )

        # Keep only positive class evidence.  Negative similarity to a wrong class
        # already acts against that class, so WPS does not suppress it further.
        positive_similarity = F.relu(similarity)

        # Remove the GT class from the competition WITHOUT penalizing it.
        # [B,1,C] broadcasts across the R spatial regions.
        gt_mask = F.one_hot(
            labels,
            num_classes=self.num_classes,
        ).to(dtype=torch.bool)[:, None, :]

        wrong_positive_similarity = positive_similarity.masked_fill(
            gt_mask,
            0.0,
        )

        # For each spatial patch, find the strongest positive WRONG-class evidence.
        # [B,R]
        strongest_wrong_positive = (
            wrong_positive_similarity
            .max(dim=2)
            .values
        )

        # Only FINAL nuisance patches participate.
        nuisance_wrong_positive = strongest_wrong_positive[
            spurious_mask
        ]

        if nuisance_wrong_positive.numel() == 0:
            return student_regions.sum() * 0.0

        # L = mean [max_{c != y} ReLU(cos(z_s, w_c))]^2
        loss_wps = (
            nuisance_wrong_positive
            .pow(2)
            .mean()
        )

        return loss_wps

    # ------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------
    @staticmethod
    def _vis_to_numpy_image(
        image: torch.Tensor,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Convert [C,H,W] tensor to RGB numpy image in [0,1]."""
        x = image.detach().cpu().float().clone()
        if x.ndim != 3:
            raise ValueError("image must be [C,H,W].")

        if mean is not None and std is not None:
            mean_t = torch.tensor(mean, dtype=x.dtype).view(-1, 1, 1)
            std_t = torch.tensor(std, dtype=x.dtype).view(-1, 1, 1)
            if mean_t.shape[0] == 1 and x.shape[0] == 3:
                mean_t = mean_t.repeat(3, 1, 1)
                std_t = std_t.repeat(3, 1, 1)
            if mean_t.shape[0] != x.shape[0]:
                raise ValueError(
                    "mean/std channel count {} does not match image channels {}.".format(
                        mean_t.shape[0], x.shape[0]
                    )
                )
            x = x * std_t + mean_t
        else:
            # Medical images are often not in [0,1]. For display only, use
            # sample-wise min-max scaling when needed.
            xmin = float(x.min().item())
            xmax = float(x.max().item())
            if xmin < 0.0 or xmax > 1.0:
                x = (x - x.min()) / (x.max() - x.min() + 1e-8)

        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        elif x.shape[0] >= 3:
            x = x[:3]
        else:
            raise ValueError("Only 1-channel or >=3-channel images are supported.")

        x = x.clamp(0.0, 1.0)
        return x.permute(1, 2, 0).numpy()

    @staticmethod
    def _vis_upsample_map(
        x: torch.Tensor,
        size: Tuple[int, int],
        mode: str,
    ) -> torch.Tensor:
        """Upsample a [H,W] map to image resolution."""
        x = x[None, None].float()
        if mode in ("bilinear", "bicubic"):
            x = F.interpolate(x, size=size, mode=mode, align_corners=False)
        else:
            x = F.interpolate(x, size=size, mode=mode)
        return x[0, 0]

    @staticmethod
    def _vis_draw_patch_boxes(
        ax,
        patch_mask: np.ndarray,
        image_h: int,
        image_w: int,
        edgecolor: str = "white",
        linewidth: float = 1.4,
    ) -> None:
        """Draw feature-grid patch boxes on an image axis."""
        hf, wf = patch_mask.shape
        for rr in range(hf):
            for cc in range(wf):
                if not bool(patch_mask[rr, cc]):
                    continue
                x0 = cc * image_w / wf
                x1 = (cc + 1) * image_w / wf
                y0 = rr * image_h / hf
                y1 = (rr + 1) * image_h / hf
                ax.add_patch(
                    Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        fill=False,
                        edgecolor=edgecolor,
                        linewidth=linewidth,
                    )
                )

    @staticmethod
    def _vis_crop_patch(
        pil_img: Image.Image,
        patch_index: int,
        hf: int,
        wf: int,
    ) -> Image.Image:
        """Crop one layer4 spatial token from the original image."""
        image_w, image_h = pil_img.size
        rr = int(patch_index) // wf
        cc = int(patch_index) % wf
        y0 = int(round(rr * image_h / hf))
        y1 = int(round((rr + 1) * image_h / hf))
        x0 = int(round(cc * image_w / wf))
        x1 = int(round((cc + 1) * image_w / wf))
        return pil_img.crop((x0, y0, x1, y1))

    def _compute_standard_cam(
        self,
        features: torch.Tensor,
        classifier: nn.Linear,
        class_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute standard CAM for a GAP + linear classifier.

        features: [B,D,Hf,Wf]
        class_ids: [B]
        return: [B,Hf,Wf] normalized to [0,1]
        """
        if classifier.weight.shape[1] != features.shape[1]:
            raise ValueError(
                "classifier dim {} != layer4 dim {}.".format(
                    classifier.weight.shape[1], features.shape[1]
                )
            )
        weight = classifier.weight.detach().to(
            device=features.device,
            dtype=features.dtype,
        )
        selected = weight.index_select(0, class_ids.long())
        cam = (features * selected[:, :, None, None]).sum(dim=1)
        cam = F.relu(cam)
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + self.eps)
        return cam

    @torch.no_grad()
    def visualize_nuisance_regions(
        self,
        student_model: nn.Module,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        save_dir: str = "./nuisance_visualization",
        file_prefix: str = "sample",
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        cam_target: str = "gt",
        cam_threshold: float = 0.35,
        foreground_masks: Optional[torch.Tensor] = None,
        foreground_patch_threshold: float = 0.25,
        max_images: Optional[int] = 8,
        save_patch_crops: bool = True,
        max_patches_per_image: int = 32,
    ) -> List[Dict[str, object]]:
        """
        High-resolution visualization of the FINAL nuisance regions after
        rel_num priority-relevant protection.

        The selection order is exactly the same as Stage 2 training:
            1) select Top-rel_num regions most similar to w_{y_i};
            2) obtain raw nuisance-cluster mask;
            3) remove priority-relevant regions from the raw nuisance mask;
            4) visualize only the remaining FINAL nuisance regions.

        Panels:
            Original
            Cluster IDs
            Priority Relevant (Top-rel_num)
            Final Nuisance
            CAM + Final Nuisance
            optional Segmentation + Final Nuisance
        """
        if labels is None:
            raise ValueError(
                "labels are required because priority-relevant regions are "
                "defined using the ground-truth class direction w_y."
            )
        if cam_target not in ("pred", "gt"):
            raise ValueError("cam_target must be 'pred' or 'gt'.")
        if not (0.0 <= cam_threshold <= 1.0):
            raise ValueError("cam_threshold must be in [0,1].")
        if not (0.0 <= foreground_patch_threshold <= 1.0):
            raise ValueError("foreground_patch_threshold must be in [0,1].")

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        base = _unwrap_model(student_model)
        was_training = base.training
        base.eval()

        try:
            device = next(base.parameters()).device
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            logits, features = _capture_layer4_and_forward(student_model, inputs)
            student_regions = self._regions_from_features(features)
            classifier = _get_classifier(student_model)

            priority_mask, gt_similarity = self._priority_relevant_mask(
                student_regions=student_regions,
                labels=labels,
                classifier=classifier,
            )
            cluster_ids, raw_spurious_mask, _ = self._stage2_region_partition(inputs)
            cluster_ids = cluster_ids.to(device)
            raw_spurious_mask = raw_spurious_mask.to(device)
            final_spurious_mask, relevant_mask, protected_mask = (
                self._apply_priority_protection(
                    raw_spurious_mask=raw_spurious_mask,
                    priority_relevant_mask=priority_mask,
                )
            )

            target_ids = labels if cam_target == "gt" else logits.argmax(dim=1)
            cams = self._compute_standard_cam(
                features=features,
                classifier=classifier,
                class_ids=target_ids,
            )

            b, _, image_h, image_w = inputs.shape
            _, _, hf, wf = features.shape
            if int(cluster_ids.shape[1]) != hf * wf:
                raise ValueError(
                    "cluster_ids regions={} but student layer4 grid={}x{}.".format(
                        cluster_ids.shape[1], hf, wf
                    )
                )

            fg_patch_fraction = None
            fg_masks_cpu = None
            if foreground_masks is not None:
                fg = foreground_masks.detach().float()
                if fg.ndim == 3:
                    fg = fg.unsqueeze(1)
                elif fg.ndim == 4 and fg.shape[1] != 1:
                    fg = fg[:, :1]
                if fg.ndim != 4:
                    raise ValueError(
                        "foreground_masks must be [B,H,W] or [B,1,H,W]."
                    )
                if fg.shape[0] != b:
                    raise ValueError("foreground_masks batch size must match inputs.")
                fg = fg.to(device)
                fg_patch_fraction = F.adaptive_avg_pool2d(fg, (hf, wf))[:, 0]
                fg_masks_cpu = fg[:, 0].detach().cpu()

            n_show = b if max_images is None else min(int(max_images), b)
            summaries: List[Dict[str, object]] = []

            for i in range(n_show):
                img_np = self._vis_to_numpy_image(inputs[i], mean=mean, std=std)
                cluster_grid = cluster_ids[i].view(hf, wf).detach().cpu()
                priority_grid = priority_mask[i].view(hf, wf).detach().cpu()
                raw_spur_grid = raw_spurious_mask[i].view(hf, wf).detach().cpu()
                spur_grid = final_spurious_mask[i].view(hf, wf).detach().cpu()
                protected_grid = protected_mask[i].view(hf, wf).detach().cpu()
                cam_grid = cams[i].detach().cpu()

                nuisance_cam = cam_grid[spur_grid]
                num_raw = int(raw_spur_grid.sum().item())
                num_priority = int(priority_grid.sum().item())
                num_protected = int(protected_grid.sum().item())
                num_nuisance = int(spur_grid.sum().item())
                num_total = int(hf * wf)
                nuisance_patch_ratio = num_nuisance / max(num_total, 1)

                if nuisance_cam.numel() > 0:
                    low_cam_ratio = float(
                        (nuisance_cam < cam_threshold).float().mean().item()
                    )
                    high_cam_ratio = float(
                        (nuisance_cam >= cam_threshold).float().mean().item()
                    )
                    mean_cam = float(nuisance_cam.mean().item())
                else:
                    low_cam_ratio = 0.0
                    high_cam_ratio = 0.0
                    mean_cam = 0.0

                priority_sim = gt_similarity[i][priority_mask[i]].detach().cpu()
                mean_priority_sim = (
                    float(priority_sim.mean().item()) if priority_sim.numel() > 0 else 0.0
                )

                seg_bg_ratio = None
                seg_fg_ratio = None
                mean_fg_fraction = None
                if fg_patch_fraction is not None:
                    nuisance_fg = fg_patch_fraction[i].detach().cpu()[spur_grid]
                    if nuisance_fg.numel() > 0:
                        seg_bg_ratio = float(
                            (nuisance_fg < foreground_patch_threshold)
                            .float().mean().item()
                        )
                        seg_fg_ratio = 1.0 - seg_bg_ratio
                        mean_fg_fraction = float(nuisance_fg.mean().item())
                    else:
                        seg_bg_ratio = 0.0
                        seg_fg_ratio = 0.0
                        mean_fg_fraction = 0.0

                priority_up = self._vis_upsample_map(
                    priority_grid.float(), (image_h, image_w), "nearest"
                ).numpy()
                spur_up = self._vis_upsample_map(
                    spur_grid.float(), (image_h, image_w), "nearest"
                ).numpy()
                cam_up = self._vis_upsample_map(
                    cam_grid, (image_h, image_w), "bilinear"
                ).numpy()
                cluster_up = self._vis_upsample_map(
                    cluster_grid.float(), (image_h, image_w), "nearest"
                ).numpy()

                ncols = 6 if fg_masks_cpu is not None else 5
                fig, axes = plt.subplots(
                    1, ncols, figsize=(5.0 * ncols, 5.2), squeeze=False
                )
                axes = axes[0]

                axes[0].imshow(img_np)
                axes[0].set_title("Original", fontsize=13)
                axes[0].axis("off")

                cluster_plot = axes[1].imshow(
                    cluster_up, cmap="tab20", interpolation="nearest"
                )
                axes[1].set_title("Cluster IDs", fontsize=13)
                axes[1].axis("off")
                fig.colorbar(cluster_plot, ax=axes[1], fraction=0.046, pad=0.04)

                axes[2].imshow(img_np)
                axes[2].imshow(
                    priority_up,
                    cmap="Greens",
                    alpha=0.36,
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                )
                self._vis_draw_patch_boxes(
                    axes[2], priority_grid.numpy(), image_h, image_w,
                    edgecolor="lime", linewidth=1.8,
                )
                axes[2].set_title(
                    "Priority Relevant (Top-{})\nmean cos(z,w_y)={:.3f}".format(
                        self.rel_num, mean_priority_sim
                    ),
                    fontsize=12,
                )
                axes[2].axis("off")

                axes[3].imshow(img_np)
                axes[3].imshow(
                    spur_up,
                    cmap="Reds",
                    alpha=0.40,
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                )
                self._vis_draw_patch_boxes(
                    axes[3], spur_grid.numpy(), image_h, image_w,
                    edgecolor="white", linewidth=1.6,
                )
                axes[3].set_title(
                    "Final Nuisance\nraw={} - protected={} = {} | ratio={:.3f}".format(
                        num_raw, num_protected, num_nuisance, nuisance_patch_ratio
                    ),
                    fontsize=12,
                )
                axes[3].axis("off")

                axes[4].imshow(img_np)
                axes[4].imshow(
                    cam_up, cmap="jet", alpha=0.42, vmin=0.0, vmax=1.0
                )
                self._vis_draw_patch_boxes(
                    axes[4], spur_grid.numpy(), image_h, image_w,
                    edgecolor="white", linewidth=1.6,
                )
                axes[4].set_title(
                    "CAM + Final Nuisance\nlow={:.3f}, high={:.3f}, mean={:.3f}".format(
                        low_cam_ratio, high_cam_ratio, mean_cam
                    ),
                    fontsize=12,
                )
                axes[4].axis("off")

                if fg_masks_cpu is not None:
                    fg_up = F.interpolate(
                        fg_masks_cpu[i][None, None].float(),
                        size=(image_h, image_w),
                        mode="nearest",
                    )[0, 0].numpy()
                    axes[5].imshow(img_np)
                    axes[5].imshow(
                        fg_up, cmap="Greens", alpha=0.30, vmin=0.0, vmax=1.0
                    )
                    self._vis_draw_patch_boxes(
                        axes[5], spur_grid.numpy(), image_h, image_w,
                        edgecolor="white", linewidth=1.6,
                    )
                    axes[5].set_title(
                        "Segmentation + Final Nuisance\nsegBG={:.3f}, segFG={:.3f}".format(
                            seg_bg_ratio, seg_fg_ratio
                        ),
                        fontsize=12,
                    )
                    axes[5].axis("off")

                pred_id = int(logits[i].argmax().item())
                gt_id = int(labels[i].item())
                fig.suptitle(
                    "idx={} | gt={} | pred={} | rel_num={} | raw_nui={} | protected={} | final_nui={}".format(
                        i, gt_id, pred_id, self.rel_num,
                        num_raw, num_protected, num_nuisance,
                    ),
                    fontsize=14,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.95])

                figure_path = os.path.join(
                    save_dir, "{}_{:03d}.png".format(file_prefix, i)
                )
                fig.savefig(
                    figure_path,
                    dpi=600,
                    bbox_inches="tight",
                    pad_inches=0.04,
                )
                plt.close(fig)

                patch_paths: List[str] = []
                patch_dir = None
                if save_patch_crops:
                    patch_dir = os.path.join(
                        save_dir, "{}_{:03d}_final_nuisance_patches".format(file_prefix, i)
                    )
                    Path(patch_dir).mkdir(parents=True, exist_ok=True)
                    pil_img = Image.fromarray((img_np * 255.0).astype(np.uint8))
                    nuisance_indices = (
                        final_spurious_mask[i]
                        .nonzero(as_tuple=False)
                        .squeeze(1)
                        .detach().cpu().tolist()
                    )[: int(max_patches_per_image)]

                    for order, patch_index in enumerate(nuisance_indices):
                        crop = self._vis_crop_patch(
                            pil_img=pil_img,
                            patch_index=int(patch_index),
                            hf=hf,
                            wf=wf,
                        )
                        cid = int(cluster_ids[i, patch_index].item())
                        rr = int(patch_index) // wf
                        cc = int(patch_index) % wf
                        crop_path = os.path.join(
                            patch_dir,
                            "patch_{:02d}_cluster_{}_r{}_c{}.png".format(
                                order, cid, rr, cc
                            ),
                        )
                        crop.save(crop_path, format="PNG", optimize=True)
                        patch_paths.append(crop_path)

                summary = {
                    "index": i,
                    "gt": gt_id,
                    "pred": pred_id,
                    "rel_num": int(self.rel_num),
                    "num_priority_relevant": num_priority,
                    "num_raw_nuisance_regions": num_raw,
                    "num_protected_from_nuisance": num_protected,
                    "num_nuisance_regions": num_nuisance,
                    "num_relevant_regions": int(relevant_mask[i].sum().item()),
                    "nuisance_patch_ratio": float(nuisance_patch_ratio),
                    "mean_priority_gt_similarity": mean_priority_sim,
                    "nuisance_low_cam_ratio": low_cam_ratio,
                    "nuisance_high_cam_ratio": high_cam_ratio,
                    "nuisance_mean_cam": mean_cam,
                    "figure_path": figure_path,
                    "patch_dir": patch_dir,
                    "patch_paths": patch_paths,
                }
                if seg_bg_ratio is not None:
                    summary.update({
                        "nuisance_seg_background_ratio": seg_bg_ratio,
                        "nuisance_seg_foreground_ratio": seg_fg_ratio,
                        "nuisance_mean_foreground_fraction": mean_fg_fraction,
                    })
                summaries.append(summary)

            return summaries
        finally:
            if was_training:
                base.train()
            else:
                base.eval()

    @torch.no_grad()
    def visualize_nuisance_cluster_exemplars(
        self,
        data_loader,
        device,
        student_model: Optional[nn.Module] = None,
        save_dir: str = "./nuisance_cluster_exemplars",
        file_prefix: str = "cluster",
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        target_clusters: Optional[List[int]] = None,
        max_patches_per_cluster: int = 64,
        max_batches: Optional[int] = 100,
        columns: int = 8,
    ) -> Dict[int, str]:
        """
        Save exemplar montages using FINAL nuisance patches only.

        Patches protected by Top-rel_num similarity to w_y are explicitly
        excluded even if they belong to a selected nuisance cluster.

        student_model:
            current model used to select priority-relevant patches. If None,
            the frozen Stage-1 assignment model is used for both priority
            selection and cluster assignment.
        """
        if self._assignment_model is None:
            raise RuntimeError(
                "No frozen assignment model. Run discover(..., "
                "make_assignment_snapshot=True) first."
            )
        if target_clusters is None:
            target_clusters = list(self.top_spurious_clusters)
        target_clusters = [int(c) for c in target_clusters]
        if len(target_clusters) == 0:
            raise RuntimeError("No nuisance clusters are available.")

        if student_model is None:
            student_model = self._assignment_model
        base = _unwrap_model(student_model)
        student_was_training = base.training
        base.eval()

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        buckets = {c: [] for c in target_clusters}

        try:
            for batch_idx, batch in enumerate(data_loader):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise TypeError("data_loader must return (inputs, labels, ...).")

                inputs = batch[0]
                labels = batch[1]
                if isinstance(inputs, (tuple, list)):
                    inputs = inputs[0]
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()

                _, features = _capture_layer4_and_forward(student_model, inputs)
                student_regions = self._regions_from_features(features)
                classifier = _get_classifier(student_model)
                priority_mask, _ = self._priority_relevant_mask(
                    student_regions, labels, classifier
                )

                cluster_ids, raw_spurious_mask, _ = self._stage2_region_partition(inputs)
                cluster_ids = cluster_ids.to(device)
                raw_spurious_mask = raw_spurious_mask.to(device)
                final_spurious_mask, _, _ = self._apply_priority_protection(
                    raw_spurious_mask, priority_mask
                )

                b, _, hf, wf = features.shape
                if int(cluster_ids.shape[1]) != hf * wf:
                    raise ValueError(
                        "cluster_ids={} but student grid={}x{}.".format(
                            cluster_ids.shape[1], hf, wf
                        )
                    )

                for i in range(b):
                    img_np = self._vis_to_numpy_image(inputs[i], mean=mean, std=std)
                    pil_img = Image.fromarray((img_np * 255.0).astype(np.uint8))
                    ids_i = cluster_ids[i]
                    final_i = final_spurious_mask[i]

                    for cid in target_clusters:
                        if len(buckets[cid]) >= int(max_patches_per_cluster):
                            continue
                        patch_indices = (
                            (ids_i.eq(cid) & final_i)
                            .nonzero(as_tuple=False)
                            .squeeze(1)
                            .detach().cpu().tolist()
                        )
                        for patch_index in patch_indices:
                            if len(buckets[cid]) >= int(max_patches_per_cluster):
                                break
                            crop = self._vis_crop_patch(
                                pil_img=pil_img,
                                patch_index=int(patch_index),
                                hf=hf,
                                wf=wf,
                            )
                            buckets[cid].append(crop.copy())

                if all(
                    len(v) >= int(max_patches_per_cluster)
                    for v in buckets.values()
                ):
                    break
        finally:
            if student_was_training:
                base.train()
            else:
                base.eval()

        output_paths: Dict[int, str] = {}
        for cid, crops in buckets.items():
            if len(crops) == 0:
                continue
            n = len(crops)
            ncols = min(int(columns), n)
            nrows = int(math.ceil(n / ncols))
            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(2.3 * ncols, 2.3 * nrows),
                squeeze=False,
            )
            for ax in axes.ravel():
                ax.axis("off")
            for j, crop in enumerate(crops):
                rr = j // ncols
                cc = j % ncols
                axes[rr, cc].imshow(crop)
                axes[rr, cc].axis("off")
            fig.suptitle(
                "Final nuisance cluster {} exemplars after Top-{} protection (n={})".format(
                    cid, self.rel_num, n
                ),
                fontsize=15,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            path = os.path.join(save_dir, "{}_{}.png".format(file_prefix, cid))
            fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.04)
            plt.close(fig)
            output_paths[cid] = path
        return output_paths

    @torch.no_grad()
    def evaluate_nuisance_cam_overlap(
        self,
        student_model: nn.Module,
        data_loader,
        device,
        cam_target: str = "gt",
        cam_threshold: float = 0.35,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Dataset-level CAM statistics for FINAL nuisance patches only."""
        if cam_target not in ("pred", "gt"):
            raise ValueError("cam_target must be 'pred' or 'gt'.")

        base = _unwrap_model(student_model)
        was_training = base.training
        base.eval()
        total_raw = 0
        total_protected = 0
        total_nuisance = 0
        total_low_cam = 0
        total_high_cam = 0
        sum_cam = 0.0

        try:
            for batch_idx, batch in enumerate(data_loader):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise TypeError("data_loader must return (inputs, labels, ...).")

                inputs = batch[0]
                labels = batch[1]
                if isinstance(inputs, (tuple, list)):
                    inputs = inputs[0]
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()

                logits, features = _capture_layer4_and_forward(student_model, inputs)
                student_regions = self._regions_from_features(features)
                classifier = _get_classifier(student_model)
                priority_mask, _ = self._priority_relevant_mask(
                    student_regions, labels, classifier
                )
                cluster_ids, raw_spurious_mask, _ = self._stage2_region_partition(inputs)
                raw_spurious_mask = raw_spurious_mask.to(device)
                final_spurious_mask, _, protected_mask = self._apply_priority_protection(
                    raw_spurious_mask, priority_mask
                )

                target_ids = labels if cam_target == "gt" else logits.argmax(dim=1)
                cams = self._compute_standard_cam(features, classifier, target_ids)
                b, _, hf, wf = features.shape
                if int(cluster_ids.shape[1]) != hf * wf:
                    raise ValueError(
                        "Student feature grid and assignment region count do not match."
                    )
                spur_grid = final_spurious_mask.view(b, hf, wf)
                nuisance_cam = cams[spur_grid]

                total_raw += int(raw_spurious_mask.sum().item())
                total_protected += int(protected_mask.sum().item())
                if nuisance_cam.numel() == 0:
                    continue
                total_nuisance += int(nuisance_cam.numel())
                total_low_cam += int((nuisance_cam < cam_threshold).sum().item())
                total_high_cam += int((nuisance_cam >= cam_threshold).sum().item())
                sum_cam += float(nuisance_cam.sum().item())
        finally:
            if was_training:
                base.train()
            else:
                base.eval()

        if total_nuisance == 0:
            return {
                "num_raw_nuisance_patches": float(total_raw),
                "num_protected_patches": float(total_protected),
                "num_final_nuisance_patches": 0.0,
                "low_cam_ratio": 0.0,
                "high_cam_ratio": 0.0,
                "mean_nuisance_cam": 0.0,
            }
        return {
            "num_raw_nuisance_patches": float(total_raw),
            "num_protected_patches": float(total_protected),
            "num_final_nuisance_patches": float(total_nuisance),
            "low_cam_ratio": total_low_cam / total_nuisance,
            "high_cam_ratio": total_high_cam / total_nuisance,
            "mean_nuisance_cam": sum_cam / total_nuisance,
        }

    @torch.no_grad()
    def evaluate_nuisance_segmentation_overlap(
        self,
        data_loader,
        device,
        student_model: Optional[nn.Module] = None,
        mask_index: int = 2,
        foreground_patch_threshold: float = 0.25,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Segmentation overlap for FINAL nuisance patches after Top-rel_num protection."""
        if not (0.0 <= foreground_patch_threshold <= 1.0):
            raise ValueError("foreground_patch_threshold must be in [0,1].")
        if student_model is None:
            if self._assignment_model is None:
                raise RuntimeError("No model available for priority-region selection.")
            student_model = self._assignment_model

        base = _unwrap_model(student_model)
        was_training = base.training
        base.eval()
        total_raw = 0
        total_protected = 0
        total_nuisance = 0
        total_bg = 0
        total_fg = 0
        total_fg_fraction = 0.0

        try:
            for batch_idx, batch in enumerate(data_loader):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
                if not isinstance(batch, (tuple, list)) or len(batch) <= max(mask_index, 1):
                    raise TypeError(
                        "data_loader must return (inputs, labels, ..., mask, ...)."
                    )
                inputs = batch[0]
                labels = batch[1]
                masks = batch[mask_index]
                if isinstance(inputs, (tuple, list)):
                    inputs = inputs[0]
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()
                masks = masks.to(device, non_blocking=True).float()
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)
                elif masks.ndim == 4 and masks.shape[1] != 1:
                    masks = masks[:, :1]
                if masks.ndim != 4:
                    raise ValueError("foreground masks must be [B,H,W] or [B,1,H,W].")

                _, features = _capture_layer4_and_forward(student_model, inputs)
                student_regions = self._regions_from_features(features)
                classifier = _get_classifier(student_model)
                priority_mask, _ = self._priority_relevant_mask(
                    student_regions, labels, classifier
                )
                cluster_ids, raw_spurious_mask, _ = self._stage2_region_partition(inputs)
                raw_spurious_mask = raw_spurious_mask.to(device)
                final_spurious_mask, _, protected_mask = self._apply_priority_protection(
                    raw_spurious_mask, priority_mask
                )

                b, _, hf, wf = features.shape
                if int(cluster_ids.shape[1]) != hf * wf:
                    raise ValueError("Assignment region count does not match feature grid.")
                fg_fraction = F.adaptive_avg_pool2d(masks, (hf, wf))[:, 0]
                spur_grid = final_spurious_mask.view(b, hf, wf)
                nuisance_fg = fg_fraction[spur_grid]

                total_raw += int(raw_spurious_mask.sum().item())
                total_protected += int(protected_mask.sum().item())
                if nuisance_fg.numel() == 0:
                    continue
                total_nuisance += int(nuisance_fg.numel())
                total_bg += int(
                    (nuisance_fg < foreground_patch_threshold).sum().item()
                )
                total_fg += int(
                    (nuisance_fg >= foreground_patch_threshold).sum().item()
                )
                total_fg_fraction += float(nuisance_fg.sum().item())
        finally:
            if was_training:
                base.train()
            else:
                base.eval()

        if total_nuisance == 0:
            return {
                "num_raw_nuisance_patches": float(total_raw),
                "num_protected_patches": float(total_protected),
                "num_final_nuisance_patches": 0.0,
                "seg_background_ratio": 0.0,
                "seg_foreground_ratio": 0.0,
                "mean_foreground_fraction": 0.0,
            }
        return {
            "num_raw_nuisance_patches": float(total_raw),
            "num_protected_patches": float(total_protected),
            "num_final_nuisance_patches": float(total_nuisance),
            "seg_background_ratio": total_bg / total_nuisance,
            "seg_foreground_ratio": total_fg / total_nuisance,
            "mean_foreground_fraction": total_fg_fraction / total_nuisance,
        }

    # ------------------------------------------------------------
    # Stage 2 public forward
    # ------------------------------------------------------------
    def forward(
        self,
        student_model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> RaVLOutput:
        """
        Stage-2 forward: GLOBAL nuisance cluster + Top-rel_num protection
        + wrong-class positive evidence suppression.

        Order:
            1) current student regions -> protect Top-rel_num regions most similar to w_y
            2) frozen Stage-1 encoder -> GLOBAL nuisance-cluster assignment
            3) remove protected Top-rel_num regions from the nuisance mask
            4) for each remaining nuisance patch, suppress only its strongest
               POSITIVE response to a WRONG class direction

        Relevant patches receive NO auxiliary attraction loss.  GT-class evidence
        inside nuisance patches is also NOT directly penalized by this objective.
        """
        if self.top_spurious_cluster is None:
            raise RuntimeError(
                "Run ravl.discover(model, reference_loader) before Stage 2."
            )

        logits, student_features = _capture_layer4_and_forward(
            student_model,
            inputs,
        )
        if logits.shape[1] != self.num_classes:
            raise ValueError(
                "Expected {} classes, got {}.".format(
                    self.num_classes,
                    logits.shape[1],
                )
            )

        student_regions = self._regions_from_features(student_features)
        classifier = _get_classifier(student_model)

        # Highest-priority task-relevant regions are selected FIRST.
        priority_relevant_mask, _ = self._priority_relevant_mask(
            student_regions=student_regions,
            labels=labels,
            classifier=classifier,
        )

        # Then obtain raw nuisance-cluster membership from the frozen Stage-1 space.
        cluster_ids, raw_spurious_mask, _ = self._stage2_region_partition(inputs)
        raw_spurious_mask = raw_spurious_mask.to(priority_relevant_mask.device)
        cluster_ids = cluster_ids.to(priority_relevant_mask.device)

        # rel_num regions can never be nuisance.
        spurious_mask, relevant_mask, protected_from_nuisance = (
            self._apply_priority_protection(
                raw_spurious_mask=raw_spurious_mask,
                priority_relevant_mask=priority_relevant_mask,
            )
        )

        # loss_region, loss_R, loss_A, valid_images = self._region_aware_loss(
        #     student_regions=student_regions,
        #     labels=labels,
        #     spurious_mask=spurious_mask,
        #     relevant_mask=relevant_mask,
        #     classifier=classifier,
        # )
        
        loss_region = self._nuisance_wrong_class_suppression_loss(
            student_regions=student_regions,
            labels=labels,
            spurious_mask=spurious_mask,
            classifier=classifier,
        )

        zero = loss_region.new_zeros(())
        valid_images = spurious_mask.any(dim=1)

        return RaVLOutput(
            loss_region=loss_region,
            loss_R=zero,
            loss_A=zero,
            logits=logits,
            raw_spurious_mask=raw_spurious_mask.detach(),
            spurious_mask=spurious_mask.detach(),
            priority_relevant_mask=priority_relevant_mask.detach(),
            relevant_mask=relevant_mask.detach(),
            cluster_ids=cluster_ids.detach(),
            num_raw_spurious_regions=int(raw_spurious_mask.sum().item()),
            num_protected_regions=int(protected_from_nuisance.sum().item()),
            num_spurious_regions=int(spurious_mask.sum().item()),
            num_relevant_regions=int(relevant_mask.sum().item()),
            num_valid_images=int(valid_images.sum().item()),
        )

    __call__ = forward

    def combine_with_classification_loss(
        self,
        classification_loss: torch.Tensor,
        output: RaVLOutput,
        lambda_cl: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Keep the original public helper interface.  In this version
        output.loss_region is the wrong-class positive suppression loss.
        The numerical mixing rule is intentionally left unchanged for backward
        compatibility with the user's existing training code.
        """
        if lambda_cl is None:
            lambda_cl = self.lambda_cl
        return classification_loss + float(lambda_cl)*output.loss_region

    # ------------------------------------------------------------
    # Persistence for the discovered clusters
    # ------------------------------------------------------------
    def save_discovery(self, path: str) -> None:
        if self.medoids_raw is None or self.top_spurious_cluster is None:
            raise RuntimeError("Nothing to save; run discover first.")

        torch.save(
            {
                "num_classes": self.num_classes,
                "region_grid": self.region_grid,
                "temperature": self.temperature,
                "lambda_cl": self.lambda_cl,
                "influence_threshold": self.influence_threshold,
                "medoids_raw": self.medoids_raw,
                "medoids_norm": self.medoids_norm,
                "top_spurious_cluster": self.top_spurious_cluster,
            "top_spurious_clusters": self.top_spurious_clusters,
            "num_spurious_clusters": self.num_spurious_clusters,
                "ranked_clusters": self.ranked_clusters,
                "discovery_result": self.discovery_result,
            },
            path,
        )

    def load_discovery(
        self,
        path: str,
        model_for_assignment: Optional[nn.Module] = None,
        device=None,
    ) -> None:
        state = torch.load(path, map_location="cpu")

        if int(state["num_classes"]) != self.num_classes:
            raise ValueError(
                "Saved num_classes={} but module num_classes={}.".format(
                    state["num_classes"],
                    self.num_classes,
                )
            )

        if int(state["region_grid"]) != self.region_grid:
            raise ValueError(
                "Saved region_grid={} but module region_grid={}.".format(
                    state["region_grid"],
                    self.region_grid,
                )
            )

        self.medoids_raw = state["medoids_raw"].float()
        self.medoids_norm = state["medoids_norm"].float()
        self.top_spurious_cluster = int(
            state["top_spurious_cluster"]
        )
        self.top_spurious_clusters = state.get(
            "top_spurious_clusters",
            [self.top_spurious_cluster]
        )
        self.rel_num = int(state.get("rel_num", self.rel_num))
        self.ranked_clusters = list(state["ranked_clusters"])
        self.discovery_result = state.get("discovery_result", None)

        if model_for_assignment is not None:
            self.prepare_stage2_assignment_model(
                model_for_assignment,
                device=device,
            )


# ================================================================
# Tiny smoke test
# ================================================================
if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset

    class TinyResNet(nn.Module):
        def __init__(self, num_classes=3):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7)),
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
            )
            self.fc = nn.Linear(32, num_classes)

        def forward(self, x):
            x = self.stem(x)
            x = self.layer4(x)
            z = x.mean(dim=(2, 3))
            return self.fc(z)

    torch.manual_seed(7)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = TinyResNet(num_classes=3).to(device)

    ravl = RaVLResNet(
        num_classes=3,
        region_grid=2,
        temperature=0.07,
        rel_num=4,
        influence_threshold=0.0,  # smoke-test only
        k_min_factor=2,
        k_max_factor=2,
        max_cluster_regions=200,
        silhouette_sample_size=200,
        random_seed=7,
    )

    # Synthetic reference set.
    ref_x = torch.randn(60, 3, 32, 32)
    ref_y = torch.randint(0, 3, (60,))
    ref_loader = DataLoader(
        TensorDataset(ref_x, ref_y),
        batch_size=12,
        shuffle=False,
    )

    # Random model may still have G=0. We test the full clustering path but
    # permit H=0 for this synthetic smoke test.
    try:
        discovery = ravl.discover(
            model=model,
            reference_loader=ref_loader,
            device=device,
            verbose=False,
        )
    except RuntimeError:
        # For a random model, if all G/H degenerates, manually select the first
        # discovered cluster is not possible if discovery did not commit state.
        # Re-run is not necessary for syntax/gradient smoke testing; set a small
        # fixed bank from reference features.
        model.eval()
        with torch.no_grad():
            _, feat = _capture_layer4_and_forward(
                model,
                ref_x[:12].to(device),
            )
            regs = ravl._regions_from_features(feat)
            flat = regs.reshape(-1, regs.shape[-1])
            medoid_ids, _ = ravl._fit_kmedoids_cosine(
                flat,
                k=3,
                seed=7,
            )
            med = flat.index_select(0, medoid_ids)
            ravl.medoids_raw = med.detach().cpu()
            ravl.medoids_norm = F.normalize(
                med,
                dim=1,
            ).detach().cpu()
            ravl.top_spurious_cluster = 0
            ravl.top_spurious_clusters = [0]
            ravl.ranked_clusters = [0]
            ravl.prepare_stage2_assignment_model(
                model,
                device=device,
            )

    model.train()

    x = torch.randn(8, 3, 32, 32, device=device)
    y = torch.randint(0, 3, (8,), device=device)

    out = ravl(
        student_model=model,
        inputs=x,
        labels=y,
    )

    loss_cls = F.cross_entropy(out.logits, y)
    loss = ravl.combine_with_classification_loss(
        loss_cls,
        out,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print("Global nuisance + Top-rel_num + wrong-class positive suppression smoke test passed.")
    print(out.statistics())





