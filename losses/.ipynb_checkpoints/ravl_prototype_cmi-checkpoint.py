# from __future__ import annotations

# from dataclasses import dataclass
# import copy
# import math
# import random
# from typing import Dict, List, Optional, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# #### RaLV.
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
#             "top_spurious_clusters": self.top_spurious_clusters,
#             "num_spurious_clusters": self.num_spurious_clusters,
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
#     spurious_mask: torch.Tensor
#     relevant_mask: torch.Tensor
#     cluster_ids: torch.Tensor
#     num_spurious_regions: int
#     num_relevant_regions: int
#     num_valid_images: int

#     def statistics(self) -> Dict[str, float]:
#         return {
#             "loss_region": float(self.loss_region.detach().item()),
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
#         2) local candidate regions
#         3) K-Medoids with cosine distance
#         4) choose K by Silhouette score over [2|Y|, 5|Y|]
#         5) Cluster Influence Score H_c
#         6) prune H_c < 0.25
#         7) Cluster Performance Gap G_c
#         8) select the top-ranked spurious cluster

#       Stage 2:
#         1) keep the Stage-1 clustering model fixed
#         2) split regions into R_i^s and R_i^r
#         3) use the original RaVL L_R and L_A region-aware objectives
#         4) temperature tau = 0.07
#         5) recommended final loss:
#               0.8 * L_cls + 0.2 * L_RaVL

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
#         Original RaVL Stage-1 logic adapted to a supervised ResNet.

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
#         # Clustering pool.
#         # --------------------------------------------------------
#         total_regions = int(regions_flat.shape[0])

#         if (
#             self.max_cluster_regions is not None
#             and total_regions > int(self.max_cluster_regions)
#         ):
#             gen = torch.Generator()
#             gen.manual_seed(self.random_seed)
#             cluster_ids = torch.randperm(
#                 total_regions,
#                 generator=gen,
#             )[: int(self.max_cluster_regions)]
#             cluster_pool_cpu = regions_flat.index_select(0, cluster_ids)
#         else:
#             cluster_pool_cpu = regions_flat

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
#             print(
#                 "images={} | regions/image={} | total_regions={} | "
#                 "cluster_pool={}".format(
#                     n_img,
#                     n_region,
#                     total_regions,
#                     n_pool,
#                 )
#             )
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
#         # H_c and G_c, exactly following the paper's definitions.
#         # --------------------------------------------------------
#         influence_scores = {}
#         performance_gaps = {}
#         per_class_gaps = {}
#         per_class_influence = {}

#         for cluster_id in range(best_k):
#             present = assignments_img.eq(cluster_id).any(dim=1)

#             gap_by_class = {}
#             influence_by_class = {}

#             for y in range(self.num_classes):
#                 class_mask = labels_img.eq(y)
#                 in_mask = class_mask & present
#                 out_mask = class_mask & (~present)

#                 n_in = int(in_mask.sum().item())
#                 n_out = int(out_mask.sum().item())

#                 # p_in and p_out both need to exist.
#                 if n_in == 0 or n_out == 0:
#                     continue

#                 p_in = float(correct_img[in_mask].float().mean().item())
#                 p_out = float(correct_img[out_mask].float().mean().item())

#                 weight = (
#                     2.0
#                     * min(n_in, n_out)
#                     / float(n_in + n_out)
#                 )

#                 gap_y = weight * (p_in - p_out)
#                 gap_by_class[y] = float(gap_y)

#                 # I_c^err:
#                 # in-cluster images with class y, mispredicted,
#                 # and p_in^y < p_out^y.
#                 h_y = 0.0

#                 if p_in < p_out:
#                     err_mask = in_mask & (~correct_img)
#                     err_ids = err_mask.nonzero(
#                         as_tuple=False
#                     ).squeeze(1)

#                     if err_ids.numel() > 0:
#                         hit = 0

#                         for img_id_tensor in err_ids:
#                             img_id = int(img_id_tensor.item())
#                             y_hat = int(preds_img[img_id].item())

#                             # r_i^max:
#                             # region with the highest score for image prediction.
#                             local_region_id = int(
#                                 region_probs_img[
#                                     img_id,
#                                     :,
#                                     y_hat,
#                                 ].argmax().item()
#                             )

#                             if (
#                                 int(
#                                     assignments_img[
#                                         img_id,
#                                         local_region_id,
#                                     ].item()
#                                 )
#                                 == cluster_id
#                             ):
#                                 hit += 1

#                         h_y = hit / float(err_ids.numel())

#                 influence_by_class[y] = float(h_y)

#             H_c = (
#                 max(influence_by_class.values())
#                 if len(influence_by_class) > 0
#                 else 0.0
#             )

#             G_c = sum(
#                 abs(v) for v in gap_by_class.values()
#             )

#             influence_scores[cluster_id] = float(H_c)
#             performance_gaps[cluster_id] = float(G_c)
#             per_class_gaps[cluster_id] = gap_by_class
#             per_class_influence[cluster_id] = influence_by_class

#         candidate_clusters = [
#             c
#             for c in range(best_k)
#             if influence_scores.get(c, 0.0)
#             >= self.influence_threshold
#         ]

#         ranked_clusters = sorted(
#             candidate_clusters,
#             key=lambda c: performance_gaps.get(c, 0.0),
#             reverse=True,
#         )

#         if len(ranked_clusters) == 0:
#             raise RuntimeError(
#                 "RaVL found no cluster with H_c >= {:.3f}. "
#                 "Try inspecting the reference split or, only for diagnosis, "
#                 "lower influence_threshold.".format(
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
#                 "H threshold={:.3f}".format(
#                     self.influence_threshold
#                 )
#             )

#             for rank, cluster_id in enumerate(ranked_clusters[:10], 1):
#                 print(
#                     "Rank {:2d} | cluster {:3d} | H_c={:.4f} | G_c={:.4f}".format(
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
#     # Original RaVL L_R + L_A, adapted to FC class directions
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
#         student_regions: [B,R,D]

#         Original RaVL:
#             region embedding f(r)
#             text embedding   g(y)

#         ResNet migration:
#             f(r) = normalized local ResNet feature
#             g(y) = normalized frozen FC class weight w_y

#         The FC weights are detached in this loss, matching RaVL's locked text
#         encoder. The user's ordinary classification loss may still be computed
#         from model(inputs) outside this class.
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
#         labels = labels.to(device=device, dtype=torch.long)
#         spurious_mask = spurious_mask.to(device=device)
#         relevant_mask = relevant_mask.to(device=device)

#         # CLIP-style embedding similarity.
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
#         sim = torch.einsum(
#             "brd,cd->brc",
#             region_n,
#             class_n,
#         )

#         # sigma_m(R_i^r, y):
#         # max over NON-SPURIOUS regions.
#         masked_rel_sim = sim.masked_fill(
#             ~relevant_mask.unsqueeze(2),
#             -1e9,
#         )

#         rel_max_sim = masked_rel_sim.max(dim=1).values  # [B,C]
#         valid_images = relevant_mask.any(dim=1)

#         # All spurious regions in the batch: R_B^s
#         spur_sim = sim[spurious_mask]  # [N_spur,C]

#         tau = self.temperature

#         loss_R_terms = []
#         loss_A_terms = []

#         # Eq. 6: P(R_B^s) uses max over class labels occurring in the batch.
#         # Duplicated labels do not change a max.
#         if spur_sim.numel() > 0:
#             spur_max_over_batch_labels = spur_sim.index_select(
#                 1,
#                 labels,
#             ).max(dim=1).values / tau
#         else:
#             spur_max_over_batch_labels = None

#         for i in range(b):
#             if not bool(valid_images[i].item()):
#                 continue

#             y_i = int(labels[i].item())

#             # log sigma_m(R_i^r, y_i)
#             positive_log = rel_max_sim[i, y_i] / tau

#             # ----------------------------------------------------
#             # L_R^i, paper Eq. (3)/(5)
#             #
#             # denominator:
#             # sum_{y_j in B} sigma_m(R_i^r, y_j)
#             # + P(R_B^s)
#             # ----------------------------------------------------
#             label_terms_R = rel_max_sim[i].index_select(
#                 0,
#                 labels,
#             ) / tau

#             if spur_max_over_batch_labels is not None:
#                 denom_terms_R = torch.cat(
#                     [
#                         label_terms_R,
#                         spur_max_over_batch_labels,
#                     ],
#                     dim=0,
#                 )
#             else:
#                 denom_terms_R = label_terms_R

#             loss_R_i = (
#                 -positive_log
#                 + torch.logsumexp(denom_terms_R, dim=0)
#             )

#             # ----------------------------------------------------
#             # L_A^i, paper Eq. (4)/(9)
#             #
#             # denominator:
#             # positive
#             # + other-class images' relevant regions
#             # + all spurious regions
#             # ----------------------------------------------------
#             denom_terms_A = [positive_log.view(1)]

#             other_image_mask = (
#                 labels.ne(y_i)
#                 & valid_images
#             )

#             if bool(other_image_mask.any().item()):
#                 other_rel = (
#                     rel_max_sim[
#                         other_image_mask,
#                         y_i,
#                     ]
#                     / tau
#                 )
#                 denom_terms_A.append(other_rel)

#             if spur_sim.numel() > 0:
#                 spur_to_y = spur_sim[:, y_i] / tau
#                 denom_terms_A.append(spur_to_y)

#             denom_A = torch.logsumexp(
#                 torch.cat(denom_terms_A, dim=0),
#                 dim=0,
#             )

#             loss_A_i = -positive_log + denom_A

#             loss_R_terms.append(loss_R_i)
#             loss_A_terms.append(loss_A_i)

#         if len(loss_R_terms) == 0:
#             zero = student_regions.sum() * 0.0
#             return (
#                 zero,
#                 zero,
#                 zero,
#                 valid_images,
#             )

#         # Mean reduction for stable minibatch optimization.
#         # The paper writes a sum over i; multiplying this mean by B recovers
#         # the literal sum if desired.
#         loss_R = torch.stack(loss_R_terms).mean()
#         loss_A = torch.stack(loss_A_terms).mean()
#         loss_region = loss_R  + loss_A

#         return (
#             loss_region,
#             loss_R,
#             loss_A,
#             valid_images,
#         )

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
#         Compute RaVL Stage-2 region-aware loss.

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

#         loss_region, loss_R, loss_A, valid_images = (
#             self._region_aware_loss(
#                 student_regions=student_regions,
#                 labels=labels,
#                 spurious_mask=spurious_mask,
#                 relevant_mask=relevant_mask,
#                 classifier=classifier,
#             )
#         )

#         return RaVLOutput(
#             loss_region=loss_region,
#             loss_R=loss_R,
#             loss_A=loss_A,
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

#     print("RaVL-ResNet smoke test passed.")
#     print(out.statistics())





from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import random
from typing import Dict, List, Optional, Tuple

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
    spurious_mask: torch.Tensor
    relevant_mask: torch.Tensor
    cluster_ids: torch.Tensor
    num_spurious_regions: int
    num_relevant_regions: int
    num_valid_images: int

    def statistics(self) -> Dict[str, float]:
        return {
            "loss_region": float(self.loss_region.detach().item()),
            "loss_R": float(self.loss_R.detach().item()),
            "loss_A": float(self.loss_A.detach().item()),
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
        2) local candidate regions
        3) K-Medoids with cosine distance
        4) choose K by Silhouette score over [2|Y|, 5|Y|]
        5) class-conditional MI I(S_k; Y_hat | Y=c)
        6) harmful error-increase gating
        7) class-balanced conditional nuisance score N_k
        8) select the top-ranked nuisance cluster(s)

      Stage 2:
        1) keep the Stage-1 clustering model fixed
        2) split regions into R_i^s and R_i^r
        3) use the original RaVL L_R and L_A region-aware objectives
        4) temperature tau = 0.07
        5) recommended final loss:
              0.8 * L_cls + 0.2 * L_RaVL

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
        k_min_factor: int = 2,
        k_max_factor: int = 5,
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
        if not 0 <= lambda_cl <= 1:
            raise ValueError("lambda_cl must be in [0,1]")
        if influence_threshold < 0:
            raise ValueError("influence_threshold must be >= 0")
        if k_min_factor < 1 or k_max_factor < k_min_factor:
            raise ValueError("invalid cluster-count factors")

        self.num_classes = int(num_classes)
        self.region_grid = int(region_grid)
        self.temperature = float(temperature)
        self.lambda_cl = float(lambda_cl)
        self.influence_threshold = float(influence_threshold)
        self.num_spurious_clusters = int(num_spurious_clusters)
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
        Original RaVL Stage-1 logic adapted to a supervised ResNet.

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
        # Clustering pool.
        # --------------------------------------------------------
        total_regions = int(regions_flat.shape[0])

        if (
            self.max_cluster_regions is not None
            and total_regions > int(self.max_cluster_regions)
        ):
            gen = torch.Generator()
            gen.manual_seed(self.random_seed)
            cluster_ids = torch.randperm(
                total_regions,
                generator=gen,
            )[: int(self.max_cluster_regions)]
            cluster_pool_cpu = regions_flat.index_select(0, cluster_ids)
        else:
            cluster_pool_cpu = regions_flat

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
            print(
                "images={} | regions/image={} | total_regions={} | "
                "cluster_pool={}".format(
                    n_img,
                    n_region,
                    total_regions,
                    n_pool,
                )
            )
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

    # ------------------------------------------------------------
    # Region assignment during Stage 2
    # ------------------------------------------------------------
    @torch.no_grad()
    def _stage2_region_partition(
        self,
        inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        spurious_mask = torch.zeros_like(
            cluster_ids, dtype=torch.bool
        )

        for c in self.top_spurious_clusters:
            spurious_mask |= cluster_ids.eq(int(c))
        relevant_mask = ~spurious_mask

        return (
            cluster_ids,
            spurious_mask,
            relevant_mask,
        )

    # ------------------------------------------------------------
    # Original RaVL L_R + L_A, adapted to FC class directions
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
        student_regions: [B,R,D]

        Original RaVL:
            region embedding f(r)
            text embedding   g(y)

        ResNet migration:
            f(r) = normalized local ResNet feature
            g(y) = normalized frozen FC class weight w_y

        The FC weights are detached in this loss, matching RaVL's locked text
        encoder. The user's ordinary classification loss may still be computed
        from model(inputs) outside this class.
        """
        b, r, d = student_regions.shape

        if classifier.weight.shape != (self.num_classes, d):
            raise ValueError(
                "Classifier weight shape {} incompatible with regions [*,{},{}].".format(
                    tuple(classifier.weight.shape),
                    r,
                    d,
                )
            )

        device = student_regions.device
        labels = labels.to(device=device, dtype=torch.long)
        spurious_mask = spurious_mask.to(device=device)
        relevant_mask = relevant_mask.to(device=device)

        # CLIP-style embedding similarity.
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
        sim = torch.einsum(
            "brd,cd->brc",
            region_n,
            class_n,
        )

        # sigma_m(R_i^r, y):
        # max over NON-SPURIOUS regions.
        masked_rel_sim = sim.masked_fill(
            ~relevant_mask.unsqueeze(2),
            -1e9,
        )

        rel_max_sim = masked_rel_sim.max(dim=1).values  # [B,C]
        valid_images = relevant_mask.any(dim=1)

        # All spurious regions in the batch: R_B^s
        spur_sim = sim[spurious_mask]  # [N_spur,C]

        tau = self.temperature

        loss_R_terms = []
        loss_A_terms = []

        # Eq. 6: P(R_B^s) uses max over class labels occurring in the batch.
        # Duplicated labels do not change a max.
        if spur_sim.numel() > 0:
            spur_max_over_batch_labels = spur_sim.index_select(
                1,
                labels,
            ).max(dim=1).values / tau
        else:
            spur_max_over_batch_labels = None

        for i in range(b):
            if not bool(valid_images[i].item()):
                continue

            y_i = int(labels[i].item())

            # log sigma_m(R_i^r, y_i)
            positive_log = rel_max_sim[i, y_i] / tau

            # ----------------------------------------------------
            # L_R^i, paper Eq. (3)/(5)
            #
            # denominator:
            # sum_{y_j in B} sigma_m(R_i^r, y_j)
            # + P(R_B^s)
            # ----------------------------------------------------
            label_terms_R = rel_max_sim[i].index_select(
                0,
                labels,
            ) / tau

            if spur_max_over_batch_labels is not None:
                denom_terms_R = torch.cat(
                    [
                        label_terms_R,
                        spur_max_over_batch_labels,
                    ],
                    dim=0,
                )
            else:
                denom_terms_R = label_terms_R

            loss_R_i = (
                -positive_log
                + torch.logsumexp(denom_terms_R, dim=0)
            )

            # ----------------------------------------------------
            # L_A^i, paper Eq. (4)/(9)
            #
            # denominator:
            # positive
            # + other-class images' relevant regions
            # + all spurious regions
            # ----------------------------------------------------
            denom_terms_A = [positive_log.view(1)]

            other_image_mask = (
                labels.ne(y_i)
                & valid_images
            )

            if bool(other_image_mask.any().item()):
                other_rel = (
                    rel_max_sim[
                        other_image_mask,
                        y_i,
                    ]
                    / tau
                )
                denom_terms_A.append(other_rel)

            if spur_sim.numel() > 0:
                spur_to_y = spur_sim[:, y_i] / tau
                denom_terms_A.append(spur_to_y)

            denom_A = torch.logsumexp(
                torch.cat(denom_terms_A, dim=0),
                dim=0,
            )

            loss_A_i = -positive_log + denom_A

            loss_R_terms.append(loss_R_i)
            loss_A_terms.append(loss_A_i)

        if len(loss_R_terms) == 0:
            zero = student_regions.sum() * 0.0
            return (
                zero,
                zero,
                zero,
                valid_images,
            )

        # Mean reduction for stable minibatch optimization.
        # The paper writes a sum over i; multiplying this mean by B recovers
        # the literal sum if desired.
        loss_R = torch.stack(loss_R_terms).mean()
        loss_A = torch.stack(loss_A_terms).mean()
        loss_region = loss_R  + loss_A

        return (
            loss_region,
            loss_R,
            loss_A,
            valid_images,
        )

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
        Compute RaVL Stage-2 region-aware loss.

        This function DOES NOT update the model and DOES NOT include the user's
        ordinary classification loss.
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

        student_regions = self._regions_from_features(
            student_features
        )

        cluster_ids, spurious_mask, relevant_mask = (
            self._stage2_region_partition(inputs)
        )

        classifier = _get_classifier(student_model)

        loss_region, loss_R, loss_A, valid_images = (
            self._region_aware_loss(
                student_regions=student_regions,
                labels=labels,
                spurious_mask=spurious_mask,
                relevant_mask=relevant_mask,
                classifier=classifier,
            )
        )

        return RaVLOutput(
            loss_region=loss_region,
            loss_R=loss_R,
            loss_A=loss_A,
            logits=logits,
            spurious_mask=spurious_mask.detach(),
            relevant_mask=relevant_mask.detach(),
            cluster_ids=cluster_ids.detach(),
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
        return classification_loss +  float(self.lambda_cl)*output.loss_region

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
        self.lambda_cl
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print("RaVL-ResNet smoke test passed.")
    print(out.statistics())
