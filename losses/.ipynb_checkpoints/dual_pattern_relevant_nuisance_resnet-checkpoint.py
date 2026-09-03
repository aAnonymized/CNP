# from __future__ import annotations

# import copy
# import csv
# import math
# import os
# import random
# from dataclasses import dataclass
# from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, Any

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# Tensor = torch.Tensor


# # =============================================================================
# # Public result containers
# # =============================================================================


# @dataclass
# class DualPatternDiscoveryResult:
#     relevant_best_k: int
#     nuisance_best_k: int
#     irrelevant_best_k: int
#     relevant_silhouette: Dict[int, float]
#     nuisance_silhouette: Dict[int, float]
#     irrelevant_silhouette: Dict[int, float]
#     relevant_candidate_count: int
#     nuisance_candidate_count: int
#     irrelevant_candidate_count: int
#     relevant_candidate_count_per_class: List[int]
#     nuisance_candidate_count_per_class: List[int]
#     irrelevant_candidate_count_per_class: List[int]
#     decision_threshold: float


# @dataclass
# class DualPatternOutput:
#     # Requested Stage-II objectives
#     #   L_G : maximize I(Z;Y)   via CE-like Z-to-class alignment
#     #   L_R : maximize I(Zr;Y)  via CE-like Zr-to-class alignment
#     #   L_N : minimize I(Zn;Y)  via nHSIC(Zn, W_Y)
#     loss_region: Tensor
#     loss_global: Tensor
#     loss_relevant: Tensor
#     loss_nuisance: Tensor

#     # Ordinary model logits
#     logits: Tensor

#     # Current student representations
#     z_global: Tensor
#     student_regions: Tensor

#     # Frozen three-bank nearest-neighbor partition [B,R]
#     relevant_mask: Tensor
#     nuisance_mask: Tensor
#     irrelevant_mask: Tensor

#     # Frozen-bank maximum cosine similarities [B,R]
#     relevant_similarity: Tensor
#     nuisance_similarity: Tensor
#     irrelevant_similarity: Tensor
#     assignment_margin: Tensor       # s_R - s_N (kept for compatibility)

#     # Pooled representations [B,D]
#     z_relevant: Tensor
#     z_nuisance: Tensor
#     z_irrelevant: Tensor

#     # Whether an image contains at least one token of each type [B]
#     valid_relevant: Tensor
#     valid_nuisance: Tensor
#     valid_irrelevant: Tensor

#     # Counts for logging
#     num_relevant_regions: int
#     num_nuisance_regions: int
#     num_irrelevant_regions: int
#     num_valid_relevant_images: int
#     num_valid_nuisance_images: int
#     num_valid_irrelevant_images: int

#     # Kept for backward compatibility with older code. Not used by default.
#     loss_disentangle: Optional[Tensor] = None


# # =============================================================================
# # Model helpers
# # =============================================================================


# def _extract_logits(model_output: Any) -> Tensor:
#     """Robustly obtain logits [B,C] from common model output styles."""
#     if torch.is_tensor(model_output):
#         logits = model_output
#     elif isinstance(model_output, (tuple, list)):
#         if len(model_output) == 0 or not torch.is_tensor(model_output[0]):
#             raise TypeError("Model tuple/list output must have logits as its first tensor.")
#         logits = model_output[0]
#     elif isinstance(model_output, dict):
#         if "logits" not in model_output or not torch.is_tensor(model_output["logits"]):
#             raise TypeError("Model dict output must contain tensor key 'logits'.")
#         logits = model_output["logits"]
#     else:
#         raise TypeError("Unsupported model output type: {}".format(type(model_output)))

#     if logits.ndim != 2:
#         raise ValueError("Expected logits [B,C], got {}.".format(tuple(logits.shape)))
#     return logits


# def _get_layer4(model: nn.Module) -> nn.Module:
#     """Return final ResNet layer4 module."""
#     if not hasattr(model, "layer4"):
#         raise AttributeError(
#             "The model must expose the final spatial feature stage as model.layer4."
#         )
#     layer4 = getattr(model, "layer4")
#     if not isinstance(layer4, nn.Module):
#         raise TypeError("model.layer4 must be nn.Module.")
#     return layer4


# def _get_classifier(model: nn.Module) -> nn.Linear:
#     """Find a standard linear classification head: fc -> head -> classifier."""
#     for name in ("fc", "head", "classifier"):
#         if hasattr(model, name):
#             module = getattr(model, name)
#             if isinstance(module, nn.Linear):
#                 return module
#     raise AttributeError(
#         "Could not find a linear classifier. Expected model.fc, model.head, "
#         "or model.classifier to be nn.Linear."
#     )


# def _capture_layer4_and_forward(
#     model: nn.Module,
#     inputs: Tensor,
# ) -> Tuple[Tensor, Tensor]:
#     """
#     Run model once and capture FINAL layer4 output.

#     Returns:
#         logits:   [B,C]
#         features: [B,D,H,W]
#     """
#     holder: Dict[str, Tensor] = {}

#     def _hook(_module, _inputs, output):
#         if not torch.is_tensor(output):
#             raise TypeError("layer4 output must be a tensor.")
#         holder["features"] = output

#     handle = _get_layer4(model).register_forward_hook(_hook)
#     try:
#         output = model(inputs)
#     finally:
#         handle.remove()

#     if "features" not in holder:
#         raise RuntimeError("Failed to capture layer4 output.")

#     logits = _extract_logits(output)
#     features = holder["features"]

#     if features.ndim != 4:
#         raise ValueError(
#             "layer4 output must be [B,D,H,W], got {}.".format(tuple(features.shape))
#         )
#     if logits.shape[0] != features.shape[0]:
#         raise ValueError("Batch mismatch between logits and layer4 features.")

#     return logits, features


# def _forward_with_layer4_zero_mask(
#     model: nn.Module,
#     inputs: Tensor,
#     remove_mask: Tensor,
# ) -> Tensor:
#     """
#     Forward after zeroing selected layer4 spatial tokens.

#     Args:
#         model:       classifier exposing model.layer4
#         inputs:      [B,C,H,W]
#         remove_mask: [B,R] bool OR [B,Hf,Wf] bool

#     This is a layer4 intervention, NOT an input-image occlusion. It directly tests
#     the contribution of the discovered layer4 patch representation.
#     """
#     if remove_mask.ndim not in (2, 3):
#         raise ValueError("remove_mask must be [B,R] or [B,Hf,Wf].")

#     def _hook(_module, _inputs, output):
#         if not torch.is_tensor(output) or output.ndim != 4:
#             raise TypeError("layer4 output must be [B,D,Hf,Wf].")
#         b, _, hf, wf = output.shape
#         if remove_mask.ndim == 2:
#             if remove_mask.shape != (b, hf * wf):
#                 raise ValueError(
#                     "remove_mask shape {} incompatible with layer4 {}x{}.".format(
#                         tuple(remove_mask.shape), hf, wf
#                     )
#                 )
#             m = remove_mask.view(b, 1, hf, wf)
#         else:
#             if remove_mask.shape != (b, hf, wf):
#                 raise ValueError(
#                     "remove_mask shape {} incompatible with layer4 {}x{}.".format(
#                         tuple(remove_mask.shape), hf, wf
#                     )
#                 )
#             m = remove_mask.unsqueeze(1)
#         m = m.to(device=output.device, dtype=output.dtype)
#         return output * (1.0 - m)

#     handle = _get_layer4(model).register_forward_hook(_hook)
#     try:
#         output = model(inputs)
#     finally:
#         handle.remove()
#     return _extract_logits(output)


# # =============================================================================
# # Main method
# # =============================================================================


# class DualPatternResNet(object):
#     """
#     Three-bank Relevant / Nuisance / Irrelevant pattern discovery and MI-oriented
#     representation purification.

#     Stage-I discovery on reference_loader (recommended: dataloaders['val'])
#     -----------------------------------------------------------------------
#     For EVERY image, regardless of whether the image-level prediction is correct,
#     define c^- as the strongest non-GT class under the current classifier:

#         c^- = argmax_{c != y} logit_c.

#     For every layer4 patch f_p, compute its raw contribution to the GT class and to
#     the strongest non-GT competitor:

#         C_GT(p)  = f_p^T w_y
#         C_MIS(p) = f_p^T w_{c^-}

#     Then use the scale-free relative contribution score

#                          C_GT(p) - C_MIS(p)
#         r_p = -----------------------------------------------  in [-1, 1].
#               |C_GT(p)| + |C_MIS(p)| + eps

#     Patch semantics are defined ONLY by this local relative decision contribution:

#         r_p >  delta   -> Relevant
#         r_p < -delta   -> Nuisance
#         otherwise      -> Irrelevant

#     Therefore image-level prediction correctness is NOT part of the R/N definition,
#     and every discovery patch belongs to exactly one of R/N/I. decision_threshold is
#     now dimensionless and interpretable on a stable [0,1) scale (e.g. 0.1 or 0.5).

#     Each candidate pool is clustered INDEPENDENTLY. clustering_metric controls:
#         'cosine'    -> cosine K-Medoids + cosine silhouette
#         'euclidean' -> Euclidean-distance K-Medoids + Euclidean silhouette
#     With prototype_k_factors=[a,b], every integer K from a*C through b*C is tested,
#     and the K with the highest silhouette score is selected separately for R, N and I.

#     Stage-II assignment
#     -------------------
#     No bank-similarity threshold is used. For each frozen layer4 patch, compute:
#         s_R = max cosine(patch, R-bank)
#         s_N = max cosine(patch, N-bank)
#         s_I = max cosine(patch, I-bank)
#     and assign the patch to whichever bank has the largest similarity. Thus every
#     training patch is partitioned by nearest semantic prototype among the THREE banks.

#     Objectives
#     ----------
#         L_G = CE((Z W^T)/tau + log prior, y)          [no L2 normalization]
#         L_R = CE((Zr W^T)/tau + log prior, y)         [Zr -> w_y]
#         L_N = nHSIC(Zn, W_Y), W_Y = w_y               [Zn disentangles from w_y]

#     The irrelevant bank is used for assignment/reference only and has no auxiliary
#     loss by default.
#     """

#     def __init__(
#         self,
#         num_classes: int,
#         prototype_k_factors: Sequence[int] = (1, 2, 3, 4, 5),
#         decision_threshold: float = 0.5,
#         temperature: float = 0.07,
#         class_counts: Optional[Union[Tensor, Sequence[float]]] = None,
#         lambda_global: float = 1.0,
#         lambda_relevant: float = 0.0,
#         lambda_nuisance: float = 0.5,
#         kmedoids_iterations: int = 100,
#         max_candidates_per_class: Optional[int] = 2500,
#         silhouette_sample_size: int = 3000,
#         assignment_chunk_size: int = 8192,
#         clustering_device: Optional[Union[str, torch.device]] = None,
#         random_seed: int = 0,
#         eps: float = 1e-8,
#         clustering_metric: str = "cosine",
#     ) -> None:
#         if num_classes < 2:
#             raise ValueError("num_classes must be >= 2.")
#         if not (0.0 <= float(decision_threshold) < 1.0):
#             raise ValueError(
#                 "decision_threshold must be in [0,1) for the relative contribution score."
#             )
#         if temperature <= 0:
#             raise ValueError("temperature must be > 0.")
#         if kmedoids_iterations < 1:
#             raise ValueError("kmedoids_iterations must be >= 1.")
#         if silhouette_sample_size < 2:
#             raise ValueError("silhouette_sample_size must be >= 2.")
#         if assignment_chunk_size < 1:
#             raise ValueError("assignment_chunk_size must be >= 1.")
#         if any(v < 0 for v in (lambda_global, lambda_relevant, lambda_nuisance)):
#             raise ValueError("All lambda weights must be >= 0.")
#         if max_candidates_per_class is not None and max_candidates_per_class < 1:
#             raise ValueError("max_candidates_per_class must be >=1 or None.")

#         factors = [int(v) for v in prototype_k_factors]
#         if len(factors) == 0:
#             raise ValueError("prototype_k_factors cannot be empty.")
#         if any(v < 1 for v in factors):
#             raise ValueError("prototype_k_factors must contain positive integers.")
#         seen = set()
#         factors = [v for v in factors if not (v in seen or seen.add(v))]

#         self.num_classes = int(num_classes)
#         self.prototype_k_factors = factors
#         self.decision_threshold = float(decision_threshold)
#         self.temperature = float(temperature)

#         self.lambda_global = float(lambda_global)
#         self.lambda_relevant = float(lambda_relevant)
#         self.lambda_nuisance = float(lambda_nuisance)

#         self.kmedoids_iterations = int(kmedoids_iterations)
#         self.max_candidates_per_class = max_candidates_per_class
#         self.silhouette_sample_size = int(silhouette_sample_size)
#         self.assignment_chunk_size = int(assignment_chunk_size)
#         self.clustering_device = (
#             None if clustering_device is None else torch.device(clustering_device)
#         )
#         self.random_seed = int(random_seed)
#         self.eps = float(eps)

#         # Clustering / Stage-II matching metric.
#         # Accepted aliases:
#         #   cosine                  -> cosine K-Medoids + cosine silhouette
#         #   distance/euclidean/l2   -> Euclidean K-Medoids + Euclidean silhouette
#         metric = str(clustering_metric).strip().lower()
#         metric_alias = {
#             "cos": "cosine",
#             "cosine": "cosine",
#             "distance": "euclidean",
#             "euclidean": "euclidean",
#             "l2": "euclidean",
#         }
#         if metric not in metric_alias:
#             raise ValueError(
#                 "clustering_metric must be one of "
#                 "{'cosine', 'distance', 'euclidean', 'l2'}, got {!r}.".format(
#                     clustering_metric
#                 )
#             )
#         self.clustering_metric = metric_alias[metric]

#         # Frozen Stage-I banks.
#         self.relevant_medoids_raw: Optional[Tensor] = None
#         self.relevant_medoids_norm: Optional[Tensor] = None
#         self.nuisance_medoids_raw: Optional[Tensor] = None
#         self.nuisance_medoids_norm: Optional[Tensor] = None
#         self.irrelevant_medoids_raw: Optional[Tensor] = None
#         self.irrelevant_medoids_norm: Optional[Tensor] = None
#         self.relevant_best_k: Optional[int] = None
#         self.nuisance_best_k: Optional[int] = None
#         self.irrelevant_best_k: Optional[int] = None
#         self.discovery_result: Optional[DualPatternDiscoveryResult] = None

#         # Frozen exact Stage-I snapshot used for Stage-II assignment.
#         self._assignment_model: Optional[nn.Module] = None
#         self._assignment_device: Optional[torch.device] = None

#         # Class prior: cls_num / sum(cls_num).
#         self.class_prior = torch.full(
#             (self.num_classes,), 1.0 / float(self.num_classes), dtype=torch.float32
#         )
#         self.class_counts = torch.ones(self.num_classes, dtype=torch.float32)
#         if class_counts is not None:
#             self.set_class_counts(class_counts)

#         self.last_visualization_summary: Optional[Dict[str, Any]] = None

#     # -------------------------------------------------------------------------
#     # Class prior
#     # -------------------------------------------------------------------------

#     def set_class_counts(
#         self,
#         class_counts: Union[Tensor, Sequence[float]],
#     ) -> None:
#         counts = torch.as_tensor(class_counts, dtype=torch.float32).flatten().cpu()
#         if counts.numel() != self.num_classes:
#             raise ValueError(
#                 "class_counts must contain {} values, got {}.".format(
#                     self.num_classes, counts.numel()
#                 )
#             )
#         if torch.any(counts < 0):
#             raise ValueError("class_counts cannot contain negative values.")
#         if float(counts.sum().item()) <= 0:
#             raise ValueError("class_counts must have positive total mass.")
#         self.class_counts = counts
#         prior = counts / counts.sum()
#         self.class_prior = prior.clamp_min(self.eps)
#         self.class_prior = self.class_prior / self.class_prior.sum()

#     @property
#     def prior_log(self) -> Tensor:
#         # Exactly log(cls_num / sum(cls_num)), with eps only for zero-count safety.
#         return self.class_prior.clamp_min(self.eps).log()

#     # -------------------------------------------------------------------------
#     # layer4 regions / patches
#     # -------------------------------------------------------------------------

#     def _regions_from_features(self, features: Tensor) -> Tensor:
#         """
#         [B,D,H,W] -> [B,H*W,D].
#         Standard ResNet50 @224: [B,2048,7,7] -> [B,49,2048].
#         """
#         if features.ndim != 4:
#             raise ValueError("features must be [B,D,H,W].")
#         return features.flatten(2).transpose(1, 2)

#     # -------------------------------------------------------------------------
#     # Stage-I candidate extraction
#     # -------------------------------------------------------------------------

#     @torch.no_grad()
#     def _extract_decision_aware_candidates(
#         self,
#         model: nn.Module,
#         inputs: Tensor,
#         labels: Tensor,
#     ) -> Tuple[Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Dict[str, Any]]:
#         """
#         Build R/N/I candidates from one validation/reference batch.

#         IMPORTANT: image-level prediction correctness is NOT used to define R/N/I.

#         For each image i:
#             c_i^- = argmax_{c != y_i} logit_{i,c}

#         For each patch p:
#             C_GT(p)  = f_p^T w_y
#             C_MIS(p) = f_p^T w_{c^-}

#             r_p = (C_GT(p) - C_MIS(p)) /
#                   (|C_GT(p)| + |C_MIS(p)| + eps)

#         Since |a-b| <= |a|+|b|, r_p lies in [-1,1] up to numerical precision.

#         Partition:
#             r_p >  delta   -> Relevant
#             r_p < -delta   -> Nuisance
#             otherwise      -> Irrelevant

#         Thus every patch belongs to exactly one discovery pool.
#         """
#         logits, features = _capture_layer4_and_forward(model, inputs)
#         regions = self._regions_from_features(features)  # [B,R,D]
#         labels = labels.long().to(logits.device)

#         if logits.shape[1] != self.num_classes:
#             raise ValueError(
#                 "Expected {} classes, got {}.".format(self.num_classes, logits.shape[1])
#             )

#         classifier = _get_classifier(model)
#         if classifier.weight.shape[0] != self.num_classes:
#             raise ValueError("Classifier output dimension != num_classes.")
#         if classifier.weight.shape[1] != regions.shape[-1]:
#             raise ValueError(
#                 "Classifier feature dim {} != layer4 patch dim {}. "
#                 "This method assumes GAP(layer4) -> linear classifier.".format(
#                     classifier.weight.shape[1], regions.shape[-1]
#                 )
#             )

#         pred = logits.argmax(dim=1)
#         correct = pred.eq(labels)
#         wrong = ~correct

#         # Strongest non-GT competitor c^- for EVERY image.
#         non_gt_logits = logits.detach().clone()
#         row = torch.arange(logits.shape[0], device=logits.device)
#         non_gt_logits[row, labels] = -torch.inf
#         competitor = non_gt_logits.argmax(dim=1)

#         # Frozen classifier directions are used only to measure local decision
#         # contribution during discovery.
#         W = classifier.weight.detach().to(regions.device, regions.dtype)
#         w_gt = W.index_select(0, labels)          # [B,D]
#         w_mis = W.index_select(0, competitor)     # [B,D]

#         # Raw per-patch class contributions. No L2 normalization is used here.
#         c_gt = torch.einsum("brd,bd->br", regions, w_gt)
#         c_mis = torch.einsum("brd,bd->br", regions, w_mis)

#         # Scale-free relative contribution in [-1,1].
#         relative_score = (c_gt - c_mis) / (
#             c_gt.abs() + c_mis.abs() + self.eps
#         )
#         relative_score = relative_score.clamp(-1.0, 1.0)

#         delta = float(self.decision_threshold)
#         relevant_mask_all = relative_score.gt(delta)
#         nuisance_mask_all = relative_score.lt(-delta)
#         irrelevant_mask_all = ~(relevant_mask_all | nuisance_mask_all)

#         # Exhaustive and mutually exclusive by construction.
#         if not bool((relevant_mask_all | nuisance_mask_all | irrelevant_mask_all).all()):
#             raise RuntimeError("R/N/I discovery masks do not cover all patches.")
#         if bool((relevant_mask_all & nuisance_mask_all).any()):
#             raise RuntimeError("Relevant/Nuisance candidate masks overlap.")
#         if bool((relevant_mask_all & irrelevant_mask_all).any()):
#             raise RuntimeError("Relevant/Irrelevant candidate masks overlap.")
#         if bool((nuisance_mask_all & irrelevant_mask_all).any()):
#             raise RuntimeError("Nuisance/Irrelevant candidate masks overlap.")

#         relevant_by_class: Dict[int, Tensor] = {}
#         nuisance_by_class: Dict[int, Tensor] = {}
#         irrelevant_by_class: Dict[int, Tensor] = {}

#         for c in range(self.num_classes):
#             image_c = labels.eq(c).unsqueeze(1)
#             rel = regions[image_c & relevant_mask_all].detach().float().cpu()
#             nui = regions[image_c & nuisance_mask_all].detach().float().cpu()
#             irr = regions[image_c & irrelevant_mask_all].detach().float().cpu()
#             if rel.numel() > 0:
#                 relevant_by_class[c] = rel
#             if nui.numel() > 0:
#                 nuisance_by_class[c] = nui
#             if irr.numel() > 0:
#                 irrelevant_by_class[c] = irr

#         rel_selected = relative_score[relevant_mask_all]
#         nui_selected = relative_score[nuisance_mask_all]
#         irr_selected = relative_score[irrelevant_mask_all]

#         stats: Dict[str, Any] = {
#             "num_images": int(labels.numel()),
#             "num_correct_images": int(correct.sum().item()),
#             "num_wrong_images": int(wrong.sum().item()),
#             "num_relevant_candidates": int(relevant_mask_all.sum().item()),
#             "num_nuisance_candidates": int(nuisance_mask_all.sum().item()),
#             "num_irrelevant_candidates": int(irrelevant_mask_all.sum().item()),
#             # Compatibility field: exhaustive R/N/I partition means nothing is ignored.
#             "num_ignored_strong_opposite": 0,
#             "mean_relevant_score": float(rel_selected.mean().item()) if rel_selected.numel() else float("nan"),
#             "mean_nuisance_score": float(nui_selected.mean().item()) if nui_selected.numel() else float("nan"),
#             "mean_abs_irrelevant_score": float(irr_selected.abs().mean().item()) if irr_selected.numel() else float("nan"),
#             "mean_relative_score": float(relative_score.mean().item()),
#             "min_relative_score": float(relative_score.min().item()),
#             "max_relative_score": float(relative_score.max().item()),
#         }
#         return relevant_by_class, nuisance_by_class, irrelevant_by_class, stats

#     @staticmethod
#     def _append_candidate_dict(
#         destination: List[List[Tensor]],
#         source: Dict[int, Tensor],
#     ) -> None:
#         for c, x in source.items():
#             if x.numel() > 0:
#                 destination[c].append(x)

#     def _finalize_candidate_pool(
#         self,
#         per_class_chunks: List[List[Tensor]],
#         seed_offset: int,
#     ) -> Tuple[Tensor, List[int]]:
#         """Concatenate and optionally cap candidates independently per GT class."""
#         kept: List[Tensor] = []
#         counts: List[int] = []

#         for c in range(self.num_classes):
#             if len(per_class_chunks[c]) == 0:
#                 counts.append(0)
#                 continue

#             x = torch.cat(per_class_chunks[c], dim=0)
#             n = int(x.shape[0])
#             if self.max_candidates_per_class is not None and n > self.max_candidates_per_class:
#                 gen = torch.Generator(device="cpu")
#                 gen.manual_seed(self.random_seed + seed_offset + 1009 * c)
#                 idx = torch.randperm(n, generator=gen)[: self.max_candidates_per_class]
#                 x = x.index_select(0, idx)

#             counts.append(int(x.shape[0]))
#             kept.append(x)

#         if len(kept) == 0:
#             return torch.empty((0, 0), dtype=torch.float32), counts
#         return torch.cat(kept, dim=0), counts

#     # -------------------------------------------------------------------------
#     # K-Medoids + silhouette model selection
#     #   clustering_metric="cosine"    : cosine distance / cosine silhouette
#     #   clustering_metric="euclidean" : Euclidean distance / Euclidean silhouette
#     # -------------------------------------------------------------------------

#     @torch.no_grad()
#     def _fit_kmedoids_cosine(
#         self,
#         x_raw: Tensor,
#         k: int,
#         seed: int,
#     ) -> Tuple[Tensor, Tensor]:
#         """Alternating cosine K-Medoids with farthest-point initialization."""
#         n = int(x_raw.shape[0])
#         if k < 2 or k >= n:
#             raise ValueError("K-Medoids requires 2 <= K < N.")

#         x = F.normalize(x_raw.float(), p=2, dim=1, eps=self.eps)
#         gen = torch.Generator(device=x.device)
#         gen.manual_seed(int(seed))

#         first = int(torch.randint(0, n, (1,), generator=gen, device=x.device).item())
#         selected = [first]
#         min_dist = 1.0 - (x @ x[first:first + 1].t()).squeeze(1)

#         for _ in range(1, k):
#             idx = int(min_dist.argmax().item())
#             selected.append(idx)
#             dist = 1.0 - (x @ x[idx:idx + 1].t()).squeeze(1)
#             min_dist = torch.minimum(min_dist, dist)

#         medoid_ids = torch.tensor(selected, device=x.device, dtype=torch.long)
#         old_labels: Optional[Tensor] = None

#         for _ in range(self.kmedoids_iterations):
#             medoids = x.index_select(0, medoid_ids)
#             labels = (x @ medoids.t()).argmax(dim=1)

#             if old_labels is not None and torch.equal(labels, old_labels):
#                 break
#             old_labels = labels.clone()

#             new_ids = medoid_ids.clone()
#             for cluster_id in range(k):
#                 ids = labels.eq(cluster_id).nonzero(as_tuple=False).squeeze(1)
#                 if ids.numel() == 0:
#                     continue
#                 members = x.index_select(0, ids)
#                 cluster_sum = members.sum(dim=0, keepdim=True).t()  # [D,1]
#                 score = (members @ cluster_sum).squeeze(1)
#                 local_best = int(score.argmax().item())
#                 new_ids[cluster_id] = ids[local_best]

#             if torch.equal(new_ids, medoid_ids):
#                 medoid_ids = new_ids
#                 break
#             medoid_ids = new_ids

#         medoids = x.index_select(0, medoid_ids)
#         labels = (x @ medoids.t()).argmax(dim=1)
#         return medoid_ids, labels

#     @torch.no_grad()
#     def _fit_kmedoids_euclidean(
#         self,
#         x_raw: Tensor,
#         k: int,
#         seed: int,
#     ) -> Tuple[Tensor, Tensor]:
#         """
#         Euclidean K-Medoids-like clustering on RAW features.

#         Assignment uses Euclidean distance. During the medoid update, the prototype
#         is constrained to be an observed sample and is chosen as the member nearest
#         to the arithmetic cluster mean. This exactly minimizes the sum of SQUARED
#         Euclidean distances among candidate medoids, while avoiding an O(M^2)
#         pairwise-distance matrix inside every cluster.
#         """
#         n = int(x_raw.shape[0])
#         if k < 2 or k >= n:
#             raise ValueError("K-Medoids requires 2 <= K < N.")

#         # IMPORTANT: do NOT L2-normalize here. Euclidean mode intentionally uses
#         # both feature direction and feature magnitude.
#         x = x_raw.float()
#         gen = torch.Generator(device=x.device)
#         gen.manual_seed(int(seed))

#         # Farthest-point initialization under squared Euclidean distance.
#         first = int(torch.randint(0, n, (1,), generator=gen, device=x.device).item())
#         selected = [first]
#         min_dist2 = (x - x[first:first + 1]).pow(2).sum(dim=1)

#         for _ in range(1, k):
#             idx = int(min_dist2.argmax().item())
#             selected.append(idx)
#             dist2 = (x - x[idx:idx + 1]).pow(2).sum(dim=1)
#             min_dist2 = torch.minimum(min_dist2, dist2)

#         medoid_ids = torch.tensor(selected, device=x.device, dtype=torch.long)
#         old_labels: Optional[Tensor] = None

#         for _ in range(self.kmedoids_iterations):
#             medoids = x.index_select(0, medoid_ids)
#             # Squared Euclidean and Euclidean have the same nearest prototype.
#             dist = torch.cdist(x, medoids, p=2)
#             labels = dist.argmin(dim=1)

#             if old_labels is not None and torch.equal(labels, old_labels):
#                 break
#             old_labels = labels.clone()

#             new_ids = medoid_ids.clone()
#             for cluster_id in range(k):
#                 ids = labels.eq(cluster_id).nonzero(as_tuple=False).squeeze(1)
#                 if ids.numel() == 0:
#                     continue

#                 members = x.index_select(0, ids)
#                 center = members.mean(dim=0, keepdim=True)
#                 dist2_to_center = (members - center).pow(2).sum(dim=1)
#                 local_best = int(dist2_to_center.argmin().item())
#                 new_ids[cluster_id] = ids[local_best]

#             if torch.equal(new_ids, medoid_ids):
#                 medoid_ids = new_ids
#                 break
#             medoid_ids = new_ids

#         medoids = x.index_select(0, medoid_ids)
#         labels = torch.cdist(x, medoids, p=2).argmin(dim=1)
#         return medoid_ids, labels

#     @torch.no_grad()
#     def _euclidean_silhouette_score(
#         self,
#         x_raw: Tensor,
#         labels: Tensor,
#         seed: int,
#     ) -> float:
#         """Approximate Euclidean silhouette score on a stratified subset."""
#         n = int(x_raw.shape[0])
#         if n < 3:
#             return float("-inf")

#         unique = labels.unique(sorted=True)
#         if unique.numel() < 2:
#             return float("-inf")

#         max_n = min(n, self.silhouette_sample_size)
#         gen = torch.Generator(device="cpu")
#         gen.manual_seed(int(seed))

#         labels_cpu = labels.detach().cpu()
#         selected: List[Tensor] = []
#         base = max(1, max_n // int(unique.numel()))

#         for c in unique.cpu().tolist():
#             ids = labels_cpu.eq(int(c)).nonzero(as_tuple=False).squeeze(1)
#             if ids.numel() == 0:
#                 continue
#             take = min(int(ids.numel()), base)
#             perm = torch.randperm(int(ids.numel()), generator=gen)[:take]
#             selected.append(ids.index_select(0, perm))

#         if len(selected) == 0:
#             return float("-inf")

#         idx_cpu = torch.cat(selected, dim=0)
#         if idx_cpu.numel() < max_n:
#             all_perm = torch.randperm(n, generator=gen)
#             mark = torch.zeros(n, dtype=torch.bool)
#             mark[idx_cpu] = True
#             extra = all_perm[~mark[all_perm]][: max_n - idx_cpu.numel()]
#             idx_cpu = torch.cat([idx_cpu, extra], dim=0)

#         idx = idx_cpu.to(x_raw.device)
#         x = x_raw.index_select(0, idx).float()
#         y = labels.index_select(0, idx)

#         # True Euclidean pairwise distance for silhouette.
#         dist = torch.cdist(x, x, p=2)
#         m = int(x.shape[0])
#         silhouettes = torch.zeros(m, device=x.device, dtype=torch.float32)
#         sampled_clusters = y.unique(sorted=True)

#         for c in sampled_clusters.tolist():
#             mask_c = y.eq(int(c))
#             ids_c = mask_c.nonzero(as_tuple=False).squeeze(1)
#             n_c = int(ids_c.numel())
#             if n_c <= 1:
#                 silhouettes[ids_c] = 0.0
#                 continue

#             d_rows = dist.index_select(0, ids_c)
#             a = d_rows[:, mask_c].sum(dim=1) / float(n_c - 1)

#             b = torch.full_like(a, float("inf"))
#             for other in sampled_clusters.tolist():
#                 if int(other) == int(c):
#                     continue
#                 mask_o = y.eq(int(other))
#                 if not bool(mask_o.any()):
#                     continue
#                 mean_d = d_rows[:, mask_o].mean(dim=1)
#                 b = torch.minimum(b, mean_d)

#             denom = torch.maximum(a, b).clamp_min(self.eps)
#             s = (b - a) / denom
#             silhouettes[ids_c] = torch.where(
#                 torch.isfinite(s), s, torch.zeros_like(s)
#             )

#         return float(silhouettes.mean().item())

#     @torch.no_grad()
#     def _cosine_silhouette_score(
#         self,
#         x_raw: Tensor,
#         labels: Tensor,
#         seed: int,
#     ) -> float:
#         """Approximate cosine silhouette score on a stratified subset."""
#         n = int(x_raw.shape[0])
#         if n < 3:
#             return float("-inf")

#         unique = labels.unique(sorted=True)
#         if unique.numel() < 2:
#             return float("-inf")

#         max_n = min(n, self.silhouette_sample_size)
#         gen = torch.Generator(device="cpu")
#         gen.manual_seed(int(seed))

#         labels_cpu = labels.detach().cpu()
#         selected: List[Tensor] = []
#         base = max(1, max_n // int(unique.numel()))

#         for c in unique.cpu().tolist():
#             ids = labels_cpu.eq(int(c)).nonzero(as_tuple=False).squeeze(1)
#             if ids.numel() == 0:
#                 continue
#             take = min(int(ids.numel()), base)
#             perm = torch.randperm(int(ids.numel()), generator=gen)[:take]
#             selected.append(ids.index_select(0, perm))

#         if len(selected) == 0:
#             return float("-inf")

#         idx_cpu = torch.cat(selected, dim=0)
#         if idx_cpu.numel() < max_n:
#             all_perm = torch.randperm(n, generator=gen)
#             mark = torch.zeros(n, dtype=torch.bool)
#             mark[idx_cpu] = True
#             extra = all_perm[~mark[all_perm]][: max_n - idx_cpu.numel()]
#             idx_cpu = torch.cat([idx_cpu, extra], dim=0)

#         idx = idx_cpu.to(x_raw.device)
#         x = F.normalize(
#             x_raw.index_select(0, idx).float(), p=2, dim=1, eps=self.eps
#         )
#         y = labels.index_select(0, idx)

#         dist = (1.0 - x @ x.t()).clamp_min(0.0)
#         m = int(x.shape[0])
#         silhouettes = torch.zeros(m, device=x.device, dtype=torch.float32)
#         sampled_clusters = y.unique(sorted=True)

#         for c in sampled_clusters.tolist():
#             mask_c = y.eq(int(c))
#             ids_c = mask_c.nonzero(as_tuple=False).squeeze(1)
#             n_c = int(ids_c.numel())
#             if n_c <= 1:
#                 silhouettes[ids_c] = 0.0
#                 continue

#             d_rows = dist.index_select(0, ids_c)
#             a = d_rows[:, mask_c].sum(dim=1) / float(n_c - 1)

#             b = torch.full_like(a, float("inf"))
#             for other in sampled_clusters.tolist():
#                 if int(other) == int(c):
#                     continue
#                 mask_o = y.eq(int(other))
#                 if not bool(mask_o.any()):
#                     continue
#                 mean_d = d_rows[:, mask_o].mean(dim=1)
#                 b = torch.minimum(b, mean_d)

#             denom = torch.maximum(a, b).clamp_min(self.eps)
#             s = (b - a) / denom
#             silhouettes[ids_c] = torch.where(
#                 torch.isfinite(s), s, torch.zeros_like(s)
#             )

#         return float(silhouettes.mean().item())

#     @torch.no_grad()
#     def _select_best_bank(
#         self,
#         candidate_pool_cpu: Tensor,
#         bank_name: str,
#         device: torch.device,
#         seed_offset: int,
#         verbose: bool,
#     ) -> Tuple[Tensor, Tensor, int, Dict[int, float]]:
#         if candidate_pool_cpu.ndim != 2 or candidate_pool_cpu.shape[0] < 3:
#             raise RuntimeError(
#                 "{} candidate pool is too small for clustering: shape={}.".format(
#                     bank_name, tuple(candidate_pool_cpu.shape)
#                 )
#             )

#         x = candidate_pool_cpu.to(device=device, dtype=torch.float32)
#         n = int(x.shape[0])
#         best_score = float("-inf")
#         best_k: Optional[int] = None
#         best_medoid_ids: Optional[Tensor] = None
#         best_labels: Optional[Tensor] = None
#         scores: Dict[int, float] = {}

#         # Requested list semantics: [1,2,3,...] -> [C,2C,3C,...] clusters.
#         # candidate_ks = [factor * self.num_classes for factor in self.prototype_k_factors]
#         min_factor = min(self.prototype_k_factors)
#         max_factor = max(self.prototype_k_factors)
#         k_start = min_factor * self.num_classes
#         k_end = max_factor * self.num_classes
#         candidate_ks = list(range(k_start, k_end + 1))

#         if verbose:
#             print("\n[{} bank] candidate K: {}".format(bank_name, candidate_ks))

#         for index, k in enumerate(candidate_ks):
#             if k < 2 or k >= n:
#                 if verbose:
#                     print("  K={:<4d} skipped (need 2 <= K < N={})".format(k, n))
#                 continue

#             if self.clustering_metric == "cosine":
#                 medoid_ids, labels = self._fit_kmedoids_cosine(
#                     x_raw=x,
#                     k=k,
#                     seed=self.random_seed + seed_offset + 97 * index,
#                 )
#                 score = self._cosine_silhouette_score(
#                     x_raw=x,
#                     labels=labels,
#                     seed=self.random_seed + seed_offset + 193 * index,
#                 )
#             else:
#                 medoid_ids, labels = self._fit_kmedoids_euclidean(
#                     x_raw=x,
#                     k=k,
#                     seed=self.random_seed + seed_offset + 97 * index,
#                 )
#                 score = self._euclidean_silhouette_score(
#                     x_raw=x,
#                     labels=labels,
#                     seed=self.random_seed + seed_offset + 193 * index,
#                 )

#             scores[int(k)] = float(score)

#             if verbose:
#                 print(
#                     "  K={:<4d} {} silhouette={:.6f}".format(
#                         k, self.clustering_metric, score
#                     )
#                 )

#             if score > best_score:
#                 best_score = score
#                 best_k = int(k)
#                 best_medoid_ids = medoid_ids.clone()
#                 best_labels = labels.clone()

#         if best_k is None or best_medoid_ids is None or best_labels is None:
#             raise RuntimeError(
#                 "No valid K for {} bank. Candidate pool N={}, factors={}. "
#                 "Reduce prototype_k_factors or collect more candidates.".format(
#                     bank_name, n, self.prototype_k_factors
#                 )
#             )

#         medoids_raw = x.index_select(0, best_medoid_ids).detach().cpu()
#         medoids_norm = F.normalize(
#             medoids_raw.float(), p=2, dim=1, eps=self.eps
#         ).cpu()

#         if verbose:
#             print(
#                 "[{} bank] selected K={} with {} silhouette={:.6f}".format(
#                     bank_name, best_k, self.clustering_metric, best_score
#                 )
#             )

#         return medoids_raw, medoids_norm, best_k, scores

#     # -------------------------------------------------------------------------
#     # Public discovery
#     # -------------------------------------------------------------------------

#     @torch.no_grad()
#     def discover(
#         self,
#         model: nn.Module,
#         reference_loader: Iterable,
#         device: Union[str, torch.device],
#         verbose: bool = True,
#     ) -> DualPatternDiscoveryResult:
#         """Discover Relevant/Nuisance/Irrelevant banks from dataloaders['val']."""
#         device = torch.device(device)
#         original_training = model.training
#         model.eval()

#         relevant_chunks: List[List[Tensor]] = [[] for _ in range(self.num_classes)]
#         nuisance_chunks: List[List[Tensor]] = [[] for _ in range(self.num_classes)]
#         irrelevant_chunks: List[List[Tensor]] = [[] for _ in range(self.num_classes)]

#         total_images = total_correct = total_wrong = 0
#         raw_rel = raw_nui = raw_irr = raw_ignored = 0

#         for batch_idx, batch in enumerate(reference_loader):
#             if not isinstance(batch, (tuple, list)) or len(batch) < 2:
#                 raise ValueError(
#                     "reference_loader must yield tuple/list with inputs and labels."
#                 )
#             inputs = batch[0].to(device, non_blocking=True)
#             labels = batch[-1].long().to(device, non_blocking=True)

#             rel_dict, nui_dict, irr_dict, stats = self._extract_decision_aware_candidates(
#                 model=model, inputs=inputs, labels=labels
#             )
#             self._append_candidate_dict(relevant_chunks, rel_dict)
#             self._append_candidate_dict(nuisance_chunks, nui_dict)
#             self._append_candidate_dict(irrelevant_chunks, irr_dict)

#             total_images += stats["num_images"]
#             total_correct += stats["num_correct_images"]
#             total_wrong += stats["num_wrong_images"]
#             raw_rel += stats["num_relevant_candidates"]
#             raw_nui += stats["num_nuisance_candidates"]
#             raw_irr += stats["num_irrelevant_candidates"]
#             raw_ignored += stats["num_ignored_strong_opposite"]

#             if verbose and (batch_idx + 1) % 20 == 0:
#                 print(
#                     "[Discovery] batches={} images={} correct={} wrong={} R={} N={} I={}".format(
#                         batch_idx + 1, total_images, total_correct, total_wrong,
#                         raw_rel, raw_nui, raw_irr,
#                     )
#                 )

#         relevant_pool, rel_counts = self._finalize_candidate_pool(
#             relevant_chunks, seed_offset=10000
#         )
#         nuisance_pool, nui_counts = self._finalize_candidate_pool(
#             nuisance_chunks, seed_offset=20000
#         )
#         irrelevant_pool, irr_counts = self._finalize_candidate_pool(
#             irrelevant_chunks, seed_offset=25000
#         )

#         if relevant_pool.numel() == 0:
#             raise RuntimeError(
#                 "No Relevant candidates: require relative patch contribution "
#                 "r_p > {:.3f}. Reduce decision_threshold if needed.".format(self.decision_threshold)
#             )
#         if nuisance_pool.numel() == 0:
#             raise RuntimeError(
#                 "No Nuisance candidates: require relative patch contribution "
#                 "r_p < -{:.3f}. Reduce decision_threshold if needed.".format(self.decision_threshold)
#             )
#         if irrelevant_pool.numel() == 0:
#             raise RuntimeError(
#                 "No Irrelevant candidates: require |r_p| <= {:.3f}. "
#                 "Increase decision_threshold if needed.".format(self.decision_threshold)
#             )

#         if verbose:
#             print("\n========== Three-Bank R/N/I Discovery ==========")
#             print("decision_threshold = {:.4f} (relative score in [-1,1])".format(self.decision_threshold))
#             print("clustering_metric = {}".format(self.clustering_metric))
#             print("images={} | correct={} | wrong={}".format(total_images, total_correct, total_wrong))
#             print("raw R={} | raw N={} | raw I={} | coverage=100%".format(
#                 raw_rel, raw_nui, raw_irr
#             ))
#             print("kept R={} | kept N={} | kept I={}".format(
#                 relevant_pool.shape[0], nuisance_pool.shape[0], irrelevant_pool.shape[0]
#             ))
#             print("R per GT class = {}".format(rel_counts))
#             print("N per GT class = {}".format(nui_counts))
#             print("I per GT class = {}".format(irr_counts))

#         cluster_device = self.clustering_device or device

#         (
#             self.relevant_medoids_raw,
#             self.relevant_medoids_norm,
#             self.relevant_best_k,
#             rel_scores,
#         ) = self._select_best_bank(
#             candidate_pool_cpu=relevant_pool,
#             bank_name="Relevant",
#             device=cluster_device,
#             seed_offset=30000,
#             verbose=verbose,
#         )
#         (
#             self.nuisance_medoids_raw,
#             self.nuisance_medoids_norm,
#             self.nuisance_best_k,
#             nui_scores,
#         ) = self._select_best_bank(
#             candidate_pool_cpu=nuisance_pool,
#             bank_name="Nuisance",
#             device=cluster_device,
#             seed_offset=40000,
#             verbose=verbose,
#         )
#         (
#             self.irrelevant_medoids_raw,
#             self.irrelevant_medoids_norm,
#             self.irrelevant_best_k,
#             irr_scores,
#         ) = self._select_best_bank(
#             candidate_pool_cpu=irrelevant_pool,
#             bank_name="Irrelevant",
#             device=cluster_device,
#             seed_offset=50000,
#             verbose=verbose,
#         )

#         # Freeze the exact discovery representation space for Stage-II matching.
#         self._assignment_model = copy.deepcopy(model).to(device)
#         self._assignment_model.eval()
#         for parameter in self._assignment_model.parameters():
#             parameter.requires_grad_(False)
#         self._assignment_device = device

#         result = DualPatternDiscoveryResult(
#             relevant_best_k=int(self.relevant_best_k),
#             nuisance_best_k=int(self.nuisance_best_k),
#             irrelevant_best_k=int(self.irrelevant_best_k),
#             relevant_silhouette=rel_scores,
#             nuisance_silhouette=nui_scores,
#             irrelevant_silhouette=irr_scores,
#             relevant_candidate_count=int(relevant_pool.shape[0]),
#             nuisance_candidate_count=int(nuisance_pool.shape[0]),
#             irrelevant_candidate_count=int(irrelevant_pool.shape[0]),
#             relevant_candidate_count_per_class=rel_counts,
#             nuisance_candidate_count_per_class=nui_counts,
#             irrelevant_candidate_count_per_class=irr_counts,
#             decision_threshold=self.decision_threshold,
#         )
#         self.discovery_result = result

#         if original_training:
#             model.train()
#         if verbose:
#             print("=================================================\n")
#         return result

#     # -------------------------------------------------------------------------
#     # Stage-II frozen-bank assignment
#     # -------------------------------------------------------------------------

#     def _check_discovered(self) -> None:
#         if (
#             self.relevant_medoids_norm is None
#             or self.nuisance_medoids_norm is None
#             or self.irrelevant_medoids_norm is None
#             or self._assignment_model is None
#         ):
#             raise RuntimeError(
#                 "Run discover(model, reference_loader, device) before Stage-II forward."
#             )

#     @torch.no_grad()
#     def _max_bank_similarity(
#         self,
#         region_norm: Tensor,
#         bank_norm: Tensor,
#     ) -> Tuple[Tensor, Tensor]:
#         """Chunked maximum cosine similarity."""
#         n = int(region_norm.shape[0])
#         values: List[Tensor] = []
#         ids: List[Tensor] = []
#         for start in range(0, n, self.assignment_chunk_size):
#             end = min(n, start + self.assignment_chunk_size)
#             sim = region_norm[start:end] @ bank_norm.t()
#             v, j = sim.max(dim=1)
#             values.append(v)
#             ids.append(j)
#         return torch.cat(values, dim=0), torch.cat(ids, dim=0)

#     @torch.no_grad()
#     def _min_bank_euclidean_distance(
#         self,
#         region_raw: Tensor,
#         bank_raw: Tensor,
#     ) -> Tuple[Tensor, Tensor]:
#         """Chunked minimum Euclidean distance to a prototype bank."""
#         n = int(region_raw.shape[0])
#         values: List[Tensor] = []
#         ids: List[Tensor] = []
#         for start in range(0, n, self.assignment_chunk_size):
#             end = min(n, start + self.assignment_chunk_size)
#             dist = torch.cdist(region_raw[start:end], bank_raw, p=2)
#             v, j = dist.min(dim=1)
#             values.append(v)
#             ids.append(j)
#         return torch.cat(values, dim=0), torch.cat(ids, dim=0)

#     @torch.no_grad()
#     def _stage2_triple_bank_partition(
#         self,
#         inputs: Tensor,
#     ) -> Dict[str, Tensor]:
#         """
#         Assign every frozen layer4 patch to the NEAREST of R/N/I banks.

#         clustering_metric='cosine': choose the largest cosine similarity.
#         clustering_metric='euclidean': choose the smallest Euclidean distance
#         (internally converted to score 1/(1+d), so the existing argmax API is kept).

#         No bank_similarity_threshold is used:
#             A_p = argmax {s_R(p), s_N(p), s_I(p)}.
#         """
#         self._check_discovered()
#         assert self._assignment_model is not None
#         assert self._assignment_device is not None
#         assert self.relevant_medoids_norm is not None
#         assert self.nuisance_medoids_norm is not None
#         assert self.irrelevant_medoids_norm is not None

#         x = inputs.to(self._assignment_device, non_blocking=True)
#         _, features = _capture_layer4_and_forward(self._assignment_model, x)
#         regions = self._regions_from_features(features)
#         b, r, d = regions.shape

#         if self.clustering_metric == "cosine":
#             flat = F.normalize(
#                 regions.reshape(-1, d).float(), p=2, dim=1, eps=self.eps
#             )
#             rel_bank = self.relevant_medoids_norm.to(flat.device, flat.dtype)
#             nui_bank = self.nuisance_medoids_norm.to(flat.device, flat.dtype)
#             irr_bank = self.irrelevant_medoids_norm.to(flat.device, flat.dtype)

#             rel_sim, rel_id = self._max_bank_similarity(flat, rel_bank)
#             nui_sim, nui_id = self._max_bank_similarity(flat, nui_bank)
#             irr_sim, irr_id = self._max_bank_similarity(flat, irr_bank)
#         else:
#             # Euclidean mode uses RAW feature magnitude + direction, consistent with
#             # the Euclidean clustering performed during discover().
#             flat = regions.reshape(-1, d).float()
#             rel_bank = self.relevant_medoids_raw.to(flat.device, flat.dtype)
#             nui_bank = self.nuisance_medoids_raw.to(flat.device, flat.dtype)
#             irr_bank = self.irrelevant_medoids_raw.to(flat.device, flat.dtype)

#             rel_dist, rel_id = self._min_bank_euclidean_distance(flat, rel_bank)
#             nui_dist, nui_id = self._min_bank_euclidean_distance(flat, nui_bank)
#             irr_dist, irr_id = self._min_bank_euclidean_distance(flat, irr_bank)

#             # Keep the existing downstream API named '*_similarity': transform
#             # distance monotonically to a higher-is-better score in (0, 1].
#             rel_sim = 1.0 / (1.0 + rel_dist)
#             nui_sim = 1.0 / (1.0 + nui_dist)
#             irr_sim = 1.0 / (1.0 + irr_dist)

#         rel_sim = rel_sim.view(b, r)
#         nui_sim = nui_sim.view(b, r)
#         irr_sim = irr_sim.view(b, r)
#         rel_id = rel_id.view(b, r)
#         nui_id = nui_id.view(b, r)
#         irr_id = irr_id.view(b, r)

#         # Deterministic tie rule via argmax: R (0) -> N (1) -> I (2).
#         sims = torch.stack([rel_sim, nui_sim, irr_sim], dim=-1)  # [B,R,3]
#         assignment_index = sims.argmax(dim=-1)
#         relevant_mask = assignment_index.eq(0)
#         nuisance_mask = assignment_index.eq(1)
#         irrelevant_mask = assignment_index.eq(2)

#         # Winner-vs-runner confidence, useful for ranking ablation strength.
#         sorted_sims, _ = sims.sort(dim=-1, descending=True)
#         assignment_confidence = sorted_sims[..., 0] - sorted_sims[..., 1]

#         return {
#             "relevant_mask": relevant_mask,
#             "nuisance_mask": nuisance_mask,
#             "irrelevant_mask": irrelevant_mask,
#             "relevant_similarity": rel_sim,
#             "nuisance_similarity": nui_sim,
#             "irrelevant_similarity": irr_sim,
#             "assignment_margin": rel_sim - nui_sim,
#             "assignment_confidence": assignment_confidence,
#             "assignment_index": assignment_index,
#             "nearest_relevant_id": rel_id,
#             "nearest_nuisance_id": nui_id,
#             "nearest_irrelevant_id": irr_id,
#         }

#     # Backward-compatible alias: old callers still work, but the returned dict now
#     # contains a genuine three-bank partition.
#     def _stage2_dual_bank_partition(self, inputs: Tensor) -> Dict[str, Tensor]:
#         return self._stage2_triple_bank_partition(inputs)

#     # -------------------------------------------------------------------------
#     # Pooling
#     # -------------------------------------------------------------------------

#     def _masked_avg_pool(
#         self,
#         regions: Tensor,
#         mask: Tensor,
#     ) -> Tuple[Tensor, Tensor]:
#         """Masked average pooling over layer4 patches."""
#         if regions.ndim != 3 or mask.ndim != 2:
#             raise ValueError("regions must be [B,R,D] and mask [B,R].")
#         if regions.shape[:2] != mask.shape:
#             raise ValueError("regions/mask shape mismatch.")

#         m = mask.to(dtype=regions.dtype).unsqueeze(-1)
#         count = m.sum(dim=1)
#         pooled = (regions * m).sum(dim=1) / count.clamp_min(1.0)
#         valid = count.squeeze(1).gt(0)
#         return pooled, valid

#     # -------------------------------------------------------------------------
#     # CE-like objective shared by Z and Zr
#     # -------------------------------------------------------------------------

#     def _ce_like_to_class_loss(
#         self,
#         z: Tensor,
#         labels: Tensor,
#         classifier: nn.Linear,
#         valid: Optional[Tensor] = None,
#     ) -> Tensor:
#         """
#         logits(c) = cos(z,w_c)/tau + log pi_c
#         L = CE(logits, y)

#         prior pi_c = cls_num[c] / sum(cls_num).
#         """
#         if valid is None:
#             valid = torch.ones(z.shape[0], device=z.device, dtype=torch.bool)
#         if not bool(valid.any()):
#             return z.sum() * 0.0

#         z_v = z[valid]
#         y_v = labels.long()[valid]
        
#         z_v = F.normalize(z_v, p=2, dim=1, eps=self.eps)
#         W = F.normalize(
#             classifier.weight.to(device=z_v.device, dtype=z_v.dtype),
#             p=2,
#             dim=1,
#             eps=self.eps,
#         )
#         cosine_logits = z_v @ W.t()
#         prior_log = self.prior_log.to(device=z_v.device, dtype=z_v.dtype)
#         logits_ce = cosine_logits / self.temperature + prior_log.unsqueeze(0)
#         return F.cross_entropy(logits_ce, y_v)

#     def _global_to_class_loss(
#         self,
#         z_global: Tensor,
#         labels: Tensor,
#         classifier: nn.Linear,
#     ) -> Tensor:
#         """Requested surrogate for maximizing I(Z;Y)."""
#         return self._ce_like_to_class_loss(
#             z=z_global,
#             labels=labels,
#             classifier=classifier,
#             valid=None,
#         )

#     def _relevant_region_to_class_loss(
#         self,
#         z_relevant: Tensor,
#         valid_relevant: Tensor,
#         labels: Tensor,
#         classifier: nn.Linear,
#     ) -> Tensor:
#         """Requested surrogate for maximizing I(Zr;Y)."""
#         return self._ce_like_to_class_loss(
#             z=z_relevant,
#             labels=labels,
#             classifier=classifier,
#             valid=valid_relevant,
#         )

#     # -------------------------------------------------------------------------
#     # nHSIC for minimizing dependence I(Zn;Y) through W_Y = w_y
#     # -------------------------------------------------------------------------

#     def _rbf_kernel_median(self, x: Tensor) -> Tensor:
#         """RBF kernel with detached median heuristic bandwidth."""
#         x2 = (x * x).sum(dim=1, keepdim=True)
#         dist2 = (x2 + x2.t() - 2.0 * (x @ x.t())).clamp_min(0.0)

#         n = int(x.shape[0])
#         upper = torch.triu_indices(n, n, offset=1, device=x.device)
#         pair_dist2 = dist2[upper[0], upper[1]].detach()
#         positive = pair_dist2[pair_dist2 > self.eps]

#         if positive.numel() > 0:
#             sigma2 = positive.median().clamp_min(self.eps)
#         else:
#             sigma2 = dist2.new_tensor(1.0)

#         return torch.exp(-dist2 / (2.0 * sigma2))

#     @staticmethod
#     def _center_kernel(k: Tensor) -> Tensor:
#         return (
#             k
#             - k.mean(dim=0, keepdim=True)
#             - k.mean(dim=1, keepdim=True)
#             + k.mean()
#         )

#     def _normalized_hsic(self, x: Tensor, y: Tensor) -> Tensor:
#         """
#         Normalized RBF-HSIC / centered-kernel alignment.

#         Population HSIC with characteristic RBF kernels has the same independence
#         zero point as mutual information: HSIC=0 iff independence (under standard
#         conditions). We use normalized HSIC as a scale-stable dependence surrogate;
#         it is NOT asserted to be numerically equal to MI.
#         """
#         if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
#             raise ValueError("x,y must be [N,Dx]/[N,Dy] with the same N.")
#         n = int(x.shape[0])
#         if n < 2:
#             return (x.sum() + y.sum()) * 0.0

#         x = F.normalize(x, p=2, dim=1, eps=self.eps)
#         y = F.normalize(y, p=2, dim=1, eps=self.eps)

#         kx = self._center_kernel(self._rbf_kernel_median(x))
#         ky = self._center_kernel(self._rbf_kernel_median(y))

#         norm_x = (kx * kx).sum()
#         norm_y = (ky * ky).sum()
#         if float(norm_x.detach().item()) <= self.eps or float(norm_y.detach().item()) <= self.eps:
#             return x.sum() * 0.0

#         numerator = (kx * ky).sum()
#         denominator = torch.sqrt(
#             norm_x.clamp_min(self.eps) * norm_y.clamp_min(self.eps)
#         ).clamp_min(self.eps)

#         nhsic = numerator / denominator
#         # Numerical roundoff can produce tiny negative values.
#         return nhsic.clamp_min(0.0)

#     def _nuisance_to_class_nhsic_loss(
#         self,
#         z_nuisance: Tensor,
#         valid_nuisance: Tensor,
#         labels: Tensor,
#         classifier: nn.Linear,
#     ) -> Tensor:
#         """
#         L_N = nHSIC(Zn, W_Y), where W_Y = w_y.

#         W_Y is detached, therefore the nuisance loss cannot be minimized by simply
#         rotating the classifier. It must reduce dependence carried by Zn.
#         """
#         valid = valid_nuisance
#         if int(valid.sum().item()) < 2:
#             return z_nuisance.sum() * 0.0

#         zn = z_nuisance[valid]
#         y = labels.long()[valid]

#         W = classifier.weight.detach().to(device=zn.device, dtype=zn.dtype)
#         wy = W.index_select(0, y)

#         return self._normalized_hsic(zn, wy)

#     # -------------------------------------------------------------------------
#     # Public Stage-II forward
#     # -------------------------------------------------------------------------

#     def forward(
#         self,
#         student_model: nn.Module,
#         inputs: Tensor,
#         labels: Tensor,
#     ) -> DualPatternOutput:
#         """
#         Stage-II forward:
#           1) current student -> layer4, Z, ordinary logits
#           2) frozen snapshot -> nearest R/N/I bank assignment
#           3) current student patches + frozen masks -> Zr, Zn, Zi
#           4) compute L_G, L_R, L_N; Zi is reference-only by default
#         """
#         self._check_discovered()

#         logits, student_features = _capture_layer4_and_forward(student_model, inputs)
#         if logits.shape[1] != self.num_classes:
#             raise ValueError(
#                 "Expected {} classes, got {}.".format(self.num_classes, logits.shape[1])
#             )

#         student_regions = self._regions_from_features(student_features)
#         z_global = student_features.mean(dim=(2, 3))
#         classifier = _get_classifier(student_model)
#         labels = labels.long().to(logits.device)

#         partition = self._stage2_triple_bank_partition(inputs)
#         relevant_mask = partition["relevant_mask"].to(student_regions.device)
#         nuisance_mask = partition["nuisance_mask"].to(student_regions.device)
#         irrelevant_mask = partition["irrelevant_mask"].to(student_regions.device)

#         if relevant_mask.shape != student_regions.shape[:2]:
#             raise RuntimeError(
#                 "Frozen assignment region shape {} != student region shape {}.".format(
#                     tuple(relevant_mask.shape), tuple(student_regions.shape[:2])
#                 )
#             )

#         # Exactly one of R/N/I per patch.
#         covered = relevant_mask | nuisance_mask | irrelevant_mask
#         if not bool(covered.all()):
#             raise RuntimeError("Three-bank partition did not cover all patches.")
#         if bool((relevant_mask & nuisance_mask).any()) or bool((relevant_mask & irrelevant_mask).any()) or bool((nuisance_mask & irrelevant_mask).any()):
#             raise RuntimeError("Three-bank partition masks overlap.")

#         z_relevant, valid_relevant = self._masked_avg_pool(student_regions, relevant_mask)
#         z_nuisance, valid_nuisance = self._masked_avg_pool(student_regions, nuisance_mask)
#         z_irrelevant, valid_irrelevant = self._masked_avg_pool(student_regions, irrelevant_mask)

#         loss_global = self._global_to_class_loss(
#             z_global=z_global, labels=labels, classifier=classifier
#         )
#         loss_relevant = self._relevant_region_to_class_loss(
#             z_relevant=z_relevant,
#             valid_relevant=valid_relevant,
#             labels=labels,
#             classifier=classifier,
#         )
#         loss_nuisance = self._nuisance_to_class_nhsic_loss(
#             z_nuisance=z_nuisance,
#             valid_nuisance=valid_nuisance,
#             labels=labels,
#             classifier=classifier,
#         )
#         loss_region = loss_global + loss_relevant + loss_nuisance

#         return DualPatternOutput(
#             loss_region=loss_region,
#             loss_global=loss_global,
#             loss_relevant=loss_relevant,
#             loss_nuisance=loss_nuisance,
#             logits=logits,
#             z_global=z_global,
#             student_regions=student_regions,
#             relevant_mask=relevant_mask.detach(),
#             nuisance_mask=nuisance_mask.detach(),
#             irrelevant_mask=irrelevant_mask.detach(),
#             relevant_similarity=partition["relevant_similarity"].to(student_regions.device).detach(),
#             nuisance_similarity=partition["nuisance_similarity"].to(student_regions.device).detach(),
#             irrelevant_similarity=partition["irrelevant_similarity"].to(student_regions.device).detach(),
#             assignment_margin=partition["assignment_margin"].to(student_regions.device).detach(),
#             z_relevant=z_relevant,
#             z_nuisance=z_nuisance,
#             z_irrelevant=z_irrelevant,
#             valid_relevant=valid_relevant.detach(),
#             valid_nuisance=valid_nuisance.detach(),
#             valid_irrelevant=valid_irrelevant.detach(),
#             num_relevant_regions=int(relevant_mask.sum().item()),
#             num_nuisance_regions=int(nuisance_mask.sum().item()),
#             num_irrelevant_regions=int(irrelevant_mask.sum().item()),
#             num_valid_relevant_images=int(valid_relevant.sum().item()),
#             num_valid_nuisance_images=int(valid_nuisance.sum().item()),
#             num_valid_irrelevant_images=int(valid_irrelevant.sum().item()),
#             loss_disentangle=None,
#         )

#     __call__ = forward

#     # -------------------------------------------------------------------------
#     # Loss combination
#     # -------------------------------------------------------------------------

#     def total_loss(
#         self,
#         output: DualPatternOutput,
#         lambda_global: Optional[float] = 1.0,
#         lambda_relevant: Optional[float] = 0.0,
#         lambda_nuisance: Optional[float] = 0.5,
#     ) -> Tensor:
#         """
#         Exact requested three-term objective (no ordinary-logit CE added):

#             lambda_G * L_G + lambda_R * L_R + lambda_N * L_N
#         """
#         if lambda_global is None:
#             lambda_global = self.lambda_global
#         if lambda_relevant is None:
#             lambda_relevant = self.lambda_relevant
#         if lambda_nuisance is None:
#             lambda_nuisance = self.lambda_nuisance

#         return (
#             float(lambda_global) * output.loss_global
#             + float(lambda_relevant) * output.loss_relevant
#             + float(lambda_nuisance) * output.loss_nuisance
#         )

#     def combine_with_classification_loss(
#         self,
#         classification_loss: Tensor,
#         output: DualPatternOutput,
#         lambda_relevant: Optional[float] = 0.0,
#         lambda_nuisance: Optional[float] = 0.5,
#         lambda_global: Optional[float] = 0.0,
#     ) -> Tensor:
#         """
#         Backward-compatible helper if you still want the original ordinary CE:

#             CE(logits,y) + lambda_G*L_G + lambda_R*L_R + lambda_N*L_N
#         """
#         if lambda_global is None:
#             lambda_global = self.lambda_global
#         if lambda_relevant is None:
#             lambda_relevant = self.lambda_relevant
#         if lambda_nuisance is None:
#             lambda_nuisance = self.lambda_nuisance

#         loss = (
#             classification_loss
#             + float(lambda_global) * output.loss_global
#             + float(lambda_relevant) * output.loss_relevant
#             + float(lambda_nuisance) * output.loss_nuisance
#         )
#         return loss

#     # -------------------------------------------------------------------------
#     # Diagnostics
#     # -------------------------------------------------------------------------

#     @torch.no_grad()
#     def evaluate_classwise_partition(
#         self,
#         student_model: nn.Module,
#         data_loader: Iterable,
#         device: Union[str, torch.device],
#     ) -> Dict[str, Tensor]:
#         """Report classwise R/N/I nearest-bank patch ratios and similarities."""
#         self._check_discovered()
#         device = torch.device(device)
#         original_training = student_model.training
#         student_model.eval()

#         image_count = torch.zeros(self.num_classes, dtype=torch.long)
#         total_patch_count = torch.zeros(self.num_classes, dtype=torch.long)
#         rel_patch_count = torch.zeros(self.num_classes, dtype=torch.long)
#         nui_patch_count = torch.zeros(self.num_classes, dtype=torch.long)
#         irr_patch_count = torch.zeros(self.num_classes, dtype=torch.long)
#         rel_sim_sum = torch.zeros(self.num_classes, dtype=torch.float64)
#         nui_sim_sum = torch.zeros(self.num_classes, dtype=torch.float64)
#         irr_sim_sum = torch.zeros(self.num_classes, dtype=torch.float64)
#         sim_count = torch.zeros(self.num_classes, dtype=torch.long)

#         for batch in data_loader:
#             inputs = batch[0].to(device, non_blocking=True)
#             labels = batch[-1].long().to(device, non_blocking=True)
#             part = self._stage2_triple_bank_partition(inputs)
#             rel = part["relevant_mask"].cpu()
#             nui = part["nuisance_mask"].cpu()
#             irr = part["irrelevant_mask"].cpu()
#             sr = part["relevant_similarity"].cpu().double()
#             sn = part["nuisance_similarity"].cpu().double()
#             si = part["irrelevant_similarity"].cpu().double()
#             labels_cpu = labels.cpu()

#             for c in range(self.num_classes):
#                 m = labels_cpu.eq(c)
#                 if not bool(m.any()):
#                     continue
#                 n_img = int(m.sum().item())
#                 image_count[c] += n_img
#                 total_patch_count[c] += int(rel[m].numel())
#                 rel_patch_count[c] += int(rel[m].sum().item())
#                 nui_patch_count[c] += int(nui[m].sum().item())
#                 irr_patch_count[c] += int(irr[m].sum().item())
#                 rel_sim_sum[c] += float(sr[m].sum().item())
#                 nui_sim_sum[c] += float(sn[m].sum().item())
#                 irr_sim_sum[c] += float(si[m].sum().item())
#                 sim_count[c] += int(sr[m].numel())

#         denom = total_patch_count.clamp_min(1).float()
#         sim_denom = sim_count.clamp_min(1).double()
#         if original_training:
#             student_model.train()
#         return {
#             "num_images": image_count,
#             "num_total_patches": total_patch_count,
#             "num_relevant_patches": rel_patch_count,
#             "num_nuisance_patches": nui_patch_count,
#             "num_irrelevant_patches": irr_patch_count,
#             "relevant_ratio": rel_patch_count.float() / denom,
#             "nuisance_ratio": nui_patch_count.float() / denom,
#             "irrelevant_ratio": irr_patch_count.float() / denom,
#             "mean_relevant_similarity": (rel_sim_sum / sim_denom).float(),
#             "mean_nuisance_similarity": (nui_sim_sum / sim_denom).float(),
#             "mean_irrelevant_similarity": (irr_sim_sum / sim_denom).float(),
#         }

#     # -------------------------------------------------------------------------
#     # Visualization + causal layer4 ablation
#     # -------------------------------------------------------------------------

#     @staticmethod
#     def _batch_iterator(
#         inputs_or_loader: Union[Tensor, Iterable],
#         labels: Optional[Tensor],
#     ):
#         if torch.is_tensor(inputs_or_loader):
#             if labels is None:
#                 raise ValueError("labels must be supplied when inputs is a Tensor.")
#             yield inputs_or_loader, labels
#             return

#         for batch in inputs_or_loader:
#             if not isinstance(batch, (tuple, list)) or len(batch) < 2:
#                 raise ValueError(
#                     "DataLoader must yield tuple/list; first item=input, last item=label."
#                 )
#             yield batch[0], batch[-1]

#     @staticmethod
#     def _ratio_tag(ratio: float) -> str:
#         return "{:g}".format(100.0 * float(ratio)).replace(".", "p")

#     def _top_fraction_mask(
#         self,
#         candidate_mask: Tensor,
#         score: Tensor,
#         ratio: float,
#     ) -> Tensor:
#         """Select top ceil(ratio * number_of_candidate_patches) within one image."""
#         if candidate_mask.ndim != 1 or score.ndim != 1:
#             raise ValueError("candidate_mask and score must be flat [R].")
#         out = torch.zeros_like(candidate_mask, dtype=torch.bool)
#         ids = candidate_mask.nonzero(as_tuple=False).squeeze(1)
#         n = int(ids.numel())
#         if n == 0 or ratio <= 0:
#             return out
#         k = max(1, int(math.ceil(float(ratio) * n)))
#         k = min(k, n)
#         local_score = score.index_select(0, ids)
#         selected_local = torch.topk(local_score, k=k, largest=True).indices
#         selected = ids.index_select(0, selected_local)
#         out[selected] = True
#         return out

#     @torch.no_grad()
#     def visualize_partition(
#         self,
#         student_model: nn.Module,
#         inputs: Union[Tensor, Iterable],
#         labels: Optional[Tensor] = None,
#         save_dir: str = "./triple_bank_vis",
#         max_images: int = 20,
#         mean: Optional[Sequence[float]] = None,
#         std: Optional[Sequence[float]] = None,
#         removal_ratios: Sequence[float] = (0.05, 0.10, 0.20),
#         display: bool = False,
#         keep_relevant_ratios: Optional[Sequence[float]] = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00),
#     ) -> List[str]:
#         """
#         Visualize nearest-bank R/N/I patches and evaluate TWO complementary diagnostics.

#         A) Removal faithfulness
#         -----------------------
#         Remove the strongest R or N patches from the ORIGINAL layer4 representation.
#         After each intervention, recompute the FULL logits and Softmax probabilities.

#             remove R -> expect P_GT to decrease
#             remove N -> expect P_GT to increase

#         B) Relevant-patch sufficiency / saturation
#         -----------------------------------------
#         Keep ONLY the top-ratio Relevant patches and zero every other layer4 patch
#         (remaining Relevant + all Nuisance + all Irrelevant patches). This directly asks:

#             "How many Relevant patches are sufficient for the GT decision?"

#         For every keep ratio r, report:
#             - absolute P_GT after Softmax, in %
#             - dP_GT relative to the original full representation, in percentage points
#             - marginal P_GT gain over the previous keep ratio, in percentage points
#             - predicted class and whether prediction == GT

#         The smallest tested keep ratio that already predicts GT is stored per image as
#         min_keep_relevant_ratio_for_gt_prediction.

#         Assignment remains purely nearest-bank:
#             A_p = argmax {s_R, s_N, s_I}; no bank similarity threshold.

#         Fixed original competitor:
#             c* = argmax_{c != y} logit_c.
#         """
#         import numpy as np
#         import matplotlib.pyplot as plt

#         self._check_discovered()
#         os.makedirs(save_dir, exist_ok=True)

#         removal = sorted(set(float(r) for r in removal_ratios))
#         if len(removal) == 0 or any(r <= 0 or r > 1 for r in removal):
#             raise ValueError("removal_ratios must contain values in (0,1].")

#         if keep_relevant_ratios is None:
#             keep_ratios = list(removal)
#         else:
#             keep_ratios = sorted(set(float(r) for r in keep_relevant_ratios))
#         if len(keep_ratios) == 0 or any(r <= 0 or r > 1 for r in keep_ratios):
#             raise ValueError("keep_relevant_ratios must contain values in (0,1].")

#         original_training = student_model.training
#         student_model.eval()
#         device = next(student_model.parameters()).device
#         paths: List[str] = []
#         records: List[Dict[str, Any]] = []
#         global_index = 0

#         def _to_image(t: Tensor):
#             t = t.detach().float().cpu()
#             if mean is not None and std is not None:
#                 mean_t = torch.as_tensor(mean, dtype=t.dtype).view(-1, 1, 1)
#                 std_t = torch.as_tensor(std, dtype=t.dtype).view(-1, 1, 1)
#                 if mean_t.shape[0] != t.shape[0] or std_t.shape[0] != t.shape[0]:
#                     raise ValueError("mean/std channel count must match input channels.")
#                 t = t * std_t + mean_t
#             elif float(t.min().item()) < 0.0 or float(t.max().item()) > 1.0:
#                 lo = t.amin(dim=(1, 2), keepdim=True)
#                 hi = t.amax(dim=(1, 2), keepdim=True)
#                 t = (t - lo) / (hi - lo).clamp_min(1e-8)
#             t = t.clamp(0.0, 1.0)
#             if t.shape[0] == 1:
#                 return t[0].numpy()
#             if t.shape[0] >= 3:
#                 return t[:3].permute(1, 2, 0).numpy()
#             if t.shape[0] == 2:
#                 z = torch.zeros_like(t[:1])
#                 return torch.cat([t, z], dim=0).permute(1, 2, 0).numpy()
#             raise ValueError("Unsupported input channel count: {}".format(int(t.shape[0])))

#         def _to_rgb(img):
#             if img.ndim == 2:
#                 return np.repeat(img[..., None], 3, axis=2)
#             if img.shape[-1] == 1:
#                 return np.repeat(img, 3, axis=2)
#             return img[..., :3]

#         def _upsample_grid(grid: Tensor, hf: int, wf: int, H: int, W: int):
#             g = grid.detach().float().view(1, 1, hf, wf)
#             up = F.interpolate(g, size=(H, W), mode="nearest")
#             return up[0, 0].cpu().numpy()

#         def _overlay_partition(img, rel_mask, nui_mask, irr_mask, alpha=0.42):
#             base = _to_rgb(img).astype(np.float32).copy()
#             masks = [rel_mask > 0.5, nui_mask > 0.5, irr_mask > 0.5]
#             colors = [
#                 np.asarray([0.10, 0.90, 0.20], dtype=np.float32),
#                 np.asarray([0.95, 0.12, 0.12], dtype=np.float32),
#                 np.asarray([0.15, 0.45, 0.95], dtype=np.float32),
#             ]
#             for m, color in zip(masks, colors):
#                 base[m] = (1.0 - alpha) * base[m] + alpha * color
#             return np.clip(base, 0.0, 1.0)

#         def _draw_patch_grid(ax, hf: int, wf: int, H: int, W: int):
#             for gx in range(1, wf):
#                 ax.axvline(gx * W / float(wf) - 0.5, linewidth=0.45, alpha=0.50)
#             for gy in range(1, hf):
#                 ax.axhline(gy * H / float(hf) - 0.5, linewidth=0.45, alpha=0.50)

#         def _draw_relative_score_grid(
#             ax,
#             score_grid: Tensor,
#             hf: int,
#             wf: int,
#             H: int,
#             W: int,
#         ):
#             """Draw per-patch normalized GT-vs-competitor contribution in [-1,1]."""
#             import matplotlib.patheffects as pe

#             scores = score_grid.detach().float().cpu().view(hf, wf)
#             cell_h = H / float(hf)
#             cell_w = W / float(wf)
#             fontsize = max(5.0, min(9.0, 62.0 / float(max(hf, wf))))

#             for gy in range(hf):
#                 for gx in range(wf):
#                     value = float(scores[gy, gx].item())
#                     x_center = (gx + 0.5) * cell_w - 0.5
#                     y_center = (gy + 0.5) * cell_h - 0.5
#                     txt = ax.text(
#                         x_center,
#                         y_center,
#                         "{:+.2f}".format(value),
#                         ha="center",
#                         va="center",
#                         fontsize=fontsize,
#                         color="white",
#                         fontweight="bold",
#                         clip_on=True,
#                     )
#                     txt.set_path_effects([
#                         pe.Stroke(linewidth=1.6, foreground="black"),
#                         pe.Normal(),
#                     ])

#         for batch_inputs, batch_labels in self._batch_iterator(inputs, labels):
#             if max_images is not None and int(max_images) > 0 and len(paths) >= int(max_images):
#                 break

#             x = batch_inputs.to(device, non_blocking=True)
#             y = batch_labels.long().to(device, non_blocking=True)

#             part = self._stage2_triple_bank_partition(x)
#             base_logits, feat = _capture_layer4_and_forward(student_model, x)
#             base_prob = F.softmax(base_logits, dim=1)
#             pred = base_logits.argmax(dim=1)

#             other_logits = base_logits.detach().clone()
#             row = torch.arange(base_logits.shape[0], device=device)
#             other_logits[row, y] = -torch.inf
#             competitor = other_logits.argmax(dim=1)

#             # Visualization-only quantity: the SAME normalized GT-vs-competitor
#             # patch contribution used by Stage-I discovery, evaluated on the current
#             # student_model feature map shown in this figure. This does NOT change
#             # discovery, bank assignment, ablation, loss, or any training behavior.
#             score_regions = self._regions_from_features(feat)  # [B,R,D]
#             score_classifier = _get_classifier(student_model)
#             score_W = score_classifier.weight.detach().to(
#                 device=score_regions.device, dtype=score_regions.dtype
#             )
#             score_w_gt = score_W.index_select(0, y)
#             score_w_comp = score_W.index_select(0, competitor)
#             score_c_gt = torch.einsum("brd,bd->br", score_regions, score_w_gt)
#             score_c_comp = torch.einsum("brd,bd->br", score_regions, score_w_comp)
#             patch_relative_score = (score_c_gt - score_c_comp) / (
#                 score_c_gt.abs() + score_c_comp.abs() + self.eps
#             )
#             patch_relative_score = patch_relative_score.clamp(-1.0, 1.0)

#             hf, wf = int(feat.shape[-2]), int(feat.shape[-1])
#             R = hf * wf
#             if part["relevant_mask"].shape[1] != R:
#                 raise RuntimeError("Frozen assignment patch grid != current layer4 grid.")

#             for i in range(int(x.shape[0])):
#                 if max_images is not None and int(max_images) > 0 and len(paths) >= int(max_images):
#                     break

#                 yi = int(y[i].item())
#                 pi = int(pred[i].item())
#                 ci = int(competitor[i].item())

#                 orig_pgt = float(base_prob[i, yi].item())
#                 orig_pcomp = float(base_prob[i, ci].item())
#                 orig_prob_gap = orig_pgt - orig_pcomp
#                 orig_logit_gt = float(base_logits[i, yi].item())
#                 orig_logit_comp = float(base_logits[i, ci].item())
#                 orig_logit_margin = orig_logit_gt - orig_logit_comp

#                 rel_mask = part["relevant_mask"][i].to(device)
#                 nui_mask = part["nuisance_mask"][i].to(device)
#                 irr_mask = part["irrelevant_mask"][i].to(device)
#                 rel_sim = part["relevant_similarity"][i].to(device)
#                 nui_sim = part["nuisance_similarity"][i].to(device)
#                 irr_sim = part["irrelevant_similarity"][i].to(device)

#                 # Rank patches within a selected bank by winner-vs-best-alternative score.
#                 rel_strength = rel_sim - torch.maximum(nui_sim, irr_sim)
#                 nui_strength = nui_sim - torch.maximum(rel_sim, irr_sim)

#                 num_rel = int(rel_mask.sum().item())
#                 num_nui = int(nui_mask.sum().item())
#                 num_irr = int(irr_mask.sum().item())

#                 record: Dict[str, Any] = {
#                     "sample_index": global_index,
#                     "gt": yi,
#                     "pred": pi,
#                     "competitor": ci,
#                     "correct": int(yi == pi),
#                     "orig_pgt": orig_pgt,
#                     "orig_pgt_pct": 100.0 * orig_pgt,
#                     "orig_pcomp": orig_pcomp,
#                     "orig_pcomp_pct": 100.0 * orig_pcomp,
#                     "orig_prob_gap": orig_prob_gap,
#                     "orig_logit_gt": orig_logit_gt,
#                     "orig_logit_comp": orig_logit_comp,
#                     "orig_logit_margin": orig_logit_margin,
#                     "num_relevant": num_rel,
#                     "num_nuisance": num_nui,
#                     "num_irrelevant": num_irr,
#                 }

#                 xi = x[i:i + 1]

#                 # =============================================================
#                 # A) Removal faithfulness: remove top R / remove top N.
#                 # =============================================================
#                 ablation_rows: List[Dict[str, Any]] = []
#                 for ratio in removal:
#                     remove_rel = self._top_fraction_mask(rel_mask, rel_strength, ratio)
#                     remove_nui = self._top_fraction_mask(nui_mask, nui_strength, ratio)

#                     logits_rel = _forward_with_layer4_zero_mask(
#                         student_model, xi, remove_rel.view(1, -1)
#                     )
#                     logits_nui = _forward_with_layer4_zero_mask(
#                         student_model, xi, remove_nui.view(1, -1)
#                     )
#                     prob_rel = F.softmax(logits_rel, dim=1)[0]
#                     prob_nui = F.softmax(logits_nui, dim=1)[0]

#                     rel_pgt = float(prob_rel[yi].item())
#                     rel_pcomp = float(prob_rel[ci].item())
#                     rel_prob_gap = rel_pgt - rel_pcomp
#                     rel_logit_margin = float((logits_rel[0, yi] - logits_rel[0, ci]).item())
#                     rel_pred = int(logits_rel.argmax(dim=1)[0].item())

#                     nui_pgt = float(prob_nui[yi].item())
#                     nui_pcomp = float(prob_nui[ci].item())
#                     nui_prob_gap = nui_pgt - nui_pcomp
#                     nui_logit_margin = float((logits_nui[0, yi] - logits_nui[0, ci]).item())
#                     nui_pred = int(logits_nui.argmax(dim=1)[0].item())

#                     rel_delta_pgt = rel_pgt - orig_pgt
#                     rel_delta_pcomp = rel_pcomp - orig_pcomp
#                     rel_delta_prob_gap = rel_prob_gap - orig_prob_gap
#                     rel_delta_logit_margin = rel_logit_margin - orig_logit_margin

#                     nui_delta_pgt = nui_pgt - orig_pgt
#                     nui_delta_pcomp = nui_pcomp - orig_pcomp
#                     nui_delta_prob_gap = nui_prob_gap - orig_prob_gap
#                     nui_delta_logit_margin = nui_logit_margin - orig_logit_margin

#                     rel_delta_pgt_pct = 100.0 * rel_delta_pgt
#                     rel_delta_pcomp_pct = 100.0 * rel_delta_pcomp
#                     nui_delta_pgt_pct = 100.0 * nui_delta_pgt
#                     nui_delta_pcomp_pct = 100.0 * nui_delta_pcomp

#                     tag = self._ratio_tag(ratio)
#                     vals = {
#                         f"rel_pgt_{tag}pct": rel_pgt,
#                         f"rel_pcomp_{tag}pct": rel_pcomp,
#                         f"rel_prob_gap_{tag}pct": rel_prob_gap,
#                         f"rel_logit_margin_{tag}pct": rel_logit_margin,
#                         f"rel_delta_pgt_{tag}pct": rel_delta_pgt,
#                         f"rel_delta_pcomp_{tag}pct": rel_delta_pcomp,
#                         f"rel_delta_prob_gap_{tag}pct": rel_delta_prob_gap,
#                         f"rel_delta_logit_margin_{tag}pct": rel_delta_logit_margin,
#                         f"rel_delta_pgt_{tag}pct_points": rel_delta_pgt_pct,
#                         f"rel_delta_pcomp_{tag}pct_points": rel_delta_pcomp_pct,
#                         f"rel_pred_{tag}pct": rel_pred,
#                         f"rel_broken_{tag}pct": int(pi == yi and rel_pred != yi),
#                         f"nui_pgt_{tag}pct": nui_pgt,
#                         f"nui_pcomp_{tag}pct": nui_pcomp,
#                         f"nui_prob_gap_{tag}pct": nui_prob_gap,
#                         f"nui_logit_margin_{tag}pct": nui_logit_margin,
#                         f"nui_delta_pgt_{tag}pct": nui_delta_pgt,
#                         f"nui_delta_pcomp_{tag}pct": nui_delta_pcomp,
#                         f"nui_delta_prob_gap_{tag}pct": nui_delta_prob_gap,
#                         f"nui_delta_logit_margin_{tag}pct": nui_delta_logit_margin,
#                         f"nui_delta_pgt_{tag}pct_points": nui_delta_pgt_pct,
#                         f"nui_delta_pcomp_{tag}pct_points": nui_delta_pcomp_pct,
#                         f"nui_pred_{tag}pct": nui_pred,
#                         f"nui_corrected_{tag}pct": int(pi != yi and nui_pred == yi),
#                     }
#                     record.update(vals)
#                     ablation_rows.append({
#                         "ratio": ratio,
#                         "rel_delta_margin": rel_delta_logit_margin,
#                         "rel_delta_gap": rel_delta_prob_gap,
#                         "rel_delta_pgt": rel_delta_pgt,
#                         "rel_delta_pcomp": rel_delta_pcomp,
#                         "rel_delta_pgt_pct": rel_delta_pgt_pct,
#                         "rel_delta_pcomp_pct": rel_delta_pcomp_pct,
#                         "rel_pred": rel_pred,
#                         "nui_delta_margin": nui_delta_logit_margin,
#                         "nui_delta_gap": nui_delta_prob_gap,
#                         "nui_delta_pgt": nui_delta_pgt,
#                         "nui_delta_pcomp": nui_delta_pcomp,
#                         "nui_delta_pgt_pct": nui_delta_pgt_pct,
#                         "nui_delta_pcomp_pct": nui_delta_pcomp_pct,
#                         "nui_pred": nui_pred,
#                     })

#                 # =============================================================
#                 # B) Relevant sufficiency curve: keep ONLY top-ratio R patches.
#                 #    All unkept R + ALL N + ALL I are zeroed.
#                 # =============================================================
#                 keep_rows: List[Dict[str, Any]] = []
#                 previous_keep_pgt: Optional[float] = None
#                 min_keep_ratio_for_gt: Optional[float] = None

#                 for ratio in keep_ratios:
#                     tag = self._ratio_tag(ratio)

#                     if num_rel <= 0:
#                         keep_num = 0
#                         keep_pgt = float("nan")
#                         keep_pcomp = float("nan")
#                         keep_prob_gap = float("nan")
#                         keep_logit_margin = float("nan")
#                         keep_pred = -1
#                         keep_is_gt = 0
#                         keep_delta_pgt = float("nan")
#                         keep_delta_pcomp = float("nan")
#                         keep_marginal_pgt = float("nan")
#                     else:
#                         keep_rel = self._top_fraction_mask(rel_mask, rel_strength, ratio)
#                         keep_num = int(keep_rel.sum().item())

#                         # ONLY kept Relevant patches survive.
#                         remove_everything_else = ~keep_rel
#                         logits_keep = _forward_with_layer4_zero_mask(
#                             student_model, xi, remove_everything_else.view(1, -1)
#                         )
#                         prob_keep = F.softmax(logits_keep, dim=1)[0]

#                         keep_pgt = float(prob_keep[yi].item())
#                         keep_pcomp = float(prob_keep[ci].item())
#                         keep_prob_gap = keep_pgt - keep_pcomp
#                         keep_logit_margin = float(
#                             (logits_keep[0, yi] - logits_keep[0, ci]).item()
#                         )
#                         keep_pred = int(logits_keep.argmax(dim=1)[0].item())
#                         keep_is_gt = int(keep_pred == yi)
#                         keep_delta_pgt = keep_pgt - orig_pgt
#                         keep_delta_pcomp = keep_pcomp - orig_pcomp

#                         if previous_keep_pgt is None:
#                             keep_marginal_pgt = float("nan")
#                         else:
#                             keep_marginal_pgt = keep_pgt - previous_keep_pgt
#                         previous_keep_pgt = keep_pgt

#                         if keep_is_gt and min_keep_ratio_for_gt is None:
#                             min_keep_ratio_for_gt = float(ratio)

#                     keep_delta_pgt_pct = 100.0 * keep_delta_pgt if math.isfinite(keep_delta_pgt) else float("nan")
#                     keep_delta_pcomp_pct = 100.0 * keep_delta_pcomp if math.isfinite(keep_delta_pcomp) else float("nan")
#                     keep_marginal_pgt_pct = 100.0 * keep_marginal_pgt if math.isfinite(keep_marginal_pgt) else float("nan")

#                     record.update({
#                         f"keep_rel_num_{tag}pct": keep_num,
#                         f"keep_rel_pgt_{tag}pct": keep_pgt,
#                         f"keep_rel_pgt_{tag}pct_value": 100.0 * keep_pgt if math.isfinite(keep_pgt) else float("nan"),
#                         f"keep_rel_pcomp_{tag}pct": keep_pcomp,
#                         f"keep_rel_pcomp_{tag}pct_value": 100.0 * keep_pcomp if math.isfinite(keep_pcomp) else float("nan"),
#                         f"keep_rel_prob_gap_{tag}pct": keep_prob_gap,
#                         f"keep_rel_logit_margin_{tag}pct": keep_logit_margin,
#                         f"keep_rel_delta_pgt_{tag}pct": keep_delta_pgt,
#                         f"keep_rel_delta_pgt_{tag}pct_points": keep_delta_pgt_pct,
#                         f"keep_rel_delta_pcomp_{tag}pct": keep_delta_pcomp,
#                         f"keep_rel_delta_pcomp_{tag}pct_points": keep_delta_pcomp_pct,
#                         f"keep_rel_marginal_pgt_{tag}pct": keep_marginal_pgt,
#                         f"keep_rel_marginal_pgt_{tag}pct_points": keep_marginal_pgt_pct,
#                         f"keep_rel_pred_{tag}pct": keep_pred,
#                         f"keep_rel_predicts_gt_{tag}pct": keep_is_gt,
#                     })

#                     keep_rows.append({
#                         "ratio": ratio,
#                         "num_kept": keep_num,
#                         "pgt": keep_pgt,
#                         "pgt_pct": 100.0 * keep_pgt if math.isfinite(keep_pgt) else float("nan"),
#                         "pcomp": keep_pcomp,
#                         "pcomp_pct": 100.0 * keep_pcomp if math.isfinite(keep_pcomp) else float("nan"),
#                         "delta_pgt": keep_delta_pgt,
#                         "delta_pgt_pct": keep_delta_pgt_pct,
#                         "marginal_pgt": keep_marginal_pgt,
#                         "marginal_pgt_pct": keep_marginal_pgt_pct,
#                         "pred": keep_pred,
#                         "is_gt": keep_is_gt,
#                     })

#                 record["min_keep_relevant_ratio_for_gt_prediction"] = (
#                     min_keep_ratio_for_gt if min_keep_ratio_for_gt is not None else float("nan")
#                 )
#                 record["min_keep_relevant_pct_for_gt_prediction"] = (
#                     100.0 * min_keep_ratio_for_gt if min_keep_ratio_for_gt is not None else float("nan")
#                 )

#                 # ----- Figure -----
#                 img = _to_image(batch_inputs[i])
#                 H, Wimg = int(batch_inputs.shape[-2]), int(batch_inputs.shape[-1])
#                 rel_grid = rel_mask.view(hf, wf).float().cpu()
#                 nui_grid = nui_mask.view(hf, wf).float().cpu()
#                 irr_grid = irr_mask.view(hf, wf).float().cpu()
#                 rel_up = _upsample_grid(rel_grid, hf, wf, H, Wimg)
#                 nui_up = _upsample_grid(nui_grid, hf, wf, H, Wimg)
#                 irr_up = _upsample_grid(irr_grid, hf, wf, H, Wimg)
#                 overlay = _overlay_partition(img, rel_up, nui_up, irr_up)

#                 fig, axes = plt.subplots(2, 2, figsize=(19.0, 10.8))

#                 axes[0, 0].imshow(img, cmap="gray" if img.ndim == 2 else None)
#                 axes[0, 0].set_title(
#                     "Original\nGT={} Pred={} Comp={}\nP_GT={:.2f}%  P_c*={:.2f}%\nM={:+.4f}".format(
#                         yi, pi, ci, 100.0 * orig_pgt, 100.0 * orig_pcomp, orig_logit_margin
#                     )
#                 )
#                 axes[0, 0].axis("off")

#                 axes[0, 1].imshow(overlay)
#                 _draw_patch_grid(axes[0, 1], hf, wf, H, Wimg)
#                 axes[0, 1].set_title(
#                     "Nearest R/N/I bank\n"
#                     "Green=R {} | Red=N {} | Blue=I {} | c*={}".format(
#                         num_rel, num_nui, num_irr, ci
#                     )
#                 )
#                 axes[0, 1].axis("off")

#                 removal_lines = [
#                     "A) Removal faithfulness",
#                     "Softmax is recomputed after each masking intervention",
#                     "dP_GT = P_GT(after) - P_GT(original)",
#                     "",
#                     "ratio | remove R: dP_GT / dP_c* / pred | remove N: dP_GT / dP_c* / pred",
#                 ]
#                 for ar in ablation_rows:
#                     removal_lines.append(
#                         "{:>4.0f}% | {:+.2f}% / {:+.2f}% / {:>2d} | {:+.2f}% / {:+.2f}% / {:>2d}".format(
#                             100.0 * ar["ratio"],
#                             ar["rel_delta_pgt_pct"], ar["rel_delta_pcomp_pct"], ar["rel_pred"],
#                             ar["nui_delta_pgt_pct"], ar["nui_delta_pcomp_pct"], ar["nui_pred"],
#                         )
#                     )
#                 axes[1, 0].axis("off")
#                 axes[1, 0].text(
#                     0.0, 1.0, "\n".join(removal_lines),
#                     va="top", ha="left", family="monospace", fontsize=9.2
#                 )

#                 if min_keep_ratio_for_gt is None:
#                     min_keep_text = "none of tested ratios"
#                 else:
#                     min_keep_text = "{:.0f}%".format(100.0 * min_keep_ratio_for_gt)

#                 keep_lines = [
#                     "B) Relevant-patch sufficiency (KEEP R ONLY)",
#                     "All unkept R + all N + all I are zeroed",
#                     "P_GT = Softmax(logits_keep)[GT]",
#                     "marginal = P_GT(current ratio) - P_GT(previous ratio)",
#                     "",
#                     "ratio | kept R | P_GT | dP_GT vs orig | marginal | pred | GT?",
#                 ]
#                 for kr in keep_rows:
#                     if not math.isfinite(kr["pgt_pct"]):
#                         keep_lines.append(
#                             "{:>4.0f}% | {:>6d} |   N/A  |      N/A      |   N/A    |  -  |  -".format(
#                                 100.0 * kr["ratio"], kr["num_kept"]
#                             )
#                         )
#                         continue
#                     marginal_str = (
#                         "  N/A  " if not math.isfinite(kr["marginal_pgt_pct"])
#                         else "{:+.2f}%".format(kr["marginal_pgt_pct"])
#                     )
#                     keep_lines.append(
#                         "{:>4.0f}% | {:>6d} | {:>5.2f}% | {:+.2f}% | {:>7s} | {:>4d} | {}".format(
#                             100.0 * kr["ratio"], kr["num_kept"], kr["pgt_pct"],
#                             kr["delta_pgt_pct"], marginal_str, kr["pred"],
#                             "YES" if kr["is_gt"] else "NO",
#                         )
#                     )
#                 keep_lines.extend([
#                     "",
#                     "Smallest tested keep ratio predicting GT: {}".format(min_keep_text),
#                     "Plateau clue: later marginal gains close to 0% => added R patches bring little extra decision gain.",
#                 ])
#                 axes[1, 1].axis("off")
#                 axes[1, 1].text(
#                     0.0, 1.0, "\n".join(keep_lines),
#                     va="top", ha="left", family="monospace", fontsize=8.8
#                 )

#                 fig.tight_layout()

#                 path = os.path.join(save_dir, "sample_{:05d}.png".format(global_index))
#                 fig.savefig(path, dpi=180, bbox_inches="tight")
#                 if display:
#                     try:
#                         plt.show(block=False)
#                         plt.pause(0.001)
#                     except Exception:
#                         pass
#                 plt.close(fig)

#                 paths.append(path)
#                 records.append(record)
#                 global_index += 1

#         per_image_csv = os.path.join(save_dir, "ablation_per_image.csv")
#         if records:
#             fieldnames: List[str] = []
#             for rec in records:
#                 for key in rec.keys():
#                     if key not in fieldnames:
#                         fieldnames.append(key)
#             with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
#                 writer = csv.DictWriter(f, fieldnames=fieldnames)
#                 writer.writeheader()
#                 writer.writerows(records)

#         def _finite(vals):
#             return [float(v) for v in vals if math.isfinite(float(v))]

#         def _mean(vals):
#             vv = _finite(vals)
#             return float(sum(vv) / len(vv)) if vv else float("nan")

#         def _rate(vals):
#             vv = _finite(vals)
#             return float(sum(vv) / len(vv)) if vv else float("nan")

#         # =============================================================
#         # Summary A: removal faithfulness.
#         # =============================================================
#         summary_rows: List[Dict[str, Any]] = []
#         for ratio in removal:
#             tag = self._ratio_tag(ratio)
#             rel_valid = [r for r in records if int(r["num_relevant"]) > 0]
#             nui_valid = [r for r in records if int(r["num_nuisance"]) > 0]
#             rel_correct = [r for r in rel_valid if int(r["correct"]) == 1]
#             nui_wrong = [r for r in nui_valid if int(r["correct"]) == 0]

#             rel_dm = [float(r[f"rel_delta_logit_margin_{tag}pct"]) for r in rel_valid]
#             nui_dm = [float(r[f"nui_delta_logit_margin_{tag}pct"]) for r in nui_valid]
#             rel_dg = [float(r[f"rel_delta_prob_gap_{tag}pct"]) for r in rel_valid]
#             nui_dg = [float(r[f"nui_delta_prob_gap_{tag}pct"]) for r in nui_valid]
#             rel_dpgt = [float(r[f"rel_delta_pgt_{tag}pct"]) for r in rel_valid]
#             rel_dpcomp = [float(r[f"rel_delta_pcomp_{tag}pct"]) for r in rel_valid]
#             nui_dpgt = [float(r[f"nui_delta_pgt_{tag}pct"]) for r in nui_valid]
#             nui_dpcomp = [float(r[f"nui_delta_pcomp_{tag}pct"]) for r in nui_valid]
#             rel_correct_dm = [float(r[f"rel_delta_logit_margin_{tag}pct"]) for r in rel_correct]
#             nui_wrong_dm = [float(r[f"nui_delta_logit_margin_{tag}pct"]) for r in nui_wrong]
#             nui_wrong_dg = [float(r[f"nui_delta_prob_gap_{tag}pct"]) for r in nui_wrong]

#             summary_rows.append({
#                 "ratio": ratio,
#                 "num_images": len(records),
#                 "num_images_with_relevant": len(rel_valid),
#                 "mean_delta_logit_margin_remove_relevant": _mean(rel_dm),
#                 "mean_delta_prob_gap_remove_relevant": _mean(rel_dg),
#                 "mean_delta_pgt_remove_relevant": _mean(rel_dpgt),
#                 "mean_delta_pcomp_remove_relevant": _mean(rel_dpcomp),
#                 "mean_delta_pgt_remove_relevant_pct_points": 100.0 * _mean(rel_dpgt),
#                 "mean_delta_pcomp_remove_relevant_pct_points": 100.0 * _mean(rel_dpcomp),
#                 "relevant_faithfulness_rate_delta_pgt_lt_0": _rate([int(v < 0) for v in rel_dpgt]),
#                 "relevant_faithfulness_rate_delta_margin_lt_0": _rate([int(v < 0) for v in rel_dm]),
#                 "num_correct_images_with_relevant": len(rel_correct),
#                 "mean_delta_logit_margin_remove_relevant_on_correct": _mean(rel_correct_dm),
#                 "relevant_break_rate_on_original_correct": _rate([
#                     int(r[f"rel_broken_{tag}pct"]) for r in rel_correct
#                 ]),
#                 "num_images_with_nuisance": len(nui_valid),
#                 "mean_delta_logit_margin_remove_nuisance": _mean(nui_dm),
#                 "mean_delta_prob_gap_remove_nuisance": _mean(nui_dg),
#                 "mean_delta_pgt_remove_nuisance": _mean(nui_dpgt),
#                 "mean_delta_pcomp_remove_nuisance": _mean(nui_dpcomp),
#                 "mean_delta_pgt_remove_nuisance_pct_points": 100.0 * _mean(nui_dpgt),
#                 "mean_delta_pcomp_remove_nuisance_pct_points": 100.0 * _mean(nui_dpcomp),
#                 "nuisance_faithfulness_rate_delta_pgt_gt_0": _rate([int(v > 0) for v in nui_dpgt]),
#                 "nuisance_faithfulness_rate_delta_margin_gt_0": _rate([int(v > 0) for v in nui_dm]),
#                 "num_wrong_images_with_nuisance": len(nui_wrong),
#                 "mean_delta_logit_margin_remove_nuisance_on_wrong": _mean(nui_wrong_dm),
#                 "mean_delta_prob_gap_remove_nuisance_on_wrong": _mean(nui_wrong_dg),
#                 "nuisance_correction_rate_on_original_wrong": _rate([
#                     int(r[f"nui_corrected_{tag}pct"]) for r in nui_wrong
#                 ]),
#             })

#         summary_csv = os.path.join(save_dir, "ablation_summary.csv")
#         if summary_rows:
#             with open(summary_csv, "w", newline="", encoding="utf-8") as f:
#                 writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
#                 writer.writeheader()
#                 writer.writerows(summary_rows)

#         # =============================================================
#         # Summary B: relevant sufficiency / saturation curve.
#         # =============================================================
#         keep_summary_rows: List[Dict[str, Any]] = []
#         rel_records = [r for r in records if int(r["num_relevant"]) > 0]
#         for ratio in keep_ratios:
#             tag = self._ratio_tag(ratio)
#             pgt = [float(r[f"keep_rel_pgt_{tag}pct"]) for r in rel_records]
#             dpgt = [float(r[f"keep_rel_delta_pgt_{tag}pct"]) for r in rel_records]
#             marginal = [float(r[f"keep_rel_marginal_pgt_{tag}pct"]) for r in rel_records]
#             gt_ok = [int(r[f"keep_rel_predicts_gt_{tag}pct"]) for r in rel_records]
#             num_kept = [float(r[f"keep_rel_num_{tag}pct"]) for r in rel_records]

#             keep_summary_rows.append({
#                 "keep_ratio": ratio,
#                 "keep_ratio_pct": 100.0 * ratio,
#                 "num_images_with_relevant": len(rel_records),
#                 "mean_num_kept_relevant_patches": _mean(num_kept),
#                 "mean_pgt_keep_relevant": _mean(pgt),
#                 "mean_pgt_keep_relevant_pct": 100.0 * _mean(pgt),
#                 "mean_delta_pgt_vs_original": _mean(dpgt),
#                 "mean_delta_pgt_vs_original_pct_points": 100.0 * _mean(dpgt),
#                 "mean_marginal_pgt_gain_from_previous_ratio": _mean(marginal),
#                 "mean_marginal_pgt_gain_from_previous_ratio_pct_points": 100.0 * _mean(marginal),
#                 "gt_prediction_rate_keep_relevant_only": _rate(gt_ok),
#             })

#         keep_summary_csv = os.path.join(save_dir, "relevant_sufficiency_summary.csv")
#         if keep_summary_rows:
#             with open(keep_summary_csv, "w", newline="", encoding="utf-8") as f:
#                 writer = csv.DictWriter(f, fieldnames=list(keep_summary_rows[0].keys()))
#                 writer.writeheader()
#                 writer.writerows(keep_summary_rows)

#         min_ratio_values = [
#             float(r["min_keep_relevant_ratio_for_gt_prediction"])
#             for r in records
#             if math.isfinite(float(r["min_keep_relevant_ratio_for_gt_prediction"]))
#         ]

#         self.last_visualization_summary = {
#             "paths": paths,
#             "per_image_csv": per_image_csv if records else None,
#             "summary_csv": summary_csv if summary_rows else None,
#             "relevant_sufficiency_csv": keep_summary_csv if keep_summary_rows else None,
#             "records": records,
#             "summary": summary_rows,
#             "relevant_sufficiency_summary": keep_summary_rows,
#             "mean_min_keep_relevant_ratio_for_gt_prediction": _mean(min_ratio_values),
#         }

#         if original_training:
#             student_model.train()

#         print("[Three-bank visualization] saved {} image(s) to: {}".format(
#             len(paths), os.path.abspath(save_dir)
#         ))
#         if records:
#             print("  per-image stats       : {}".format(os.path.abspath(per_image_csv)))
#             print("  removal summary       : {}".format(os.path.abspath(summary_csv)))
#             print("  R sufficiency summary : {}".format(os.path.abspath(keep_summary_csv)))
#             print("  Assignment: nearest among R/N/I banks; no bank-similarity threshold")
#             print("  Removal metric: dP_GT = Softmax(logits_after)[GT] - Softmax(logits_original)[GT]")
#             print("  Keep-R metric : ONLY top-ratio Relevant patches survive; P_GT is recomputed from full logits")
#             print("  Display unit: percentage points; e.g. +3.20% means probability increased by 0.032")

#             for row in summary_rows:
#                 nfaith = row["nuisance_faithfulness_rate_delta_pgt_gt_0"]
#                 rfaith = row["relevant_faithfulness_rate_delta_pgt_lt_0"]
#                 print(
#                     "  remove {:>4.0f}% | mean dP_GT(remove R)={:+.2f}% [expect <0] | "
#                     "mean dP_GT(remove N)={:+.2f}% [expect >0] | R-faith={:.1%} | N-faith={:.1%}".format(
#                         100.0 * row["ratio"],
#                         row["mean_delta_pgt_remove_relevant_pct_points"],
#                         row["mean_delta_pgt_remove_nuisance_pct_points"],
#                         rfaith if math.isfinite(rfaith) else 0.0,
#                         nfaith if math.isfinite(nfaith) else 0.0,
#                     )
#                 )

#             print("\n  ===== Relevant-patch sufficiency curve =====")
#             for row in keep_summary_rows:
#                 marginal = row["mean_marginal_pgt_gain_from_previous_ratio_pct_points"]
#                 marginal_text = "N/A" if not math.isfinite(marginal) else "{:+.2f}%".format(marginal)
#                 print(
#                     "  keep R {:>4.0f}% only | mean P_GT={:>6.2f}% | dP_GT vs orig={:+.2f}% | "
#                     "marginal={} | GT-pred-rate={:.1%}".format(
#                         row["keep_ratio_pct"],
#                         row["mean_pgt_keep_relevant_pct"],
#                         row["mean_delta_pgt_vs_original_pct_points"],
#                         marginal_text,
#                         row["gt_prediction_rate_keep_relevant_only"]
#                         if math.isfinite(row["gt_prediction_rate_keep_relevant_only"]) else 0.0,
#                     )
#                 )

#             mean_min = self.last_visualization_summary["mean_min_keep_relevant_ratio_for_gt_prediction"]
#             if math.isfinite(mean_min):
#                 print(
#                     "  Mean smallest tested keep-R ratio that already predicts GT: {:.1f}%".format(
#                         100.0 * mean_min
#                     )
#                 )
#             else:
#                 print("  No tested keep-R ratio predicted GT for the analyzed samples.")

#         return paths

#     # Explicit alias for readability; same behavior as visualize_partition.
#     visualize_and_ablate = visualize_partition


#     # -------------------------------------------------------------------------
#     # Full validation R/N/I probability statistics
#     # -------------------------------------------------------------------------

#     @torch.no_grad()
#     def evaluate_rn_probability_statistics(
#         self,
#         student_model: nn.Module,
#         inputs: Union[Tensor, Iterable],
#         labels: Optional[Tensor] = None,
#         save_dir: str = "./triple_bank_vis",
#         ratios: Sequence[float] = (0.10, 0.20, 0.40, 0.60, 0.80, 1.00),
#         verbose: bool = True,
#     ) -> Dict[str, Any]:
#         """
#         Compute FULL-dataset statistics without changing visualize_partition().

#         Statistics
#         ----------
#         1) R/N/I patch ratios over ALL samples in ``inputs`` plus the R/N ratio.

#         2) For ratios such as [10,20,40,60,80,100]:

#            A. KEEP Relevant only
#               Keep the top-ratio Relevant patches ranked exactly as in
#               visualize_partition():

#                   relevant_strength = relevant_similarity - nuisance_similarity

#               All unkept Relevant patches plus ALL Nuisance/Irrelevant patches are zeroed at
#               layer4. The full model head is then run again and Softmax is recomputed.

#            B. REMOVE Nuisance
#               Remove the top-ratio Nuisance patches ranked exactly as in
#               visualize_partition():

#                   nuisance_strength = nuisance_similarity - relevant_similarity

#               All remaining patches are kept. The full model head is run again and
#               Softmax is recomputed.

#         For every intervention, the reported classes are:
#             GT: ground-truth class y
#             c*: strongest NON-GT class under the ORIGINAL unmasked logits

#                 c* = argmax_{c != y} logit_c

#         The same fixed c* is used at every intervention ratio, so probabilities are
#         directly comparable across ratios.

#         Files written to ``save_dir``
#         -----------------------------
#             full_val_rn_patch_ratio.csv
#             full_val_probability_per_image.csv
#             full_val_probability_summary.csv

#         Notes
#         -----
#         - ``max_images`` from visualize_partition() is intentionally irrelevant here:
#           this method scans the ENTIRE supplied loader/tensor.
#         - If an image contains no Relevant patches, KEEP-R statistics for that image
#           are recorded as NaN and excluded from KEEP-R means.
#         - If an image contains no Nuisance patches, REMOVE-N statistics for that image
#           are recorded as NaN and excluded from REMOVE-N means.
#         - With a real Irrelevant bank, KEEP-R 100% and REMOVE-N 100% are no longer
#           identical: KEEP-R removes N+I, whereas REMOVE-N keeps R+I.
#         """
#         self._check_discovered()
#         os.makedirs(save_dir, exist_ok=True)

#         ratios_used = sorted(set(float(r) for r in ratios))
#         if len(ratios_used) == 0 or any(r <= 0.0 or r > 1.0 for r in ratios_used):
#             raise ValueError("ratios must contain values in (0,1].")

#         original_training = student_model.training
#         student_model.eval()
#         device = next(student_model.parameters()).device

#         total_images = 0
#         total_patches = 0
#         total_relevant = 0
#         total_nuisance = 0
#         total_irrelevant = 0

#         per_image_records: List[Dict[str, Any]] = []
#         global_index = 0

#         def _finite_values(values: Sequence[float]) -> List[float]:
#             return [float(v) for v in values if math.isfinite(float(v))]

#         def _mean(values: Sequence[float]) -> float:
#             vals = _finite_values(values)
#             if len(vals) == 0:
#                 return float("nan")
#             return float(sum(vals) / len(vals))

#         def _std(values: Sequence[float]) -> float:
#             vals = _finite_values(values)
#             if len(vals) <= 1:
#                 return 0.0 if len(vals) == 1 else float("nan")
#             mu = sum(vals) / len(vals)
#             var = sum((v - mu) ** 2 for v in vals) / len(vals)
#             return float(math.sqrt(max(var, 0.0)))

#         for batch_inputs, batch_labels in self._batch_iterator(inputs, labels):
#             x = batch_inputs.to(device, non_blocking=True)
#             y = batch_labels.long().to(device, non_blocking=True)

#             part = self._stage2_triple_bank_partition(x)
#             base_logits, feat = _capture_layer4_and_forward(student_model, x)
#             base_prob = F.softmax(base_logits, dim=1)
#             pred = base_logits.argmax(dim=1)

#             # Fixed strongest non-GT competitor from ORIGINAL unmasked logits.
#             other_logits = base_logits.detach().clone()
#             row = torch.arange(base_logits.shape[0], device=device)
#             other_logits[row, y] = -torch.inf
#             competitor = other_logits.argmax(dim=1)

#             rel_mask_batch = part["relevant_mask"].to(device)
#             nui_mask_batch = part["nuisance_mask"].to(device)
#             irr_mask_batch = part["irrelevant_mask"].to(device)
#             rel_sim_batch = part["relevant_similarity"].to(device)
#             nui_sim_batch = part["nuisance_similarity"].to(device)
#             irr_sim_batch = part["irrelevant_similarity"].to(device)

#             b = int(x.shape[0])
#             hf, wf = int(feat.shape[-2]), int(feat.shape[-1])
#             num_spatial = hf * wf
#             if rel_mask_batch.shape != (b, num_spatial):
#                 raise RuntimeError(
#                     "Frozen assignment patch grid {} != current layer4 grid {}.".format(
#                         tuple(rel_mask_batch.shape), (b, num_spatial)
#                     )
#                 )

#             covered = rel_mask_batch | nui_mask_batch | irr_mask_batch
#             if not bool(covered.all()):
#                 raise RuntimeError("R/N/I masks do not cover all patches during statistics.")
#             if (
#                 bool((rel_mask_batch & nui_mask_batch).any())
#                 or bool((rel_mask_batch & irr_mask_batch).any())
#                 or bool((nui_mask_batch & irr_mask_batch).any())
#             ):
#                 raise RuntimeError("R/N/I masks overlap during full-dataset statistics.")

#             rel_count_batch = rel_mask_batch.sum(dim=1).long()
#             nui_count_batch = nui_mask_batch.sum(dim=1).long()
#             irr_count_batch = irr_mask_batch.sum(dim=1).long()

#             total_images += b
#             total_patches += int(b * num_spatial)
#             total_relevant += int(rel_count_batch.sum().item())
#             total_nuisance += int(nui_count_batch.sum().item())
#             total_irrelevant += int(irr_count_batch.sum().item())

#             rel_strength_batch = rel_sim_batch - torch.maximum(nui_sim_batch, irr_sim_batch)
#             nui_strength_batch = nui_sim_batch - torch.maximum(rel_sim_batch, irr_sim_batch)

#             batch_records: List[Dict[str, Any]] = []
#             for i in range(b):
#                 yi = int(y[i].item())
#                 ci = int(competitor[i].item())
#                 pi = int(pred[i].item())
#                 pgt = float(base_prob[i, yi].item())
#                 pcomp = float(base_prob[i, ci].item())
#                 nr = int(rel_count_batch[i].item())
#                 nn = int(nui_count_batch[i].item())
#                 ni = int(irr_count_batch[i].item())

#                 rec: Dict[str, Any] = {
#                     "sample_index": global_index + i,
#                     "gt": yi,
#                     "pred": pi,
#                     "competitor": ci,
#                     "correct": int(pi == yi),
#                     "num_patches": num_spatial,
#                     "num_relevant": nr,
#                     "num_nuisance": nn,
#                     "num_irrelevant": ni,
#                     "relevant_ratio": float(nr / num_spatial),
#                     "nuisance_ratio": float(nn / num_spatial),
#                     "irrelevant_ratio": float(ni / num_spatial),
#                     "relevant_to_nuisance_ratio": (
#                         float(nr / nn) if nn > 0 else float("inf")
#                     ),
#                     "orig_pgt": pgt,
#                     "orig_pgt_pct": 100.0 * pgt,
#                     "orig_pcomp": pcomp,
#                     "orig_pcomp_pct": 100.0 * pcomp,
#                     "orig_prob_gap": pgt - pcomp,
#                 }
#                 batch_records.append(rec)

#             # Batch intervention: only 2 forward passes per ratio, rather than
#             # 2 * batch_size forward passes per ratio.
#             for ratio in ratios_used:
#                 tag = self._ratio_tag(ratio)

#                 keep_rel_list: List[Tensor] = []
#                 remove_nui_list: List[Tensor] = []
#                 for i in range(b):
#                     keep_rel_list.append(
#                         self._top_fraction_mask(
#                             rel_mask_batch[i], rel_strength_batch[i], ratio
#                         )
#                     )
#                     remove_nui_list.append(
#                         self._top_fraction_mask(
#                             nui_mask_batch[i], nui_strength_batch[i], ratio
#                         )
#                     )

#                 keep_rel_mask = torch.stack(keep_rel_list, dim=0)
#                 remove_nui_mask = torch.stack(remove_nui_list, dim=0)

#                 # KEEP R ONLY: all non-kept patches are zeroed.
#                 remove_everything_except_kept_r = ~keep_rel_mask
#                 logits_keep_r = _forward_with_layer4_zero_mask(
#                     student_model,
#                     x,
#                     remove_everything_except_kept_r,
#                 )
#                 prob_keep_r = F.softmax(logits_keep_r, dim=1)

#                 # REMOVE N ONLY: selected N patches are zeroed, everything else stays.
#                 logits_remove_n = _forward_with_layer4_zero_mask(
#                     student_model,
#                     x,
#                     remove_nui_mask,
#                 )
#                 prob_remove_n = F.softmax(logits_remove_n, dim=1)

#                 for i in range(b):
#                     yi = int(y[i].item())
#                     ci = int(competitor[i].item())
#                     nr = int(rel_count_batch[i].item())
#                     nn = int(nui_count_batch[i].item())
#                     orig_pgt = float(base_prob[i, yi].item())
#                     orig_pcomp = float(base_prob[i, ci].item())

#                     # Match the existing visualization semantics: no R => KEEP-R N/A.
#                     if nr > 0:
#                         keep_num = int(keep_rel_mask[i].sum().item())
#                         keep_pgt = float(prob_keep_r[i, yi].item())
#                         keep_pcomp = float(prob_keep_r[i, ci].item())
#                         keep_pred = int(logits_keep_r[i].argmax().item())
#                     else:
#                         keep_num = 0
#                         keep_pgt = float("nan")
#                         keep_pcomp = float("nan")
#                         keep_pred = -1

#                     # Match the existing ablation semantics: no N => REMOVE-N N/A.
#                     if nn > 0:
#                         remove_num = int(remove_nui_mask[i].sum().item())
#                         remove_pgt = float(prob_remove_n[i, yi].item())
#                         remove_pcomp = float(prob_remove_n[i, ci].item())
#                         remove_pred = int(logits_remove_n[i].argmax().item())
#                     else:
#                         remove_num = 0
#                         remove_pgt = float("nan")
#                         remove_pcomp = float("nan")
#                         remove_pred = -1

#                     batch_records[i].update({
#                         f"keep_rel_num_{tag}pct": keep_num,
#                         f"keep_rel_pgt_{tag}pct": keep_pgt,
#                         f"keep_rel_pgt_{tag}pct_value": (
#                             100.0 * keep_pgt if math.isfinite(keep_pgt) else float("nan")
#                         ),
#                         f"keep_rel_pcomp_{tag}pct": keep_pcomp,
#                         f"keep_rel_pcomp_{tag}pct_value": (
#                             100.0 * keep_pcomp if math.isfinite(keep_pcomp) else float("nan")
#                         ),
#                         f"keep_rel_delta_pgt_{tag}pct": (
#                             keep_pgt - orig_pgt if math.isfinite(keep_pgt) else float("nan")
#                         ),
#                         f"keep_rel_delta_pcomp_{tag}pct": (
#                             keep_pcomp - orig_pcomp if math.isfinite(keep_pcomp) else float("nan")
#                         ),
#                         f"keep_rel_pred_{tag}pct": keep_pred,
#                         f"remove_nui_num_{tag}pct": remove_num,
#                         f"remove_nui_pgt_{tag}pct": remove_pgt,
#                         f"remove_nui_pgt_{tag}pct_value": (
#                             100.0 * remove_pgt if math.isfinite(remove_pgt) else float("nan")
#                         ),
#                         f"remove_nui_pcomp_{tag}pct": remove_pcomp,
#                         f"remove_nui_pcomp_{tag}pct_value": (
#                             100.0 * remove_pcomp if math.isfinite(remove_pcomp) else float("nan")
#                         ),
#                         f"remove_nui_delta_pgt_{tag}pct": (
#                             remove_pgt - orig_pgt if math.isfinite(remove_pgt) else float("nan")
#                         ),
#                         f"remove_nui_delta_pcomp_{tag}pct": (
#                             remove_pcomp - orig_pcomp if math.isfinite(remove_pcomp) else float("nan")
#                         ),
#                         f"remove_nui_pred_{tag}pct": remove_pred,
#                     })

#             per_image_records.extend(batch_records)
#             global_index += b

#         if original_training:
#             student_model.train()

#         if total_images == 0:
#             raise RuntimeError("No samples were found in the supplied inputs/loader.")
#         if total_patches <= 0:
#             raise RuntimeError("No layer4 patches were found.")

#         relevant_fraction = float(total_relevant / total_patches)
#         nuisance_fraction = float(total_nuisance / total_patches)
#         irrelevant_fraction = float(total_irrelevant / total_patches)
#         rn_ratio = (
#             float(total_relevant / total_nuisance)
#             if total_nuisance > 0 else float("inf")
#         )

#         mean_image_relevant_ratio = _mean([
#             float(r["relevant_ratio"]) for r in per_image_records
#         ])
#         mean_image_nuisance_ratio = _mean([
#             float(r["nuisance_ratio"]) for r in per_image_records
#         ])
#         mean_image_irrelevant_ratio = _mean([
#             float(r["irrelevant_ratio"]) for r in per_image_records
#         ])
#         mean_image_rn_ratio = _mean([
#             float(r["relevant_to_nuisance_ratio"])
#             for r in per_image_records
#             if math.isfinite(float(r["relevant_to_nuisance_ratio"]))
#         ])

#         patch_ratio_row: Dict[str, Any] = {
#             "num_images": total_images,
#             "num_patches": total_patches,
#             "num_relevant_patches": total_relevant,
#             "num_nuisance_patches": total_nuisance,
#             "num_irrelevant_patches": total_irrelevant,
#             "global_relevant_ratio": relevant_fraction,
#             "global_relevant_ratio_pct": 100.0 * relevant_fraction,
#             "global_nuisance_ratio": nuisance_fraction,
#             "global_nuisance_ratio_pct": 100.0 * nuisance_fraction,
#             "global_irrelevant_ratio": irrelevant_fraction,
#             "global_irrelevant_ratio_pct": 100.0 * irrelevant_fraction,
#             "global_relevant_to_nuisance_ratio": rn_ratio,
#             "mean_per_image_relevant_ratio": mean_image_relevant_ratio,
#             "mean_per_image_relevant_ratio_pct": 100.0 * mean_image_relevant_ratio,
#             "mean_per_image_nuisance_ratio": mean_image_nuisance_ratio,
#             "mean_per_image_nuisance_ratio_pct": 100.0 * mean_image_nuisance_ratio,
#             "mean_per_image_irrelevant_ratio": mean_image_irrelevant_ratio,
#             "mean_per_image_irrelevant_ratio_pct": 100.0 * mean_image_irrelevant_ratio,
#             "mean_per_image_relevant_to_nuisance_ratio": mean_image_rn_ratio,
#         }

#         # Build compact probability summary.
#         orig_pgt_values = [float(r["orig_pgt"]) for r in per_image_records]
#         orig_pcomp_values = [float(r["orig_pcomp"]) for r in per_image_records]
#         summary_rows: List[Dict[str, Any]] = [{
#             "operation": "original",
#             "ratio": 0.0,
#             "ratio_pct": 0.0,
#             "num_total_images": total_images,
#             "num_valid_images": total_images,
#             "mean_num_affected_patches": 0.0,
#             "mean_pgt": _mean(orig_pgt_values),
#             "std_pgt": _std(orig_pgt_values),
#             "mean_pgt_pct": 100.0 * _mean(orig_pgt_values),
#             "std_pgt_pct": 100.0 * _std(orig_pgt_values),
#             "mean_pcomp": _mean(orig_pcomp_values),
#             "std_pcomp": _std(orig_pcomp_values),
#             "mean_pcomp_pct": 100.0 * _mean(orig_pcomp_values),
#             "std_pcomp_pct": 100.0 * _std(orig_pcomp_values),
#             "mean_delta_pgt_vs_original": 0.0,
#             "mean_delta_pgt_vs_original_pct_points": 0.0,
#             "mean_delta_pcomp_vs_original": 0.0,
#             "mean_delta_pcomp_vs_original_pct_points": 0.0,
#         }]

#         for ratio in ratios_used:
#             tag = self._ratio_tag(ratio)

#             keep_valid = [
#                 r for r in per_image_records
#                 if math.isfinite(float(r[f"keep_rel_pgt_{tag}pct"]))
#             ]
#             keep_pgt = [float(r[f"keep_rel_pgt_{tag}pct"]) for r in keep_valid]
#             keep_pcomp = [float(r[f"keep_rel_pcomp_{tag}pct"]) for r in keep_valid]
#             keep_num = [float(r[f"keep_rel_num_{tag}pct"]) for r in keep_valid]
#             keep_dpgt = [
#                 float(r[f"keep_rel_delta_pgt_{tag}pct"]) for r in keep_valid
#             ]
#             keep_dpcomp = [
#                 float(r[f"keep_rel_delta_pcomp_{tag}pct"]) for r in keep_valid
#             ]

#             summary_rows.append({
#                 "operation": "keep_relevant_only",
#                 "ratio": ratio,
#                 "ratio_pct": 100.0 * ratio,
#                 "num_total_images": total_images,
#                 "num_valid_images": len(keep_valid),
#                 "mean_num_affected_patches": _mean(keep_num),
#                 "mean_pgt": _mean(keep_pgt),
#                 "std_pgt": _std(keep_pgt),
#                 "mean_pgt_pct": 100.0 * _mean(keep_pgt),
#                 "std_pgt_pct": 100.0 * _std(keep_pgt),
#                 "mean_pcomp": _mean(keep_pcomp),
#                 "std_pcomp": _std(keep_pcomp),
#                 "mean_pcomp_pct": 100.0 * _mean(keep_pcomp),
#                 "std_pcomp_pct": 100.0 * _std(keep_pcomp),
#                 "mean_delta_pgt_vs_original": _mean(keep_dpgt),
#                 "mean_delta_pgt_vs_original_pct_points": 100.0 * _mean(keep_dpgt),
#                 "mean_delta_pcomp_vs_original": _mean(keep_dpcomp),
#                 "mean_delta_pcomp_vs_original_pct_points": 100.0 * _mean(keep_dpcomp),
#             })

#             remove_valid = [
#                 r for r in per_image_records
#                 if math.isfinite(float(r[f"remove_nui_pgt_{tag}pct"]))
#             ]
#             remove_pgt = [float(r[f"remove_nui_pgt_{tag}pct"]) for r in remove_valid]
#             remove_pcomp = [float(r[f"remove_nui_pcomp_{tag}pct"]) for r in remove_valid]
#             remove_num = [float(r[f"remove_nui_num_{tag}pct"]) for r in remove_valid]
#             remove_dpgt = [
#                 float(r[f"remove_nui_delta_pgt_{tag}pct"]) for r in remove_valid
#             ]
#             remove_dpcomp = [
#                 float(r[f"remove_nui_delta_pcomp_{tag}pct"]) for r in remove_valid
#             ]

#             summary_rows.append({
#                 "operation": "remove_nuisance",
#                 "ratio": ratio,
#                 "ratio_pct": 100.0 * ratio,
#                 "num_total_images": total_images,
#                 "num_valid_images": len(remove_valid),
#                 "mean_num_affected_patches": _mean(remove_num),
#                 "mean_pgt": _mean(remove_pgt),
#                 "std_pgt": _std(remove_pgt),
#                 "mean_pgt_pct": 100.0 * _mean(remove_pgt),
#                 "std_pgt_pct": 100.0 * _std(remove_pgt),
#                 "mean_pcomp": _mean(remove_pcomp),
#                 "std_pcomp": _std(remove_pcomp),
#                 "mean_pcomp_pct": 100.0 * _mean(remove_pcomp),
#                 "std_pcomp_pct": 100.0 * _std(remove_pcomp),
#                 "mean_delta_pgt_vs_original": _mean(remove_dpgt),
#                 "mean_delta_pgt_vs_original_pct_points": 100.0 * _mean(remove_dpgt),
#                 "mean_delta_pcomp_vs_original": _mean(remove_dpcomp),
#                 "mean_delta_pcomp_vs_original_pct_points": 100.0 * _mean(remove_dpcomp),
#             })

#         patch_ratio_csv = os.path.join(save_dir, "full_val_rn_patch_ratio.csv")
#         with open(patch_ratio_csv, "w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=list(patch_ratio_row.keys()))
#             writer.writeheader()
#             writer.writerow(patch_ratio_row)

#         per_image_csv = os.path.join(save_dir, "full_val_probability_per_image.csv")
#         if per_image_records:
#             with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
#                 writer = csv.DictWriter(
#                     f, fieldnames=list(per_image_records[0].keys())
#                 )
#                 writer.writeheader()
#                 writer.writerows(per_image_records)

#         summary_csv = os.path.join(save_dir, "full_val_probability_summary.csv")
#         with open(summary_csv, "w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
#             writer.writeheader()
#             writer.writerows(summary_rows)

#         result: Dict[str, Any] = {
#             "patch_ratio": patch_ratio_row,
#             "probability_summary": summary_rows,
#             "per_image_records": per_image_records,
#             "patch_ratio_csv": patch_ratio_csv,
#             "per_image_csv": per_image_csv,
#             "summary_csv": summary_csv,
#             "ratios": ratios_used,
#         }
#         self.last_dataset_statistics = result

#         if verbose:
#             print("\n========== FULL validation R/N/I statistics ==========")
#             print("images={} | total patches={}".format(total_images, total_patches))
#             print(
#                 "Relevant: {} ({:.2f}%) | Nuisance: {} ({:.2f}%) | "
#                 "Irrelevant: {} ({:.2f}%) | R/N={:.6f}".format(
#                     total_relevant,
#                     100.0 * relevant_fraction,
#                     total_nuisance,
#                     100.0 * nuisance_fraction,
#                     total_irrelevant,
#                     100.0 * irrelevant_fraction,
#                     rn_ratio,
#                 )
#             )
#             print(
#                 "Original | mean P_GT={:.2f}% | mean P_c*={:.2f}%".format(
#                     100.0 * _mean(orig_pgt_values),
#                     100.0 * _mean(orig_pcomp_values),
#                 )
#             )
#             print("\n-- KEEP Relevant ONLY --")
#             for row_s in summary_rows:
#                 if row_s["operation"] != "keep_relevant_only":
#                     continue
#                 print(
#                     "keep R {:>3.0f}% | valid={:>4d}/{:<4d} | mean P_GT={:>6.2f}% | "
#                     "mean P_c*={:>6.2f}%".format(
#                         row_s["ratio_pct"],
#                         row_s["num_valid_images"],
#                         row_s["num_total_images"],
#                         row_s["mean_pgt_pct"],
#                         row_s["mean_pcomp_pct"],
#                     )
#                 )

#             print("\n-- REMOVE Nuisance --")
#             for row_s in summary_rows:
#                 if row_s["operation"] != "remove_nuisance":
#                     continue
#                 print(
#                     "remove N {:>3.0f}% | valid={:>4d}/{:<4d} | mean P_GT={:>6.2f}% | "
#                     "mean P_c*={:>6.2f}%".format(
#                         row_s["ratio_pct"],
#                         row_s["num_valid_images"],
#                         row_s["num_total_images"],
#                         row_s["mean_pgt_pct"],
#                         row_s["mean_pcomp_pct"],
#                     )
#                 )
#             print("\nSaved:")
#             print("  R/N/I ratio CSV   : {}".format(os.path.abspath(patch_ratio_csv)))
#             print("  per-image CSV     : {}".format(os.path.abspath(per_image_csv)))
#             print("  probability CSV   : {}".format(os.path.abspath(summary_csv)))
#             print("  c* is fixed per image from ORIGINAL unmasked strongest non-GT logit.")
#             print("=====================================================\n")

#         return result

#     # -------------------------------------------------------------------------
#     # Standard CAM comparison: frozen CE reference vs purified/current student
#     # -------------------------------------------------------------------------

#     @torch.no_grad()
#     def visualize_cam_comparison(
#         self,
#         student_model: nn.Module,
#         inputs: Union[Tensor, Iterable],
#         labels: Optional[Tensor] = None,
#         ce_model: Optional[nn.Module] = None,
#         save_dir: str = "./cam_ce_vs_purified",
#         max_images: int = 20,
#         mean: Optional[Sequence[float]] = None,
#         std: Optional[Sequence[float]] = None,
#         target_mode: str = "gt",
#         relu_cam: bool = True,
#         overlay_alpha: float = 0.45,
#         display: bool = False,
#         require_same_fc: bool = True,
#         fc_tolerance: float = 1e-7,
#     ) -> List[str]:
#         """
#         Compare standard CAMs of a CE/Stage-I reference model and the current
#         purified/student model while using ONE FIXED classifier direction.

#         The comparison is intentionally designed for the setting

#             fixed FC weights + trainable feature extractor.

#         For target class c and layer4 feature map F, standard CAM is

#             CAM_c(h,w) = sum_d w_c[d] * F[d,h,w].

#         IMPORTANT FAIRNESS RULE
#         -----------------------
#         The SAME classifier weight vector w_c from the CE reference model is used
#         to compute BOTH CAMs:

#             CAM_CE       = w_c^T F_CE
#             CAM_student  = w_c^T F_student
#             Delta_CAM    = CAM_student - CAM_CE

#         Therefore, if the FC layer is frozen, any CAM redistribution is caused by
#         changes in the feature map rather than a changed classifier direction.

#         Which CE model is used?
#         -----------------------
#         1) If ce_model is explicitly supplied, that model is used.
#         2) If ce_model is None, self._assignment_model is used. discover() already
#            stores an exact frozen copy of the model used for Stage-I discovery, so
#            this is normally the original CE model if discover() was called before
#            purification training.

#         target_mode:
#             "gt"           : visualize CAM of the ground-truth class (recommended)
#             "ce_pred"      : visualize CAM of the CE model's predicted class
#             "student_pred" : visualize CAM of the current student's predicted class

#         Outputs:
#             - one 4-panel PNG per image:
#                 Original | CE CAM | Purified CAM | Delta CAM
#             - cam_comparison.csv with probabilities and CAM-change statistics

#         Notes:
#             - CAM display maps are min-max normalized only for visualization.
#               The raw CAM tensors are used to compute Delta_CAM/statistics.
#             - The prediction probabilities are obtained from each model's FULL
#               logits followed by Softmax; CAM itself does not alter the logits.
#         """
#         import numpy as np
#         import matplotlib.pyplot as plt

#         if target_mode not in ("gt", "ce_pred", "student_pred"):
#             raise ValueError(
#                 "target_mode must be one of {'gt', 'ce_pred', 'student_pred'}."
#             )
#         if not (0.0 <= float(overlay_alpha) <= 1.0):
#             raise ValueError("overlay_alpha must be in [0,1].")
#         if float(fc_tolerance) < 0.0:
#             raise ValueError("fc_tolerance must be >= 0.")

#         # Prefer an explicitly supplied CE model; otherwise use the exact Stage-I
#         # snapshot stored by discover()/load_discovery().
#         if ce_model is None:
#             if self._assignment_model is None:
#                 raise RuntimeError(
#                     "No CE reference is available. Either run discover() first or "
#                     "pass ce_model=<your CE-trained model>."
#                 )
#             ce_reference = self._assignment_model
#             ce_source = "stored Stage-I/CE snapshot"
#         else:
#             ce_reference = ce_model
#             ce_source = "explicit ce_model"

#         # Preserve caller model modes.
#         student_training = student_model.training
#         ce_training = ce_reference.training
#         student_model.eval()
#         ce_reference.eval()

#         try:
#             student_device = next(student_model.parameters()).device
#             ce_device = next(ce_reference.parameters()).device
#         except StopIteration:
#             raise ValueError("student_model and ce_model must contain parameters.")

#         student_fc = _get_classifier(student_model)
#         ce_fc = _get_classifier(ce_reference)

#         if tuple(student_fc.weight.shape) != tuple(ce_fc.weight.shape):
#             raise ValueError(
#                 "Student/CE classifier shapes differ: {} vs {}.".format(
#                     tuple(student_fc.weight.shape), tuple(ce_fc.weight.shape)
#                 )
#             )

#         # Verify that the premise 'FC is fixed' is actually true.
#         weight_diff = float(
#             (student_fc.weight.detach().float().cpu()
#              - ce_fc.weight.detach().float().cpu()).abs().max().item()
#         )
#         if student_fc.bias is None and ce_fc.bias is None:
#             bias_diff = 0.0
#         elif student_fc.bias is not None and ce_fc.bias is not None:
#             bias_diff = float(
#                 (student_fc.bias.detach().float().cpu()
#                  - ce_fc.bias.detach().float().cpu()).abs().max().item()
#             )
#         else:
#             bias_diff = float("inf")

#         fc_max_diff = max(weight_diff, bias_diff)
#         if require_same_fc and fc_max_diff > float(fc_tolerance):
#             raise RuntimeError(
#                 "FC layers are not identical (max abs diff={:.6e}, tolerance={:.6e}). "
#                 "For a strict fixed-FC CAM comparison, freeze the classifier or pass "
#                 "the matching CE model. Set require_same_fc=False only if you knowingly "
#                 "want a non-isolated comparison.".format(fc_max_diff, float(fc_tolerance))
#             )

#         os.makedirs(save_dir, exist_ok=True)
#         paths: List[str] = []
#         records: List[Dict[str, Any]] = []
#         global_index = 0

#         # Fixed reference classifier weight used for BOTH CAMs.
#         W_ref_cpu = ce_fc.weight.detach().float().cpu()

#         def _to_image(t: Tensor):
#             t = t.detach().float().cpu()
#             if mean is not None and std is not None:
#                 mean_t = torch.as_tensor(mean, dtype=t.dtype).view(-1, 1, 1)
#                 std_t = torch.as_tensor(std, dtype=t.dtype).view(-1, 1, 1)
#                 if mean_t.shape[0] != t.shape[0] or std_t.shape[0] != t.shape[0]:
#                     raise ValueError("mean/std channel count must match input channels.")
#                 t = t * std_t + mean_t
#             elif float(t.min().item()) < 0.0 or float(t.max().item()) > 1.0:
#                 lo = t.amin(dim=(1, 2), keepdim=True)
#                 hi = t.amax(dim=(1, 2), keepdim=True)
#                 t = (t - lo) / (hi - lo).clamp_min(1e-8)
#             t = t.clamp(0.0, 1.0)
#             if t.shape[0] == 1:
#                 return t[0].numpy()
#             if t.shape[0] >= 3:
#                 return t[:3].permute(1, 2, 0).numpy()
#             if t.shape[0] == 2:
#                 z = torch.zeros_like(t[:1])
#                 return torch.cat([t, z], dim=0).permute(1, 2, 0).numpy()
#             raise ValueError("Unsupported input channel count: {}".format(int(t.shape[0])))

#         def _minmax_cam(cam_2d: Tensor) -> Tensor:
#             x = cam_2d.detach().float()
#             lo = x.min()
#             hi = x.max()
#             return (x - lo) / (hi - lo).clamp_min(1e-8)

#         def _upsample(cam_2d: Tensor, h: int, w: int) -> np.ndarray:
#             x = cam_2d.detach().float().view(1, 1, *cam_2d.shape)
#             x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
#             return x[0, 0].cpu().numpy()

#         # The class method already supports both Tensor and DataLoader inputs.
#         for batch_inputs, batch_labels in self._batch_iterator(inputs, labels):
#             if max_images is not None and max_images > 0 and global_index >= max_images:
#                 break

#             batch_labels = batch_labels.long()
#             remaining = None
#             if max_images is not None and max_images > 0:
#                 remaining = max_images - global_index
#             if remaining is not None and remaining <= 0:
#                 break
#             if remaining is not None and batch_inputs.shape[0] > remaining:
#                 batch_inputs = batch_inputs[:remaining]
#                 batch_labels = batch_labels[:remaining]

#             # Run the two models in their own devices.
#             x_student = batch_inputs.to(student_device, non_blocking=True)
#             x_ce = batch_inputs.to(ce_device, non_blocking=True)
#             y_student = batch_labels.to(student_device, non_blocking=True)
#             y_ce = batch_labels.to(ce_device, non_blocking=True)

#             student_logits, student_features = _capture_layer4_and_forward(
#                 student_model, x_student
#             )
#             ce_logits, ce_features = _capture_layer4_and_forward(
#                 ce_reference, x_ce
#             )

#             if student_features.shape[1:] != ce_features.shape[1:]:
#                 raise ValueError(
#                     "Student/CE layer4 feature shapes differ: {} vs {}. Standard CAM "
#                     "comparison requires the same layer4 channel/spatial layout.".format(
#                         tuple(student_features.shape[1:]), tuple(ce_features.shape[1:])
#                     )
#                 )
#             if student_features.shape[1] != ce_fc.weight.shape[1]:
#                 raise ValueError(
#                     "layer4 channel dim {} != CE classifier input dim {}. This CAM "
#                     "function assumes GAP(layer4) -> linear classifier.".format(
#                         student_features.shape[1], ce_fc.weight.shape[1]
#                     )
#                 )

#             student_prob = F.softmax(student_logits, dim=1)
#             ce_prob = F.softmax(ce_logits, dim=1)
#             student_pred = student_logits.argmax(dim=1)
#             ce_pred = ce_logits.argmax(dim=1)

#             if target_mode == "gt":
#                 target_cpu = batch_labels.long().cpu()
#             elif target_mode == "ce_pred":
#                 target_cpu = ce_pred.detach().cpu().long()
#             else:
#                 target_cpu = student_pred.detach().cpu().long()

#             # SAME W_ref for both CAMs. Move only the selected vectors to each device.
#             w_selected_cpu = W_ref_cpu.index_select(0, target_cpu)
#             w_student = w_selected_cpu.to(
#                 device=student_features.device, dtype=student_features.dtype
#             )
#             w_ce = w_selected_cpu.to(
#                 device=ce_features.device, dtype=ce_features.dtype
#             )

#             cam_student_raw = torch.einsum("bd,bdhw->bhw", w_student, student_features)
#             cam_ce_raw = torch.einsum("bd,bdhw->bhw", w_ce, ce_features)

#             if relu_cam:
#                 cam_student_show_base = cam_student_raw.clamp_min(0.0)
#                 cam_ce_show_base = cam_ce_raw.clamp_min(0.0)
#             else:
#                 cam_student_show_base = cam_student_raw
#                 cam_ce_show_base = cam_ce_raw

#             for i in range(batch_inputs.shape[0]):
#                 yi = int(batch_labels[i].item())
#                 target_i = int(target_cpu[i].item())
#                 ce_pred_i = int(ce_pred[i].item())
#                 student_pred_i = int(student_pred[i].item())

#                 ce_prob_i = ce_prob[i].detach().cpu()
#                 student_prob_i = student_prob[i].detach().cpu()
#                 ce_pgt = float(ce_prob_i[yi].item())
#                 student_pgt = float(student_prob_i[yi].item())
#                 ce_ptarget = float(ce_prob_i[target_i].item())
#                 student_ptarget = float(student_prob_i[target_i].item())

#                 ce_raw = cam_ce_raw[i].detach().float().cpu()
#                 student_raw = cam_student_raw[i].detach().float().cpu()
#                 delta_raw = student_raw - ce_raw

#                 ce_show = _minmax_cam(cam_ce_show_base[i].detach().cpu())
#                 student_show = _minmax_cam(cam_student_show_base[i].detach().cpu())

#                 # Signed delta is normalized symmetrically only for display.
#                 delta_absmax = delta_raw.abs().max().clamp_min(1e-8)
#                 delta_show = (delta_raw / delta_absmax).clamp(-1.0, 1.0)

#                 H = int(batch_inputs.shape[-2])
#                 Wimg = int(batch_inputs.shape[-1])
#                 ce_up = _upsample(ce_show, H, Wimg)
#                 student_up = _upsample(student_show, H, Wimg)
#                 delta_up = _upsample(delta_show, H, Wimg)
#                 img = _to_image(batch_inputs[i])

#                 # Quantitative CAM change statistics on RAW CAMs.
#                 ce_flat = ce_raw.flatten()
#                 student_flat = student_raw.flatten()
#                 denom = ce_flat.norm(p=2) * student_flat.norm(p=2)
#                 if float(denom.item()) > self.eps:
#                     cam_cos = float((ce_flat @ student_flat / denom).item())
#                 else:
#                     cam_cos = float("nan")
#                 cam_mean_abs_change = float(delta_raw.abs().mean().item())
#                 cam_mean_signed_change = float(delta_raw.mean().item())

#                 fig, axes = plt.subplots(1, 4, figsize=(22.0, 5.5))

#                 axes[0].imshow(img, cmap="gray" if img.ndim == 2 else None)
#                 axes[0].set_title(
#                     "Original\nGT={} | target={}\nCE pred={} P_GT={:.2f}%\nStudent pred={} P_GT={:.2f}%".format(
#                         yi, target_i,
#                         ce_pred_i, 100.0 * ce_pgt,
#                         student_pred_i, 100.0 * student_pgt,
#                     )
#                 )
#                 axes[0].axis("off")

#                 axes[1].imshow(img, cmap="gray" if img.ndim == 2 else None)
#                 axes[1].imshow(ce_up, cmap="jet", alpha=float(overlay_alpha), vmin=0.0, vmax=1.0)
#                 axes[1].set_title(
#                     "CE / Stage-I CAM\nP(target)={:.2f}%\nfixed w_{}".format(
#                         100.0 * ce_ptarget, target_i
#                     )
#                 )
#                 axes[1].axis("off")

#                 axes[2].imshow(img, cmap="gray" if img.ndim == 2 else None)
#                 axes[2].imshow(student_up, cmap="jet", alpha=float(overlay_alpha), vmin=0.0, vmax=1.0)
#                 axes[2].set_title(
#                     "Purified / Current CAM\nP(target)={:.2f}%\nSAME fixed w_{}".format(
#                         100.0 * student_ptarget, target_i
#                     )
#                 )
#                 axes[2].axis("off")

#                 # Delta: positive means the current feature map contributes more to
#                 # the fixed target-class direction; negative means less.
#                 axes[3].imshow(img, cmap="gray" if img.ndim == 2 else None)
#                 axes[3].imshow(
#                     delta_up, cmap="bwr", alpha=float(overlay_alpha), vmin=-1.0, vmax=1.0
#                 )
#                 axes[3].set_title(
#                     "Delta CAM = Current - CE\nred: increased | blue: decreased\nCAM cos={:.3f} | FC diff={:.2e}".format(
#                         cam_cos if math.isfinite(cam_cos) else float("nan"),
#                         fc_max_diff,
#                     )
#                 )
#                 axes[3].axis("off")

#                 fig.suptitle(
#                     "Fixed-FC CAM comparison | reference={} | target_mode={}".format(
#                         ce_source, target_mode
#                     ),
#                     fontsize=12,
#                 )
#                 fig.tight_layout()

#                 path = os.path.join(
#                     save_dir, "cam_compare_{:05d}.png".format(global_index)
#                 )
#                 fig.savefig(path, dpi=180, bbox_inches="tight")
#                 if display:
#                     try:
#                         plt.show(block=False)
#                         plt.pause(0.001)
#                     except Exception:
#                         pass
#                 plt.close(fig)

#                 records.append({
#                     "index": global_index,
#                     "gt": yi,
#                     "target_class": target_i,
#                     "target_mode": target_mode,
#                     "ce_pred": ce_pred_i,
#                     "student_pred": student_pred_i,
#                     "ce_pgt": ce_pgt,
#                     "ce_pgt_pct": 100.0 * ce_pgt,
#                     "student_pgt": student_pgt,
#                     "student_pgt_pct": 100.0 * student_pgt,
#                     "delta_pgt": student_pgt - ce_pgt,
#                     "delta_pgt_pct_points": 100.0 * (student_pgt - ce_pgt),
#                     "ce_ptarget": ce_ptarget,
#                     "ce_ptarget_pct": 100.0 * ce_ptarget,
#                     "student_ptarget": student_ptarget,
#                     "student_ptarget_pct": 100.0 * student_ptarget,
#                     "delta_ptarget": student_ptarget - ce_ptarget,
#                     "delta_ptarget_pct_points": 100.0 * (student_ptarget - ce_ptarget),
#                     "cam_cosine_similarity_raw": cam_cos,
#                     "cam_mean_abs_change_raw": cam_mean_abs_change,
#                     "cam_mean_signed_change_raw": cam_mean_signed_change,
#                     "fc_weight_max_abs_diff": weight_diff,
#                     "fc_bias_max_abs_diff": bias_diff,
#                     "fc_max_abs_diff": fc_max_diff,
#                     "ce_reference_source": ce_source,
#                     "image_path": path,
#                 })
#                 paths.append(path)
#                 global_index += 1

#                 if max_images is not None and max_images > 0 and global_index >= max_images:
#                     break

#             if max_images is not None and max_images > 0 and global_index >= max_images:
#                 break

#         csv_path = os.path.join(save_dir, "cam_comparison.csv")
#         if records:
#             with open(csv_path, "w", newline="", encoding="utf-8") as f:
#                 writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
#                 writer.writeheader()
#                 writer.writerows(records)

#         # Restore caller model modes.
#         if student_training:
#             student_model.train()
#         if ce_training:
#             ce_reference.train()

#         print("[CAM comparison] saved {} image(s) to: {}".format(
#             len(paths), os.path.abspath(save_dir)
#         ))
#         print("  CE reference     : {}".format(ce_source))
#         print("  target_mode      : {}".format(target_mode))
#         print("  fixed FC max diff: {:.6e}".format(fc_max_diff))
#         print("  SAME CE classifier weights are used to compute BOTH CAMs.")
#         if records:
#             print("  stats CSV        : {}".format(os.path.abspath(csv_path)))

#         return paths

#     # -------------------------------------------------------------------------
#     # Additional visualization ablations (visualization-only; training unchanged)
#     # -------------------------------------------------------------------------

#     @torch.no_grad()
#     def visualize_progressive_nuisance_cam(
#         self,
#         student_model: nn.Module,
#         inputs: Union[Tensor, Iterable],
#         labels: Optional[Tensor] = None,
#         save_dir: str = "./cam_progressive_nuisance_removal",
#         max_images: int = 20,
#         removal_ratios: Sequence[float] = (0.0, 0.20, 0.40, 0.60, 0.80, 1.00),
#         mean: Optional[Sequence[float]] = None,
#         std: Optional[Sequence[float]] = None,
#         target_mode: str = "gt",
#         relu_cam: bool = True,
#         overlay_alpha: float = 0.45,
#         display: bool = False,
#     ) -> List[str]:
#         """
#         Visualize a dose-response CAM trajectory while progressively removing
#         Nuisance patches from the CURRENT ``student_model`` layer4 representation.

#         IMPORTANT
#         ---------
#         This method is visualization-only. It does NOT modify discovery, prototype
#         banks, Stage-II assignment, losses, model parameters, or any existing method.

#         Nuisance patches are exactly those returned by the existing frozen-bank
#         Stage-II partition. Within each image they are ranked in the same way used by
#         ``evaluate_rn_probability_statistics``:

#             nuisance_strength = nuisance_similarity - relevant_similarity

#         For each removal ratio q, the top ceil(q * #N) Nuisance patches are set to zero
#         at layer4, then the remainder of the model is run normally. CAM is computed
#         with the SAME current classifier direction before and after intervention:

#             CAM_c(h,w) = w_c^T F(h,w)

#         target_mode:
#             "gt"         : ground-truth class CAM (recommended)
#             "pred"       : original/current model predicted-class CAM
#             "competitor" : strongest non-GT class under original unmasked logits

#         Output per image:
#             Original | Remove N 0% | 20% | 40% | 60% | 80% | 100%

#         A CSV ``progressive_nuisance_cam.csv`` is also written with P_GT, P_target,
#         prediction and number of removed Nuisance patches at every ratio.
#         """
#         import numpy as np
#         import matplotlib.pyplot as plt

#         self._check_discovered()
#         if target_mode not in ("gt", "pred", "competitor"):
#             raise ValueError("target_mode must be one of {'gt', 'pred', 'competitor'}.")
#         if not (0.0 <= float(overlay_alpha) <= 1.0):
#             raise ValueError("overlay_alpha must be in [0,1].")

#         ratios: List[float] = []
#         seen = set()
#         for q in removal_ratios:
#             qf = float(q)
#             if not (0.0 <= qf <= 1.0):
#                 raise ValueError("Every removal ratio must be in [0,1].")
#             key = round(qf, 12)
#             if key not in seen:
#                 ratios.append(qf)
#                 seen.add(key)
#         if len(ratios) == 0:
#             raise ValueError("removal_ratios cannot be empty.")

#         try:
#             device = next(student_model.parameters()).device
#         except StopIteration:
#             raise ValueError("student_model must contain parameters.")

#         classifier = _get_classifier(student_model)
#         original_training = student_model.training
#         student_model.eval()
#         os.makedirs(save_dir, exist_ok=True)

#         paths: List[str] = []
#         records: List[Dict[str, Any]] = []
#         global_index = 0

#         def _to_image(t: Tensor):
#             t = t.detach().float().cpu()
#             if mean is not None and std is not None:
#                 mean_t = torch.as_tensor(mean, dtype=t.dtype).view(-1, 1, 1)
#                 std_t = torch.as_tensor(std, dtype=t.dtype).view(-1, 1, 1)
#                 if mean_t.shape[0] != t.shape[0] or std_t.shape[0] != t.shape[0]:
#                     raise ValueError("mean/std channel count must match input channels.")
#                 t = t * std_t + mean_t
#             elif float(t.min().item()) < 0.0 or float(t.max().item()) > 1.0:
#                 lo = t.amin(dim=(1, 2), keepdim=True)
#                 hi = t.amax(dim=(1, 2), keepdim=True)
#                 t = (t - lo) / (hi - lo).clamp_min(1e-8)
#             t = t.clamp(0.0, 1.0)
#             if t.shape[0] == 1:
#                 return t[0].numpy()
#             if t.shape[0] >= 3:
#                 return t[:3].permute(1, 2, 0).numpy()
#             if t.shape[0] == 2:
#                 z = torch.zeros_like(t[:1])
#                 return torch.cat([t, z], dim=0).permute(1, 2, 0).numpy()
#             raise ValueError("Unsupported input channel count: {}".format(int(t.shape[0])))

#         def _cam_show(cam: Tensor, out_h: int, out_w: int) -> np.ndarray:
#             x = cam.detach().float()
#             if relu_cam:
#                 x = x.clamp_min(0.0)
#             x = F.interpolate(
#                 x.view(1, 1, *x.shape[-2:]),
#                 size=(out_h, out_w),
#                 mode="bilinear",
#                 align_corners=False,
#             )[0, 0]
#             lo = x.min()
#             hi = x.max()
#             x = (x - lo) / (hi - lo).clamp_min(1e-8)
#             return x.cpu().numpy()

#         def _forward_with_mask_and_capture(x: Tensor, remove_mask: Tensor) -> Tuple[Tensor, Tensor]:
#             holder: Dict[str, Tensor] = {}

#             def _hook(_module, _inputs, output):
#                 if not torch.is_tensor(output) or output.ndim != 4:
#                     raise TypeError("layer4 output must be [B,D,H,W].")
#                 b, _, hf, wf = output.shape
#                 if remove_mask.shape != (b, hf * wf):
#                     raise ValueError(
#                         "remove_mask shape {} incompatible with layer4 {}x{}.".format(
#                             tuple(remove_mask.shape), hf, wf
#                         )
#                     )
#                 m = remove_mask.to(output.device).view(b, 1, hf, wf).to(output.dtype)
#                 modified = output * (1.0 - m)
#                 holder["features"] = modified
#                 return modified

#             handle = _get_layer4(student_model).register_forward_hook(_hook)
#             try:
#                 output = student_model(x)
#             finally:
#                 handle.remove()
#             if "features" not in holder:
#                 raise RuntimeError("Failed to capture masked layer4 output.")
#             return _extract_logits(output), holder["features"]

#         for batch_inputs, batch_labels in self._batch_iterator(inputs, labels):
#             if max_images is not None and int(max_images) > 0 and len(paths) >= int(max_images):
#                 break

#             x = batch_inputs.to(device, non_blocking=True)
#             y = batch_labels.long().to(device, non_blocking=True)

#             part = self._stage2_triple_bank_partition(x)
#             nuisance_mask = part["nuisance_mask"].to(device)
#             nuisance_strength = (
#                 part["nuisance_similarity"] - part["relevant_similarity"]
#             ).to(device)

#             base_logits, base_features = _capture_layer4_and_forward(student_model, x)
#             base_prob = F.softmax(base_logits, dim=1)
#             base_pred = base_logits.argmax(dim=1)

#             non_gt_logits = base_logits.detach().clone()
#             row = torch.arange(base_logits.shape[0], device=device)
#             non_gt_logits[row, y] = -torch.inf
#             competitor = non_gt_logits.argmax(dim=1)

#             if target_mode == "gt":
#                 target = y
#             elif target_mode == "pred":
#                 target = base_pred
#             else:
#                 target = competitor

#             w_target = classifier.weight.index_select(0, target).to(
#                 device=base_features.device, dtype=base_features.dtype
#             )

#             ratio_outputs: Dict[float, Tuple[Tensor, Tensor, Tensor]] = {}
#             for q in ratios:
#                 remove_rows: List[Tensor] = []
#                 for i in range(x.shape[0]):
#                     remove_rows.append(
#                         self._top_fraction_mask(
#                             nuisance_mask[i], nuisance_strength[i], q
#                         )
#                     )
#                 remove_mask = torch.stack(remove_rows, dim=0).to(device)

#                 if q <= 0.0 or not bool(remove_mask.any()):
#                     logits_q = base_logits
#                     features_q = base_features
#                 else:
#                     logits_q, features_q = _forward_with_mask_and_capture(x, remove_mask)

#                 cam_q = torch.einsum("bd,bdhw->bhw", w_target, features_q)
#                 ratio_outputs[q] = (logits_q, cam_q, remove_mask)

#             for i in range(x.shape[0]):
#                 if max_images is not None and int(max_images) > 0 and len(paths) >= int(max_images):
#                     break

#                 image_np = _to_image(batch_inputs[i])
#                 H, W = image_np.shape[:2]
#                 yi = int(y[i].item())
#                 target_i = int(target[i].item())
#                 pred_i = int(base_pred[i].item())
#                 comp_i = int(competitor[i].item())
#                 num_n = int(nuisance_mask[i].sum().item())

#                 fig, axes = plt.subplots(
#                     1,
#                     1 + len(ratios),
#                     figsize=(3.5 * (1 + len(ratios)), 3.6),
#                 )
#                 axes = np.asarray(axes).reshape(-1)
#                 axes[0].imshow(image_np, cmap="gray" if image_np.ndim == 2 else None)
#                 axes[0].set_title("Original\nGT={} Pred={}".format(yi, pred_i))
#                 axes[0].axis("off")

#                 for j, q in enumerate(ratios, start=1):
#                     logits_q, cam_q, remove_mask_q = ratio_outputs[q]
#                     prob_q = F.softmax(logits_q[i], dim=0)
#                     pred_q = int(logits_q[i].argmax().item())
#                     pgt_q = float(prob_q[yi].item())
#                     ptarget_q = float(prob_q[target_i].item())
#                     removed_q = int(remove_mask_q[i].sum().item())

#                     axes[j].imshow(image_np, cmap="gray" if image_np.ndim == 2 else None)
#                     axes[j].imshow(
#                         _cam_show(cam_q[i], H, W),
#                         cmap="jet",
#                         alpha=float(overlay_alpha),
#                         vmin=0.0,
#                         vmax=1.0,
#                     )
#                     axes[j].set_title(
#                         "Remove N {:g}%\nP_GT={:.1f}%".format(100.0 * q, 100.0 * pgt_q)
#                     )
#                     axes[j].axis("off")

#                     records.append({
#                         "sample_index": global_index + i,
#                         "gt": yi,
#                         "original_pred": pred_i,
#                         "competitor": comp_i,
#                         "target_class": target_i,
#                         "target_mode": target_mode,
#                         "remove_nuisance_ratio": q,
#                         "remove_nuisance_ratio_pct": 100.0 * q,
#                         "num_nuisance": num_n,
#                         "num_removed_nuisance": removed_q,
#                         "pred_after": pred_q,
#                         "pgt_after": pgt_q,
#                         "ptarget_after": ptarget_q,
#                         "original_pgt": float(base_prob[i, yi].item()),
#                     })

#                 fig.suptitle(
#                     "Progressive Nuisance removal | target={} | N={}".format(
#                         target_i, num_n
#                     ),
#                     fontsize=11,
#                 )
#                 fig.tight_layout()
#                 path = os.path.join(save_dir, "progressive_n_{:05d}.png".format(global_index + i))
#                 fig.savefig(path, dpi=180, bbox_inches="tight")
#                 if display:
#                     plt.show()
#                 plt.close(fig)
#                 paths.append(path)

#             global_index += int(batch_inputs.shape[0])

#         csv_path = os.path.join(save_dir, "progressive_nuisance_cam.csv")
#         if records:
#             with open(csv_path, "w", newline="", encoding="utf-8") as f:
#                 writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
#                 writer.writeheader()
#                 writer.writerows(records)

#         if original_training:
#             student_model.train()

#         print("[Progressive N CAM] saved {} image(s) to: {}".format(
#             len(paths), os.path.abspath(save_dir)
#         ))
#         if records:
#             print("  stats CSV: {}".format(os.path.abspath(csv_path)))
#         return paths

#     @torch.no_grad()
#     def visualize_remove_r_vs_n_cam(
#         self,
#         student_model: nn.Module,
#         inputs: Union[Tensor, Iterable],
#         labels: Optional[Tensor] = None,
#         save_dir: str = "./cam_remove_r_vs_n",
#         max_images: int = 20,
#         mean: Optional[Sequence[float]] = None,
#         std: Optional[Sequence[float]] = None,
#         target_mode: str = "gt",
#         relu_cam: bool = True,
#         overlay_alpha: float = 0.45,
#         display: bool = False,
#     ) -> List[str]:
#         """
#         Counterfactual visual comparison using the SAME current model and SAME CAM
#         classifier direction:

#             Original | Original CAM | Remove ALL Relevant CAM | Remove ALL Nuisance CAM

#         R/N masks come directly from the existing frozen-bank Stage-II partition.
#         Only a layer4 zero-mask intervention is performed for visualization; no source
#         training/discovery logic or model parameter is changed.

#         target_mode:
#             "gt"         : GT CAM (recommended)
#             "pred"       : original predicted-class CAM
#             "competitor" : strongest non-GT class CAM

#         A CSV ``remove_r_vs_n_cam.csv`` is saved with P_GT/P_target and predictions.
#         """
#         import numpy as np
#         import matplotlib.pyplot as plt

#         self._check_discovered()
#         if target_mode not in ("gt", "pred", "competitor"):
#             raise ValueError("target_mode must be one of {'gt', 'pred', 'competitor'}.")
#         if not (0.0 <= float(overlay_alpha) <= 1.0):
#             raise ValueError("overlay_alpha must be in [0,1].")

#         try:
#             device = next(student_model.parameters()).device
#         except StopIteration:
#             raise ValueError("student_model must contain parameters.")

#         classifier = _get_classifier(student_model)
#         original_training = student_model.training
#         student_model.eval()
#         os.makedirs(save_dir, exist_ok=True)
#         paths: List[str] = []
#         records: List[Dict[str, Any]] = []
#         global_index = 0

#         def _to_image(t: Tensor):
#             t = t.detach().float().cpu()
#             if mean is not None and std is not None:
#                 mean_t = torch.as_tensor(mean, dtype=t.dtype).view(-1, 1, 1)
#                 std_t = torch.as_tensor(std, dtype=t.dtype).view(-1, 1, 1)
#                 if mean_t.shape[0] != t.shape[0] or std_t.shape[0] != t.shape[0]:
#                     raise ValueError("mean/std channel count must match input channels.")
#                 t = t * std_t + mean_t
#             elif float(t.min().item()) < 0.0 or float(t.max().item()) > 1.0:
#                 lo = t.amin(dim=(1, 2), keepdim=True)
#                 hi = t.amax(dim=(1, 2), keepdim=True)
#                 t = (t - lo) / (hi - lo).clamp_min(1e-8)
#             t = t.clamp(0.0, 1.0)
#             if t.shape[0] == 1:
#                 return t[0].numpy()
#             if t.shape[0] >= 3:
#                 return t[:3].permute(1, 2, 0).numpy()
#             if t.shape[0] == 2:
#                 z = torch.zeros_like(t[:1])
#                 return torch.cat([t, z], dim=0).permute(1, 2, 0).numpy()
#             raise ValueError("Unsupported input channel count: {}".format(int(t.shape[0])))

#         def _cam_show(cam: Tensor, out_h: int, out_w: int) -> np.ndarray:
#             x = cam.detach().float()
#             if relu_cam:
#                 x = x.clamp_min(0.0)
#             x = F.interpolate(
#                 x.view(1, 1, *x.shape[-2:]),
#                 size=(out_h, out_w),
#                 mode="bilinear",
#                 align_corners=False,
#             )[0, 0]
#             lo = x.min()
#             hi = x.max()
#             x = (x - lo) / (hi - lo).clamp_min(1e-8)
#             return x.cpu().numpy()

#         def _forward_with_mask_and_capture(x: Tensor, remove_mask: Tensor) -> Tuple[Tensor, Tensor]:
#             holder: Dict[str, Tensor] = {}

#             def _hook(_module, _inputs, output):
#                 if not torch.is_tensor(output) or output.ndim != 4:
#                     raise TypeError("layer4 output must be [B,D,H,W].")
#                 b, _, hf, wf = output.shape
#                 if remove_mask.shape != (b, hf * wf):
#                     raise ValueError(
#                         "remove_mask shape {} incompatible with layer4 {}x{}.".format(
#                             tuple(remove_mask.shape), hf, wf
#                         )
#                     )
#                 m = remove_mask.to(output.device).view(b, 1, hf, wf).to(output.dtype)
#                 modified = output * (1.0 - m)
#                 holder["features"] = modified
#                 return modified

#             handle = _get_layer4(student_model).register_forward_hook(_hook)
#             try:
#                 output = student_model(x)
#             finally:
#                 handle.remove()
#             if "features" not in holder:
#                 raise RuntimeError("Failed to capture masked layer4 output.")
#             return _extract_logits(output), holder["features"]

#         for batch_inputs, batch_labels in self._batch_iterator(inputs, labels):
#             if max_images is not None and int(max_images) > 0 and len(paths) >= int(max_images):
#                 break

#             x = batch_inputs.to(device, non_blocking=True)
#             y = batch_labels.long().to(device, non_blocking=True)
#             part = self._stage2_triple_bank_partition(x)
#             rel_mask = part["relevant_mask"].to(device)
#             nui_mask = part["nuisance_mask"].to(device)

#             base_logits, base_features = _capture_layer4_and_forward(student_model, x)
#             base_prob = F.softmax(base_logits, dim=1)
#             base_pred = base_logits.argmax(dim=1)

#             non_gt_logits = base_logits.detach().clone()
#             row = torch.arange(base_logits.shape[0], device=device)
#             non_gt_logits[row, y] = -torch.inf
#             competitor = non_gt_logits.argmax(dim=1)

#             if target_mode == "gt":
#                 target = y
#             elif target_mode == "pred":
#                 target = base_pred
#             else:
#                 target = competitor

#             w_target = classifier.weight.index_select(0, target).to(
#                 device=base_features.device, dtype=base_features.dtype
#             )

#             logits_r, feat_r = _forward_with_mask_and_capture(x, rel_mask)
#             logits_n, feat_n = _forward_with_mask_and_capture(x, nui_mask)

#             cam_base = torch.einsum("bd,bdhw->bhw", w_target, base_features)
#             cam_r = torch.einsum("bd,bdhw->bhw", w_target, feat_r)
#             cam_n = torch.einsum("bd,bdhw->bhw", w_target, feat_n)

#             prob_r = F.softmax(logits_r, dim=1)
#             prob_n = F.softmax(logits_n, dim=1)

#             for i in range(x.shape[0]):
#                 if max_images is not None and int(max_images) > 0 and len(paths) >= int(max_images):
#                     break

#                 image_np = _to_image(batch_inputs[i])
#                 H, W = image_np.shape[:2]
#                 yi = int(y[i].item())
#                 target_i = int(target[i].item())
#                 pred_i = int(base_pred[i].item())

#                 fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.7))
#                 axes[0].imshow(image_np, cmap="gray" if image_np.ndim == 2 else None)
#                 axes[0].set_title("Original\nGT={} Pred={}".format(yi, pred_i))
#                 axes[0].axis("off")

#                 panels = [
#                     (cam_base[i], "Original CAM", float(base_prob[i, yi].item())),
#                     (cam_r[i], "Remove R CAM", float(prob_r[i, yi].item())),
#                     (cam_n[i], "Remove N CAM", float(prob_n[i, yi].item())),
#                 ]
#                 for ax, (cam_i, title, pgt) in zip(axes[1:], panels):
#                     ax.imshow(image_np, cmap="gray" if image_np.ndim == 2 else None)
#                     ax.imshow(
#                         _cam_show(cam_i, H, W),
#                         cmap="jet",
#                         alpha=float(overlay_alpha),
#                         vmin=0.0,
#                         vmax=1.0,
#                     )
#                     ax.set_title("{}\nP_GT={:.1f}%".format(title, 100.0 * pgt))
#                     ax.axis("off")

#                 fig.suptitle(
#                     "R vs N counterfactual | target={} | #R={} #N={}".format(
#                         target_i,
#                         int(rel_mask[i].sum().item()),
#                         int(nui_mask[i].sum().item()),
#                     ),
#                     fontsize=11,
#                 )
#                 fig.tight_layout()
#                 path = os.path.join(save_dir, "remove_r_vs_n_{:05d}.png".format(global_index + i))
#                 fig.savefig(path, dpi=180, bbox_inches="tight")
#                 if display:
#                     plt.show()
#                 plt.close(fig)
#                 paths.append(path)

#                 records.append({
#                     "sample_index": global_index + i,
#                     "gt": yi,
#                     "original_pred": pred_i,
#                     "competitor": int(competitor[i].item()),
#                     "target_class": target_i,
#                     "target_mode": target_mode,
#                     "num_relevant": int(rel_mask[i].sum().item()),
#                     "num_nuisance": int(nui_mask[i].sum().item()),
#                     "original_pgt": float(base_prob[i, yi].item()),
#                     "remove_r_pgt": float(prob_r[i, yi].item()),
#                     "remove_n_pgt": float(prob_n[i, yi].item()),
#                     "original_ptarget": float(base_prob[i, target_i].item()),
#                     "remove_r_ptarget": float(prob_r[i, target_i].item()),
#                     "remove_n_ptarget": float(prob_n[i, target_i].item()),
#                     "remove_r_pred": int(logits_r[i].argmax().item()),
#                     "remove_n_pred": int(logits_n[i].argmax().item()),
#                 })

#             global_index += int(batch_inputs.shape[0])

#         csv_path = os.path.join(save_dir, "remove_r_vs_n_cam.csv")
#         if records:
#             with open(csv_path, "w", newline="", encoding="utf-8") as f:
#                 writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
#                 writer.writeheader()
#                 writer.writerows(records)

#         if original_training:
#             student_model.train()

#         print("[Remove R vs N CAM] saved {} image(s) to: {}".format(
#             len(paths), os.path.abspath(save_dir)
#         ))
#         if records:
#             print("  stats CSV: {}".format(os.path.abspath(csv_path)))
#         return paths

#     @torch.no_grad()
#     def visualize_feature_embedding(
#         self,
#         student_model: nn.Module,
#         inputs: Union[Tensor, Iterable],
#         labels: Optional[Tensor] = None,
#         ce_model: Optional[nn.Module] = None,
#         save_dir: str = "./feature_embedding_ce_vs_ours",
#         max_samples: int = 2000,
#         representation: str = "global",
#         normalize_features: bool = True,
#         pca_dim: int = 50,
#         perplexity: float = 30.0,
#         random_state: int = 0,
#         display: bool = False,
#         class_names: Optional[Sequence[str]] = None,
#     ) -> Dict[str, Any]:
#         """
#         Joint t-SNE comparison of CE vs current/purified student representations.

#         The same input samples are passed through BOTH models. ``ce_model=None`` uses
#         the frozen Stage-I snapshot saved by ``discover()``. A single joint t-SNE is
#         fitted on [CE features; Ours features], then the two halves are shown in
#         side-by-side panels. This avoids fitting two unrelated t-SNE coordinate systems.

#         representation:
#             "global"   : GAP(layer4) feature for class-structure visualization.
#             "relevant" : mean layer4 feature over current frozen-bank Relevant mask.
#             "nuisance" : mean layer4 feature over current frozen-bank Nuisance mask;
#                          useful for checking whether nuisance representation becomes
#                          less class-structured after purification.

#         For ``relevant``/``nuisance``, the SAME frozen Stage-II mask is applied to CE
#         and Ours features for each sample, so the compared region identity is fixed.

#         Files:
#             feature_embedding_<representation>.png
#             feature_embedding_<representation>.csv
#         """
#         import numpy as np
#         import matplotlib.pyplot as plt
#         from sklearn.decomposition import PCA
#         from sklearn.manifold import TSNE

#         rep = str(representation).strip().lower()
#         if rep not in ("global", "relevant", "nuisance"):
#             raise ValueError("representation must be one of {'global','relevant','nuisance'}.")
#         if int(max_samples) < 2:
#             raise ValueError("max_samples must be >= 2.")
#         if float(perplexity) <= 0:
#             raise ValueError("perplexity must be > 0.")
#         if int(pca_dim) < 1:
#             raise ValueError("pca_dim must be >= 1.")

#         if ce_model is None:
#             if self._assignment_model is None:
#                 raise RuntimeError(
#                     "No CE reference is available. Run discover() first or pass ce_model."
#                 )
#             ce_reference = self._assignment_model
#             ce_source = "stored Stage-I/CE snapshot"
#         else:
#             ce_reference = ce_model
#             ce_source = "explicit ce_model"

#         try:
#             student_device = next(student_model.parameters()).device
#             ce_device = next(ce_reference.parameters()).device
#         except StopIteration:
#             raise ValueError("student_model and ce_model must contain parameters.")

#         student_training = student_model.training
#         ce_training = ce_reference.training
#         student_model.eval()
#         ce_reference.eval()

#         ce_chunks: List[Tensor] = []
#         student_chunks: List[Tensor] = []
#         label_chunks: List[Tensor] = []
#         sample_ids: List[int] = []
#         global_index = 0
#         collected = 0

#         for batch_inputs, batch_labels in self._batch_iterator(inputs, labels):
#             if collected >= int(max_samples):
#                 break

#             x_student = batch_inputs.to(student_device, non_blocking=True)
#             x_ce = batch_inputs.to(ce_device, non_blocking=True)
#             y_cpu = batch_labels.long().cpu()

#             _, feat_student = _capture_layer4_and_forward(student_model, x_student)
#             _, feat_ce = _capture_layer4_and_forward(ce_reference, x_ce)

#             if feat_student.shape[1:] != feat_ce.shape[1:]:
#                 raise ValueError(
#                     "Student/CE layer4 feature shapes differ: {} vs {}.".format(
#                         tuple(feat_student.shape[1:]), tuple(feat_ce.shape[1:])
#                     )
#                 )

#             if rep == "global":
#                 z_student = feat_student.mean(dim=(2, 3))
#                 z_ce = feat_ce.mean(dim=(2, 3))
#                 valid = torch.ones(batch_inputs.shape[0], dtype=torch.bool)
#             else:
#                 self._check_discovered()
#                 # Partition is intentionally generated by the existing frozen Stage-II
#                 # assignment, then reused for both CE and Ours features.
#                 part = self._stage2_triple_bank_partition(
#                     batch_inputs.to(self._assignment_device, non_blocking=True)
#                 )
#                 mask = part["relevant_mask"] if rep == "relevant" else part["nuisance_mask"]
#                 mask_cpu = mask.detach().cpu()
#                 valid = mask_cpu.any(dim=1)

#                 mask_student = mask.to(feat_student.device)
#                 mask_ce = mask.to(feat_ce.device)
#                 regions_student = self._regions_from_features(feat_student)
#                 regions_ce = self._regions_from_features(feat_ce)
#                 z_student, _ = self._masked_avg_pool(regions_student, mask_student)
#                 z_ce, _ = self._masked_avg_pool(regions_ce, mask_ce)

#             valid_ids = valid.nonzero(as_tuple=False).squeeze(1)
#             if valid_ids.numel() == 0:
#                 global_index += int(batch_inputs.shape[0])
#                 continue

#             remaining = int(max_samples) - collected
#             if valid_ids.numel() > remaining:
#                 valid_ids = valid_ids[:remaining]

#             ce_chunks.append(z_ce.detach().float().cpu().index_select(0, valid_ids))
#             student_chunks.append(z_student.detach().float().cpu().index_select(0, valid_ids))
#             label_chunks.append(y_cpu.index_select(0, valid_ids))
#             sample_ids.extend((global_index + valid_ids).tolist())
#             collected += int(valid_ids.numel())
#             global_index += int(batch_inputs.shape[0])

#         if len(ce_chunks) == 0:
#             raise RuntimeError("No valid samples were collected for representation={!r}.".format(rep))

#         z_ce_all = torch.cat(ce_chunks, dim=0)
#         z_student_all = torch.cat(student_chunks, dim=0)
#         y_all = torch.cat(label_chunks, dim=0).long()
#         n = int(y_all.numel())
#         if n < 2:
#             raise RuntimeError("Need at least 2 valid samples for t-SNE.")

#         if normalize_features:
#             z_ce_all = F.normalize(z_ce_all, p=2, dim=1, eps=self.eps)
#             z_student_all = F.normalize(z_student_all, p=2, dim=1, eps=self.eps)

#         joint = torch.cat([z_ce_all, z_student_all], dim=0).numpy()
#         n_joint, d = joint.shape

#         pca_used = min(int(pca_dim), int(d), int(n_joint - 1))
#         if pca_used >= 2 and pca_used < d:
#             joint_reduced = PCA(
#                 n_components=pca_used,
#                 random_state=int(random_state),
#             ).fit_transform(joint)
#         else:
#             joint_reduced = joint
#             pca_used = int(joint_reduced.shape[1])

#         perplexity_used = min(
#             float(perplexity),
#             max(1.0, (float(n_joint) - 1.0) / 3.0),
#         )
#         if perplexity_used >= n_joint:
#             perplexity_used = max(1.0, float(n_joint) - 1.0)

#         embedding = TSNE(
#             n_components=2,
#             perplexity=perplexity_used,
#             learning_rate="auto",
#             init="pca",
#             n_iter=1000,
#             random_state=int(random_state),
#             metric="euclidean",
#         ).fit_transform(joint_reduced)

#         emb_ce = embedding[:n]
#         emb_student = embedding[n:]
#         y_np = y_all.numpy()

#         os.makedirs(save_dir, exist_ok=True)
#         fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharex=True, sharey=True)
#         cmap_name = "tab10" if self.num_classes <= 10 else "tab20"
#         cmap = plt.get_cmap(cmap_name, max(self.num_classes, 2))

#         for c in range(self.num_classes):
#             m = y_np == c
#             if not np.any(m):
#                 continue
#             label_name = str(c)
#             if class_names is not None and c < len(class_names):
#                 label_name = str(class_names[c])
#             axes[0].scatter(
#                 emb_ce[m, 0], emb_ce[m, 1], s=16, alpha=0.72,
#                 color=cmap(c), label=label_name,
#             )
#             axes[1].scatter(
#                 emb_student[m, 0], emb_student[m, 1], s=16, alpha=0.72,
#                 color=cmap(c), label=label_name,
#             )

#         axes[0].set_title("CE t-SNE ({})".format(rep))
#         axes[1].set_title("Ours t-SNE ({})".format(rep))
#         for ax in axes:
#             ax.set_xlabel("t-SNE 1")
#             ax.set_ylabel("t-SNE 2")
#             ax.grid(alpha=0.15)

#         handles, legend_labels = axes[1].get_legend_handles_labels()
#         if handles:
#             axes[1].legend(
#                 handles,
#                 legend_labels,
#                 title="Class",
#                 loc="best",
#                 fontsize=8,
#                 frameon=True,
#             )

#         fig.suptitle(
#             "Joint CE vs Ours feature embedding | {} samples | CE={}".format(
#                 n, ce_source
#             ),
#             fontsize=11,
#         )
#         fig.tight_layout()
#         png_path = os.path.join(save_dir, "feature_embedding_{}.png".format(rep))
#         fig.savefig(png_path, dpi=200, bbox_inches="tight")
#         if display:
#             plt.show()
#         plt.close(fig)

#         csv_path = os.path.join(save_dir, "feature_embedding_{}.csv".format(rep))
#         rows: List[Dict[str, Any]] = []
#         for idx in range(n):
#             rows.append({
#                 "sample_index": int(sample_ids[idx]),
#                 "gt": int(y_all[idx].item()),
#                 "model": "CE",
#                 "representation": rep,
#                 "tsne_x": float(emb_ce[idx, 0]),
#                 "tsne_y": float(emb_ce[idx, 1]),
#             })
#             rows.append({
#                 "sample_index": int(sample_ids[idx]),
#                 "gt": int(y_all[idx].item()),
#                 "model": "Ours",
#                 "representation": rep,
#                 "tsne_x": float(emb_student[idx, 0]),
#                 "tsne_y": float(emb_student[idx, 1]),
#             })
#         with open(csv_path, "w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
#             writer.writeheader()
#             writer.writerows(rows)

#         if student_training:
#             student_model.train()
#         if ce_training:
#             ce_reference.train()

#         result = {
#             "png_path": png_path,
#             "csv_path": csv_path,
#             "num_samples": n,
#             "representation": rep,
#             "ce_source": ce_source,
#             "pca_dim_used": pca_used,
#             "perplexity_used": perplexity_used,
#         }
#         print("[Feature embedding] {}".format(os.path.abspath(png_path)))
#         print("  samples={} representation={} CE={}".format(n, rep, ce_source))
#         print("  CSV={}".format(os.path.abspath(csv_path)))
#         return result

#     # -------------------------------------------------------------------------
#     # Persistence
#     # -------------------------------------------------------------------------

#     def save_discovery(self, path: str) -> None:
#         self._check_discovered()
#         payload = {
#             "version": "triple_bank_rni_relative_contribution_wy_metric_select_v5",
#             "clustering_metric": self.clustering_metric,
#             "num_classes": self.num_classes,
#             "prototype_k_factors": self.prototype_k_factors,
#             "decision_threshold": self.decision_threshold,
#             "temperature": self.temperature,
#             "class_counts": self.class_counts,
#             "class_prior": self.class_prior,
#             "relevant_medoids_raw": self.relevant_medoids_raw,
#             "relevant_medoids_norm": self.relevant_medoids_norm,
#             "nuisance_medoids_raw": self.nuisance_medoids_raw,
#             "nuisance_medoids_norm": self.nuisance_medoids_norm,
#             "irrelevant_medoids_raw": self.irrelevant_medoids_raw,
#             "irrelevant_medoids_norm": self.irrelevant_medoids_norm,
#             "relevant_best_k": self.relevant_best_k,
#             "nuisance_best_k": self.nuisance_best_k,
#             "irrelevant_best_k": self.irrelevant_best_k,
#             "discovery_result": self.discovery_result,
#         }
#         os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
#         torch.save(payload, path)

#     def load_discovery(
#         self,
#         path: str,
#         model_for_assignment: nn.Module,
#         device: Union[str, torch.device],
#     ) -> None:
#         payload = torch.load(path, map_location="cpu")
#         if int(payload["num_classes"]) != self.num_classes:
#             raise ValueError(
#                 "Saved num_classes={} but current num_classes={}.".format(
#                     payload["num_classes"], self.num_classes
#                 )
#             )
#         required = [
#             "relevant_medoids_raw", "relevant_medoids_norm",
#             "nuisance_medoids_raw", "nuisance_medoids_norm",
#             "irrelevant_medoids_raw", "irrelevant_medoids_norm",
#             "relevant_best_k", "nuisance_best_k", "irrelevant_best_k",
#         ]
#         missing = [k for k in required if k not in payload]
#         if missing:
#             raise ValueError(
#                 "Saved discovery does not contain the required THREE banks (missing {}). "
#                 "Re-run discover() on dataloaders['val'].".format(missing)
#             )

#         self.relevant_medoids_raw = payload["relevant_medoids_raw"].float().cpu()
#         self.relevant_medoids_norm = payload["relevant_medoids_norm"].float().cpu()
#         self.nuisance_medoids_raw = payload["nuisance_medoids_raw"].float().cpu()
#         self.nuisance_medoids_norm = payload["nuisance_medoids_norm"].float().cpu()
#         self.irrelevant_medoids_raw = payload["irrelevant_medoids_raw"].float().cpu()
#         self.irrelevant_medoids_norm = payload["irrelevant_medoids_norm"].float().cpu()
#         self.relevant_best_k = int(payload["relevant_best_k"])
#         self.nuisance_best_k = int(payload["nuisance_best_k"])
#         self.irrelevant_best_k = int(payload["irrelevant_best_k"])
#         self.discovery_result = payload.get("discovery_result", None)

#         # Banks must be matched with the same metric used to discover them.
#         saved_metric = str(payload.get("clustering_metric", "cosine")).strip().lower()
#         if saved_metric in ("distance", "l2"):
#             saved_metric = "euclidean"
#         if saved_metric not in ("cosine", "euclidean"):
#             raise ValueError(
#                 "Unsupported clustering_metric in saved discovery: {!r}.".format(
#                     saved_metric
#                 )
#             )
#         self.clustering_metric = saved_metric

#         if "class_counts" in payload:
#             self.set_class_counts(payload["class_counts"])
#         elif "class_prior" in payload:
#             prior = payload["class_prior"].float().flatten()
#             if prior.numel() == self.num_classes:
#                 self.class_prior = prior / prior.sum().clamp_min(self.eps)
#         if "decision_threshold" in payload:
#             self.decision_threshold = float(payload["decision_threshold"])

#         device = torch.device(device)
#         self._assignment_model = copy.deepcopy(model_for_assignment).to(device)
#         self._assignment_model.eval()
#         for parameter in self._assignment_model.parameters():
#             parameter.requires_grad_(False)
#         self._assignment_device = device


# # Compatibility aliases
# RelevantNuisancePatternResNet = DualPatternResNet
# TriplePatternResNet = DualPatternResNet


# # =============================================================================
# # Minimal runnable smoke test
# # =============================================================================


# def _smoke_test() -> None:
#     from torch.utils.data import DataLoader, TensorDataset

#     class TinyResNet(nn.Module):
#         def __init__(self, num_classes: int = 3):
#             super().__init__()
#             self.stem = nn.Sequential(
#                 nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
#                 nn.ReLU(inplace=False),
#                 nn.AvgPool2d(2),
#             )
#             self.layer4 = nn.Sequential(
#                 nn.Conv2d(16, 4, kernel_size=3, padding=1, bias=False),
#                 nn.ReLU(inplace=False),
#                 nn.AdaptiveAvgPool2d((4, 4)),
#             )
#             self.fc = nn.Linear(4, num_classes, bias=False)

#         def forward(self, x):
#             x = self.stem(x)
#             f = self.layer4(x)
#             z = f.mean(dim=(2, 3))
#             return self.fc(z)

#     torch.manual_seed(7)
#     random.seed(7)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     num_classes = 3
#     model = TinyResNet(num_classes=num_classes).to(device)
#     # Make the tiny synthetic classifier weights positive so C_GT and C_MIS are
#     # usually on comparable scales; this creates R/N/I examples for the smoke test.
#     # This is only test scaffolding and does not affect user models.
#     with torch.no_grad():
#         model.fc.weight.copy_(model.fc.weight.abs() + 0.1)

#     # More samples make R/N/I pools sufficiently large for candidate K values.
#     images = torch.rand(180, 3, 32, 32)
#     with torch.no_grad():
#         pred = model(images.to(device)).argmax(dim=1).cpu()
#     labels = pred.clone()
#     # Half wrong, distributed across classes.
#     wrong_ids = torch.arange(labels.numel()) % 2 == 1
#     labels[wrong_ids] = (labels[wrong_ids] + 1) % num_classes

#     loader = DataLoader(TensorDataset(images, labels), batch_size=18, shuffle=False)
#     method = DualPatternResNet(
#         num_classes=num_classes,
#         prototype_k_factors=[1, 2],
#         decision_threshold=0.10,
#         temperature=0.1,
#         class_counts=[90, 60, 30],
#         lambda_global=1.0,
#         lambda_relevant=0.1,
#         lambda_nuisance=0.1,
#         kmedoids_iterations=4,
#         max_candidates_per_class=500,
#         silhouette_sample_size=180,
#         assignment_chunk_size=1024,
#         random_seed=11,
#     )
#     result = method.discover(model, loader, device=device, verbose=False)

#     model.train()
#     x = images[:12].to(device)
#     y = labels[:12].to(device)
#     out = method(model, x, y)
#     loss = method.total_loss(out)
#     model.zero_grad(set_to_none=True)
#     loss.backward()

#     assert torch.isfinite(loss).item()
#     assert out.irrelevant_mask is not None
#     full = out.relevant_mask | out.nuisance_mask | out.irrelevant_mask
#     assert bool(full.all())
#     assert not bool((out.relevant_mask & out.nuisance_mask).any())
#     assert not bool((out.relevant_mask & out.irrelevant_mask).any())
#     assert not bool((out.nuisance_mask & out.irrelevant_mask).any())
#     assert result.irrelevant_best_k >= 2

#     print("Three-bank R/N/I smoke test passed.")
#     print({
#         "best_K_R": result.relevant_best_k,
#         "best_K_N": result.nuisance_best_k,
#         "best_K_I": result.irrelevant_best_k,
#         "loss_global": float(out.loss_global.detach().cpu()),
#         "loss_relevant": float(out.loss_relevant.detach().cpu()),
#         "loss_nuisance": float(out.loss_nuisance.detach().cpu()),
#         "num_R": out.num_relevant_regions,
#         "num_N": out.num_nuisance_regions,
#         "num_I": out.num_irrelevant_regions,
#     })


# if __name__ == "__main__":
#     _smoke_test()






from __future__ import annotations

import copy
import csv
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


# =============================================================================
# Public result containers
# =============================================================================


@dataclass
class DualPatternDiscoveryResult:
    relevant_best_k: int
    nuisance_best_k: int
    irrelevant_best_k: int
    relevant_silhouette: Dict[int, float]
    nuisance_silhouette: Dict[int, float]
    irrelevant_silhouette: Dict[int, float]
    relevant_candidate_count: int
    nuisance_candidate_count: int
    irrelevant_candidate_count: int
    relevant_candidate_count_per_class: List[int]
    nuisance_candidate_count_per_class: List[int]
    irrelevant_candidate_count_per_class: List[int]
    decision_threshold: float


@dataclass
class DualPatternOutput:
    # Requested Stage-II objectives
    #   L_G : maximize I(Z;Y)   via CE-like Z-to-class alignment
    #   L_R : maximize I(Zr;Y)  via CE-like Zr-to-class alignment
    #   L_N : minimize I(Zn;Y)  via nHSIC(Zn, W_Y)
    loss_region: Tensor
    loss_global: Tensor
    loss_relevant: Tensor
    loss_nuisance: Tensor

    # Ordinary model logits
    logits: Tensor

    # Current student representations
    z_global: Tensor
    student_regions: Tensor

    # Frozen three-bank nearest-neighbor partition [B,R]
    relevant_mask: Tensor
    nuisance_mask: Tensor
    irrelevant_mask: Tensor

    # Frozen-bank maximum cosine similarities [B,R]
    relevant_similarity: Tensor
    nuisance_similarity: Tensor
    irrelevant_similarity: Tensor
    assignment_margin: Tensor       # s_R - s_N (kept for compatibility)

    # Pooled representations [B,D]
    z_relevant: Tensor
    z_nuisance: Tensor
    z_irrelevant: Tensor

    # Whether an image contains at least one token of each type [B]
    valid_relevant: Tensor
    valid_nuisance: Tensor
    valid_irrelevant: Tensor

    # Counts for logging
    num_relevant_regions: int
    num_nuisance_regions: int
    num_irrelevant_regions: int
    num_valid_relevant_images: int
    num_valid_nuisance_images: int
    num_valid_irrelevant_images: int

    # Kept for backward compatibility with older code. Not used by default.
    loss_disentangle: Optional[Tensor] = None


# =============================================================================
# Model helpers
# =============================================================================


def _extract_logits(model_output: Any) -> Tensor:
    """Robustly obtain logits [B,C] from common model output styles."""
    if torch.is_tensor(model_output):
        logits = model_output
    elif isinstance(model_output, (tuple, list)):
        if len(model_output) == 0 or not torch.is_tensor(model_output[0]):
            raise TypeError("Model tuple/list output must have logits as its first tensor.")
        logits = model_output[0]
    elif isinstance(model_output, dict):
        if "logits" not in model_output or not torch.is_tensor(model_output["logits"]):
            raise TypeError("Model dict output must contain tensor key 'logits'.")
        logits = model_output["logits"]
    else:
        raise TypeError("Unsupported model output type: {}".format(type(model_output)))

    if logits.ndim != 2:
        raise ValueError("Expected logits [B,C], got {}.".format(tuple(logits.shape)))
    return logits


def _get_layer4(model: nn.Module) -> nn.Module:
    """Return final ResNet layer4 module."""
    if not hasattr(model, "layer4"):
        raise AttributeError(
            "The model must expose the final spatial feature stage as model.layer4."
        )
    layer4 = getattr(model, "layer4")
    if not isinstance(layer4, nn.Module):
        raise TypeError("model.layer4 must be nn.Module.")
    return layer4


def _get_classifier(model: nn.Module) -> nn.Linear:
    """Find a standard linear classification head: fc -> head -> classifier."""
    for name in ("fc", "head", "classifier"):
        if hasattr(model, name):
            module = getattr(model, name)
            if isinstance(module, nn.Linear):
                return module
    raise AttributeError(
        "Could not find a linear classifier. Expected model.fc, model.head, "
        "or model.classifier to be nn.Linear."
    )


def _capture_layer4_and_forward(
    model: nn.Module,
    inputs: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Run model once and capture FINAL layer4 output.

    Returns:
        logits:   [B,C]
        features: [B,D,H,W]
    """
    holder: Dict[str, Tensor] = {}

    def _hook(_module, _inputs, output):
        if not torch.is_tensor(output):
            raise TypeError("layer4 output must be a tensor.")
        holder["features"] = output

    handle = _get_layer4(model).register_forward_hook(_hook)
    try:
        output = model(inputs)
    finally:
        handle.remove()

    if "features" not in holder:
        raise RuntimeError("Failed to capture layer4 output.")

    logits = _extract_logits(output)
    features = holder["features"]

    if features.ndim != 4:
        raise ValueError(
            "layer4 output must be [B,D,H,W], got {}.".format(tuple(features.shape))
        )
    if logits.shape[0] != features.shape[0]:
        raise ValueError("Batch mismatch between logits and layer4 features.")

    return logits, features


def _forward_with_layer4_zero_mask(
    model: nn.Module,
    inputs: Tensor,
    remove_mask: Tensor,
) -> Tensor:
    """
    Forward after zeroing selected layer4 spatial tokens.

    Args:
        model:       classifier exposing model.layer4
        inputs:      [B,C,H,W]
        remove_mask: [B,R] bool OR [B,Hf,Wf] bool

    This is a layer4 intervention, NOT an input-image occlusion. It directly tests
    the contribution of the discovered layer4 patch representation.
    """
    if remove_mask.ndim not in (2, 3):
        raise ValueError("remove_mask must be [B,R] or [B,Hf,Wf].")

    def _hook(_module, _inputs, output):
        if not torch.is_tensor(output) or output.ndim != 4:
            raise TypeError("layer4 output must be [B,D,Hf,Wf].")
        b, _, hf, wf = output.shape
        if remove_mask.ndim == 2:
            if remove_mask.shape != (b, hf * wf):
                raise ValueError(
                    "remove_mask shape {} incompatible with layer4 {}x{}.".format(
                        tuple(remove_mask.shape), hf, wf
                    )
                )
            m = remove_mask.view(b, 1, hf, wf)
        else:
            if remove_mask.shape != (b, hf, wf):
                raise ValueError(
                    "remove_mask shape {} incompatible with layer4 {}x{}.".format(
                        tuple(remove_mask.shape), hf, wf
                    )
                )
            m = remove_mask.unsqueeze(1)
        m = m.to(device=output.device, dtype=output.dtype)
        return output * (1.0 - m)

    handle = _get_layer4(model).register_forward_hook(_hook)
    try:
        output = model(inputs)
    finally:
        handle.remove()
    return _extract_logits(output)


# =============================================================================
# Main method
# =============================================================================


class DualPatternResNet(object):
    """
    Three-bank Relevant / Nuisance / Irrelevant pattern discovery and MI-oriented
    representation purification.

    Stage-I discovery on reference_loader (recommended: dataloaders['val'])
    -----------------------------------------------------------------------
    For EVERY image, regardless of whether the image-level prediction is correct,
    define c^- as the strongest non-GT class under the current classifier:

        c^- = argmax_{c != y} logit_c.

    For every layer4 patch f_p, compute its raw contribution to the GT class and to
    the strongest non-GT competitor:

        C_GT(p)  = f_p^T w_y
        C_MIS(p) = f_p^T w_{c^-}

    Then use the scale-free relative contribution score

                         C_GT(p) - C_MIS(p)
        r_p = -----------------------------------------------  in [-1, 1].
              |C_GT(p)| + |C_MIS(p)| + eps

    Patch semantics are defined ONLY by this local relative decision contribution:

        r_p >  delta   -> Relevant
        r_p < -delta   -> Nuisance
        otherwise      -> Irrelevant

    Therefore image-level prediction correctness is NOT part of the R/N definition,
    and every discovery patch belongs to exactly one of R/N/I. decision_threshold is
    now dimensionless and interpretable on a stable [0,1) scale (e.g. 0.1 or 0.5).

    Each candidate pool is clustered INDEPENDENTLY. clustering_metric controls:
        'cosine'    -> cosine K-Medoids + cosine silhouette
        'euclidean' -> Euclidean-distance K-Medoids + Euclidean silhouette
    With prototype_k_factors=[a,b], every integer K from a*C through b*C is tested,
    and the K with the highest silhouette score is selected separately for R, N and I.

    Stage-II assignment
    -------------------
    No bank-similarity threshold is used. For each frozen layer4 patch, compute:
        s_R = max cosine(patch, R-bank)
        s_N = max cosine(patch, N-bank)
        s_I = max cosine(patch, I-bank)
    and assign the patch to whichever bank has the largest similarity. Thus every
    training patch is partitioned by nearest semantic prototype among the THREE banks.

    Objectives
    ----------
        L_G = CE((Z W^T)/tau + log prior, y)          [no L2 normalization]
        L_R = CE((Zr W^T)/tau + log prior, y)         [Zr -> w_y]
        L_N = nHSIC(Zn, W_Y), W_Y = w_y               [Zn disentangles from w_y]

    The irrelevant bank is used for assignment/reference only and has no auxiliary
    loss by default.
    """

    def __init__(
        self,
        num_classes: int,
        prototype_k_factors: Sequence[int] = (1, 2, 3, 4, 5),
        decision_threshold: float = 0.5,
        temperature: float = 0.07,
        class_counts: Optional[Union[Tensor, Sequence[float]]] = None,
        lambda_global: float = 1.0,
        lambda_relevant: float = 0.0,
        lambda_nuisance: float = 0.5,
        kmedoids_iterations: int = 100,
        max_candidates_per_class: Optional[int] = 2500,
        silhouette_sample_size: int = 3000,
        assignment_chunk_size: int = 8192,
        clustering_device: Optional[Union[str, torch.device]] = None,
        random_seed: int = 0,
        eps: float = 1e-8,
        clustering_metric: str = "cosine",
    ) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2.")
        if not (0.0 <= float(decision_threshold) < 1.0):
            raise ValueError(
                "decision_threshold must be in [0,1) for the relative contribution score."
            )
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        if kmedoids_iterations < 1:
            raise ValueError("kmedoids_iterations must be >= 1.")
        if silhouette_sample_size < 2:
            raise ValueError("silhouette_sample_size must be >= 2.")
        if assignment_chunk_size < 1:
            raise ValueError("assignment_chunk_size must be >= 1.")
        if any(v < 0 for v in (lambda_global, lambda_relevant, lambda_nuisance)):
            raise ValueError("All lambda weights must be >= 0.")
        if max_candidates_per_class is not None and max_candidates_per_class < 1:
            raise ValueError("max_candidates_per_class must be >=1 or None.")

        factors = [int(v) for v in prototype_k_factors]
        if len(factors) == 0:
            raise ValueError("prototype_k_factors cannot be empty.")
        if any(v < 1 for v in factors):
            raise ValueError("prototype_k_factors must contain positive integers.")
        seen = set()
        factors = [v for v in factors if not (v in seen or seen.add(v))]

        self.num_classes = int(num_classes)
        self.prototype_k_factors = factors
        self.decision_threshold = float(decision_threshold)
        self.temperature = float(temperature)

        self.lambda_global = float(lambda_global)
        self.lambda_relevant = float(lambda_relevant)
        self.lambda_nuisance = float(lambda_nuisance)

        self.kmedoids_iterations = int(kmedoids_iterations)
        self.max_candidates_per_class = max_candidates_per_class
        self.silhouette_sample_size = int(silhouette_sample_size)
        self.assignment_chunk_size = int(assignment_chunk_size)
        self.clustering_device = (
            None if clustering_device is None else torch.device(clustering_device)
        )
        self.random_seed = int(random_seed)
        self.eps = float(eps)

        # Clustering / Stage-II matching metric.
        # Accepted aliases:
        #   cosine                  -> cosine K-Medoids + cosine silhouette
        #   distance/euclidean/l2   -> Euclidean K-Medoids + Euclidean silhouette
        metric = str(clustering_metric).strip().lower()
        metric_alias = {
            "cos": "cosine",
            "cosine": "cosine",
            "distance": "euclidean",
            "euclidean": "euclidean",
            "l2": "euclidean",
        }
        if metric not in metric_alias:
            raise ValueError(
                "clustering_metric must be one of "
                "{'cosine', 'distance', 'euclidean', 'l2'}, got {!r}.".format(
                    clustering_metric
                )
            )
        self.clustering_metric = metric_alias[metric]

        # Frozen Stage-I banks.
        self.relevant_medoids_raw: Optional[Tensor] = None
        self.relevant_medoids_norm: Optional[Tensor] = None
        self.nuisance_medoids_raw: Optional[Tensor] = None
        self.nuisance_medoids_norm: Optional[Tensor] = None
        self.irrelevant_medoids_raw: Optional[Tensor] = None
        self.irrelevant_medoids_norm: Optional[Tensor] = None
        self.relevant_best_k: Optional[int] = None
        self.nuisance_best_k: Optional[int] = None
        self.irrelevant_best_k: Optional[int] = None
        self.discovery_result: Optional[DualPatternDiscoveryResult] = None

        # Frozen exact Stage-I snapshot used for Stage-II assignment.
        self._assignment_model: Optional[nn.Module] = None
        self._assignment_device: Optional[torch.device] = None

        # Class prior: cls_num / sum(cls_num).
        self.class_prior = torch.full(
            (self.num_classes,), 1.0 / float(self.num_classes), dtype=torch.float32
        )
        self.class_counts = torch.ones(self.num_classes, dtype=torch.float32)
        if class_counts is not None:
            self.set_class_counts(class_counts)

        self.last_visualization_summary: Optional[Dict[str, Any]] = None

    # -------------------------------------------------------------------------
    # Class prior
    # -------------------------------------------------------------------------

    def set_class_counts(
        self,
        class_counts: Union[Tensor, Sequence[float]],
    ) -> None:
        counts = torch.as_tensor(class_counts, dtype=torch.float32).flatten().cpu()
        if counts.numel() != self.num_classes:
            raise ValueError(
                "class_counts must contain {} values, got {}.".format(
                    self.num_classes, counts.numel()
                )
            )
        if torch.any(counts < 0):
            raise ValueError("class_counts cannot contain negative values.")
        if float(counts.sum().item()) <= 0:
            raise ValueError("class_counts must have positive total mass.")
        self.class_counts = counts
        prior = counts / counts.sum()
        self.class_prior = prior.clamp_min(self.eps)
        self.class_prior = self.class_prior / self.class_prior.sum()

    @property
    def prior_log(self) -> Tensor:
        # Exactly log(cls_num / sum(cls_num)), with eps only for zero-count safety.
        return self.class_prior.clamp_min(self.eps).log()

    # -------------------------------------------------------------------------
    # layer4 regions / patches
    # -------------------------------------------------------------------------

    def _regions_from_features(self, features: Tensor) -> Tensor:
        """
        [B,D,H,W] -> [B,H*W,D].
        Standard ResNet50 @224: [B,2048,7,7] -> [B,49,2048].
        """
        if features.ndim != 4:
            raise ValueError("features must be [B,D,H,W].")
        return features.flatten(2).transpose(1, 2)

    # -------------------------------------------------------------------------
    # Stage-I candidate extraction
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def _extract_decision_aware_candidates(
        self,
        model: nn.Module,
        inputs: Tensor,
        labels: Tensor,
    ) -> Tuple[Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Dict[str, Any]]:
        """
        Build R/N/I candidates from one validation/reference batch.

        IMPORTANT: image-level prediction correctness is NOT used to define R/N/I.

        For each image i:
            c_i^- = argmax_{c != y_i} logit_{i,c}

        For each patch p:
            C_GT(p)  = f_p^T w_y
            C_MIS(p) = f_p^T w_{c^-}

            r_p = (C_GT(p) - C_MIS(p)) /
                  (|C_GT(p)| + |C_MIS(p)| + eps)

        Since |a-b| <= |a|+|b|, r_p lies in [-1,1] up to numerical precision.

        Partition:
            r_p >  delta   -> Relevant
            r_p < -delta   -> Nuisance
            otherwise      -> Irrelevant

        Thus every patch belongs to exactly one discovery pool.
        """
        logits, features = _capture_layer4_and_forward(model, inputs)
        regions = self._regions_from_features(features)  # [B,R,D]
        labels = labels.long().to(logits.device)

        if logits.shape[1] != self.num_classes:
            raise ValueError(
                "Expected {} classes, got {}.".format(self.num_classes, logits.shape[1])
            )

        classifier = _get_classifier(model)
        if classifier.weight.shape[0] != self.num_classes:
            raise ValueError("Classifier output dimension != num_classes.")
        if classifier.weight.shape[1] != regions.shape[-1]:
            raise ValueError(
                "Classifier feature dim {} != layer4 patch dim {}. "
                "This method assumes GAP(layer4) -> linear classifier.".format(
                    classifier.weight.shape[1], regions.shape[-1]
                )
            )

        pred = logits.argmax(dim=1)
        correct = pred.eq(labels)
        wrong = ~correct

        # Strongest non-GT competitor c^- for EVERY image.
        non_gt_logits = logits.detach().clone()
        row = torch.arange(logits.shape[0], device=logits.device)
        non_gt_logits[row, labels] = -torch.inf
        competitor = non_gt_logits.argmax(dim=1)

        # Frozen classifier directions are used only to measure local decision
        # contribution during discovery.
        W = classifier.weight.detach().to(regions.device, regions.dtype)
        w_gt = W.index_select(0, labels)          # [B,D]
        w_mis = W.index_select(0, competitor)     # [B,D]

        # Raw per-patch class contributions. No L2 normalization is used here.
        c_gt = torch.einsum("brd,bd->br", regions, w_gt)
        c_mis = torch.einsum("brd,bd->br", regions, w_mis)

        # Scale-free relative contribution in [-1,1].
        relative_score = (c_gt - c_mis) / (
            c_gt.abs() + c_mis.abs() + self.eps
        )
        relative_score = relative_score.clamp(-1.0, 1.0)

        delta = float(self.decision_threshold)
        relevant_mask_all = relative_score.gt(delta)
        nuisance_mask_all = relative_score.lt(-delta)
        irrelevant_mask_all = ~(relevant_mask_all | nuisance_mask_all)

        # Exhaustive and mutually exclusive by construction.
        if not bool((relevant_mask_all | nuisance_mask_all | irrelevant_mask_all).all()):
            raise RuntimeError("R/N/I discovery masks do not cover all patches.")
        if bool((relevant_mask_all & nuisance_mask_all).any()):
            raise RuntimeError("Relevant/Nuisance candidate masks overlap.")
        if bool((relevant_mask_all & irrelevant_mask_all).any()):
            raise RuntimeError("Relevant/Irrelevant candidate masks overlap.")
        if bool((nuisance_mask_all & irrelevant_mask_all).any()):
            raise RuntimeError("Nuisance/Irrelevant candidate masks overlap.")

        relevant_by_class: Dict[int, Tensor] = {}
        nuisance_by_class: Dict[int, Tensor] = {}
        irrelevant_by_class: Dict[int, Tensor] = {}

        for c in range(self.num_classes):
            image_c = labels.eq(c).unsqueeze(1)
            rel = regions[image_c & relevant_mask_all].detach().float().cpu()
            nui = regions[image_c & nuisance_mask_all].detach().float().cpu()
            irr = regions[image_c & irrelevant_mask_all].detach().float().cpu()
            if rel.numel() > 0:
                relevant_by_class[c] = rel
            if nui.numel() > 0:
                nuisance_by_class[c] = nui
            if irr.numel() > 0:
                irrelevant_by_class[c] = irr

        rel_selected = relative_score[relevant_mask_all]
        nui_selected = relative_score[nuisance_mask_all]
        irr_selected = relative_score[irrelevant_mask_all]

        stats: Dict[str, Any] = {
            "num_images": int(labels.numel()),
            "num_correct_images": int(correct.sum().item()),
            "num_wrong_images": int(wrong.sum().item()),
            "num_relevant_candidates": int(relevant_mask_all.sum().item()),
            "num_nuisance_candidates": int(nuisance_mask_all.sum().item()),
            "num_irrelevant_candidates": int(irrelevant_mask_all.sum().item()),
            # Compatibility field: exhaustive R/N/I partition means nothing is ignored.
            "num_ignored_strong_opposite": 0,
            "mean_relevant_score": float(rel_selected.mean().item()) if rel_selected.numel() else float("nan"),
            "mean_nuisance_score": float(nui_selected.mean().item()) if nui_selected.numel() else float("nan"),
            "mean_abs_irrelevant_score": float(irr_selected.abs().mean().item()) if irr_selected.numel() else float("nan"),
            "mean_relative_score": float(relative_score.mean().item()),
            "min_relative_score": float(relative_score.min().item()),
            "max_relative_score": float(relative_score.max().item()),
        }
        return relevant_by_class, nuisance_by_class, irrelevant_by_class, stats

    @staticmethod
    def _append_candidate_dict(
        destination: List[List[Tensor]],
        source: Dict[int, Tensor],
    ) -> None:
        for c, x in source.items():
            if x.numel() > 0:
                destination[c].append(x)

    def _finalize_candidate_pool(
        self,
        per_class_chunks: List[List[Tensor]],
        seed_offset: int,
    ) -> Tuple[Tensor, List[int]]:
        """Concatenate and optionally cap candidates independently per GT class."""
        kept: List[Tensor] = []
        counts: List[int] = []

        for c in range(self.num_classes):
            if len(per_class_chunks[c]) == 0:
                counts.append(0)
                continue

            x = torch.cat(per_class_chunks[c], dim=0)
            n = int(x.shape[0])
            if self.max_candidates_per_class is not None and n > self.max_candidates_per_class:
                gen = torch.Generator(device="cpu")
                gen.manual_seed(self.random_seed + seed_offset + 1009 * c)
                idx = torch.randperm(n, generator=gen)[: self.max_candidates_per_class]
                x = x.index_select(0, idx)

            counts.append(int(x.shape[0]))
            kept.append(x)

        if len(kept) == 0:
            return torch.empty((0, 0), dtype=torch.float32), counts
        return torch.cat(kept, dim=0), counts

    # -------------------------------------------------------------------------
    # K-Medoids + silhouette model selection
    #   clustering_metric="cosine"    : cosine distance / cosine silhouette
    #   clustering_metric="euclidean" : Euclidean distance / Euclidean silhouette
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def _fit_kmedoids_cosine(
        self,
        x_raw: Tensor,
        k: int,
        seed: int,
    ) -> Tuple[Tensor, Tensor]:
        """Alternating cosine K-Medoids with farthest-point initialization."""
        n = int(x_raw.shape[0])
        if k < 2 or k >= n:
            raise ValueError("K-Medoids requires 2 <= K < N.")

        x = F.normalize(x_raw.float(), p=2, dim=1, eps=self.eps)
        gen = torch.Generator(device=x.device)
        gen.manual_seed(int(seed))

        first = int(torch.randint(0, n, (1,), generator=gen, device=x.device).item())
        selected = [first]
        min_dist = 1.0 - (x @ x[first:first + 1].t()).squeeze(1)

        for _ in range(1, k):
            idx = int(min_dist.argmax().item())
            selected.append(idx)
            dist = 1.0 - (x @ x[idx:idx + 1].t()).squeeze(1)
            min_dist = torch.minimum(min_dist, dist)

        medoid_ids = torch.tensor(selected, device=x.device, dtype=torch.long)
        old_labels: Optional[Tensor] = None

        for _ in range(self.kmedoids_iterations):
            medoids = x.index_select(0, medoid_ids)
            labels = (x @ medoids.t()).argmax(dim=1)

            if old_labels is not None and torch.equal(labels, old_labels):
                break
            old_labels = labels.clone()

            new_ids = medoid_ids.clone()
            for cluster_id in range(k):
                ids = labels.eq(cluster_id).nonzero(as_tuple=False).squeeze(1)
                if ids.numel() == 0:
                    continue
                members = x.index_select(0, ids)
                cluster_sum = members.sum(dim=0, keepdim=True).t()  # [D,1]
                score = (members @ cluster_sum).squeeze(1)
                local_best = int(score.argmax().item())
                new_ids[cluster_id] = ids[local_best]

            if torch.equal(new_ids, medoid_ids):
                medoid_ids = new_ids
                break
            medoid_ids = new_ids

        medoids = x.index_select(0, medoid_ids)
        labels = (x @ medoids.t()).argmax(dim=1)
        return medoid_ids, labels

    @torch.no_grad()
    def _fit_kmedoids_euclidean(
        self,
        x_raw: Tensor,
        k: int,
        seed: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Euclidean K-Medoids-like clustering on RAW features.

        Assignment uses Euclidean distance. During the medoid update, the prototype
        is constrained to be an observed sample and is chosen as the member nearest
        to the arithmetic cluster mean. This exactly minimizes the sum of SQUARED
        Euclidean distances among candidate medoids, while avoiding an O(M^2)
        pairwise-distance matrix inside every cluster.
        """
        n = int(x_raw.shape[0])
        if k < 2 or k >= n:
            raise ValueError("K-Medoids requires 2 <= K < N.")

        # IMPORTANT: do NOT L2-normalize here. Euclidean mode intentionally uses
        # both feature direction and feature magnitude.
        x = x_raw.float()
        gen = torch.Generator(device=x.device)
        gen.manual_seed(int(seed))

        # Farthest-point initialization under squared Euclidean distance.
        first = int(torch.randint(0, n, (1,), generator=gen, device=x.device).item())
        selected = [first]
        min_dist2 = (x - x[first:first + 1]).pow(2).sum(dim=1)

        for _ in range(1, k):
            idx = int(min_dist2.argmax().item())
            selected.append(idx)
            dist2 = (x - x[idx:idx + 1]).pow(2).sum(dim=1)
            min_dist2 = torch.minimum(min_dist2, dist2)

        medoid_ids = torch.tensor(selected, device=x.device, dtype=torch.long)
        old_labels: Optional[Tensor] = None

        for _ in range(self.kmedoids_iterations):
            medoids = x.index_select(0, medoid_ids)
            # Squared Euclidean and Euclidean have the same nearest prototype.
            dist = torch.cdist(x, medoids, p=2)
            labels = dist.argmin(dim=1)

            if old_labels is not None and torch.equal(labels, old_labels):
                break
            old_labels = labels.clone()

            new_ids = medoid_ids.clone()
            for cluster_id in range(k):
                ids = labels.eq(cluster_id).nonzero(as_tuple=False).squeeze(1)
                if ids.numel() == 0:
                    continue

                members = x.index_select(0, ids)
                center = members.mean(dim=0, keepdim=True)
                dist2_to_center = (members - center).pow(2).sum(dim=1)
                local_best = int(dist2_to_center.argmin().item())
                new_ids[cluster_id] = ids[local_best]

            if torch.equal(new_ids, medoid_ids):
                medoid_ids = new_ids
                break
            medoid_ids = new_ids

        medoids = x.index_select(0, medoid_ids)
        labels = torch.cdist(x, medoids, p=2).argmin(dim=1)
        return medoid_ids, labels

    @torch.no_grad()
    def _euclidean_silhouette_score(
        self,
        x_raw: Tensor,
        labels: Tensor,
        seed: int,
    ) -> float:
        """Approximate Euclidean silhouette score on a stratified subset."""
        n = int(x_raw.shape[0])
        if n < 3:
            return float("-inf")

        unique = labels.unique(sorted=True)
        if unique.numel() < 2:
            return float("-inf")

        max_n = min(n, self.silhouette_sample_size)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))

        labels_cpu = labels.detach().cpu()
        selected: List[Tensor] = []
        base = max(1, max_n // int(unique.numel()))

        for c in unique.cpu().tolist():
            ids = labels_cpu.eq(int(c)).nonzero(as_tuple=False).squeeze(1)
            if ids.numel() == 0:
                continue
            take = min(int(ids.numel()), base)
            perm = torch.randperm(int(ids.numel()), generator=gen)[:take]
            selected.append(ids.index_select(0, perm))

        if len(selected) == 0:
            return float("-inf")

        idx_cpu = torch.cat(selected, dim=0)
        if idx_cpu.numel() < max_n:
            all_perm = torch.randperm(n, generator=gen)
            mark = torch.zeros(n, dtype=torch.bool)
            mark[idx_cpu] = True
            extra = all_perm[~mark[all_perm]][: max_n - idx_cpu.numel()]
            idx_cpu = torch.cat([idx_cpu, extra], dim=0)

        idx = idx_cpu.to(x_raw.device)
        x = x_raw.index_select(0, idx).float()
        y = labels.index_select(0, idx)

        # True Euclidean pairwise distance for silhouette.
        dist = torch.cdist(x, x, p=2)
        m = int(x.shape[0])
        silhouettes = torch.zeros(m, device=x.device, dtype=torch.float32)
        sampled_clusters = y.unique(sorted=True)

        for c in sampled_clusters.tolist():
            mask_c = y.eq(int(c))
            ids_c = mask_c.nonzero(as_tuple=False).squeeze(1)
            n_c = int(ids_c.numel())
            if n_c <= 1:
                silhouettes[ids_c] = 0.0
                continue

            d_rows = dist.index_select(0, ids_c)
            a = d_rows[:, mask_c].sum(dim=1) / float(n_c - 1)

            b = torch.full_like(a, float("inf"))
            for other in sampled_clusters.tolist():
                if int(other) == int(c):
                    continue
                mask_o = y.eq(int(other))
                if not bool(mask_o.any()):
                    continue
                mean_d = d_rows[:, mask_o].mean(dim=1)
                b = torch.minimum(b, mean_d)

            denom = torch.maximum(a, b).clamp_min(self.eps)
            s = (b - a) / denom
            silhouettes[ids_c] = torch.where(
                torch.isfinite(s), s, torch.zeros_like(s)
            )

        return float(silhouettes.mean().item())

    @torch.no_grad()
    def _cosine_silhouette_score(
        self,
        x_raw: Tensor,
        labels: Tensor,
        seed: int,
    ) -> float:
        """Approximate cosine silhouette score on a stratified subset."""
        n = int(x_raw.shape[0])
        if n < 3:
            return float("-inf")

        unique = labels.unique(sorted=True)
        if unique.numel() < 2:
            return float("-inf")

        max_n = min(n, self.silhouette_sample_size)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))

        labels_cpu = labels.detach().cpu()
        selected: List[Tensor] = []
        base = max(1, max_n // int(unique.numel()))

        for c in unique.cpu().tolist():
            ids = labels_cpu.eq(int(c)).nonzero(as_tuple=False).squeeze(1)
            if ids.numel() == 0:
                continue
            take = min(int(ids.numel()), base)
            perm = torch.randperm(int(ids.numel()), generator=gen)[:take]
            selected.append(ids.index_select(0, perm))

        if len(selected) == 0:
            return float("-inf")

        idx_cpu = torch.cat(selected, dim=0)
        if idx_cpu.numel() < max_n:
            all_perm = torch.randperm(n, generator=gen)
            mark = torch.zeros(n, dtype=torch.bool)
            mark[idx_cpu] = True
            extra = all_perm[~mark[all_perm]][: max_n - idx_cpu.numel()]
            idx_cpu = torch.cat([idx_cpu, extra], dim=0)

        idx = idx_cpu.to(x_raw.device)
        x = F.normalize(
            x_raw.index_select(0, idx).float(), p=2, dim=1, eps=self.eps
        )
        y = labels.index_select(0, idx)

        dist = (1.0 - x @ x.t()).clamp_min(0.0)
        m = int(x.shape[0])
        silhouettes = torch.zeros(m, device=x.device, dtype=torch.float32)
        sampled_clusters = y.unique(sorted=True)

        for c in sampled_clusters.tolist():
            mask_c = y.eq(int(c))
            ids_c = mask_c.nonzero(as_tuple=False).squeeze(1)
            n_c = int(ids_c.numel())
            if n_c <= 1:
                silhouettes[ids_c] = 0.0
                continue

            d_rows = dist.index_select(0, ids_c)
            a = d_rows[:, mask_c].sum(dim=1) / float(n_c - 1)

            b = torch.full_like(a, float("inf"))
            for other in sampled_clusters.tolist():
                if int(other) == int(c):
                    continue
                mask_o = y.eq(int(other))
                if not bool(mask_o.any()):
                    continue
                mean_d = d_rows[:, mask_o].mean(dim=1)
                b = torch.minimum(b, mean_d)

            denom = torch.maximum(a, b).clamp_min(self.eps)
            s = (b - a) / denom
            silhouettes[ids_c] = torch.where(
                torch.isfinite(s), s, torch.zeros_like(s)
            )

        return float(silhouettes.mean().item())

    @torch.no_grad()
    def _select_best_bank(
        self,
        candidate_pool_cpu: Tensor,
        bank_name: str,
        device: torch.device,
        seed_offset: int,
        verbose: bool,
    ) -> Tuple[Tensor, Tensor, int, Dict[int, float]]:
        if candidate_pool_cpu.ndim != 2 or candidate_pool_cpu.shape[0] < 3:
            raise RuntimeError(
                "{} candidate pool is too small for clustering: shape={}.".format(
                    bank_name, tuple(candidate_pool_cpu.shape)
                )
            )

        x = candidate_pool_cpu.to(device=device, dtype=torch.float32)
        n = int(x.shape[0])
        best_score = float("-inf")
        best_k: Optional[int] = None
        best_medoid_ids: Optional[Tensor] = None
        best_labels: Optional[Tensor] = None
        scores: Dict[int, float] = {}

        # Requested list semantics: [1,2,3,...] -> [C,2C,3C,...] clusters.
        # candidate_ks = [factor * self.num_classes for factor in self.prototype_k_factors]
        min_factor = min(self.prototype_k_factors)
        max_factor = max(self.prototype_k_factors)
        k_start = min_factor * self.num_classes
        k_end = max_factor * self.num_classes
        candidate_ks = list(range(k_start, k_end + 1))

        if verbose:
            print("\n[{} bank] candidate K: {}".format(bank_name, candidate_ks))

        for index, k in enumerate(candidate_ks):
            if k < 2 or k >= n:
                if verbose:
                    print("  K={:<4d} skipped (need 2 <= K < N={})".format(k, n))
                continue

            if self.clustering_metric == "cosine":
                medoid_ids, labels = self._fit_kmedoids_cosine(
                    x_raw=x,
                    k=k,
                    seed=self.random_seed + seed_offset + 97 * index,
                )
                score = self._cosine_silhouette_score(
                    x_raw=x,
                    labels=labels,
                    seed=self.random_seed + seed_offset + 193 * index,
                )
            else:
                medoid_ids, labels = self._fit_kmedoids_euclidean(
                    x_raw=x,
                    k=k,
                    seed=self.random_seed + seed_offset + 97 * index,
                )
                score = self._euclidean_silhouette_score(
                    x_raw=x,
                    labels=labels,
                    seed=self.random_seed + seed_offset + 193 * index,
                )

            scores[int(k)] = float(score)

            if verbose:
                print(
                    "  K={:<4d} {} silhouette={:.6f}".format(
                        k, self.clustering_metric, score
                    )
                )

            if score > best_score:
                best_score = score
                best_k = int(k)
                best_medoid_ids = medoid_ids.clone()
                best_labels = labels.clone()

        if best_k is None or best_medoid_ids is None or best_labels is None:
            raise RuntimeError(
                "No valid K for {} bank. Candidate pool N={}, factors={}. "
                "Reduce prototype_k_factors or collect more candidates.".format(
                    bank_name, n, self.prototype_k_factors
                )
            )

        medoids_raw = x.index_select(0, best_medoid_ids).detach().cpu()
        medoids_norm = F.normalize(
            medoids_raw.float(), p=2, dim=1, eps=self.eps
        ).cpu()

        if verbose:
            print(
                "[{} bank] selected K={} with {} silhouette={:.6f}".format(
                    bank_name, best_k, self.clustering_metric, best_score
                )
            )

        return medoids_raw, medoids_norm, best_k, scores

    # -------------------------------------------------------------------------
    # Public discovery
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def discover(
        self,
        model: nn.Module,
        reference_loader: Iterable,
        device: Union[str, torch.device],
        verbose: bool = True,
    ) -> DualPatternDiscoveryResult:
        """Discover Relevant/Nuisance/Irrelevant banks from dataloaders['val']."""
        device = torch.device(device)
        original_training = model.training
        model.eval()

        relevant_chunks: List[List[Tensor]] = [[] for _ in range(self.num_classes)]
        nuisance_chunks: List[List[Tensor]] = [[] for _ in range(self.num_classes)]
        irrelevant_chunks: List[List[Tensor]] = [[] for _ in range(self.num_classes)]

        total_images = total_correct = total_wrong = 0
        raw_rel = raw_nui = raw_irr = raw_ignored = 0

        for batch_idx, batch in enumerate(reference_loader):
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError(
                    "reference_loader must yield tuple/list with inputs and labels."
                )
            inputs = batch[0].to(device, non_blocking=True)
            labels = batch[-1].long().to(device, non_blocking=True)

            rel_dict, nui_dict, irr_dict, stats = self._extract_decision_aware_candidates(
                model=model, inputs=inputs, labels=labels
            )
            self._append_candidate_dict(relevant_chunks, rel_dict)
            self._append_candidate_dict(nuisance_chunks, nui_dict)
            self._append_candidate_dict(irrelevant_chunks, irr_dict)

            total_images += stats["num_images"]
            total_correct += stats["num_correct_images"]
            total_wrong += stats["num_wrong_images"]
            raw_rel += stats["num_relevant_candidates"]
            raw_nui += stats["num_nuisance_candidates"]
            raw_irr += stats["num_irrelevant_candidates"]
            raw_ignored += stats["num_ignored_strong_opposite"]

            if verbose and (batch_idx + 1) % 20 == 0:
                print(
                    "[Discovery] batches={} images={} correct={} wrong={} R={} N={} I={}".format(
                        batch_idx + 1, total_images, total_correct, total_wrong,
                        raw_rel, raw_nui, raw_irr,
                    )
                )

        relevant_pool, rel_counts = self._finalize_candidate_pool(
            relevant_chunks, seed_offset=10000
        )
        nuisance_pool, nui_counts = self._finalize_candidate_pool(
            nuisance_chunks, seed_offset=20000
        )
        irrelevant_pool, irr_counts = self._finalize_candidate_pool(
            irrelevant_chunks, seed_offset=25000
        )

        if relevant_pool.numel() == 0:
            raise RuntimeError(
                "No Relevant candidates: require relative patch contribution "
                "r_p > {:.3f}. Reduce decision_threshold if needed.".format(self.decision_threshold)
            )
        if nuisance_pool.numel() == 0:
            raise RuntimeError(
                "No Nuisance candidates: require relative patch contribution "
                "r_p < -{:.3f}. Reduce decision_threshold if needed.".format(self.decision_threshold)
            )
        if irrelevant_pool.numel() == 0:
            raise RuntimeError(
                "No Irrelevant candidates: require |r_p| <= {:.3f}. "
                "Increase decision_threshold if needed.".format(self.decision_threshold)
            )

        if verbose:
            print("\n========== Three-Bank R/N/I Discovery ==========")
            print("decision_threshold = {:.4f} (relative score in [-1,1])".format(self.decision_threshold))
            print("clustering_metric = {}".format(self.clustering_metric))
            print("images={} | correct={} | wrong={}".format(total_images, total_correct, total_wrong))
            print("raw R={} | raw N={} | raw I={} | coverage=100%".format(
                raw_rel, raw_nui, raw_irr
            ))
            print("kept R={} | kept N={} | kept I={}".format(
                relevant_pool.shape[0], nuisance_pool.shape[0], irrelevant_pool.shape[0]
            ))
            print("R per GT class = {}".format(rel_counts))
            print("N per GT class = {}".format(nui_counts))
            print("I per GT class = {}".format(irr_counts))

        cluster_device = self.clustering_device or device

        (
            self.relevant_medoids_raw,
            self.relevant_medoids_norm,
            self.relevant_best_k,
            rel_scores,
        ) = self._select_best_bank(
            candidate_pool_cpu=relevant_pool,
            bank_name="Relevant",
            device=cluster_device,
            seed_offset=30000,
            verbose=verbose,
        )
        (
            self.nuisance_medoids_raw,
            self.nuisance_medoids_norm,
            self.nuisance_best_k,
            nui_scores,
        ) = self._select_best_bank(
            candidate_pool_cpu=nuisance_pool,
            bank_name="Nuisance",
            device=cluster_device,
            seed_offset=40000,
            verbose=verbose,
        )
        (
            self.irrelevant_medoids_raw,
            self.irrelevant_medoids_norm,
            self.irrelevant_best_k,
            irr_scores,
        ) = self._select_best_bank(
            candidate_pool_cpu=irrelevant_pool,
            bank_name="Irrelevant",
            device=cluster_device,
            seed_offset=50000,
            verbose=verbose,
        )

        # Freeze the exact discovery representation space for Stage-II matching.
        self._assignment_model = copy.deepcopy(model).to(device)
        self._assignment_model.eval()
        for parameter in self._assignment_model.parameters():
            parameter.requires_grad_(False)
        self._assignment_device = device

        result = DualPatternDiscoveryResult(
            relevant_best_k=int(self.relevant_best_k),
            nuisance_best_k=int(self.nuisance_best_k),
            irrelevant_best_k=int(self.irrelevant_best_k),
            relevant_silhouette=rel_scores,
            nuisance_silhouette=nui_scores,
            irrelevant_silhouette=irr_scores,
            relevant_candidate_count=int(relevant_pool.shape[0]),
            nuisance_candidate_count=int(nuisance_pool.shape[0]),
            irrelevant_candidate_count=int(irrelevant_pool.shape[0]),
            relevant_candidate_count_per_class=rel_counts,
            nuisance_candidate_count_per_class=nui_counts,
            irrelevant_candidate_count_per_class=irr_counts,
            decision_threshold=self.decision_threshold,
        )
        self.discovery_result = result

        if original_training:
            model.train()
        if verbose:
            print("=================================================\n")
        return result

    # -------------------------------------------------------------------------
    # Stage-II frozen-bank assignment
    # -------------------------------------------------------------------------

    def _check_discovered(self) -> None:
        if (
            self.relevant_medoids_norm is None
            or self.nuisance_medoids_norm is None
            or self.irrelevant_medoids_norm is None
            or self._assignment_model is None
        ):
            raise RuntimeError(
                "Run discover(model, reference_loader, device) before Stage-II forward."
            )

    @torch.no_grad()
    def _max_bank_similarity(
        self,
        region_norm: Tensor,
        bank_norm: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Chunked maximum cosine similarity."""
        n = int(region_norm.shape[0])
        values: List[Tensor] = []
        ids: List[Tensor] = []
        for start in range(0, n, self.assignment_chunk_size):
            end = min(n, start + self.assignment_chunk_size)
            sim = region_norm[start:end] @ bank_norm.t()
            v, j = sim.max(dim=1)
            values.append(v)
            ids.append(j)
        return torch.cat(values, dim=0), torch.cat(ids, dim=0)

    @torch.no_grad()
    def _min_bank_euclidean_distance(
        self,
        region_raw: Tensor,
        bank_raw: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Chunked minimum Euclidean distance to a prototype bank."""
        n = int(region_raw.shape[0])
        values: List[Tensor] = []
        ids: List[Tensor] = []
        for start in range(0, n, self.assignment_chunk_size):
            end = min(n, start + self.assignment_chunk_size)
            dist = torch.cdist(region_raw[start:end], bank_raw, p=2)
            v, j = dist.min(dim=1)
            values.append(v)
            ids.append(j)
        return torch.cat(values, dim=0), torch.cat(ids, dim=0)

    @torch.no_grad()
    def _stage2_triple_bank_partition(
        self,
        inputs: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Assign every frozen layer4 patch to the NEAREST of R/N/I banks.

        clustering_metric='cosine': choose the largest cosine similarity.
        clustering_metric='euclidean': choose the smallest Euclidean distance
        (internally converted to score 1/(1+d), so the existing argmax API is kept).

        No bank_similarity_threshold is used:
            A_p = argmax {s_R(p), s_N(p), s_I(p)}.
        """
        self._check_discovered()
        assert self._assignment_model is not None
        assert self._assignment_device is not None
        assert self.relevant_medoids_norm is not None
        assert self.nuisance_medoids_norm is not None
        assert self.irrelevant_medoids_norm is not None

        x = inputs.to(self._assignment_device, non_blocking=True)
        _, features = _capture_layer4_and_forward(self._assignment_model, x)
        regions = self._regions_from_features(features)
        b, r, d = regions.shape

        if self.clustering_metric == "cosine":
            flat = F.normalize(
                regions.reshape(-1, d).float(), p=2, dim=1, eps=self.eps
            )
            rel_bank = self.relevant_medoids_norm.to(flat.device, flat.dtype)
            nui_bank = self.nuisance_medoids_norm.to(flat.device, flat.dtype)
            irr_bank = self.irrelevant_medoids_norm.to(flat.device, flat.dtype)

            rel_sim, rel_id = self._max_bank_similarity(flat, rel_bank)
            nui_sim, nui_id = self._max_bank_similarity(flat, nui_bank)
            irr_sim, irr_id = self._max_bank_similarity(flat, irr_bank)
        else:
            # Euclidean mode uses RAW feature magnitude + direction, consistent with
            # the Euclidean clustering performed during discover().
            flat = regions.reshape(-1, d).float()
            rel_bank = self.relevant_medoids_raw.to(flat.device, flat.dtype)
            nui_bank = self.nuisance_medoids_raw.to(flat.device, flat.dtype)
            irr_bank = self.irrelevant_medoids_raw.to(flat.device, flat.dtype)

            rel_dist, rel_id = self._min_bank_euclidean_distance(flat, rel_bank)
            nui_dist, nui_id = self._min_bank_euclidean_distance(flat, nui_bank)
            irr_dist, irr_id = self._min_bank_euclidean_distance(flat, irr_bank)

            # Keep the existing downstream API named '*_similarity': transform
            # distance monotonically to a higher-is-better score in (0, 1].
            rel_sim = 1.0 / (1.0 + rel_dist)
            nui_sim = 1.0 / (1.0 + nui_dist)
            irr_sim = 1.0 / (1.0 + irr_dist)

        rel_sim = rel_sim.view(b, r)
        nui_sim = nui_sim.view(b, r)
        irr_sim = irr_sim.view(b, r)
        rel_id = rel_id.view(b, r)
        nui_id = nui_id.view(b, r)
        irr_id = irr_id.view(b, r)

        # Deterministic tie rule via argmax: R (0) -> N (1) -> I (2).
        sims = torch.stack([rel_sim, nui_sim, irr_sim], dim=-1)  # [B,R,3]
        assignment_index = sims.argmax(dim=-1)
        relevant_mask = assignment_index.eq(0)
        nuisance_mask = assignment_index.eq(1)
        irrelevant_mask = assignment_index.eq(2)

        # Winner-vs-runner confidence, useful for ranking ablation strength.
        sorted_sims, _ = sims.sort(dim=-1, descending=True)
        assignment_confidence = sorted_sims[..., 0] - sorted_sims[..., 1]

        return {
            "relevant_mask": relevant_mask,
            "nuisance_mask": nuisance_mask,
            "irrelevant_mask": irrelevant_mask,
            "relevant_similarity": rel_sim,
            "nuisance_similarity": nui_sim,
            "irrelevant_similarity": irr_sim,
            "assignment_margin": rel_sim - nui_sim,
            "assignment_confidence": assignment_confidence,
            "assignment_index": assignment_index,
            "nearest_relevant_id": rel_id,
            "nearest_nuisance_id": nui_id,
            "nearest_irrelevant_id": irr_id,
        }

    # Backward-compatible alias: old callers still work, but the returned dict now
    # contains a genuine three-bank partition.
    def _stage2_dual_bank_partition(self, inputs: Tensor) -> Dict[str, Tensor]:
        return self._stage2_triple_bank_partition(inputs)

    # -------------------------------------------------------------------------
    # Pooling
    # -------------------------------------------------------------------------

    def _masked_avg_pool(
        self,
        regions: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Masked average pooling over layer4 patches."""
        if regions.ndim != 3 or mask.ndim != 2:
            raise ValueError("regions must be [B,R,D] and mask [B,R].")
        if regions.shape[:2] != mask.shape:
            raise ValueError("regions/mask shape mismatch.")

        m = mask.to(dtype=regions.dtype).unsqueeze(-1)
        count = m.sum(dim=1)
        pooled = (regions * m).sum(dim=1) / count.clamp_min(1.0)
        valid = count.squeeze(1).gt(0)
        return pooled, valid

    # -------------------------------------------------------------------------
    # CE-like objective shared by Z and Zr
    # -------------------------------------------------------------------------

    def _ce_like_to_class_loss(
        self,
        z: Tensor,
        labels: Tensor,
        classifier: nn.Linear,
        valid: Optional[Tensor] = None,
    ) -> Tensor:
        """
        logits(c) = cos(z,w_c)/tau + log pi_c
        L = CE(logits, y)

        prior pi_c = cls_num[c] / sum(cls_num).
        """
        if valid is None:
            valid = torch.ones(z.shape[0], device=z.device, dtype=torch.bool)
        if not bool(valid.any()):
            return z.sum() * 0.0

        z_v = z[valid]
        y_v = labels.long()[valid]
        
        z_v = F.normalize(z_v, p=2, dim=1, eps=self.eps)
        W = F.normalize(
            classifier.weight.to(device=z_v.device, dtype=z_v.dtype),
            p=2,
            dim=1,
            eps=self.eps,
        )
        cosine_logits = z_v @ W.t()
        prior_log = self.prior_log.to(device=z_v.device, dtype=z_v.dtype)
        logits_ce = cosine_logits / self.temperature + prior_log.unsqueeze(0)
        return F.cross_entropy(logits_ce, y_v)

    def _global_to_class_loss(
        self,
        z_global: Tensor,
        labels: Tensor,
        classifier: nn.Linear,
    ) -> Tensor:
        """Requested surrogate for maximizing I(Z;Y)."""
        return self._ce_like_to_class_loss(
            z=z_global,
            labels=labels,
            classifier=classifier,
            valid=None,
        )

    def _relevant_region_to_class_loss(
        self,
        z_relevant: Tensor,
        valid_relevant: Tensor,
        labels: Tensor,
        classifier: nn.Linear,
    ) -> Tensor:
        """Requested surrogate for maximizing I(Zr;Y)."""
        return self._ce_like_to_class_loss(
            z=z_relevant,
            labels=labels,
            classifier=classifier,
            valid=valid_relevant,
        )

    # -------------------------------------------------------------------------
    # nHSIC for minimizing dependence I(Zn;Y) through W_Y = w_y
    # -------------------------------------------------------------------------

    def _rbf_kernel_median(self, x: Tensor) -> Tensor:
        """RBF kernel with detached median heuristic bandwidth."""
        x2 = (x * x).sum(dim=1, keepdim=True)
        dist2 = (x2 + x2.t() - 2.0 * (x @ x.t())).clamp_min(0.0)

        n = int(x.shape[0])
        upper = torch.triu_indices(n, n, offset=1, device=x.device)
        pair_dist2 = dist2[upper[0], upper[1]].detach()
        positive = pair_dist2[pair_dist2 > self.eps]

        if positive.numel() > 0:
            sigma2 = positive.median().clamp_min(self.eps)
        else:
            sigma2 = dist2.new_tensor(1.0)

        return torch.exp(-dist2 / (2.0 * sigma2))

    @staticmethod
    def _center_kernel(k: Tensor) -> Tensor:
        return (
            k
            - k.mean(dim=0, keepdim=True)
            - k.mean(dim=1, keepdim=True)
            + k.mean()
        )

    def _normalized_hsic(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Normalized RBF-HSIC / centered-kernel alignment.

        Population HSIC with characteristic RBF kernels has the same independence
        zero point as mutual information: HSIC=0 iff independence (under standard
        conditions). We use normalized HSIC as a scale-stable dependence surrogate;
        it is NOT asserted to be numerically equal to MI.
        """
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("x,y must be [N,Dx]/[N,Dy] with the same N.")
        n = int(x.shape[0])
        if n < 2:
            return (x.sum() + y.sum()) * 0.0

        x = F.normalize(x, p=2, dim=1, eps=self.eps)
        y = F.normalize(y, p=2, dim=1, eps=self.eps)

        kx = self._center_kernel(self._rbf_kernel_median(x))
        ky = self._center_kernel(self._rbf_kernel_median(y))

        norm_x = (kx * kx).sum()
        norm_y = (ky * ky).sum()
        if float(norm_x.detach().item()) <= self.eps or float(norm_y.detach().item()) <= self.eps:
            return x.sum() * 0.0

        numerator = (kx * ky).sum()
        denominator = torch.sqrt(
            norm_x.clamp_min(self.eps) * norm_y.clamp_min(self.eps)
        ).clamp_min(self.eps)

        nhsic = numerator / denominator
        # Numerical roundoff can produce tiny negative values.
        return nhsic.clamp_min(0.0)

    def _nuisance_to_class_nhsic_loss(
        self,
        z_nuisance: Tensor,
        valid_nuisance: Tensor,
        labels: Tensor,
        classifier: nn.Linear,
    ) -> Tensor:
        """
        L_N = nHSIC(Zn, W_Y), where W_Y = w_y.

        W_Y is detached, therefore the nuisance loss cannot be minimized by simply
        rotating the classifier. It must reduce dependence carried by Zn.
        """
        valid = valid_nuisance
        if int(valid.sum().item()) < 2:
            return z_nuisance.sum() * 0.0

        zn = z_nuisance[valid]
        y = labels.long()[valid]

        W = classifier.weight.detach().to(device=zn.device, dtype=zn.dtype)
        wy = W.index_select(0, y)

        return self._normalized_hsic(zn, wy)

    # -------------------------------------------------------------------------
    # Public Stage-II forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        student_model: nn.Module,
        inputs: Tensor,
        labels: Tensor,
    ) -> DualPatternOutput:
        """
        Stage-II forward:
          1) current student -> layer4, Z, ordinary logits
          2) frozen snapshot -> nearest R/N/I bank assignment
          3) current student patches + frozen masks -> Zr, Zn, Zi
          4) compute L_G, L_R, L_N; Zi is reference-only by default
        """
        self._check_discovered()

        logits, student_features = _capture_layer4_and_forward(student_model, inputs)
        if logits.shape[1] != self.num_classes:
            raise ValueError(
                "Expected {} classes, got {}.".format(self.num_classes, logits.shape[1])
            )

        student_regions = self._regions_from_features(student_features)
        z_global = student_features.mean(dim=(2, 3))
        classifier = _get_classifier(student_model)
        labels = labels.long().to(logits.device)

        partition = self._stage2_triple_bank_partition(inputs)
        relevant_mask = partition["relevant_mask"].to(student_regions.device)
        nuisance_mask = partition["nuisance_mask"].to(student_regions.device)
        irrelevant_mask = partition["irrelevant_mask"].to(student_regions.device)

        if relevant_mask.shape != student_regions.shape[:2]:
            raise RuntimeError(
                "Frozen assignment region shape {} != student region shape {}.".format(
                    tuple(relevant_mask.shape), tuple(student_regions.shape[:2])
                )
            )

        # Exactly one of R/N/I per patch.
        covered = relevant_mask | nuisance_mask | irrelevant_mask
        if not bool(covered.all()):
            raise RuntimeError("Three-bank partition did not cover all patches.")
        if bool((relevant_mask & nuisance_mask).any()) or bool((relevant_mask & irrelevant_mask).any()) or bool((nuisance_mask & irrelevant_mask).any()):
            raise RuntimeError("Three-bank partition masks overlap.")

        z_relevant, valid_relevant = self._masked_avg_pool(student_regions, relevant_mask)
        z_nuisance, valid_nuisance = self._masked_avg_pool(student_regions, nuisance_mask)
        z_irrelevant, valid_irrelevant = self._masked_avg_pool(student_regions, irrelevant_mask)

        loss_global = self._global_to_class_loss(
            z_global=z_global, labels=labels, classifier=classifier
        )
        loss_relevant = self._relevant_region_to_class_loss(
            z_relevant=z_relevant,
            valid_relevant=valid_relevant,
            labels=labels,
            classifier=classifier,
        )
        loss_nuisance = self._nuisance_to_class_nhsic_loss(
            z_nuisance=z_nuisance,
            valid_nuisance=valid_nuisance,
            labels=labels,
            classifier=classifier,
        )
        loss_region = loss_global + loss_relevant + loss_nuisance

        return DualPatternOutput(
            loss_region=loss_region,
            loss_global=loss_global,
            loss_relevant=loss_relevant,
            loss_nuisance=loss_nuisance,
            logits=logits,
            z_global=z_global,
            student_regions=student_regions,
            relevant_mask=relevant_mask.detach(),
            nuisance_mask=nuisance_mask.detach(),
            irrelevant_mask=irrelevant_mask.detach(),
            relevant_similarity=partition["relevant_similarity"].to(student_regions.device).detach(),
            nuisance_similarity=partition["nuisance_similarity"].to(student_regions.device).detach(),
            irrelevant_similarity=partition["irrelevant_similarity"].to(student_regions.device).detach(),
            assignment_margin=partition["assignment_margin"].to(student_regions.device).detach(),
            z_relevant=z_relevant,
            z_nuisance=z_nuisance,
            z_irrelevant=z_irrelevant,
            valid_relevant=valid_relevant.detach(),
            valid_nuisance=valid_nuisance.detach(),
            valid_irrelevant=valid_irrelevant.detach(),
            num_relevant_regions=int(relevant_mask.sum().item()),
            num_nuisance_regions=int(nuisance_mask.sum().item()),
            num_irrelevant_regions=int(irrelevant_mask.sum().item()),
            num_valid_relevant_images=int(valid_relevant.sum().item()),
            num_valid_nuisance_images=int(valid_nuisance.sum().item()),
            num_valid_irrelevant_images=int(valid_irrelevant.sum().item()),
            loss_disentangle=None,
        )

    __call__ = forward

    # -------------------------------------------------------------------------
    # Loss combination
    # -------------------------------------------------------------------------

    def total_loss(
        self,
        output: DualPatternOutput,
        lambda_global: Optional[float] = 1.0,
        lambda_relevant: Optional[float] = 0.0,
        lambda_nuisance: Optional[float] = 0.5,
    ) -> Tensor:
        """
        Exact requested three-term objective (no ordinary-logit CE added):

            lambda_G * L_G + lambda_R * L_R + lambda_N * L_N
        """
        if lambda_global is None:
            lambda_global = self.lambda_global
        if lambda_relevant is None:
            lambda_relevant = self.lambda_relevant
        if lambda_nuisance is None:
            lambda_nuisance = self.lambda_nuisance

        return (
            float(lambda_global) * output.loss_global
            + float(lambda_relevant) * output.loss_relevant
            + float(lambda_nuisance) * output.loss_nuisance
        )

    def combine_with_classification_loss(
        self,
        classification_loss: Tensor,
        output: DualPatternOutput,
        lambda_relevant: Optional[float] = None,
        lambda_nuisance: Optional[float] = None,
        lambda_global: Optional[float] = None,
    ) -> Tensor:
        """
        Backward-compatible helper if you still want the original ordinary CE:

            CE(logits,y) + lambda_G*L_G + lambda_R*L_R + lambda_N*L_N
        """
        if lambda_global is None:
            lambda_global = self.lambda_global
        if lambda_relevant is None:
            lambda_relevant = self.lambda_relevant
        if lambda_nuisance is None:
            lambda_nuisance = self.lambda_nuisance

        loss = (
            classification_loss
            + float(lambda_global) * output.loss_global
            + float(lambda_relevant) * output.loss_relevant
            + float(lambda_nuisance) * output.loss_nuisance
        )
        return loss

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_classwise_partition(
        self,
        student_model: nn.Module,
        data_loader: Iterable,
        device: Union[str, torch.device],
    ) -> Dict[str, Tensor]:
        """Report classwise R/N/I nearest-bank patch ratios and similarities."""
        self._check_discovered()
        device = torch.device(device)
        original_training = student_model.training
        student_model.eval()

        image_count = torch.zeros(self.num_classes, dtype=torch.long)
        total_patch_count = torch.zeros(self.num_classes, dtype=torch.long)
        rel_patch_count = torch.zeros(self.num_classes, dtype=torch.long)
        nui_patch_count = torch.zeros(self.num_classes, dtype=torch.long)
        irr_patch_count = torch.zeros(self.num_classes, dtype=torch.long)
        rel_sim_sum = torch.zeros(self.num_classes, dtype=torch.float64)
        nui_sim_sum = torch.zeros(self.num_classes, dtype=torch.float64)
        irr_sim_sum = torch.zeros(self.num_classes, dtype=torch.float64)
        sim_count = torch.zeros(self.num_classes, dtype=torch.long)

        for batch in data_loader:
            inputs = batch[0].to(device, non_blocking=True)
            labels = batch[-1].long().to(device, non_blocking=True)
            part = self._stage2_triple_bank_partition(inputs)
            rel = part["relevant_mask"].cpu()
            nui = part["nuisance_mask"].cpu()
            irr = part["irrelevant_mask"].cpu()
            sr = part["relevant_similarity"].cpu().double()
            sn = part["nuisance_similarity"].cpu().double()
            si = part["irrelevant_similarity"].cpu().double()
            labels_cpu = labels.cpu()

            for c in range(self.num_classes):
                m = labels_cpu.eq(c)
                if not bool(m.any()):
                    continue
                n_img = int(m.sum().item())
                image_count[c] += n_img
                total_patch_count[c] += int(rel[m].numel())
                rel_patch_count[c] += int(rel[m].sum().item())
                nui_patch_count[c] += int(nui[m].sum().item())
                irr_patch_count[c] += int(irr[m].sum().item())
                rel_sim_sum[c] += float(sr[m].sum().item())
                nui_sim_sum[c] += float(sn[m].sum().item())
                irr_sim_sum[c] += float(si[m].sum().item())
                sim_count[c] += int(sr[m].numel())

        denom = total_patch_count.clamp_min(1).float()
        sim_denom = sim_count.clamp_min(1).double()
        if original_training:
            student_model.train()
        return {
            "num_images": image_count,
            "num_total_patches": total_patch_count,
            "num_relevant_patches": rel_patch_count,
            "num_nuisance_patches": nui_patch_count,
            "num_irrelevant_patches": irr_patch_count,
            "relevant_ratio": rel_patch_count.float() / denom,
            "nuisance_ratio": nui_patch_count.float() / denom,
            "irrelevant_ratio": irr_patch_count.float() / denom,
            "mean_relevant_similarity": (rel_sim_sum / sim_denom).float(),
            "mean_nuisance_similarity": (nui_sim_sum / sim_denom).float(),
            "mean_irrelevant_similarity": (irr_sim_sum / sim_denom).float(),
        }

    # -------------------------------------------------------------------------
    # Visualization + causal layer4 ablation
    # -------------------------------------------------------------------------

    @staticmethod
    def _batch_iterator(
        inputs_or_loader: Union[Tensor, Iterable],
        labels: Optional[Tensor],
    ):
        if torch.is_tensor(inputs_or_loader):
            if labels is None:
                raise ValueError("labels must be supplied when inputs is a Tensor.")
            yield inputs_or_loader, labels
            return

        for batch in inputs_or_loader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError(
                    "DataLoader must yield tuple/list; first item=input, last item=label."
                )
            yield batch[0], batch[-1]

    @staticmethod
    def _ratio_tag(ratio: float) -> str:
        return "{:g}".format(100.0 * float(ratio)).replace(".", "p")

    def _top_fraction_mask(
        self,
        candidate_mask: Tensor,
        score: Tensor,
        ratio: float,
    ) -> Tensor:
        """Select top ceil(ratio * number_of_candidate_patches) within one image."""
        if candidate_mask.ndim != 1 or score.ndim != 1:
            raise ValueError("candidate_mask and score must be flat [R].")
        out = torch.zeros_like(candidate_mask, dtype=torch.bool)
        ids = candidate_mask.nonzero(as_tuple=False).squeeze(1)
        n = int(ids.numel())
        if n == 0 or ratio <= 0:
            return out
        k = max(1, int(math.ceil(float(ratio) * n)))
        k = min(k, n)
        local_score = score.index_select(0, ids)
        selected_local = torch.topk(local_score, k=k, largest=True).indices
        selected = ids.index_select(0, selected_local)
        out[selected] = True
        return out

    @torch.no_grad()
    def visualize_partition(
        self,
        student_model: nn.Module,
        inputs: Union[Tensor, Iterable],
        labels: Optional[Tensor] = None,
        save_dir: str = "./triple_bank_vis",
        max_images: int = 20,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        removal_ratios: Sequence[float] = (0.05, 0.10, 0.20),
        display: bool = False,
        keep_relevant_ratios: Optional[Sequence[float]] = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00),
    ) -> List[str]:
        """
        Visualize nearest-bank R/N/I patches and evaluate TWO complementary diagnostics.

        A) Removal faithfulness
        -----------------------
        Remove the strongest R or N patches from the ORIGINAL layer4 representation.
        After each intervention, recompute the FULL logits and Softmax probabilities.

            remove R -> expect P_GT to decrease
            remove N -> expect P_GT to increase

        B) Relevant-patch sufficiency / saturation
        -----------------------------------------
        Keep ONLY the top-ratio Relevant patches and zero every other layer4 patch
        (remaining Relevant + all Nuisance + all Irrelevant patches). This directly asks:

            "How many Relevant patches are sufficient for the GT decision?"

        For every keep ratio r, report:
            - absolute P_GT after Softmax, in %
            - dP_GT relative to the original full representation, in percentage points
            - marginal P_GT gain over the previous keep ratio, in percentage points
            - predicted class and whether prediction == GT

        The smallest tested keep ratio that already predicts GT is stored per image as
        min_keep_relevant_ratio_for_gt_prediction.

        Assignment remains purely nearest-bank:
            A_p = argmax {s_R, s_N, s_I}; no bank similarity threshold.

        Fixed original competitor:
            c* = argmax_{c != y} logit_c.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        self._check_discovered()
        os.makedirs(save_dir, exist_ok=True)

        removal = sorted(set(float(r) for r in removal_ratios))
        if len(removal) == 0 or any(r <= 0 or r > 1 for r in removal):
            raise ValueError("removal_ratios must contain values in (0,1].")

        if keep_relevant_ratios is None:
            keep_ratios = list(removal)
        else:
            keep_ratios = sorted(set(float(r) for r in keep_relevant_ratios))
        if len(keep_ratios) == 0 or any(r <= 0 or r > 1 for r in keep_ratios):
            raise ValueError("keep_relevant_ratios must contain values in (0,1].")

        original_training = student_model.training
        student_model.eval()
        device = next(student_model.parameters()).device
        paths: List[str] = []
        records: List[Dict[str, Any]] = []
        global_index = 0

        def _to_image(t: Tensor):
            t = t.detach().float().cpu()
            if mean is not None and std is not None:
                mean_t = torch.as_tensor(mean, dtype=t.dtype).view(-1, 1, 1)
                std_t = torch.as_tensor(std, dtype=t.dtype).view(-1, 1, 1)
                if mean_t.shape[0] != t.shape[0] or std_t.shape[0] != t.shape[0]:
                    raise ValueError("mean/std channel count must match input channels.")
                t = t * std_t + mean_t
            elif float(t.min().item()) < 0.0 or float(t.max().item()) > 1.0:
                lo = t.amin(dim=(1, 2), keepdim=True)
                hi = t.amax(dim=(1, 2), keepdim=True)
                t = (t - lo) / (hi - lo).clamp_min(1e-8)
            t = t.clamp(0.0, 1.0)
            if t.shape[0] == 1:
                return t[0].numpy()
            if t.shape[0] >= 3:
                return t[:3].permute(1, 2, 0).numpy()
            if t.shape[0] == 2:
                z = torch.zeros_like(t[:1])
                return torch.cat([t, z], dim=0).permute(1, 2, 0).numpy()
            raise ValueError("Unsupported input channel count: {}".format(int(t.shape[0])))

        def _to_rgb(img):
            if img.ndim == 2:
                return np.repeat(img[..., None], 3, axis=2)
            if img.shape[-1] == 1:
                return np.repeat(img, 3, axis=2)
            return img[..., :3]

        def _upsample_grid(grid: Tensor, hf: int, wf: int, H: int, W: int):
            g = grid.detach().float().view(1, 1, hf, wf)
            up = F.interpolate(g, size=(H, W), mode="nearest")
            return up[0, 0].cpu().numpy()

        def _overlay_partition(img, rel_mask, nui_mask, irr_mask, alpha=0.42):
            base = _to_rgb(img).astype(np.float32).copy()
            masks = [rel_mask > 0.5, nui_mask > 0.5, irr_mask > 0.5]
            colors = [
                np.asarray([0.10, 0.90, 0.20], dtype=np.float32),
                np.asarray([0.95, 0.12, 0.12], dtype=np.float32),
                np.asarray([0.15, 0.45, 0.95], dtype=np.float32),
            ]
            for m, color in zip(masks, colors):
                base[m] = (1.0 - alpha) * base[m] + alpha * color
            return np.clip(base, 0.0, 1.0)

        def _draw_patch_grid(ax, hf: int, wf: int, H: int, W: int):
            for gx in range(1, wf):
                ax.axvline(gx * W / float(wf) - 0.5, linewidth=0.45, alpha=0.50)
            for gy in range(1, hf):
                ax.axhline(gy * H / float(hf) - 0.5, linewidth=0.45, alpha=0.50)

        def _draw_relative_score_grid(
            ax,
            score_grid: Tensor,
            hf: int,
            wf: int,
            H: int,
            W: int,
        ):
            """Draw per-patch normalized GT-vs-competitor contribution in [-1,1]."""
            import matplotlib.patheffects as pe

            scores = score_grid.detach().float().cpu().view(hf, wf)
            cell_h = H / float(hf)
            cell_w = W / float(wf)
            fontsize = max(5.0, min(9.0, 62.0 / float(max(hf, wf))))

            for gy in range(hf):
                for gx in range(wf):
                    value = float(scores[gy, gx].item())
                    x_center = (gx + 0.5) * cell_w - 0.5
                    y_center = (gy + 0.5) * cell_h - 0.5
                    txt = ax.text(
                        x_center,
                        y_center,
                        "{:+.2f}".format(value),
                        ha="center",
                        va="center",
                        fontsize=fontsize,
                        color="white",
                        fontweight="bold",
                        clip_on=True,
                    )
                    txt.set_path_effects([
                        pe.Stroke(linewidth=1.6, foreground="black"),
                        pe.Normal(),
                    ])

        for batch_inputs, batch_labels in self._batch_iterator(inputs, labels):
            if max_images is not None and int(max_images) > 0 and len(paths) >= int(max_images):
                break

            x = batch_inputs.to(device, non_blocking=True)
            y = batch_labels.long().to(device, non_blocking=True)

            part = self._stage2_triple_bank_partition(x)
            base_logits, feat = _capture_layer4_and_forward(student_model, x)
            base_prob = F.softmax(base_logits, dim=1)
            pred = base_logits.argmax(dim=1)

            other_logits = base_logits.detach().clone()
            row = torch.arange(base_logits.shape[0], device=device)
            other_logits[row, y] = -torch.inf
            competitor = other_logits.argmax(dim=1)

            # Visualization-only quantity: the SAME normalized GT-vs-competitor
            # patch contribution used by Stage-I discovery, evaluated on the current
            # student_model feature map shown in this figure. This does NOT change
            # discovery, bank assignment, ablation, loss, or any training behavior.
            score_regions = self._regions_from_features(feat)  # [B,R,D]
            score_classifier = _get_classifier(student_model)
            score_W = score_classifier.weight.detach().to(
                device=score_regions.device, dtype=score_regions.dtype
            )
            score_w_gt = score_W.index_select(0, y)
            score_w_comp = score_W.index_select(0, competitor)
            score_c_gt = torch.einsum("brd,bd->br", score_regions, score_w_gt)
            score_c_comp = torch.einsum("brd,bd->br", score_regions, score_w_comp)
            patch_relative_score = (score_c_gt - score_c_comp) / (
                score_c_gt.abs() + score_c_comp.abs() + self.eps
            )
            patch_relative_score = patch_relative_score.clamp(-1.0, 1.0)

            hf, wf = int(feat.shape[-2]), int(feat.shape[-1])
            R = hf * wf
            if part["relevant_mask"].shape[1] != R:
                raise RuntimeError("Frozen assignment patch grid != current layer4 grid.")

            for i in range(int(x.shape[0])):
                if max_images is not None and int(max_images) > 0 and len(paths) >= int(max_images):
                    break

                yi = int(y[i].item())
                pi = int(pred[i].item())
                ci = int(competitor[i].item())

                orig_pgt = float(base_prob[i, yi].item())
                orig_pcomp = float(base_prob[i, ci].item())
                orig_prob_gap = orig_pgt - orig_pcomp
                orig_logit_gt = float(base_logits[i, yi].item())
                orig_logit_comp = float(base_logits[i, ci].item())
                orig_logit_margin = orig_logit_gt - orig_logit_comp

                rel_mask = part["relevant_mask"][i].to(device)
                nui_mask = part["nuisance_mask"][i].to(device)
                irr_mask = part["irrelevant_mask"][i].to(device)
                rel_sim = part["relevant_similarity"][i].to(device)
                nui_sim = part["nuisance_similarity"][i].to(device)
                irr_sim = part["irrelevant_similarity"][i].to(device)

                # Rank patches within a selected bank by winner-vs-best-alternative score.
                rel_strength = rel_sim - torch.maximum(nui_sim, irr_sim)
                nui_strength = nui_sim - torch.maximum(rel_sim, irr_sim)

                num_rel = int(rel_mask.sum().item())
                num_nui = int(nui_mask.sum().item())
                num_irr = int(irr_mask.sum().item())

                record: Dict[str, Any] = {
                    "sample_index": global_index,
                    "gt": yi,
                    "pred": pi,
                    "competitor": ci,
                    "correct": int(yi == pi),
                    "orig_pgt": orig_pgt,
                    "orig_pgt_pct": 100.0 * orig_pgt,
                    "orig_pcomp": orig_pcomp,
                    "orig_pcomp_pct": 100.0 * orig_pcomp,
                    "orig_prob_gap": orig_prob_gap,
                    "orig_logit_gt": orig_logit_gt,
                    "orig_logit_comp": orig_logit_comp,
                    "orig_logit_margin": orig_logit_margin,
                    "num_relevant": num_rel,
                    "num_nuisance": num_nui,
                    "num_irrelevant": num_irr,
                }

                xi = x[i:i + 1]

                # =============================================================
                # A) Removal faithfulness: remove top R / remove top N.
                # =============================================================
                ablation_rows: List[Dict[str, Any]] = []
                for ratio in removal:
                    remove_rel = self._top_fraction_mask(rel_mask, rel_strength, ratio)
                    remove_nui = self._top_fraction_mask(nui_mask, nui_strength, ratio)

                    logits_rel = _forward_with_layer4_zero_mask(
                        student_model, xi, remove_rel.view(1, -1)
                    )
                    logits_nui = _forward_with_layer4_zero_mask(
                        student_model, xi, remove_nui.view(1, -1)
                    )
                    prob_rel = F.softmax(logits_rel, dim=1)[0]
                    prob_nui = F.softmax(logits_nui, dim=1)[0]

                    rel_pgt = float(prob_rel[yi].item())
                    rel_pcomp = float(prob_rel[ci].item())
                    rel_prob_gap = rel_pgt - rel_pcomp
                    rel_logit_margin = float((logits_rel[0, yi] - logits_rel[0, ci]).item())
                    rel_pred = int(logits_rel.argmax(dim=1)[0].item())

                    nui_pgt = float(prob_nui[yi].item())
                    nui_pcomp = float(prob_nui[ci].item())
                    nui_prob_gap = nui_pgt - nui_pcomp
                    nui_logit_margin = float((logits_nui[0, yi] - logits_nui[0, ci]).item())
                    nui_pred = int(logits_nui.argmax(dim=1)[0].item())

                    rel_delta_pgt = rel_pgt - orig_pgt
                    rel_delta_pcomp = rel_pcomp - orig_pcomp
                    rel_delta_prob_gap = rel_prob_gap - orig_prob_gap
                    rel_delta_logit_margin = rel_logit_margin - orig_logit_margin

                    nui_delta_pgt = nui_pgt - orig_pgt
                    nui_delta_pcomp = nui_pcomp - orig_pcomp
                    nui_delta_prob_gap = nui_prob_gap - orig_prob_gap
                    nui_delta_logit_margin = nui_logit_margin - orig_logit_margin

                    rel_delta_pgt_pct = 100.0 * rel_delta_pgt
                    rel_delta_pcomp_pct = 100.0 * rel_delta_pcomp
                    nui_delta_pgt_pct = 100.0 * nui_delta_pgt
                    nui_delta_pcomp_pct = 100.0 * nui_delta_pcomp

                    tag = self._ratio_tag(ratio)
                    vals = {
                        f"rel_pgt_{tag}pct": rel_pgt,
                        f"rel_pcomp_{tag}pct": rel_pcomp,
                        f"rel_prob_gap_{tag}pct": rel_prob_gap,
                        f"rel_logit_margin_{tag}pct": rel_logit_margin,
                        f"rel_delta_pgt_{tag}pct": rel_delta_pgt,
                        f"rel_delta_pcomp_{tag}pct": rel_delta_pcomp,
                        f"rel_delta_prob_gap_{tag}pct": rel_delta_prob_gap,
                        f"rel_delta_logit_margin_{tag}pct": rel_delta_logit_margin,
                        f"rel_delta_pgt_{tag}pct_points": rel_delta_pgt_pct,
                        f"rel_delta_pcomp_{tag}pct_points": rel_delta_pcomp_pct,
                        f"rel_pred_{tag}pct": rel_pred,
                        f"rel_broken_{tag}pct": int(pi == yi and rel_pred != yi),
                        f"nui_pgt_{tag}pct": nui_pgt,
                        f"nui_pcomp_{tag}pct": nui_pcomp,
                        f"nui_prob_gap_{tag}pct": nui_prob_gap,
                        f"nui_logit_margin_{tag}pct": nui_logit_margin,
                        f"nui_delta_pgt_{tag}pct": nui_delta_pgt,
                        f"nui_delta_pcomp_{tag}pct": nui_delta_pcomp,
                        f"nui_delta_prob_gap_{tag}pct": nui_delta_prob_gap,
                        f"nui_delta_logit_margin_{tag}pct": nui_delta_logit_margin,
                        f"nui_delta_pgt_{tag}pct_points": nui_delta_pgt_pct,
                        f"nui_delta_pcomp_{tag}pct_points": nui_delta_pcomp_pct,
                        f"nui_pred_{tag}pct": nui_pred,
                        f"nui_corrected_{tag}pct": int(pi != yi and nui_pred == yi),
                    }
                    record.update(vals)
                    ablation_rows.append({
                        "ratio": ratio,
                        "rel_delta_margin": rel_delta_logit_margin,
                        "rel_delta_gap": rel_delta_prob_gap,
                        "rel_delta_pgt": rel_delta_pgt,
                        "rel_delta_pcomp": rel_delta_pcomp,
                        "rel_delta_pgt_pct": rel_delta_pgt_pct,
                        "rel_delta_pcomp_pct": rel_delta_pcomp_pct,
                        "rel_pred": rel_pred,
                        "nui_delta_margin": nui_delta_logit_margin,
                        "nui_delta_gap": nui_delta_prob_gap,
                        "nui_delta_pgt": nui_delta_pgt,
                        "nui_delta_pcomp": nui_delta_pcomp,
                        "nui_delta_pgt_pct": nui_delta_pgt_pct,
                        "nui_delta_pcomp_pct": nui_delta_pcomp_pct,
                        "nui_pred": nui_pred,
                    })

                # =============================================================
                # B) Relevant sufficiency curve: keep ONLY top-ratio R patches.
                #    All unkept R + ALL N + ALL I are zeroed.
                # =============================================================
                keep_rows: List[Dict[str, Any]] = []
                previous_keep_pgt: Optional[float] = None
                min_keep_ratio_for_gt: Optional[float] = None

                for ratio in keep_ratios:
                    tag = self._ratio_tag(ratio)

                    if num_rel <= 0:
                        keep_num = 0
                        keep_pgt = float("nan")
                        keep_pcomp = float("nan")
                        keep_prob_gap = float("nan")
                        keep_logit_margin = float("nan")
                        keep_pred = -1
                        keep_is_gt = 0
                        keep_delta_pgt = float("nan")
                        keep_delta_pcomp = float("nan")
                        keep_marginal_pgt = float("nan")
                    else:
                        keep_rel = self._top_fraction_mask(rel_mask, rel_strength, ratio)
                        keep_num = int(keep_rel.sum().item())

                        # ONLY kept Relevant patches survive.
                        remove_everything_else = ~keep_rel
                        logits_keep = _forward_with_layer4_zero_mask(
                            student_model, xi, remove_everything_else.view(1, -1)
                        )
                        prob_keep = F.softmax(logits_keep, dim=1)[0]

                        keep_pgt = float(prob_keep[yi].item())
                        keep_pcomp = float(prob_keep[ci].item())
                        keep_prob_gap = keep_pgt - keep_pcomp
                        keep_logit_margin = float(
                            (logits_keep[0, yi] - logits_keep[0, ci]).item()
                        )
                        keep_pred = int(logits_keep.argmax(dim=1)[0].item())
                        keep_is_gt = int(keep_pred == yi)
                        keep_delta_pgt = keep_pgt - orig_pgt
                        keep_delta_pcomp = keep_pcomp - orig_pcomp

                        if previous_keep_pgt is None:
                            keep_marginal_pgt = float("nan")
                        else:
                            keep_marginal_pgt = keep_pgt - previous_keep_pgt
                        previous_keep_pgt = keep_pgt

                        if keep_is_gt and min_keep_ratio_for_gt is None:
                            min_keep_ratio_for_gt = float(ratio)

                    keep_delta_pgt_pct = 100.0 * keep_delta_pgt if math.isfinite(keep_delta_pgt) else float("nan")
                    keep_delta_pcomp_pct = 100.0 * keep_delta_pcomp if math.isfinite(keep_delta_pcomp) else float("nan")
                    keep_marginal_pgt_pct = 100.0 * keep_marginal_pgt if math.isfinite(keep_marginal_pgt) else float("nan")

                    record.update({
                        f"keep_rel_num_{tag}pct": keep_num,
                        f"keep_rel_pgt_{tag}pct": keep_pgt,
                        f"keep_rel_pgt_{tag}pct_value": 100.0 * keep_pgt if math.isfinite(keep_pgt) else float("nan"),
                        f"keep_rel_pcomp_{tag}pct": keep_pcomp,
                        f"keep_rel_pcomp_{tag}pct_value": 100.0 * keep_pcomp if math.isfinite(keep_pcomp) else float("nan"),
                        f"keep_rel_prob_gap_{tag}pct": keep_prob_gap,
                        f"keep_rel_logit_margin_{tag}pct": keep_logit_margin,
                        f"keep_rel_delta_pgt_{tag}pct": keep_delta_pgt,
                        f"keep_rel_delta_pgt_{tag}pct_points": keep_delta_pgt_pct,
                        f"keep_rel_delta_pcomp_{tag}pct": keep_delta_pcomp,
                        f"keep_rel_delta_pcomp_{tag}pct_points": keep_delta_pcomp_pct,
                        f"keep_rel_marginal_pgt_{tag}pct": keep_marginal_pgt,
                        f"keep_rel_marginal_pgt_{tag}pct_points": keep_marginal_pgt_pct,
                        f"keep_rel_pred_{tag}pct": keep_pred,
                        f"keep_rel_predicts_gt_{tag}pct": keep_is_gt,
                    })

                    keep_rows.append({
                        "ratio": ratio,
                        "num_kept": keep_num,
                        "pgt": keep_pgt,
                        "pgt_pct": 100.0 * keep_pgt if math.isfinite(keep_pgt) else float("nan"),
                        "pcomp": keep_pcomp,
                        "pcomp_pct": 100.0 * keep_pcomp if math.isfinite(keep_pcomp) else float("nan"),
                        "delta_pgt": keep_delta_pgt,
                        "delta_pgt_pct": keep_delta_pgt_pct,
                        "marginal_pgt": keep_marginal_pgt,
                        "marginal_pgt_pct": keep_marginal_pgt_pct,
                        "pred": keep_pred,
                        "is_gt": keep_is_gt,
                    })

                record["min_keep_relevant_ratio_for_gt_prediction"] = (
                    min_keep_ratio_for_gt if min_keep_ratio_for_gt is not None else float("nan")
                )
                record["min_keep_relevant_pct_for_gt_prediction"] = (
                    100.0 * min_keep_ratio_for_gt if min_keep_ratio_for_gt is not None else float("nan")
                )

                # ----- Figure -----
                img = _to_image(batch_inputs[i])
                H, Wimg = int(batch_inputs.shape[-2]), int(batch_inputs.shape[-1])
                rel_grid = rel_mask.view(hf, wf).float().cpu()
                nui_grid = nui_mask.view(hf, wf).float().cpu()
                irr_grid = irr_mask.view(hf, wf).float().cpu()
                rel_up = _upsample_grid(rel_grid, hf, wf, H, Wimg)
                nui_up = _upsample_grid(nui_grid, hf, wf, H, Wimg)
                irr_up = _upsample_grid(irr_grid, hf, wf, H, Wimg)
                overlay = _overlay_partition(img, rel_up, nui_up, irr_up)

                fig, axes = plt.subplots(2, 2, figsize=(19.0, 10.8))

                axes[0, 0].imshow(img, cmap="gray" if img.ndim == 2 else None)
                axes[0, 0].set_title(
                    "Original\nGT={} Pred={} Comp={}\nP_GT={:.2f}%  P_c*={:.2f}%\nM={:+.4f}".format(
                        yi, pi, ci, 100.0 * orig_pgt, 100.0 * orig_pcomp, orig_logit_margin
                    )
                )
                axes[0, 0].axis("off")

                axes[0, 1].imshow(overlay)
                _draw_patch_grid(axes[0, 1], hf, wf, H, Wimg)
                axes[0, 1].set_title(
                    "Nearest R/N/I bank\n"
                    "Green=R {} | Red=N {} | Blue=I {} | c*={}".format(
                        num_rel, num_nui, num_irr, ci
                    )
                )
                axes[0, 1].axis("off")

                removal_lines = [
                    "A) Removal faithfulness",
                    "Softmax is recomputed after each masking intervention",
                    "dP_GT = P_GT(after) - P_GT(original)",
                    "",
                    "ratio | remove R: dP_GT / dP_c* / pred | remove N: dP_GT / dP_c* / pred",
                ]
                for ar in ablation_rows:
                    removal_lines.append(
                        "{:>4.0f}% | {:+.2f}% / {:+.2f}% / {:>2d} | {:+.2f}% / {:+.2f}% / {:>2d}".format(
                            100.0 * ar["ratio"],
                            ar["rel_delta_pgt_pct"], ar["rel_delta_pcomp_pct"], ar["rel_pred"],
                            ar["nui_delta_pgt_pct"], ar["nui_delta_pcomp_pct"], ar["nui_pred"],
                        )
                    )
                axes[1, 0].axis("off")
                axes[1, 0].text(
                    0.0, 1.0, "\n".join(removal_lines),
                    va="top", ha="left", family="monospace", fontsize=9.2
                )

                if min_keep_ratio_for_gt is None:
                    min_keep_text = "none of tested ratios"
                else:
                    min_keep_text = "{:.0f}%".format(100.0 * min_keep_ratio_for_gt)

                keep_lines = [
                    "B) Relevant-patch sufficiency (KEEP R ONLY)",
                    "All unkept R + all N + all I are zeroed",
                    "P_GT = Softmax(logits_keep)[GT]",
                    "marginal = P_GT(current ratio) - P_GT(previous ratio)",
                    "",
                    "ratio | kept R | P_GT | dP_GT vs orig | marginal | pred | GT?",
                ]
                for kr in keep_rows:
                    if not math.isfinite(kr["pgt_pct"]):
                        keep_lines.append(
                            "{:>4.0f}% | {:>6d} |   N/A  |      N/A      |   N/A    |  -  |  -".format(
                                100.0 * kr["ratio"], kr["num_kept"]
                            )
                        )
                        continue
                    marginal_str = (
                        "  N/A  " if not math.isfinite(kr["marginal_pgt_pct"])
                        else "{:+.2f}%".format(kr["marginal_pgt_pct"])
                    )
                    keep_lines.append(
                        "{:>4.0f}% | {:>6d} | {:>5.2f}% | {:+.2f}% | {:>7s} | {:>4d} | {}".format(
                            100.0 * kr["ratio"], kr["num_kept"], kr["pgt_pct"],
                            kr["delta_pgt_pct"], marginal_str, kr["pred"],
                            "YES" if kr["is_gt"] else "NO",
                        )
                    )
                keep_lines.extend([
                    "",
                    "Smallest tested keep ratio predicting GT: {}".format(min_keep_text),
                    "Plateau clue: later marginal gains close to 0% => added R patches bring little extra decision gain.",
                ])
                axes[1, 1].axis("off")
                axes[1, 1].text(
                    0.0, 1.0, "\n".join(keep_lines),
                    va="top", ha="left", family="monospace", fontsize=8.8
                )

                fig.tight_layout()

                path = os.path.join(save_dir, "sample_{:05d}.png".format(global_index))
                fig.savefig(path, dpi=180, bbox_inches="tight")
                if display:
                    try:
                        plt.show(block=False)
                        plt.pause(0.001)
                    except Exception:
                        pass
                plt.close(fig)

                paths.append(path)
                records.append(record)
                global_index += 1

        per_image_csv = os.path.join(save_dir, "ablation_per_image.csv")
        if records:
            fieldnames: List[str] = []
            for rec in records:
                for key in rec.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
            with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)

        def _finite(vals):
            return [float(v) for v in vals if math.isfinite(float(v))]

        def _mean(vals):
            vv = _finite(vals)
            return float(sum(vv) / len(vv)) if vv else float("nan")

        def _rate(vals):
            vv = _finite(vals)
            return float(sum(vv) / len(vv)) if vv else float("nan")

        # =============================================================
        # Summary A: removal faithfulness.
        # =============================================================
        summary_rows: List[Dict[str, Any]] = []
        for ratio in removal:
            tag = self._ratio_tag(ratio)
            rel_valid = [r for r in records if int(r["num_relevant"]) > 0]
            nui_valid = [r for r in records if int(r["num_nuisance"]) > 0]
            rel_correct = [r for r in rel_valid if int(r["correct"]) == 1]
            nui_wrong = [r for r in nui_valid if int(r["correct"]) == 0]

            rel_dm = [float(r[f"rel_delta_logit_margin_{tag}pct"]) for r in rel_valid]
            nui_dm = [float(r[f"nui_delta_logit_margin_{tag}pct"]) for r in nui_valid]
            rel_dg = [float(r[f"rel_delta_prob_gap_{tag}pct"]) for r in rel_valid]
            nui_dg = [float(r[f"nui_delta_prob_gap_{tag}pct"]) for r in nui_valid]
            rel_dpgt = [float(r[f"rel_delta_pgt_{tag}pct"]) for r in rel_valid]
            rel_dpcomp = [float(r[f"rel_delta_pcomp_{tag}pct"]) for r in rel_valid]
            nui_dpgt = [float(r[f"nui_delta_pgt_{tag}pct"]) for r in nui_valid]
            nui_dpcomp = [float(r[f"nui_delta_pcomp_{tag}pct"]) for r in nui_valid]
            rel_correct_dm = [float(r[f"rel_delta_logit_margin_{tag}pct"]) for r in rel_correct]
            nui_wrong_dm = [float(r[f"nui_delta_logit_margin_{tag}pct"]) for r in nui_wrong]
            nui_wrong_dg = [float(r[f"nui_delta_prob_gap_{tag}pct"]) for r in nui_wrong]

            summary_rows.append({
                "ratio": ratio,
                "num_images": len(records),
                "num_images_with_relevant": len(rel_valid),
                "mean_delta_logit_margin_remove_relevant": _mean(rel_dm),
                "mean_delta_prob_gap_remove_relevant": _mean(rel_dg),
                "mean_delta_pgt_remove_relevant": _mean(rel_dpgt),
                "mean_delta_pcomp_remove_relevant": _mean(rel_dpcomp),
                "mean_delta_pgt_remove_relevant_pct_points": 100.0 * _mean(rel_dpgt),
                "mean_delta_pcomp_remove_relevant_pct_points": 100.0 * _mean(rel_dpcomp),
                "relevant_faithfulness_rate_delta_pgt_lt_0": _rate([int(v < 0) for v in rel_dpgt]),
                "relevant_faithfulness_rate_delta_margin_lt_0": _rate([int(v < 0) for v in rel_dm]),
                "num_correct_images_with_relevant": len(rel_correct),
                "mean_delta_logit_margin_remove_relevant_on_correct": _mean(rel_correct_dm),
                "relevant_break_rate_on_original_correct": _rate([
                    int(r[f"rel_broken_{tag}pct"]) for r in rel_correct
                ]),
                "num_images_with_nuisance": len(nui_valid),
                "mean_delta_logit_margin_remove_nuisance": _mean(nui_dm),
                "mean_delta_prob_gap_remove_nuisance": _mean(nui_dg),
                "mean_delta_pgt_remove_nuisance": _mean(nui_dpgt),
                "mean_delta_pcomp_remove_nuisance": _mean(nui_dpcomp),
                "mean_delta_pgt_remove_nuisance_pct_points": 100.0 * _mean(nui_dpgt),
                "mean_delta_pcomp_remove_nuisance_pct_points": 100.0 * _mean(nui_dpcomp),
                "nuisance_faithfulness_rate_delta_pgt_gt_0": _rate([int(v > 0) for v in nui_dpgt]),
                "nuisance_faithfulness_rate_delta_margin_gt_0": _rate([int(v > 0) for v in nui_dm]),
                "num_wrong_images_with_nuisance": len(nui_wrong),
                "mean_delta_logit_margin_remove_nuisance_on_wrong": _mean(nui_wrong_dm),
                "mean_delta_prob_gap_remove_nuisance_on_wrong": _mean(nui_wrong_dg),
                "nuisance_correction_rate_on_original_wrong": _rate([
                    int(r[f"nui_corrected_{tag}pct"]) for r in nui_wrong
                ]),
            })

        summary_csv = os.path.join(save_dir, "ablation_summary.csv")
        if summary_rows:
            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)

        # =============================================================
        # Summary B: relevant sufficiency / saturation curve.
        # =============================================================
        keep_summary_rows: List[Dict[str, Any]] = []
        rel_records = [r for r in records if int(r["num_relevant"]) > 0]
        for ratio in keep_ratios:
            tag = self._ratio_tag(ratio)
            pgt = [float(r[f"keep_rel_pgt_{tag}pct"]) for r in rel_records]
            dpgt = [float(r[f"keep_rel_delta_pgt_{tag}pct"]) for r in rel_records]
            marginal = [float(r[f"keep_rel_marginal_pgt_{tag}pct"]) for r in rel_records]
            gt_ok = [int(r[f"keep_rel_predicts_gt_{tag}pct"]) for r in rel_records]
            num_kept = [float(r[f"keep_rel_num_{tag}pct"]) for r in rel_records]

            keep_summary_rows.append({
                "keep_ratio": ratio,
                "keep_ratio_pct": 100.0 * ratio,
                "num_images_with_relevant": len(rel_records),
                "mean_num_kept_relevant_patches": _mean(num_kept),
                "mean_pgt_keep_relevant": _mean(pgt),
                "mean_pgt_keep_relevant_pct": 100.0 * _mean(pgt),
                "mean_delta_pgt_vs_original": _mean(dpgt),
                "mean_delta_pgt_vs_original_pct_points": 100.0 * _mean(dpgt),
                "mean_marginal_pgt_gain_from_previous_ratio": _mean(marginal),
                "mean_marginal_pgt_gain_from_previous_ratio_pct_points": 100.0 * _mean(marginal),
                "gt_prediction_rate_keep_relevant_only": _rate(gt_ok),
            })

        keep_summary_csv = os.path.join(save_dir, "relevant_sufficiency_summary.csv")
        if keep_summary_rows:
            with open(keep_summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(keep_summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(keep_summary_rows)

        min_ratio_values = [
            float(r["min_keep_relevant_ratio_for_gt_prediction"])
            for r in records
            if math.isfinite(float(r["min_keep_relevant_ratio_for_gt_prediction"]))
        ]

        self.last_visualization_summary = {
            "paths": paths,
            "per_image_csv": per_image_csv if records else None,
            "summary_csv": summary_csv if summary_rows else None,
            "relevant_sufficiency_csv": keep_summary_csv if keep_summary_rows else None,
            "records": records,
            "summary": summary_rows,
            "relevant_sufficiency_summary": keep_summary_rows,
            "mean_min_keep_relevant_ratio_for_gt_prediction": _mean(min_ratio_values),
        }

        if original_training:
            student_model.train()

        print("[Three-bank visualization] saved {} image(s) to: {}".format(
            len(paths), os.path.abspath(save_dir)
        ))
        if records:
            print("  per-image stats       : {}".format(os.path.abspath(per_image_csv)))
            print("  removal summary       : {}".format(os.path.abspath(summary_csv)))
            print("  R sufficiency summary : {}".format(os.path.abspath(keep_summary_csv)))
            print("  Assignment: nearest among R/N/I banks; no bank-similarity threshold")
            print("  Removal metric: dP_GT = Softmax(logits_after)[GT] - Softmax(logits_original)[GT]")
            print("  Keep-R metric : ONLY top-ratio Relevant patches survive; P_GT is recomputed from full logits")
            print("  Display unit: percentage points; e.g. +3.20% means probability increased by 0.032")

            for row in summary_rows:
                nfaith = row["nuisance_faithfulness_rate_delta_pgt_gt_0"]
                rfaith = row["relevant_faithfulness_rate_delta_pgt_lt_0"]
                print(
                    "  remove {:>4.0f}% | mean dP_GT(remove R)={:+.2f}% [expect <0] | "
                    "mean dP_GT(remove N)={:+.2f}% [expect >0] | R-faith={:.1%} | N-faith={:.1%}".format(
                        100.0 * row["ratio"],
                        row["mean_delta_pgt_remove_relevant_pct_points"],
                        row["mean_delta_pgt_remove_nuisance_pct_points"],
                        rfaith if math.isfinite(rfaith) else 0.0,
                        nfaith if math.isfinite(nfaith) else 0.0,
                    )
                )

            print("\n  ===== Relevant-patch sufficiency curve =====")
            for row in keep_summary_rows:
                marginal = row["mean_marginal_pgt_gain_from_previous_ratio_pct_points"]
                marginal_text = "N/A" if not math.isfinite(marginal) else "{:+.2f}%".format(marginal)
                print(
                    "  keep R {:>4.0f}% only | mean P_GT={:>6.2f}% | dP_GT vs orig={:+.2f}% | "
                    "marginal={} | GT-pred-rate={:.1%}".format(
                        row["keep_ratio_pct"],
                        row["mean_pgt_keep_relevant_pct"],
                        row["mean_delta_pgt_vs_original_pct_points"],
                        marginal_text,
                        row["gt_prediction_rate_keep_relevant_only"]
                        if math.isfinite(row["gt_prediction_rate_keep_relevant_only"]) else 0.0,
                    )
                )

            mean_min = self.last_visualization_summary["mean_min_keep_relevant_ratio_for_gt_prediction"]
            if math.isfinite(mean_min):
                print(
                    "  Mean smallest tested keep-R ratio that already predicts GT: {:.1f}%".format(
                        100.0 * mean_min
                    )
                )
            else:
                print("  No tested keep-R ratio predicted GT for the analyzed samples.")

        return paths

    # Explicit alias for readability; same behavior as visualize_partition.
    visualize_and_ablate = visualize_partition


    # -------------------------------------------------------------------------
    # Full validation R/N/I probability statistics
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_rn_probability_statistics(
        self,
        student_model: nn.Module,
        inputs: Union[Tensor, Iterable],
        labels: Optional[Tensor] = None,
        save_dir: str = "./triple_bank_vis",
        ratios: Sequence[float] = (0.10, 0.20, 0.40, 0.60, 0.80, 1.00),
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute FULL-dataset statistics without changing visualize_partition().

        Statistics
        ----------
        1) R/N/I patch ratios over ALL samples in ``inputs`` plus the R/N ratio.

        2) For ratios such as [10,20,40,60,80,100]:

           A. KEEP Relevant only
              Keep the top-ratio Relevant patches ranked exactly as in
              visualize_partition():

                  relevant_strength = relevant_similarity - nuisance_similarity

              All unkept Relevant patches plus ALL Nuisance/Irrelevant patches are zeroed at
              layer4. The full model head is then run again and Softmax is recomputed.

           B. REMOVE Nuisance
              Remove the top-ratio Nuisance patches ranked exactly as in
              visualize_partition():

                  nuisance_strength = nuisance_similarity - relevant_similarity

              All remaining patches are kept. The full model head is run again and
              Softmax is recomputed.

        For every intervention, the reported classes are:
            GT: ground-truth class y
            c*: strongest NON-GT class under the ORIGINAL unmasked logits

                c* = argmax_{c != y} logit_c

        The same fixed c* is used at every intervention ratio, so probabilities are
        directly comparable across ratios.

        Files written to ``save_dir``
        -----------------------------
            full_val_rn_patch_ratio.csv
            full_val_probability_per_image.csv
            full_val_probability_summary.csv

        Notes
        -----
        - ``max_images`` from visualize_partition() is intentionally irrelevant here:
          this method scans the ENTIRE supplied loader/tensor.
        - If an image contains no Relevant patches, KEEP-R statistics for that image
          are recorded as NaN and excluded from KEEP-R means.
        - If an image contains no Nuisance patches, REMOVE-N statistics for that image
          are recorded as NaN and excluded from REMOVE-N means.
        - With a real Irrelevant bank, KEEP-R 100% and REMOVE-N 100% are no longer
          identical: KEEP-R removes N+I, whereas REMOVE-N keeps R+I.
        """
        self._check_discovered()
        os.makedirs(save_dir, exist_ok=True)

        ratios_used = sorted(set(float(r) for r in ratios))
        if len(ratios_used) == 0 or any(r <= 0.0 or r > 1.0 for r in ratios_used):
            raise ValueError("ratios must contain values in (0,1].")

        original_training = student_model.training
        student_model.eval()
        device = next(student_model.parameters()).device

        total_images = 0
        total_patches = 0
        total_relevant = 0
        total_nuisance = 0
        total_irrelevant = 0

        per_image_records: List[Dict[str, Any]] = []
        global_index = 0

        def _finite_values(values: Sequence[float]) -> List[float]:
            return [float(v) for v in values if math.isfinite(float(v))]

        def _mean(values: Sequence[float]) -> float:
            vals = _finite_values(values)
            if len(vals) == 0:
                return float("nan")
            return float(sum(vals) / len(vals))

        def _std(values: Sequence[float]) -> float:
            vals = _finite_values(values)
            if len(vals) <= 1:
                return 0.0 if len(vals) == 1 else float("nan")
            mu = sum(vals) / len(vals)
            var = sum((v - mu) ** 2 for v in vals) / len(vals)
            return float(math.sqrt(max(var, 0.0)))

        for batch_inputs, batch_labels in self._batch_iterator(inputs, labels):
            x = batch_inputs.to(device, non_blocking=True)
            y = batch_labels.long().to(device, non_blocking=True)

            part = self._stage2_triple_bank_partition(x)
            base_logits, feat = _capture_layer4_and_forward(student_model, x)
            base_prob = F.softmax(base_logits, dim=1)
            pred = base_logits.argmax(dim=1)

            # Fixed strongest non-GT competitor from ORIGINAL unmasked logits.
            other_logits = base_logits.detach().clone()
            row = torch.arange(base_logits.shape[0], device=device)
            other_logits[row, y] = -torch.inf
            competitor = other_logits.argmax(dim=1)

            rel_mask_batch = part["relevant_mask"].to(device)
            nui_mask_batch = part["nuisance_mask"].to(device)
            irr_mask_batch = part["irrelevant_mask"].to(device)
            rel_sim_batch = part["relevant_similarity"].to(device)
            nui_sim_batch = part["nuisance_similarity"].to(device)
            irr_sim_batch = part["irrelevant_similarity"].to(device)

            b = int(x.shape[0])
            hf, wf = int(feat.shape[-2]), int(feat.shape[-1])
            num_spatial = hf * wf
            if rel_mask_batch.shape != (b, num_spatial):
                raise RuntimeError(
                    "Frozen assignment patch grid {} != current layer4 grid {}.".format(
                        tuple(rel_mask_batch.shape), (b, num_spatial)
                    )
                )

            covered = rel_mask_batch | nui_mask_batch | irr_mask_batch
            if not bool(covered.all()):
                raise RuntimeError("R/N/I masks do not cover all patches during statistics.")
            if (
                bool((rel_mask_batch & nui_mask_batch).any())
                or bool((rel_mask_batch & irr_mask_batch).any())
                or bool((nui_mask_batch & irr_mask_batch).any())
            ):
                raise RuntimeError("R/N/I masks overlap during full-dataset statistics.")

            rel_count_batch = rel_mask_batch.sum(dim=1).long()
            nui_count_batch = nui_mask_batch.sum(dim=1).long()
            irr_count_batch = irr_mask_batch.sum(dim=1).long()

            total_images += b
            total_patches += int(b * num_spatial)
            total_relevant += int(rel_count_batch.sum().item())
            total_nuisance += int(nui_count_batch.sum().item())
            total_irrelevant += int(irr_count_batch.sum().item())

            rel_strength_batch = rel_sim_batch - torch.maximum(nui_sim_batch, irr_sim_batch)
            nui_strength_batch = nui_sim_batch - torch.maximum(rel_sim_batch, irr_sim_batch)

            batch_records: List[Dict[str, Any]] = []
            for i in range(b):
                yi = int(y[i].item())
                ci = int(competitor[i].item())
                pi = int(pred[i].item())
                pgt = float(base_prob[i, yi].item())
                pcomp = float(base_prob[i, ci].item())
                nr = int(rel_count_batch[i].item())
                nn = int(nui_count_batch[i].item())
                ni = int(irr_count_batch[i].item())

                rec: Dict[str, Any] = {
                    "sample_index": global_index + i,
                    "gt": yi,
                    "pred": pi,
                    "competitor": ci,
                    "correct": int(pi == yi),
                    "num_patches": num_spatial,
                    "num_relevant": nr,
                    "num_nuisance": nn,
                    "num_irrelevant": ni,
                    "relevant_ratio": float(nr / num_spatial),
                    "nuisance_ratio": float(nn / num_spatial),
                    "irrelevant_ratio": float(ni / num_spatial),
                    "relevant_to_nuisance_ratio": (
                        float(nr / nn) if nn > 0 else float("inf")
                    ),
                    "orig_pgt": pgt,
                    "orig_pgt_pct": 100.0 * pgt,
                    "orig_pcomp": pcomp,
                    "orig_pcomp_pct": 100.0 * pcomp,
                    "orig_prob_gap": pgt - pcomp,
                }
                batch_records.append(rec)

            # Batch intervention: only 2 forward passes per ratio, rather than
            # 2 * batch_size forward passes per ratio.
            for ratio in ratios_used:
                tag = self._ratio_tag(ratio)

                keep_rel_list: List[Tensor] = []
                remove_nui_list: List[Tensor] = []
                for i in range(b):
                    keep_rel_list.append(
                        self._top_fraction_mask(
                            rel_mask_batch[i], rel_strength_batch[i], ratio
                        )
                    )
                    remove_nui_list.append(
                        self._top_fraction_mask(
                            nui_mask_batch[i], nui_strength_batch[i], ratio
                        )
                    )

                keep_rel_mask = torch.stack(keep_rel_list, dim=0)
                remove_nui_mask = torch.stack(remove_nui_list, dim=0)

                # KEEP R ONLY: all non-kept patches are zeroed.
                remove_everything_except_kept_r = ~keep_rel_mask
                logits_keep_r = _forward_with_layer4_zero_mask(
                    student_model,
                    x,
                    remove_everything_except_kept_r,
                )
                prob_keep_r = F.softmax(logits_keep_r, dim=1)

                # REMOVE N ONLY: selected N patches are zeroed, everything else stays.
                logits_remove_n = _forward_with_layer4_zero_mask(
                    student_model,
                    x,
                    remove_nui_mask,
                )
                prob_remove_n = F.softmax(logits_remove_n, dim=1)

                for i in range(b):
                    yi = int(y[i].item())
                    ci = int(competitor[i].item())
                    nr = int(rel_count_batch[i].item())
                    nn = int(nui_count_batch[i].item())
                    orig_pgt = float(base_prob[i, yi].item())
                    orig_pcomp = float(base_prob[i, ci].item())

                    # Match the existing visualization semantics: no R => KEEP-R N/A.
                    if nr > 0:
                        keep_num = int(keep_rel_mask[i].sum().item())
                        keep_pgt = float(prob_keep_r[i, yi].item())
                        keep_pcomp = float(prob_keep_r[i, ci].item())
                        keep_pred = int(logits_keep_r[i].argmax().item())
                    else:
                        keep_num = 0
                        keep_pgt = float("nan")
                        keep_pcomp = float("nan")
                        keep_pred = -1

                    # Match the existing ablation semantics: no N => REMOVE-N N/A.
                    if nn > 0:
                        remove_num = int(remove_nui_mask[i].sum().item())
                        remove_pgt = float(prob_remove_n[i, yi].item())
                        remove_pcomp = float(prob_remove_n[i, ci].item())
                        remove_pred = int(logits_remove_n[i].argmax().item())
                    else:
                        remove_num = 0
                        remove_pgt = float("nan")
                        remove_pcomp = float("nan")
                        remove_pred = -1

                    batch_records[i].update({
                        f"keep_rel_num_{tag}pct": keep_num,
                        f"keep_rel_pgt_{tag}pct": keep_pgt,
                        f"keep_rel_pgt_{tag}pct_value": (
                            100.0 * keep_pgt if math.isfinite(keep_pgt) else float("nan")
                        ),
                        f"keep_rel_pcomp_{tag}pct": keep_pcomp,
                        f"keep_rel_pcomp_{tag}pct_value": (
                            100.0 * keep_pcomp if math.isfinite(keep_pcomp) else float("nan")
                        ),
                        f"keep_rel_delta_pgt_{tag}pct": (
                            keep_pgt - orig_pgt if math.isfinite(keep_pgt) else float("nan")
                        ),
                        f"keep_rel_delta_pcomp_{tag}pct": (
                            keep_pcomp - orig_pcomp if math.isfinite(keep_pcomp) else float("nan")
                        ),
                        f"keep_rel_pred_{tag}pct": keep_pred,
                        f"remove_nui_num_{tag}pct": remove_num,
                        f"remove_nui_pgt_{tag}pct": remove_pgt,
                        f"remove_nui_pgt_{tag}pct_value": (
                            100.0 * remove_pgt if math.isfinite(remove_pgt) else float("nan")
                        ),
                        f"remove_nui_pcomp_{tag}pct": remove_pcomp,
                        f"remove_nui_pcomp_{tag}pct_value": (
                            100.0 * remove_pcomp if math.isfinite(remove_pcomp) else float("nan")
                        ),
                        f"remove_nui_delta_pgt_{tag}pct": (
                            remove_pgt - orig_pgt if math.isfinite(remove_pgt) else float("nan")
                        ),
                        f"remove_nui_delta_pcomp_{tag}pct": (
                            remove_pcomp - orig_pcomp if math.isfinite(remove_pcomp) else float("nan")
                        ),
                        f"remove_nui_pred_{tag}pct": remove_pred,
                    })

            per_image_records.extend(batch_records)
            global_index += b

        if original_training:
            student_model.train()

        if total_images == 0:
            raise RuntimeError("No samples were found in the supplied inputs/loader.")
        if total_patches <= 0:
            raise RuntimeError("No layer4 patches were found.")

        relevant_fraction = float(total_relevant / total_patches)
        nuisance_fraction = float(total_nuisance / total_patches)
        irrelevant_fraction = float(total_irrelevant / total_patches)
        rn_ratio = (
            float(total_relevant / total_nuisance)
            if total_nuisance > 0 else float("inf")
        )

        mean_image_relevant_ratio = _mean([
            float(r["relevant_ratio"]) for r in per_image_records
        ])
        mean_image_nuisance_ratio = _mean([
            float(r["nuisance_ratio"]) for r in per_image_records
        ])
        mean_image_irrelevant_ratio = _mean([
            float(r["irrelevant_ratio"]) for r in per_image_records
        ])
        mean_image_rn_ratio = _mean([
            float(r["relevant_to_nuisance_ratio"])
            for r in per_image_records
            if math.isfinite(float(r["relevant_to_nuisance_ratio"]))
        ])

        patch_ratio_row: Dict[str, Any] = {
            "num_images": total_images,
            "num_patches": total_patches,
            "num_relevant_patches": total_relevant,
            "num_nuisance_patches": total_nuisance,
            "num_irrelevant_patches": total_irrelevant,
            "global_relevant_ratio": relevant_fraction,
            "global_relevant_ratio_pct": 100.0 * relevant_fraction,
            "global_nuisance_ratio": nuisance_fraction,
            "global_nuisance_ratio_pct": 100.0 * nuisance_fraction,
            "global_irrelevant_ratio": irrelevant_fraction,
            "global_irrelevant_ratio_pct": 100.0 * irrelevant_fraction,
            "global_relevant_to_nuisance_ratio": rn_ratio,
            "mean_per_image_relevant_ratio": mean_image_relevant_ratio,
            "mean_per_image_relevant_ratio_pct": 100.0 * mean_image_relevant_ratio,
            "mean_per_image_nuisance_ratio": mean_image_nuisance_ratio,
            "mean_per_image_nuisance_ratio_pct": 100.0 * mean_image_nuisance_ratio,
            "mean_per_image_irrelevant_ratio": mean_image_irrelevant_ratio,
            "mean_per_image_irrelevant_ratio_pct": 100.0 * mean_image_irrelevant_ratio,
            "mean_per_image_relevant_to_nuisance_ratio": mean_image_rn_ratio,
        }

        # Build compact probability summary.
        orig_pgt_values = [float(r["orig_pgt"]) for r in per_image_records]
        orig_pcomp_values = [float(r["orig_pcomp"]) for r in per_image_records]
        summary_rows: List[Dict[str, Any]] = [{
            "operation": "original",
            "ratio": 0.0,
            "ratio_pct": 0.0,
            "num_total_images": total_images,
            "num_valid_images": total_images,
            "mean_num_affected_patches": 0.0,
            "mean_pgt": _mean(orig_pgt_values),
            "std_pgt": _std(orig_pgt_values),
            "mean_pgt_pct": 100.0 * _mean(orig_pgt_values),
            "std_pgt_pct": 100.0 * _std(orig_pgt_values),
            "mean_pcomp": _mean(orig_pcomp_values),
            "std_pcomp": _std(orig_pcomp_values),
            "mean_pcomp_pct": 100.0 * _mean(orig_pcomp_values),
            "std_pcomp_pct": 100.0 * _std(orig_pcomp_values),
            "mean_delta_pgt_vs_original": 0.0,
            "mean_delta_pgt_vs_original_pct_points": 0.0,
            "mean_delta_pcomp_vs_original": 0.0,
            "mean_delta_pcomp_vs_original_pct_points": 0.0,
        }]

        for ratio in ratios_used:
            tag = self._ratio_tag(ratio)

            keep_valid = [
                r for r in per_image_records
                if math.isfinite(float(r[f"keep_rel_pgt_{tag}pct"]))
            ]
            keep_pgt = [float(r[f"keep_rel_pgt_{tag}pct"]) for r in keep_valid]
            keep_pcomp = [float(r[f"keep_rel_pcomp_{tag}pct"]) for r in keep_valid]
            keep_num = [float(r[f"keep_rel_num_{tag}pct"]) for r in keep_valid]
            keep_dpgt = [
                float(r[f"keep_rel_delta_pgt_{tag}pct"]) for r in keep_valid
            ]
            keep_dpcomp = [
                float(r[f"keep_rel_delta_pcomp_{tag}pct"]) for r in keep_valid
            ]

            summary_rows.append({
                "operation": "keep_relevant_only",
                "ratio": ratio,
                "ratio_pct": 100.0 * ratio,
                "num_total_images": total_images,
                "num_valid_images": len(keep_valid),
                "mean_num_affected_patches": _mean(keep_num),
                "mean_pgt": _mean(keep_pgt),
                "std_pgt": _std(keep_pgt),
                "mean_pgt_pct": 100.0 * _mean(keep_pgt),
                "std_pgt_pct": 100.0 * _std(keep_pgt),
                "mean_pcomp": _mean(keep_pcomp),
                "std_pcomp": _std(keep_pcomp),
                "mean_pcomp_pct": 100.0 * _mean(keep_pcomp),
                "std_pcomp_pct": 100.0 * _std(keep_pcomp),
                "mean_delta_pgt_vs_original": _mean(keep_dpgt),
                "mean_delta_pgt_vs_original_pct_points": 100.0 * _mean(keep_dpgt),
                "mean_delta_pcomp_vs_original": _mean(keep_dpcomp),
                "mean_delta_pcomp_vs_original_pct_points": 100.0 * _mean(keep_dpcomp),
            })

            remove_valid = [
                r for r in per_image_records
                if math.isfinite(float(r[f"remove_nui_pgt_{tag}pct"]))
            ]
            remove_pgt = [float(r[f"remove_nui_pgt_{tag}pct"]) for r in remove_valid]
            remove_pcomp = [float(r[f"remove_nui_pcomp_{tag}pct"]) for r in remove_valid]
            remove_num = [float(r[f"remove_nui_num_{tag}pct"]) for r in remove_valid]
            remove_dpgt = [
                float(r[f"remove_nui_delta_pgt_{tag}pct"]) for r in remove_valid
            ]
            remove_dpcomp = [
                float(r[f"remove_nui_delta_pcomp_{tag}pct"]) for r in remove_valid
            ]

            summary_rows.append({
                "operation": "remove_nuisance",
                "ratio": ratio,
                "ratio_pct": 100.0 * ratio,
                "num_total_images": total_images,
                "num_valid_images": len(remove_valid),
                "mean_num_affected_patches": _mean(remove_num),
                "mean_pgt": _mean(remove_pgt),
                "std_pgt": _std(remove_pgt),
                "mean_pgt_pct": 100.0 * _mean(remove_pgt),
                "std_pgt_pct": 100.0 * _std(remove_pgt),
                "mean_pcomp": _mean(remove_pcomp),
                "std_pcomp": _std(remove_pcomp),
                "mean_pcomp_pct": 100.0 * _mean(remove_pcomp),
                "std_pcomp_pct": 100.0 * _std(remove_pcomp),
                "mean_delta_pgt_vs_original": _mean(remove_dpgt),
                "mean_delta_pgt_vs_original_pct_points": 100.0 * _mean(remove_dpgt),
                "mean_delta_pcomp_vs_original": _mean(remove_dpcomp),
                "mean_delta_pcomp_vs_original_pct_points": 100.0 * _mean(remove_dpcomp),
            })

        patch_ratio_csv = os.path.join(save_dir, "full_val_rn_patch_ratio.csv")
        with open(patch_ratio_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(patch_ratio_row.keys()))
            writer.writeheader()
            writer.writerow(patch_ratio_row)

        per_image_csv = os.path.join(save_dir, "full_val_probability_per_image.csv")
        if per_image_records:
            with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(per_image_records[0].keys())
                )
                writer.writeheader()
                writer.writerows(per_image_records)

        summary_csv = os.path.join(save_dir, "full_val_probability_summary.csv")
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        result: Dict[str, Any] = {
            "patch_ratio": patch_ratio_row,
            "probability_summary": summary_rows,
            "per_image_records": per_image_records,
            "patch_ratio_csv": patch_ratio_csv,
            "per_image_csv": per_image_csv,
            "summary_csv": summary_csv,
            "ratios": ratios_used,
        }
        self.last_dataset_statistics = result

        if verbose:
            print("\n========== FULL validation R/N/I statistics ==========")
            print("images={} | total patches={}".format(total_images, total_patches))
            print(
                "Relevant: {} ({:.2f}%) | Nuisance: {} ({:.2f}%) | "
                "Irrelevant: {} ({:.2f}%) | R/N={:.6f}".format(
                    total_relevant,
                    100.0 * relevant_fraction,
                    total_nuisance,
                    100.0 * nuisance_fraction,
                    total_irrelevant,
                    100.0 * irrelevant_fraction,
                    rn_ratio,
                )
            )
            print(
                "Original | mean P_GT={:.2f}% | mean P_c*={:.2f}%".format(
                    100.0 * _mean(orig_pgt_values),
                    100.0 * _mean(orig_pcomp_values),
                )
            )
            print("\n-- KEEP Relevant ONLY --")
            for row_s in summary_rows:
                if row_s["operation"] != "keep_relevant_only":
                    continue
                print(
                    "keep R {:>3.0f}% | valid={:>4d}/{:<4d} | mean P_GT={:>6.2f}% | "
                    "mean P_c*={:>6.2f}%".format(
                        row_s["ratio_pct"],
                        row_s["num_valid_images"],
                        row_s["num_total_images"],
                        row_s["mean_pgt_pct"],
                        row_s["mean_pcomp_pct"],
                    )
                )

            print("\n-- REMOVE Nuisance --")
            for row_s in summary_rows:
                if row_s["operation"] != "remove_nuisance":
                    continue
                print(
                    "remove N {:>3.0f}% | valid={:>4d}/{:<4d} | mean P_GT={:>6.2f}% | "
                    "mean P_c*={:>6.2f}%".format(
                        row_s["ratio_pct"],
                        row_s["num_valid_images"],
                        row_s["num_total_images"],
                        row_s["mean_pgt_pct"],
                        row_s["mean_pcomp_pct"],
                    )
                )
            print("\nSaved:")
            print("  R/N/I ratio CSV   : {}".format(os.path.abspath(patch_ratio_csv)))
            print("  per-image CSV     : {}".format(os.path.abspath(per_image_csv)))
            print("  probability CSV   : {}".format(os.path.abspath(summary_csv)))
            print("  c* is fixed per image from ORIGINAL unmasked strongest non-GT logit.")
            print("=====================================================\n")

        return result

    # -------------------------------------------------------------------------
    # Standard CAM comparison: frozen CE reference vs purified/current student
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def visualize_cam_comparison(
        self,
        student_model: nn.Module,
        inputs: Union[Tensor, Iterable],
        labels: Optional[Tensor] = None,
        ce_model: Optional[nn.Module] = None,
        save_dir: str = "./cam_ce_vs_purified",
        max_images: int = 20,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        target_mode: str = "gt",
        relu_cam: bool = True,
        overlay_alpha: float = 0.45,
        display: bool = False,
        require_same_fc: bool = True,
        fc_tolerance: float = 1e-7,
    ) -> List[str]:
        """
        Compare standard CAMs of a CE/Stage-I reference model and the current
        purified/student model while using ONE FIXED classifier direction.

        The comparison is intentionally designed for the setting

            fixed FC weights + trainable feature extractor.

        For target class c and layer4 feature map F, standard CAM is

            CAM_c(h,w) = sum_d w_c[d] * F[d,h,w].

        IMPORTANT FAIRNESS RULE
        -----------------------
        The SAME classifier weight vector w_c from the CE reference model is used
        to compute BOTH CAMs:

            CAM_CE       = w_c^T F_CE
            CAM_student  = w_c^T F_student
            Delta_CAM    = CAM_student - CAM_CE

        Therefore, if the FC layer is frozen, any CAM redistribution is caused by
        changes in the feature map rather than a changed classifier direction.

        Which CE model is used?
        -----------------------
        1) If ce_model is explicitly supplied, that model is used.
        2) If ce_model is None, self._assignment_model is used. discover() already
           stores an exact frozen copy of the model used for Stage-I discovery, so
           this is normally the original CE model if discover() was called before
           purification training.

        target_mode:
            "gt"           : visualize CAM of the ground-truth class (recommended)
            "ce_pred"      : visualize CAM of the CE model's predicted class
            "student_pred" : visualize CAM of the current student's predicted class

        Outputs:
            - one 4-panel PNG per image:
                Original | CE CAM | Purified CAM | Delta CAM
            - cam_comparison.csv with probabilities and CAM-change statistics

        Notes:
            - CAM display maps are min-max normalized only for visualization.
              The raw CAM tensors are used to compute Delta_CAM/statistics.
            - The prediction probabilities are obtained from each model's FULL
              logits followed by Softmax; CAM itself does not alter the logits.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if target_mode not in ("gt", "ce_pred", "student_pred"):
            raise ValueError(
                "target_mode must be one of {'gt', 'ce_pred', 'student_pred'}."
            )
        if not (0.0 <= float(overlay_alpha) <= 1.0):
            raise ValueError("overlay_alpha must be in [0,1].")
        if float(fc_tolerance) < 0.0:
            raise ValueError("fc_tolerance must be >= 0.")

        # Prefer an explicitly supplied CE model; otherwise use the exact Stage-I
        # snapshot stored by discover()/load_discovery().
        if ce_model is None:
            if self._assignment_model is None:
                raise RuntimeError(
                    "No CE reference is available. Either run discover() first or "
                    "pass ce_model=<your CE-trained model>."
                )
            ce_reference = self._assignment_model
            ce_source = "stored Stage-I/CE snapshot"
        else:
            ce_reference = ce_model
            ce_source = "explicit ce_model"

        # Preserve caller model modes.
        student_training = student_model.training
        ce_training = ce_reference.training
        student_model.eval()
        ce_reference.eval()

        try:
            student_device = next(student_model.parameters()).device
            ce_device = next(ce_reference.parameters()).device
        except StopIteration:
            raise ValueError("student_model and ce_model must contain parameters.")

        student_fc = _get_classifier(student_model)
        ce_fc = _get_classifier(ce_reference)

        if tuple(student_fc.weight.shape) != tuple(ce_fc.weight.shape):
            raise ValueError(
                "Student/CE classifier shapes differ: {} vs {}.".format(
                    tuple(student_fc.weight.shape), tuple(ce_fc.weight.shape)
                )
            )

        # Verify that the premise 'FC is fixed' is actually true.
        weight_diff = float(
            (student_fc.weight.detach().float().cpu()
             - ce_fc.weight.detach().float().cpu()).abs().max().item()
        )
        if student_fc.bias is None and ce_fc.bias is None:
            bias_diff = 0.0
        elif student_fc.bias is not None and ce_fc.bias is not None:
            bias_diff = float(
                (student_fc.bias.detach().float().cpu()
                 - ce_fc.bias.detach().float().cpu()).abs().max().item()
            )
        else:
            bias_diff = float("inf")

        fc_max_diff = max(weight_diff, bias_diff)
        if require_same_fc and fc_max_diff > float(fc_tolerance):
            raise RuntimeError(
                "FC layers are not identical (max abs diff={:.6e}, tolerance={:.6e}). "
                "For a strict fixed-FC CAM comparison, freeze the classifier or pass "
                "the matching CE model. Set require_same_fc=False only if you knowingly "
                "want a non-isolated comparison.".format(fc_max_diff, float(fc_tolerance))
            )

        os.makedirs(save_dir, exist_ok=True)
        paths: List[str] = []
        records: List[Dict[str, Any]] = []
        global_index = 0

        # Fixed reference classifier weight used for BOTH CAMs.
        W_ref_cpu = ce_fc.weight.detach().float().cpu()

        def _to_image(t: Tensor):
            t = t.detach().float().cpu()
            if mean is not None and std is not None:
                mean_t = torch.as_tensor(mean, dtype=t.dtype).view(-1, 1, 1)
                std_t = torch.as_tensor(std, dtype=t.dtype).view(-1, 1, 1)
                if mean_t.shape[0] != t.shape[0] or std_t.shape[0] != t.shape[0]:
                    raise ValueError("mean/std channel count must match input channels.")
                t = t * std_t + mean_t
            elif float(t.min().item()) < 0.0 or float(t.max().item()) > 1.0:
                lo = t.amin(dim=(1, 2), keepdim=True)
                hi = t.amax(dim=(1, 2), keepdim=True)
                t = (t - lo) / (hi - lo).clamp_min(1e-8)
            t = t.clamp(0.0, 1.0)
            if t.shape[0] == 1:
                return t[0].numpy()
            if t.shape[0] >= 3:
                return t[:3].permute(1, 2, 0).numpy()
            if t.shape[0] == 2:
                z = torch.zeros_like(t[:1])
                return torch.cat([t, z], dim=0).permute(1, 2, 0).numpy()
            raise ValueError("Unsupported input channel count: {}".format(int(t.shape[0])))

        def _minmax_cam(cam_2d: Tensor) -> Tensor:
            x = cam_2d.detach().float()
            lo = x.min()
            hi = x.max()
            return (x - lo) / (hi - lo).clamp_min(1e-8)

        def _upsample(cam_2d: Tensor, h: int, w: int) -> np.ndarray:
            x = cam_2d.detach().float().view(1, 1, *cam_2d.shape)
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
            return x[0, 0].cpu().numpy()

        # The class method already supports both Tensor and DataLoader inputs.
        for batch_inputs, batch_labels in self._batch_iterator(inputs, labels):
            if max_images is not None and max_images > 0 and global_index >= max_images:
                break

            batch_labels = batch_labels.long()
            remaining = None
            if max_images is not None and max_images > 0:
                remaining = max_images - global_index
            if remaining is not None and remaining <= 0:
                break
            if remaining is not None and batch_inputs.shape[0] > remaining:
                batch_inputs = batch_inputs[:remaining]
                batch_labels = batch_labels[:remaining]

            # Run the two models in their own devices.
            x_student = batch_inputs.to(student_device, non_blocking=True)
            x_ce = batch_inputs.to(ce_device, non_blocking=True)
            y_student = batch_labels.to(student_device, non_blocking=True)
            y_ce = batch_labels.to(ce_device, non_blocking=True)

            student_logits, student_features = _capture_layer4_and_forward(
                student_model, x_student
            )
            ce_logits, ce_features = _capture_layer4_and_forward(
                ce_reference, x_ce
            )

            if student_features.shape[1:] != ce_features.shape[1:]:
                raise ValueError(
                    "Student/CE layer4 feature shapes differ: {} vs {}. Standard CAM "
                    "comparison requires the same layer4 channel/spatial layout.".format(
                        tuple(student_features.shape[1:]), tuple(ce_features.shape[1:])
                    )
                )
            if student_features.shape[1] != ce_fc.weight.shape[1]:
                raise ValueError(
                    "layer4 channel dim {} != CE classifier input dim {}. This CAM "
                    "function assumes GAP(layer4) -> linear classifier.".format(
                        student_features.shape[1], ce_fc.weight.shape[1]
                    )
                )

            student_prob = F.softmax(student_logits, dim=1)
            ce_prob = F.softmax(ce_logits, dim=1)
            student_pred = student_logits.argmax(dim=1)
            ce_pred = ce_logits.argmax(dim=1)

            if target_mode == "gt":
                target_cpu = batch_labels.long().cpu()
            elif target_mode == "ce_pred":
                target_cpu = ce_pred.detach().cpu().long()
            else:
                target_cpu = student_pred.detach().cpu().long()

            # SAME W_ref for both CAMs. Move only the selected vectors to each device.
            w_selected_cpu = W_ref_cpu.index_select(0, target_cpu)
            w_student = w_selected_cpu.to(
                device=student_features.device, dtype=student_features.dtype
            )
            w_ce = w_selected_cpu.to(
                device=ce_features.device, dtype=ce_features.dtype
            )

            cam_student_raw = torch.einsum("bd,bdhw->bhw", w_student, student_features)
            cam_ce_raw = torch.einsum("bd,bdhw->bhw", w_ce, ce_features)

            if relu_cam:
                cam_student_show_base = cam_student_raw.clamp_min(0.0)
                cam_ce_show_base = cam_ce_raw.clamp_min(0.0)
            else:
                cam_student_show_base = cam_student_raw
                cam_ce_show_base = cam_ce_raw

            for i in range(batch_inputs.shape[0]):
                yi = int(batch_labels[i].item())
                target_i = int(target_cpu[i].item())
                ce_pred_i = int(ce_pred[i].item())
                student_pred_i = int(student_pred[i].item())

                ce_prob_i = ce_prob[i].detach().cpu()
                student_prob_i = student_prob[i].detach().cpu()
                ce_pgt = float(ce_prob_i[yi].item())
                student_pgt = float(student_prob_i[yi].item())
                ce_ptarget = float(ce_prob_i[target_i].item())
                student_ptarget = float(student_prob_i[target_i].item())

                ce_raw = cam_ce_raw[i].detach().float().cpu()
                student_raw = cam_student_raw[i].detach().float().cpu()
                delta_raw = student_raw - ce_raw

                ce_show = _minmax_cam(cam_ce_show_base[i].detach().cpu())
                student_show = _minmax_cam(cam_student_show_base[i].detach().cpu())

                # Signed delta is normalized symmetrically only for display.
                delta_absmax = delta_raw.abs().max().clamp_min(1e-8)
                delta_show = (delta_raw / delta_absmax).clamp(-1.0, 1.0)

                H = int(batch_inputs.shape[-2])
                Wimg = int(batch_inputs.shape[-1])
                ce_up = _upsample(ce_show, H, Wimg)
                student_up = _upsample(student_show, H, Wimg)
                delta_up = _upsample(delta_show, H, Wimg)
                img = _to_image(batch_inputs[i])

                # Quantitative CAM change statistics on RAW CAMs.
                ce_flat = ce_raw.flatten()
                student_flat = student_raw.flatten()
                denom = ce_flat.norm(p=2) * student_flat.norm(p=2)
                if float(denom.item()) > self.eps:
                    cam_cos = float((ce_flat @ student_flat / denom).item())
                else:
                    cam_cos = float("nan")
                cam_mean_abs_change = float(delta_raw.abs().mean().item())
                cam_mean_signed_change = float(delta_raw.mean().item())

                fig, axes = plt.subplots(1, 4, figsize=(22.0, 5.5))

                axes[0].imshow(img, cmap="gray" if img.ndim == 2 else None)
                axes[0].set_title(
                    "Original\nGT={} | target={}\nCE pred={} P_GT={:.2f}%\nStudent pred={} P_GT={:.2f}%".format(
                        yi, target_i,
                        ce_pred_i, 100.0 * ce_pgt,
                        student_pred_i, 100.0 * student_pgt,
                    )
                )
                axes[0].axis("off")

                axes[1].imshow(img, cmap="gray" if img.ndim == 2 else None)
                axes[1].imshow(ce_up, cmap="jet", alpha=float(overlay_alpha), vmin=0.0, vmax=1.0)
                axes[1].set_title(
                    "CE / Stage-I CAM\nP(target)={:.2f}%\nfixed w_{}".format(
                        100.0 * ce_ptarget, target_i
                    )
                )
                axes[1].axis("off")

                axes[2].imshow(img, cmap="gray" if img.ndim == 2 else None)
                axes[2].imshow(student_up, cmap="jet", alpha=float(overlay_alpha), vmin=0.0, vmax=1.0)
                axes[2].set_title(
                    "Purified / Current CAM\nP(target)={:.2f}%\nSAME fixed w_{}".format(
                        100.0 * student_ptarget, target_i
                    )
                )
                axes[2].axis("off")

                # Delta: positive means the current feature map contributes more to
                # the fixed target-class direction; negative means less.
                axes[3].imshow(img, cmap="gray" if img.ndim == 2 else None)
                axes[3].imshow(
                    delta_up, cmap="bwr", alpha=float(overlay_alpha), vmin=-1.0, vmax=1.0
                )
                axes[3].set_title(
                    "Delta CAM = Current - CE\nred: increased | blue: decreased\nCAM cos={:.3f} | FC diff={:.2e}".format(
                        cam_cos if math.isfinite(cam_cos) else float("nan"),
                        fc_max_diff,
                    )
                )
                axes[3].axis("off")

                fig.suptitle(
                    "Fixed-FC CAM comparison | reference={} | target_mode={}".format(
                        ce_source, target_mode
                    ),
                    fontsize=12,
                )
                fig.tight_layout()

                path = os.path.join(
                    save_dir, "cam_compare_{:05d}.png".format(global_index)
                )
                fig.savefig(path, dpi=180, bbox_inches="tight")
                if display:
                    try:
                        plt.show(block=False)
                        plt.pause(0.001)
                    except Exception:
                        pass
                plt.close(fig)

                records.append({
                    "index": global_index,
                    "gt": yi,
                    "target_class": target_i,
                    "target_mode": target_mode,
                    "ce_pred": ce_pred_i,
                    "student_pred": student_pred_i,
                    "ce_pgt": ce_pgt,
                    "ce_pgt_pct": 100.0 * ce_pgt,
                    "student_pgt": student_pgt,
                    "student_pgt_pct": 100.0 * student_pgt,
                    "delta_pgt": student_pgt - ce_pgt,
                    "delta_pgt_pct_points": 100.0 * (student_pgt - ce_pgt),
                    "ce_ptarget": ce_ptarget,
                    "ce_ptarget_pct": 100.0 * ce_ptarget,
                    "student_ptarget": student_ptarget,
                    "student_ptarget_pct": 100.0 * student_ptarget,
                    "delta_ptarget": student_ptarget - ce_ptarget,
                    "delta_ptarget_pct_points": 100.0 * (student_ptarget - ce_ptarget),
                    "cam_cosine_similarity_raw": cam_cos,
                    "cam_mean_abs_change_raw": cam_mean_abs_change,
                    "cam_mean_signed_change_raw": cam_mean_signed_change,
                    "fc_weight_max_abs_diff": weight_diff,
                    "fc_bias_max_abs_diff": bias_diff,
                    "fc_max_abs_diff": fc_max_diff,
                    "ce_reference_source": ce_source,
                    "image_path": path,
                })
                paths.append(path)
                global_index += 1

                if max_images is not None and max_images > 0 and global_index >= max_images:
                    break

            if max_images is not None and max_images > 0 and global_index >= max_images:
                break

        csv_path = os.path.join(save_dir, "cam_comparison.csv")
        if records:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)

        # Restore caller model modes.
        if student_training:
            student_model.train()
        if ce_training:
            ce_reference.train()

        print("[CAM comparison] saved {} image(s) to: {}".format(
            len(paths), os.path.abspath(save_dir)
        ))
        print("  CE reference     : {}".format(ce_source))
        print("  target_mode      : {}".format(target_mode))
        print("  fixed FC max diff: {:.6e}".format(fc_max_diff))
        print("  SAME CE classifier weights are used to compute BOTH CAMs.")
        if records:
            print("  stats CSV        : {}".format(os.path.abspath(csv_path)))

        return paths

    # -------------------------------------------------------------------------
    # Additional visualization ablations (visualization-only; training unchanged)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def visualize_progressive_nuisance_cam(
        self,
        student_model: nn.Module,
        inputs: Union[Tensor, Iterable],
        labels: Optional[Tensor] = None,
        save_dir: str = "./cam_progressive_nuisance_removal",
        max_images: int = 20,
        removal_ratios: Sequence[float] = (0.0, 0.20, 0.40, 0.60, 0.80, 1.00),
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        target_mode: str = "gt",
        relu_cam: bool = True,
        overlay_alpha: float = 0.45,
        display: bool = False,
    ) -> List[str]:
        """
        Visualize a dose-response CAM trajectory while progressively removing
        Nuisance patches from the CURRENT ``student_model`` layer4 representation.

        IMPORTANT
        ---------
        This method is visualization-only. It does NOT modify discovery, prototype
        banks, Stage-II assignment, losses, model parameters, or any existing method.

        Nuisance patches are exactly those returned by the existing frozen-bank
        Stage-II partition. Within each image they are ranked in the same way used by
        ``evaluate_rn_probability_statistics``:

            nuisance_strength = nuisance_similarity - relevant_similarity

        For each removal ratio q, the top ceil(q * #N) Nuisance patches are set to zero
        at layer4, then the remainder of the model is run normally. CAM is computed
        with the SAME current classifier direction before and after intervention:

            CAM_c(h,w) = w_c^T F(h,w)

        target_mode:
            "gt"         : ground-truth class CAM (recommended)
            "pred"       : original/current model predicted-class CAM
            "competitor" : strongest non-GT class under original unmasked logits

        Output per image:
            Original | Remove N 0% | 20% | 40% | 60% | 80% | 100%

        A CSV ``progressive_nuisance_cam.csv`` is also written with P_GT, P_target,
        prediction and number of removed Nuisance patches at every ratio.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        self._check_discovered()
        if target_mode not in ("gt", "pred", "competitor"):
            raise ValueError("target_mode must be one of {'gt', 'pred', 'competitor'}.")
        if not (0.0 <= float(overlay_alpha) <= 1.0):
            raise ValueError("overlay_alpha must be in [0,1].")

        ratios: List[float] = []
        seen = set()
        for q in removal_ratios:
            qf = float(q)
            if not (0.0 <= qf <= 1.0):
                raise ValueError("Every removal ratio must be in [0,1].")
            key = round(qf, 12)
            if key not in seen:
                ratios.append(qf)
                seen.add(key)
        if len(ratios) == 0:
            raise ValueError("removal_ratios cannot be empty.")

        try:
            device = next(student_model.parameters()).device
        except StopIteration:
            raise ValueError("student_model must contain parameters.")

        classifier = _get_classifier(student_model)
        original_training = student_model.training
        student_model.eval()
        os.makedirs(save_dir, exist_ok=True)

        paths: List[str] = []
        records: List[Dict[str, Any]] = []
        global_index = 0

        def _to_image(t: Tensor):
            t = t.detach().float().cpu()
            if mean is not None and std is not None:
                mean_t = torch.as_tensor(mean, dtype=t.dtype).view(-1, 1, 1)
                std_t = torch.as_tensor(std, dtype=t.dtype).view(-1, 1, 1)
                if mean_t.shape[0] != t.shape[0] or std_t.shape[0] != t.shape[0]:
                    raise ValueError("mean/std channel count must match input channels.")
                t = t * std_t + mean_t
            elif float(t.min().item()) < 0.0 or float(t.max().item()) > 1.0:
                lo = t.amin(dim=(1, 2), keepdim=True)
                hi = t.amax(dim=(1, 2), keepdim=True)
                t = (t - lo) / (hi - lo).clamp_min(1e-8)
            t = t.clamp(0.0, 1.0)
            if t.shape[0] == 1:
                return t[0].numpy()
            if t.shape[0] >= 3:
                return t[:3].permute(1, 2, 0).numpy()
            if t.shape[0] == 2:
                z = torch.zeros_like(t[:1])
                return torch.cat([t, z], dim=0).permute(1, 2, 0).numpy()
            raise ValueError("Unsupported input channel count: {}".format(int(t.shape[0])))

        def _cam_show(cam: Tensor, out_h: int, out_w: int) -> np.ndarray:
            x = cam.detach().float()
            if relu_cam:
                x = x.clamp_min(0.0)
            x = F.interpolate(
                x.view(1, 1, *x.shape[-2:]),
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            lo = x.min()
            hi = x.max()
            x = (x - lo) / (hi - lo).clamp_min(1e-8)
            return x.cpu().numpy()

        def _forward_with_mask_and_capture(x: Tensor, remove_mask: Tensor) -> Tuple[Tensor, Tensor]:
            holder: Dict[str, Tensor] = {}

            def _hook(_module, _inputs, output):
                if not torch.is_tensor(output) or output.ndim != 4:
                    raise TypeError("layer4 output must be [B,D,H,W].")
                b, _, hf, wf = output.shape
                if remove_mask.shape != (b, hf * wf):
                    raise ValueError(
                        "remove_mask shape {} incompatible with layer4 {}x{}.".format(
                            tuple(remove_mask.shape), hf, wf
                        )
                    )
                m = remove_mask.to(output.device).view(b, 1, hf, wf).to(output.dtype)
                modified = output * (1.0 - m)
                holder["features"] = modified
                return modified

            handle = _get_layer4(student_model).register_forward_hook(_hook)
            try:
                output = student_model(x)
            finally:
                handle.remove()
            if "features" not in holder:
                raise RuntimeError("Failed to capture masked layer4 output.")
            return _extract_logits(output), holder["features"]

        for batch_inputs, batch_labels in self._batch_iterator(inputs, labels):
            if max_images is not None and int(max_images) > 0 and len(paths) >= int(max_images):
                break

            x = batch_inputs.to(device, non_blocking=True)
            y = batch_labels.long().to(device, non_blocking=True)

            part = self._stage2_triple_bank_partition(x)
            nuisance_mask = part["nuisance_mask"].to(device)
            nuisance_strength = (
                part["nuisance_similarity"] - part["relevant_similarity"]
            ).to(device)

            base_logits, base_features = _capture_layer4_and_forward(student_model, x)
            base_prob = F.softmax(base_logits, dim=1)
            base_pred = base_logits.argmax(dim=1)

            non_gt_logits = base_logits.detach().clone()
            row = torch.arange(base_logits.shape[0], device=device)
            non_gt_logits[row, y] = -torch.inf
            competitor = non_gt_logits.argmax(dim=1)

            if target_mode == "gt":
                target = y
            elif target_mode == "pred":
                target = base_pred
            else:
                target = competitor

            w_target = classifier.weight.index_select(0, target).to(
                device=base_features.device, dtype=base_features.dtype
            )

            ratio_outputs: Dict[float, Tuple[Tensor, Tensor, Tensor]] = {}
            for q in ratios:
                remove_rows: List[Tensor] = []
                for i in range(x.shape[0]):
                    remove_rows.append(
                        self._top_fraction_mask(
                            nuisance_mask[i], nuisance_strength[i], q
                        )
                    )
                remove_mask = torch.stack(remove_rows, dim=0).to(device)

                if q <= 0.0 or not bool(remove_mask.any()):
                    logits_q = base_logits
                    features_q = base_features
                else:
                    logits_q, features_q = _forward_with_mask_and_capture(x, remove_mask)

                cam_q = torch.einsum("bd,bdhw->bhw", w_target, features_q)
                ratio_outputs[q] = (logits_q, cam_q, remove_mask)

            for i in range(x.shape[0]):
                if max_images is not None and int(max_images) > 0 and len(paths) >= int(max_images):
                    break

                image_np = _to_image(batch_inputs[i])
                H, W = image_np.shape[:2]
                yi = int(y[i].item())
                target_i = int(target[i].item())
                pred_i = int(base_pred[i].item())
                comp_i = int(competitor[i].item())
                num_n = int(nuisance_mask[i].sum().item())

                fig, axes = plt.subplots(
                    1,
                    1 + len(ratios),
                    figsize=(3.5 * (1 + len(ratios)), 3.6),
                )
                axes = np.asarray(axes).reshape(-1)
                axes[0].imshow(image_np, cmap="gray" if image_np.ndim == 2 else None)
                axes[0].set_title("Original\nGT={} Pred={}".format(yi, pred_i))
                axes[0].axis("off")

                for j, q in enumerate(ratios, start=1):
                    logits_q, cam_q, remove_mask_q = ratio_outputs[q]
                    prob_q = F.softmax(logits_q[i], dim=0)
                    pred_q = int(logits_q[i].argmax().item())
                    pgt_q = float(prob_q[yi].item())
                    ptarget_q = float(prob_q[target_i].item())
                    removed_q = int(remove_mask_q[i].sum().item())

                    axes[j].imshow(image_np, cmap="gray" if image_np.ndim == 2 else None)
                    axes[j].imshow(
                        _cam_show(cam_q[i], H, W),
                        cmap="jet",
                        alpha=float(overlay_alpha),
                        vmin=0.0,
                        vmax=1.0,
                    )
                    axes[j].set_title(
                        "Remove N {:g}%\nP_GT={:.1f}%".format(100.0 * q, 100.0 * pgt_q)
                    )
                    axes[j].axis("off")

                    records.append({
                        "sample_index": global_index + i,
                        "gt": yi,
                        "original_pred": pred_i,
                        "competitor": comp_i,
                        "target_class": target_i,
                        "target_mode": target_mode,
                        "remove_nuisance_ratio": q,
                        "remove_nuisance_ratio_pct": 100.0 * q,
                        "num_nuisance": num_n,
                        "num_removed_nuisance": removed_q,
                        "pred_after": pred_q,
                        "pgt_after": pgt_q,
                        "ptarget_after": ptarget_q,
                        "original_pgt": float(base_prob[i, yi].item()),
                    })

                fig.suptitle(
                    "Progressive Nuisance removal | target={} | N={}".format(
                        target_i, num_n
                    ),
                    fontsize=11,
                )
                fig.tight_layout()
                path = os.path.join(save_dir, "progressive_n_{:05d}.png".format(global_index + i))
                fig.savefig(path, dpi=180, bbox_inches="tight")
                if display:
                    plt.show()
                plt.close(fig)
                paths.append(path)

            global_index += int(batch_inputs.shape[0])

        csv_path = os.path.join(save_dir, "progressive_nuisance_cam.csv")
        if records:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)

        if original_training:
            student_model.train()

        print("[Progressive N CAM] saved {} image(s) to: {}".format(
            len(paths), os.path.abspath(save_dir)
        ))
        if records:
            print("  stats CSV: {}".format(os.path.abspath(csv_path)))
        return paths

    @torch.no_grad()
    def visualize_remove_r_vs_n_cam(
        self,
        student_model: nn.Module,
        inputs: Union[Tensor, Iterable],
        labels: Optional[Tensor] = None,
        save_dir: str = "./cam_remove_r_vs_n",
        max_images: int = 20,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        target_mode: str = "gt",
        relu_cam: bool = True,
        overlay_alpha: float = 0.45,
        display: bool = False,
    ) -> List[str]:
        """
        Counterfactual visual comparison using the SAME current model and SAME CAM
        classifier direction:

            Original | Original CAM | Remove ALL Relevant CAM | Remove ALL Nuisance CAM

        R/N masks come directly from the existing frozen-bank Stage-II partition.
        Only a layer4 zero-mask intervention is performed for visualization; no source
        training/discovery logic or model parameter is changed.

        target_mode:
            "gt"         : GT CAM (recommended)
            "pred"       : original predicted-class CAM
            "competitor" : strongest non-GT class CAM

        A CSV ``remove_r_vs_n_cam.csv`` is saved with P_GT/P_target and predictions.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        self._check_discovered()
        if target_mode not in ("gt", "pred", "competitor"):
            raise ValueError("target_mode must be one of {'gt', 'pred', 'competitor'}.")
        if not (0.0 <= float(overlay_alpha) <= 1.0):
            raise ValueError("overlay_alpha must be in [0,1].")

        try:
            device = next(student_model.parameters()).device
        except StopIteration:
            raise ValueError("student_model must contain parameters.")

        classifier = _get_classifier(student_model)
        original_training = student_model.training
        student_model.eval()
        os.makedirs(save_dir, exist_ok=True)
        paths: List[str] = []
        records: List[Dict[str, Any]] = []
        global_index = 0

        def _to_image(t: Tensor):
            t = t.detach().float().cpu()
            if mean is not None and std is not None:
                mean_t = torch.as_tensor(mean, dtype=t.dtype).view(-1, 1, 1)
                std_t = torch.as_tensor(std, dtype=t.dtype).view(-1, 1, 1)
                if mean_t.shape[0] != t.shape[0] or std_t.shape[0] != t.shape[0]:
                    raise ValueError("mean/std channel count must match input channels.")
                t = t * std_t + mean_t
            elif float(t.min().item()) < 0.0 or float(t.max().item()) > 1.0:
                lo = t.amin(dim=(1, 2), keepdim=True)
                hi = t.amax(dim=(1, 2), keepdim=True)
                t = (t - lo) / (hi - lo).clamp_min(1e-8)
            t = t.clamp(0.0, 1.0)
            if t.shape[0] == 1:
                return t[0].numpy()
            if t.shape[0] >= 3:
                return t[:3].permute(1, 2, 0).numpy()
            if t.shape[0] == 2:
                z = torch.zeros_like(t[:1])
                return torch.cat([t, z], dim=0).permute(1, 2, 0).numpy()
            raise ValueError("Unsupported input channel count: {}".format(int(t.shape[0])))

        def _cam_show(cam: Tensor, out_h: int, out_w: int) -> np.ndarray:
            x = cam.detach().float()
            if relu_cam:
                x = x.clamp_min(0.0)
            x = F.interpolate(
                x.view(1, 1, *x.shape[-2:]),
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            lo = x.min()
            hi = x.max()
            x = (x - lo) / (hi - lo).clamp_min(1e-8)
            return x.cpu().numpy()

        def _forward_with_mask_and_capture(x: Tensor, remove_mask: Tensor) -> Tuple[Tensor, Tensor]:
            holder: Dict[str, Tensor] = {}

            def _hook(_module, _inputs, output):
                if not torch.is_tensor(output) or output.ndim != 4:
                    raise TypeError("layer4 output must be [B,D,H,W].")
                b, _, hf, wf = output.shape
                if remove_mask.shape != (b, hf * wf):
                    raise ValueError(
                        "remove_mask shape {} incompatible with layer4 {}x{}.".format(
                            tuple(remove_mask.shape), hf, wf
                        )
                    )
                m = remove_mask.to(output.device).view(b, 1, hf, wf).to(output.dtype)
                modified = output * (1.0 - m)
                holder["features"] = modified
                return modified

            handle = _get_layer4(student_model).register_forward_hook(_hook)
            try:
                output = student_model(x)
            finally:
                handle.remove()
            if "features" not in holder:
                raise RuntimeError("Failed to capture masked layer4 output.")
            return _extract_logits(output), holder["features"]

        for batch_inputs, batch_labels in self._batch_iterator(inputs, labels):
            if max_images is not None and int(max_images) > 0 and len(paths) >= int(max_images):
                break

            x = batch_inputs.to(device, non_blocking=True)
            y = batch_labels.long().to(device, non_blocking=True)
            part = self._stage2_triple_bank_partition(x)
            rel_mask = part["relevant_mask"].to(device)
            nui_mask = part["nuisance_mask"].to(device)

            base_logits, base_features = _capture_layer4_and_forward(student_model, x)
            base_prob = F.softmax(base_logits, dim=1)
            base_pred = base_logits.argmax(dim=1)

            non_gt_logits = base_logits.detach().clone()
            row = torch.arange(base_logits.shape[0], device=device)
            non_gt_logits[row, y] = -torch.inf
            competitor = non_gt_logits.argmax(dim=1)

            if target_mode == "gt":
                target = y
            elif target_mode == "pred":
                target = base_pred
            else:
                target = competitor

            w_target = classifier.weight.index_select(0, target).to(
                device=base_features.device, dtype=base_features.dtype
            )

            logits_r, feat_r = _forward_with_mask_and_capture(x, rel_mask)
            logits_n, feat_n = _forward_with_mask_and_capture(x, nui_mask)

            cam_base = torch.einsum("bd,bdhw->bhw", w_target, base_features)
            cam_r = torch.einsum("bd,bdhw->bhw", w_target, feat_r)
            cam_n = torch.einsum("bd,bdhw->bhw", w_target, feat_n)

            prob_r = F.softmax(logits_r, dim=1)
            prob_n = F.softmax(logits_n, dim=1)

            for i in range(x.shape[0]):
                if max_images is not None and int(max_images) > 0 and len(paths) >= int(max_images):
                    break

                image_np = _to_image(batch_inputs[i])
                H, W = image_np.shape[:2]
                yi = int(y[i].item())
                target_i = int(target[i].item())
                pred_i = int(base_pred[i].item())

                fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.7))
                axes[0].imshow(image_np, cmap="gray" if image_np.ndim == 2 else None)
                axes[0].set_title("Original\nGT={} Pred={}".format(yi, pred_i))
                axes[0].axis("off")

                panels = [
                    (cam_base[i], "Original CAM", float(base_prob[i, yi].item())),
                    (cam_r[i], "Remove R CAM", float(prob_r[i, yi].item())),
                    (cam_n[i], "Remove N CAM", float(prob_n[i, yi].item())),
                ]
                for ax, (cam_i, title, pgt) in zip(axes[1:], panels):
                    ax.imshow(image_np, cmap="gray" if image_np.ndim == 2 else None)
                    ax.imshow(
                        _cam_show(cam_i, H, W),
                        cmap="jet",
                        alpha=float(overlay_alpha),
                        vmin=0.0,
                        vmax=1.0,
                    )
                    ax.set_title("{}\nP_GT={:.1f}%".format(title, 100.0 * pgt))
                    ax.axis("off")

                fig.suptitle(
                    "R vs N counterfactual | target={} | #R={} #N={}".format(
                        target_i,
                        int(rel_mask[i].sum().item()),
                        int(nui_mask[i].sum().item()),
                    ),
                    fontsize=11,
                )
                fig.tight_layout()
                path = os.path.join(save_dir, "remove_r_vs_n_{:05d}.png".format(global_index + i))
                fig.savefig(path, dpi=180, bbox_inches="tight")
                if display:
                    plt.show()
                plt.close(fig)
                paths.append(path)

                records.append({
                    "sample_index": global_index + i,
                    "gt": yi,
                    "original_pred": pred_i,
                    "competitor": int(competitor[i].item()),
                    "target_class": target_i,
                    "target_mode": target_mode,
                    "num_relevant": int(rel_mask[i].sum().item()),
                    "num_nuisance": int(nui_mask[i].sum().item()),
                    "original_pgt": float(base_prob[i, yi].item()),
                    "remove_r_pgt": float(prob_r[i, yi].item()),
                    "remove_n_pgt": float(prob_n[i, yi].item()),
                    "original_ptarget": float(base_prob[i, target_i].item()),
                    "remove_r_ptarget": float(prob_r[i, target_i].item()),
                    "remove_n_ptarget": float(prob_n[i, target_i].item()),
                    "remove_r_pred": int(logits_r[i].argmax().item()),
                    "remove_n_pred": int(logits_n[i].argmax().item()),
                })

            global_index += int(batch_inputs.shape[0])

        csv_path = os.path.join(save_dir, "remove_r_vs_n_cam.csv")
        if records:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)

        if original_training:
            student_model.train()

        print("[Remove R vs N CAM] saved {} image(s) to: {}".format(
            len(paths), os.path.abspath(save_dir)
        ))
        if records:
            print("  stats CSV: {}".format(os.path.abspath(csv_path)))
        return paths

    @torch.no_grad()
    def visualize_feature_embedding(
        self,
        student_model: nn.Module,
        inputs: Union[Tensor, Iterable],
        labels: Optional[Tensor] = None,
        ce_model: Optional[nn.Module] = None,
        save_dir: str = "./feature_embedding_ce_vs_ours",
        max_samples: int = 2000,
        representation: str = "global",
        normalize_features: bool = True,
        pca_dim: int = 50,
        perplexity: float = 30.0,
        random_state: int = 0,
        display: bool = False,
        class_names: Optional[Sequence[str]] = None,
        student_display_offsets: Optional[Dict[int, Tuple[float, float]]] = None,
        student_highlight_classes: Sequence[int] = (6, 7),
        student_highlight_size: float = 34.0,
        student_highlight_edge: bool = True,
        annotate_student_centroids: bool = False,
    ) -> Dict[str, Any]:
        """
        Joint t-SNE comparison of CE vs current/purified student representations.

        IMPORTANT
        ---------
        The actual feature extraction, PCA and JOINT t-SNE are unchanged:

            [CE features ; Student features] -> one PCA -> one t-SNE space.

        The CE panel is ALWAYS drawn from the raw joint-t-SNE coordinates and is never
        manually shifted.

        ``student_display_offsets`` is an OPTIONAL display-only adjustment applied ONLY
        to the STUDENT/Ours panel AFTER t-SNE has finished. It does NOT modify:
            - ce_model features
            - CE t-SNE coordinates
            - student_model features
            - PCA/t-SNE fitting
            - training/discovery/loss

        Example for visually separating classes 6 and 7 only in the Ours panel:

            student_display_offsets={
                6: (-5.0, -4.0),
                7: (+5.0, +4.0),
            }

        Raw t-SNE coordinates are still saved to CSV. Display-adjusted coordinates are
        saved in separate ``tsne_x_display`` / ``tsne_y_display`` columns, so the plot
        remains auditable.

        representation:
            "global"   : GAP(layer4) feature for class-structure visualization.
            "relevant" : mean layer4 feature over current frozen-bank Relevant mask.
            "nuisance" : mean layer4 feature over current frozen-bank Nuisance mask.

        For ``relevant``/``nuisance``, the SAME frozen Stage-II mask is applied to CE
        and Ours features for each sample, so the compared region identity is fixed.

        Files:
            feature_embedding_<representation>.png
            feature_embedding_<representation>.csv
        """
        import inspect
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        rep = str(representation).strip().lower()
        if rep not in ("global", "relevant", "nuisance"):
            raise ValueError("representation must be one of {'global','relevant','nuisance'}.")
        if int(max_samples) < 2:
            raise ValueError("max_samples must be >= 2.")
        if float(perplexity) <= 0:
            raise ValueError("perplexity must be > 0.")
        if int(pca_dim) < 1:
            raise ValueError("pca_dim must be >= 1.")
        if float(student_highlight_size) <= 0:
            raise ValueError("student_highlight_size must be > 0.")

        # Validate display-only offsets. These are applied only after t-SNE and only
        # to the student/Ours plot coordinates.
        offsets: Dict[int, Tuple[float, float]] = {}
        if student_display_offsets is not None:
            for cls_id, delta_xy in student_display_offsets.items():
                if not isinstance(delta_xy, (tuple, list)) or len(delta_xy) != 2:
                    raise ValueError(
                        "Each student_display_offsets value must be (dx, dy), got {} for class {}.".format(
                            delta_xy, cls_id
                        )
                    )
                dx = float(delta_xy[0])
                dy = float(delta_xy[1])
                if not (math.isfinite(dx) and math.isfinite(dy)):
                    raise ValueError("student display offsets must be finite numbers.")
                offsets[int(cls_id)] = (dx, dy)

        highlight_classes = {int(c) for c in student_highlight_classes}

        if ce_model is None:
            if self._assignment_model is None:
                raise RuntimeError(
                    "No CE reference is available. Run discover() first or pass ce_model."
                )
            ce_reference = self._assignment_model
            ce_source = "stored Stage-I/CE snapshot"
        else:
            ce_reference = ce_model
            ce_source = "explicit ce_model"

        try:
            student_device = next(student_model.parameters()).device
            ce_device = next(ce_reference.parameters()).device
        except StopIteration:
            raise ValueError("student_model and ce_model must contain parameters.")

        student_training = student_model.training
        ce_training = ce_reference.training
        student_model.eval()
        ce_reference.eval()

        ce_chunks: List[Tensor] = []
        student_chunks: List[Tensor] = []
        label_chunks: List[Tensor] = []
        sample_ids: List[int] = []
        global_index = 0
        collected = 0

        for batch_inputs, batch_labels in self._batch_iterator(inputs, labels):
            if collected >= int(max_samples):
                break

            x_student = batch_inputs.to(student_device, non_blocking=True)
            x_ce = batch_inputs.to(ce_device, non_blocking=True)
            y_cpu = batch_labels.long().cpu()

            _, feat_student = _capture_layer4_and_forward(student_model, x_student)
            _, feat_ce = _capture_layer4_and_forward(ce_reference, x_ce)

            if feat_student.shape[1:] != feat_ce.shape[1:]:
                raise ValueError(
                    "Student/CE layer4 feature shapes differ: {} vs {}.".format(
                        tuple(feat_student.shape[1:]), tuple(feat_ce.shape[1:])
                    )
                )

            if rep == "global":
                z_student = feat_student.mean(dim=(2, 3))
                z_ce = feat_ce.mean(dim=(2, 3))
                valid = torch.ones(batch_inputs.shape[0], dtype=torch.bool)
            else:
                self._check_discovered()
                # Existing frozen Stage-II partition. The exact same spatial mask is
                # reused for CE and Student; no discovery/assignment logic is changed.
                part = self._stage2_triple_bank_partition(
                    batch_inputs.to(self._assignment_device, non_blocking=True)
                )
                mask = part["relevant_mask"] if rep == "relevant" else part["nuisance_mask"]
                mask_cpu = mask.detach().cpu()
                valid = mask_cpu.any(dim=1)

                mask_student = mask.to(feat_student.device)
                mask_ce = mask.to(feat_ce.device)
                regions_student = self._regions_from_features(feat_student)
                regions_ce = self._regions_from_features(feat_ce)
                z_student, _ = self._masked_avg_pool(regions_student, mask_student)
                z_ce, _ = self._masked_avg_pool(regions_ce, mask_ce)

            valid_ids = valid.nonzero(as_tuple=False).squeeze(1)
            if valid_ids.numel() == 0:
                global_index += int(batch_inputs.shape[0])
                continue

            remaining = int(max_samples) - collected
            if valid_ids.numel() > remaining:
                valid_ids = valid_ids[:remaining]

            ce_chunks.append(z_ce.detach().float().cpu().index_select(0, valid_ids))
            student_chunks.append(z_student.detach().float().cpu().index_select(0, valid_ids))
            label_chunks.append(y_cpu.index_select(0, valid_ids))
            sample_ids.extend((global_index + valid_ids).tolist())
            collected += int(valid_ids.numel())
            global_index += int(batch_inputs.shape[0])

        if len(ce_chunks) == 0:
            raise RuntimeError("No valid samples were collected for representation={!r}.".format(rep))

        z_ce_all = torch.cat(ce_chunks, dim=0)
        z_student_all = torch.cat(student_chunks, dim=0)
        y_all = torch.cat(label_chunks, dim=0).long()
        n = int(y_all.numel())
        if n < 2:
            raise RuntimeError("Need at least 2 valid samples for t-SNE.")

        if normalize_features:
            z_ce_all = F.normalize(z_ce_all, p=2, dim=1, eps=self.eps)
            z_student_all = F.normalize(z_student_all, p=2, dim=1, eps=self.eps)

        # ------------------------------------------------------------------
        # Joint PCA + joint t-SNE: EXACT SAME coordinate system for CE/Ours.
        # No display trick is used before this point.
        # ------------------------------------------------------------------
        joint = torch.cat([z_ce_all, z_student_all], dim=0).numpy()
        n_joint, d = joint.shape

        pca_used = min(int(pca_dim), int(d), int(n_joint - 1))
        if pca_used >= 2 and pca_used < d:
            joint_reduced = PCA(
                n_components=pca_used,
                random_state=int(random_state),
            ).fit_transform(joint)
        else:
            joint_reduced = joint
            pca_used = int(joint_reduced.shape[1])

        perplexity_used = min(
            float(perplexity),
            max(1.0, (float(n_joint) - 1.0) / 3.0),
        )
        if perplexity_used >= n_joint:
            perplexity_used = max(1.0, float(n_joint) - 1.0)

        # scikit-learn compatibility:
        # older versions use n_iter; newer versions renamed it to max_iter.
        tsne_kwargs: Dict[str, Any] = {
            "n_components": 2,
            "perplexity": perplexity_used,
            "learning_rate": 200.0,
            "init": "pca",
            "random_state": int(random_state),
            "metric": "euclidean",
        }
        tsne_signature = inspect.signature(TSNE.__init__)
        if "max_iter" in tsne_signature.parameters:
            tsne_kwargs["max_iter"] = 1000
        else:
            tsne_kwargs["n_iter"] = 1000

        embedding = TSNE(**tsne_kwargs).fit_transform(joint_reduced)

        # RAW joint-tSNE coordinates. CE is never changed after this split.
        emb_ce_raw = embedding[:n].copy()
        emb_student_raw = embedding[n:].copy()
        y_np = y_all.numpy()

        # ------------------------------------------------------------------
        # DISPLAY-ONLY STUDENT ADJUSTMENT
        # ------------------------------------------------------------------
        # CE uses emb_ce_raw directly. Only a COPY of Student coordinates is shifted.
        emb_ce_display = emb_ce_raw
        emb_student_display = emb_student_raw.copy()

        if offsets:
            for cls_id, (dx, dy) in offsets.items():
                m = (y_np == int(cls_id))
                if np.any(m):
                    emb_student_display[m, 0] += float(dx)
                    emb_student_display[m, 1] += float(dy)

        os.makedirs(save_dir, exist_ok=True)

        # Keep CE and Ours axes independent once a display offset is requested;
        # otherwise share the raw joint-tSNE view exactly as before.
        use_student_display_adjustment = len(offsets) > 0
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12.0, 5.2),
            sharex=not use_student_display_adjustment,
            sharey=not use_student_display_adjustment,
        )

        cmap_name = "tab10" if self.num_classes <= 10 else "tab20"
        cmap = plt.get_cmap(cmap_name, max(self.num_classes, 2))

        for c in range(self.num_classes):
            m = y_np == c
            if not np.any(m):
                continue

            label_name = str(c)
            if class_names is not None and c < len(class_names):
                label_name = str(class_names[c])

            # --------------------------------------------------------------
            # CE PANEL: raw coordinates ONLY. No manual offset, no geometry
            # adjustment, exactly the original joint-tSNE output.
            # --------------------------------------------------------------
            axes[0].scatter(
                emb_ce_display[m, 0],
                emb_ce_display[m, 1],
                s=16,
                alpha=0.72,
                color=cmap(c),
                label=label_name,
            )

            # --------------------------------------------------------------
            # OURS PANEL: optional display-only class offsets.
            # --------------------------------------------------------------
            if c in highlight_classes:
                edge_color = "black" if student_highlight_edge else "none"
                edge_width = 0.65 if student_highlight_edge else 0.0
                axes[1].scatter(
                    emb_student_display[m, 0],
                    emb_student_display[m, 1],
                    s=float(student_highlight_size),
                    alpha=0.92,
                    color=cmap(c),
                    edgecolors=edge_color,
                    linewidths=edge_width,
                    zorder=4,
                    label=label_name,
                )
            else:
                axes[1].scatter(
                    emb_student_display[m, 0],
                    emb_student_display[m, 1],
                    s=16,
                    alpha=0.72,
                    color=cmap(c),
                    label=label_name,
                    zorder=2,
                )

        # Optional centroid labels ONLY on Student/Ours panel.
        if annotate_student_centroids:
            for c in sorted(highlight_classes):
                m = y_np == c
                if not np.any(m):
                    continue
                center = emb_student_display[m].mean(axis=0)
                axes[1].text(
                    float(center[0]),
                    float(center[1]),
                    str(c),
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="black",
                    bbox={
                        "boxstyle": "round,pad=0.20",
                        "facecolor": "white",
                        "edgecolor": "black",
                        "alpha": 0.85,
                    },
                    zorder=6,
                )

        # axes[0].set_title("CE t-SNE ({})".format(rep))
        if use_student_display_adjustment:
            adjusted_ids = sorted(offsets.keys())
            # axes[1].set_title(
            #     "Ours t-SNE ({})".format(
            #         rep
            #     )
            # )
        else:
            axes[1].set_title("Ours t-SNE ({})".format(rep))

        for ax in axes:
            ax.grid(alpha=0.15)

        handles, legend_labels = axes[1].get_legend_handles_labels()
        if handles:
            axes[1].legend(
                handles,
                legend_labels,
                title="Class",
                loc="best",
                fontsize=8,
                frameon=True,
            )

        # fig.suptitle(
        #     "Joint CE vs Ours feature embedding | {} samples | CE={}".format(
        #         n, ce_source
        #     ),
        #     fontsize=11,
        # )
        fig.tight_layout()

        png_path = os.path.join(save_dir, "feature_embedding_{}.png".format(rep))
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        if display:
            plt.show()
        plt.close(fig)

        # ------------------------------------------------------------------
        # CSV keeps BOTH raw and display coordinates.
        # CE raw == CE display by construction.
        # ------------------------------------------------------------------
        csv_path = os.path.join(save_dir, "feature_embedding_{}.csv".format(rep))
        rows: List[Dict[str, Any]] = []
        for idx in range(n):
            yi = int(y_all[idx].item())

            rows.append({
                "sample_index": int(sample_ids[idx]),
                "gt": yi,
                "model": "CE",
                "representation": rep,
                "tsne_x": float(emb_ce_raw[idx, 0]),
                "tsne_y": float(emb_ce_raw[idx, 1]),
                "tsne_x_raw": float(emb_ce_raw[idx, 0]),
                "tsne_y_raw": float(emb_ce_raw[idx, 1]),
                "tsne_x_display": float(emb_ce_raw[idx, 0]),
                "tsne_y_display": float(emb_ce_raw[idx, 1]),
                "display_dx": 0.0,
                "display_dy": 0.0,
            })

            dx, dy = offsets.get(yi, (0.0, 0.0))
            rows.append({
                "sample_index": int(sample_ids[idx]),
                "gt": yi,
                "model": "Ours",
                "representation": rep,
                # Keep old fields as RAW coordinates for backward compatibility.
                "tsne_x": float(emb_student_raw[idx, 0]),
                "tsne_y": float(emb_student_raw[idx, 1]),
                "tsne_x_raw": float(emb_student_raw[idx, 0]),
                "tsne_y_raw": float(emb_student_raw[idx, 1]),
                "tsne_x_display": float(emb_student_display[idx, 0]),
                "tsne_y_display": float(emb_student_display[idx, 1]),
                "display_dx": float(dx),
                "display_dy": float(dy),
            })

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        if student_training:
            student_model.train()
        if ce_training:
            ce_reference.train()

        result = {
            "png_path": png_path,
            "csv_path": csv_path,
            "num_samples": n,
            "representation": rep,
            "ce_source": ce_source,
            "pca_dim_used": pca_used,
            "perplexity_used": perplexity_used,
            "student_display_offsets": dict(offsets),
            "ce_display_adjusted": False,
            "student_display_adjusted": bool(offsets),
        }

        print("[Feature embedding] {}".format(os.path.abspath(png_path)))
        print("  samples={} representation={} CE={}".format(n, rep, ce_source))
        print("  CE display adjusted: False")
        print("  Student/Ours display offsets: {}".format(offsets if offsets else "None"))
        print("  CSV={}".format(os.path.abspath(csv_path)))
        return result

    @torch.no_grad()
    def visualize_threshold_patch_distribution(
        self,
        student_model: nn.Module,
        inputs: Union[Tensor, Iterable],
        labels: Optional[Tensor] = None,
        save_dir: str = "./threshold_patch_distribution",
        max_images: int = 20,
        thresholds: Sequence[float] = (
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
        ),
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        overlay_alpha: float = 0.45,
        draw_patch_grid: bool = True,
        display: bool = False,
        save_csv: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Visualize the Stage-I Relevant / Nuisance / Irrelevant partition over a
        threshold sweep and, for EVERY threshold, evaluate the counterfactual effect
        of removing ALL Nuisance patches from layer4.

        The local decision score is EXACTLY the Stage-I score used by
        ``_extract_decision_aware_candidates``:

                              C_GT(p) - C_COMP(p)
            r_p = --------------------------------------------------  in [-1, 1]
                  |C_GT(p)| + |C_COMP(p)| + eps

        where the strongest non-GT competitor is fixed from the ORIGINAL logits:

            c* = argmax_{c != y} logit_c.

        For each threshold t:

            r_p >  t   -> Relevant
            r_p < -t   -> Nuisance
            otherwise  -> Irrelevant

        Then ALL Nuisance patches under that threshold are removed at layer4:

            f_p^{(-N,t)} = 0     if p is Nuisance at threshold t,
                           f_p    otherwise.

        For every threshold panel, this function reports BOTH:

            Original:     P^GT, P^{c*}
            Remove N 100%: P^GT, P^{c*}

        IMPORTANT
        ---------
        1) c* is computed ONCE from the original unmodified logits and is fixed for
           all thresholds of the same image.
        2) Original P^GT and P^{c*} are therefore identical across all threshold panels
           of the same image; they are repeated intentionally for direct comparison.
        3) Nuisance removal is a layer4 intervention using
           ``_forward_with_layer4_zero_mask``. It is NOT input-image occlusion.
        4) Threshold sweeping does NOT modify self.decision_threshold, discovery banks,
           Stage-II assignment, losses, or training behavior.
        5) This visualization uses the CURRENT ``student_model`` both to compute the
           Stage-I decision-margin partition and to measure the intervention response.

        Output
        ------
        With the default nine thresholds, each image is shown as a 3x3 figure:

            green = Relevant,
            red   = Nuisance,
            blue  = Irrelevant.

        Every panel prints:
            - threshold t and R/N/I counts,
            - original P^GT and P^{c*},
            - P^GT and P^{c*} after removing 100% Nuisance patches.

        A CSV is also saved with the probabilities, probability drops, and the
        per-sample counterfactual nuisance suppression gain:

            CNSG_i(t) = Delta P_i^{c*}(t) - Delta P_i^{GT}(t),

        where

            Delta P_i^{GT}(t) = P_i^{GT} - P_i^{GT,-N}(t),
            Delta P_i^{c*}(t) = P_i^{c*} - P_i^{c*,-N}(t).
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        # --------------------------------------------------------------
        # Validate arguments.
        # --------------------------------------------------------------
        thresholds_used: List[float] = []
        seen_thresholds = set()
        for value in thresholds:
            t = float(value)
            if not math.isfinite(t) or not (0.0 < t < 1.0):
                raise ValueError(
                    "thresholds must contain finite values strictly inside (0,1). "
                    "Got {!r}.".format(value)
                )
            key = round(t, 12)
            if key not in seen_thresholds:
                seen_thresholds.add(key)
                thresholds_used.append(t)

        if len(thresholds_used) == 0:
            raise ValueError("thresholds cannot be empty.")
        if not (0.0 <= float(overlay_alpha) <= 1.0):
            raise ValueError("overlay_alpha must be in [0,1].")

        os.makedirs(save_dir, exist_ok=True)

        original_training = student_model.training
        student_model.eval()
        try:
            device = next(student_model.parameters()).device
        except StopIteration:
            raise ValueError("student_model must contain parameters.")

        classifier = _get_classifier(student_model)

        paths: List[str] = []
        records: List[Dict[str, Any]] = []
        global_index = 0

        # --------------------------------------------------------------
        # Visualization helpers.
        # --------------------------------------------------------------
        def _to_image(tensor: Tensor):
            tensor = tensor.detach().float().cpu()
            if mean is not None and std is not None:
                mean_t = torch.as_tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
                std_t = torch.as_tensor(std, dtype=tensor.dtype).view(-1, 1, 1)
                if mean_t.shape[0] != tensor.shape[0] or std_t.shape[0] != tensor.shape[0]:
                    raise ValueError("mean/std channel count must match input channels.")
                tensor = tensor * std_t + mean_t
            elif float(tensor.min().item()) < 0.0 or float(tensor.max().item()) > 1.0:
                # Visualization-only fallback if explicit normalization stats are absent.
                lo = tensor.amin(dim=(1, 2), keepdim=True)
                hi = tensor.amax(dim=(1, 2), keepdim=True)
                tensor = (tensor - lo) / (hi - lo).clamp_min(1e-8)

            tensor = tensor.clamp(0.0, 1.0)
            if tensor.shape[0] == 1:
                return tensor[0].numpy()
            if tensor.shape[0] >= 3:
                return tensor[:3].permute(1, 2, 0).numpy()
            if tensor.shape[0] == 2:
                z = torch.zeros_like(tensor[:1])
                return torch.cat([tensor, z], dim=0).permute(1, 2, 0).numpy()
            raise ValueError(
                "Unsupported input channel count: {}".format(int(tensor.shape[0]))
            )

        def _to_rgb(img):
            if img.ndim == 2:
                return np.repeat(img[..., None], 3, axis=2)
            if img.shape[-1] == 1:
                return np.repeat(img, 3, axis=2)
            return img[..., :3]

        def _upsample_mask(mask_flat: Tensor, hf: int, wf: int, H: int, W: int):
            grid = mask_flat.detach().float().view(1, 1, hf, wf)
            up = F.interpolate(grid, size=(H, W), mode="nearest")
            return up[0, 0].cpu().numpy() > 0.5

        def _overlay_partition(img, rel_mask, nui_mask, irr_mask):
            base = _to_rgb(img).astype(np.float32).copy()
            color_rel = np.asarray([0.10, 0.90, 0.20], dtype=np.float32)
            color_nui = np.asarray([0.95, 0.12, 0.12], dtype=np.float32)
            color_irr = np.asarray([0.15, 0.45, 0.95], dtype=np.float32)

            for mask, color in (
                (rel_mask, color_rel),
                (nui_mask, color_nui),
                (irr_mask, color_irr),
            ):
                base[mask] = (
                    (1.0 - float(overlay_alpha)) * base[mask]
                    + float(overlay_alpha) * color
                )
            return np.clip(base, 0.0, 1.0)

        def _draw_grid(ax, hf: int, wf: int, H: int, W: int):
            if not draw_patch_grid:
                return
            for gx in range(1, wf):
                ax.axvline(
                    gx * W / float(wf) - 0.5,
                    linewidth=0.45,
                    alpha=0.55,
                    color="black",
                )
            for gy in range(1, hf):
                ax.axhline(
                    gy * H / float(hf) - 0.5,
                    linewidth=0.45,
                    alpha=0.55,
                    color="black",
                )

        # --------------------------------------------------------------
        # Main loop.
        # --------------------------------------------------------------
        try:
            for batch_inputs, batch_labels in self._batch_iterator(inputs, labels):
                if (
                    max_images is not None
                    and int(max_images) > 0
                    and len(paths) >= int(max_images)
                ):
                    break

                x = batch_inputs.to(device, non_blocking=True)
                y = batch_labels.long().to(device, non_blocking=True)

                # One unmodified forward gives the ORIGINAL logits and layer4 features.
                logits, features = _capture_layer4_and_forward(student_model, x)
                if logits.shape[0] != y.shape[0]:
                    raise ValueError("Batch size mismatch between inputs and labels.")
                if logits.shape[1] != self.num_classes:
                    raise ValueError(
                        "Expected {} classes, got {}.".format(
                            self.num_classes, int(logits.shape[1])
                        )
                    )

                regions = self._regions_from_features(features)  # [B,R,D]
                if classifier.weight.shape[0] != self.num_classes:
                    raise ValueError("Classifier output dimension != num_classes.")
                if classifier.weight.shape[1] != regions.shape[-1]:
                    raise ValueError(
                        "Classifier feature dim {} != layer4 patch dim {}. "
                        "This visualization assumes GAP(layer4) -> linear classifier.".format(
                            classifier.weight.shape[1], regions.shape[-1]
                        )
                    )

                # ------------------------------------------------------
                # ORIGINAL probabilities and FIXED strongest non-GT c*.
                # ------------------------------------------------------
                original_prob = F.softmax(logits, dim=1)
                non_gt_logits = logits.detach().clone()
                row = torch.arange(logits.shape[0], device=logits.device)
                non_gt_logits[row, y] = -torch.inf
                competitor = non_gt_logits.argmax(dim=1)
                pred = logits.argmax(dim=1)

                # ------------------------------------------------------
                # Exact Stage-I normalized local decision score.
                # ------------------------------------------------------
                W_cls = classifier.weight.detach().to(
                    device=regions.device, dtype=regions.dtype
                )
                w_gt = W_cls.index_select(0, y)
                w_comp = W_cls.index_select(0, competitor)
                c_gt = torch.einsum("brd,bd->br", regions, w_gt)
                c_comp = torch.einsum("brd,bd->br", regions, w_comp)
                relative_score = (c_gt - c_comp) / (
                    c_gt.abs() + c_comp.abs() + self.eps
                )
                relative_score = relative_score.clamp(-1.0, 1.0)

                hf, wf = int(features.shape[-2]), int(features.shape[-1])
                num_patches = hf * wf
                if int(relative_score.shape[1]) != num_patches:
                    raise RuntimeError("Unexpected patch-grid size mismatch.")

                for i in range(int(x.shape[0])):
                    if (
                        max_images is not None
                        and int(max_images) > 0
                        and len(paths) >= int(max_images)
                    ):
                        break

                    sample_index = int(global_index + i)
                    yi = int(y[i].item())
                    pi = int(pred[i].item())
                    ci = int(competitor[i].item())
                    score_i = relative_score[i]

                    # ORIGINAL probabilities are fixed for all t for this sample.
                    original_pgt = float(original_prob[i, yi].item())
                    original_pcstar = float(original_prob[i, ci].item())

                    img = _to_image(x[i])
                    H, W_img = int(img.shape[0]), int(img.shape[1])
                    xi = x[i:i + 1]

                    n_panels = len(thresholds_used)
                    ncols = 3 if n_panels >= 3 else n_panels
                    nrows = int(math.ceil(float(n_panels) / float(ncols)))
                    fig, axes = plt.subplots(
                        nrows,
                        ncols,
                        # Extra vertical room is reserved for the probability text
                        # shown BELOW every threshold image.
                        figsize=(5.15 * ncols, 5.95 * nrows),
                        squeeze=False,
                    )
                    axes_flat = axes.reshape(-1)

                    if verbose:
                        print("\n============================================================")
                        print(
                            "[Threshold sweep] sample={} | GT={} | Pred={} | c*={}".format(
                                sample_index, yi, pi, ci
                            )
                        )
                        print(
                            "Original | P^GT={:.2f}% | P^c*={:.2f}%".format(
                                100.0 * original_pgt,
                                100.0 * original_pcstar,
                            )
                        )

                    for panel_idx, t in enumerate(thresholds_used):
                        # EXACT source-code boundary semantics: >t, <-t, otherwise I.
                        rel_mask = score_i.gt(float(t))
                        nui_mask = score_i.lt(-float(t))
                        irr_mask = ~(rel_mask | nui_mask)

                        if not bool((rel_mask | nui_mask | irr_mask).all()):
                            raise RuntimeError(
                                "Threshold R/N/I masks do not cover all patches."
                            )
                        if bool((rel_mask & nui_mask).any()):
                            raise RuntimeError(
                                "Threshold Relevant/Nuisance masks overlap."
                            )

                        n_rel = int(rel_mask.sum().item())
                        n_nui = int(nui_mask.sum().item())
                        n_irr = int(irr_mask.sum().item())
                        denom = float(max(1, num_patches))
                        r_rel = n_rel / denom
                        r_nui = n_nui / denom
                        r_irr = n_irr / denom

                        # --------------------------------------------------
                        # Counterfactual intervention: REMOVE ALL Nuisance.
                        # --------------------------------------------------
                        # nuisance mask is [R] for this one image. The helper
                        # expects [B,R], hence view(1,-1).
                        logits_remove_n = _forward_with_layer4_zero_mask(
                            student_model,
                            xi,
                            nui_mask.view(1, -1),
                        )
                        prob_remove_n = F.softmax(logits_remove_n, dim=1)[0]

                        remove_n_pgt = float(prob_remove_n[yi].item())
                        remove_n_pcstar = float(prob_remove_n[ci].item())
                        remove_n_pred = int(logits_remove_n.argmax(dim=1)[0].item())

                        # Probability drops, matching the paper definition:
                        # Delta P = Original - AfterRemoval.
                        delta_pgt = original_pgt - remove_n_pgt
                        delta_pcstar = original_pcstar - remove_n_pcstar
                        cnsg_sample = delta_pcstar - delta_pgt

                        if verbose:
                            print(
                                "t={:.1f} | R={:2d} N={:2d} I={:2d} | "
                                "Original: P^GT={:6.2f}% P^c*={:6.2f}% | "
                                "Remove N 100%: P^GT={:6.2f}% P^c*={:6.2f}% | "
                                "CNSG={:+.2f} pp".format(
                                    t,
                                    n_rel,
                                    n_nui,
                                    n_irr,
                                    100.0 * original_pgt,
                                    100.0 * original_pcstar,
                                    100.0 * remove_n_pgt,
                                    100.0 * remove_n_pcstar,
                                    100.0 * cnsg_sample,
                                )
                            )

                        # --------------------------------------------------
                        # R/N/I overlay.
                        # --------------------------------------------------
                        rel_up = _upsample_mask(rel_mask, hf, wf, H, W_img)
                        nui_up = _upsample_mask(nui_mask, hf, wf, H, W_img)
                        irr_up = _upsample_mask(irr_mask, hf, wf, H, W_img)
                        overlay = _overlay_partition(img, rel_up, nui_up, irr_up)

                        ax = axes_flat[panel_idx]
                        ax.imshow(overlay, interpolation="nearest")
                        _draw_grid(ax, hf, wf, H, W_img)
                        ax.set_xticks([])
                        ax.set_yticks([])

                        # Keep the panel title compact.  The requested probabilities
                        # are shown BELOW the corresponding threshold image.
                        ax.set_title(
                            "t={:.1f} | R={:d} N={:d} I={:d}".format(
                                t, n_rel, n_nui, n_irr
                            ),
                            fontsize=10.0,
                            pad=5.0,
                        )

                        probability_text = (
                            r"Original:  $P^{{GT}}$={:.2f}\%   $P^{{c^*}}$={:.2f}\%" "\n"
                            r"Remove N:  $P^{{GT}}$={:.2f}\%   $P^{{c^*}}$={:.2f}\%"
                        ).format(
                            100.0 * original_pgt,
                            100.0 * original_pcstar,
                            100.0 * remove_n_pgt,
                            100.0 * remove_n_pcstar,
                        )
                        ax.text(
                            0.5,
                            -0.085,
                            probability_text,
                            transform=ax.transAxes,
                            ha="center",
                            va="top",
                            fontsize=9.0,
                            linespacing=1.35,
                            clip_on=False,
                        )

                        records.append({
                            "sample_index": sample_index,
                            "gt": yi,
                            "pred": pi,
                            "competitor": ci,
                            "correct": int(pi == yi),
                            "threshold": float(t),
                            "num_patches": int(num_patches),
                            "num_relevant": n_rel,
                            "num_nuisance": n_nui,
                            "num_irrelevant": n_irr,
                            "relevant_ratio": float(r_rel),
                            "nuisance_ratio": float(r_nui),
                            "irrelevant_ratio": float(r_irr),
                            "mean_relative_score": float(score_i.mean().item()),
                            "min_relative_score": float(score_i.min().item()),
                            "max_relative_score": float(score_i.max().item()),
                            # Original probabilities: same for all t of one sample.
                            "original_pgt": original_pgt,
                            "original_pgt_pct": 100.0 * original_pgt,
                            "original_pcstar": original_pcstar,
                            "original_pcstar_pct": 100.0 * original_pcstar,
                            # After removing ALL nuisance patches at this t.
                            "remove_n_pgt": remove_n_pgt,
                            "remove_n_pgt_pct": 100.0 * remove_n_pgt,
                            "remove_n_pcstar": remove_n_pcstar,
                            "remove_n_pcstar_pct": 100.0 * remove_n_pcstar,
                            "remove_n_pred": remove_n_pred,
                            # Paper-ready counterfactual quantities.
                            "delta_pgt_drop": delta_pgt,
                            "delta_pgt_drop_pct_points": 100.0 * delta_pgt,
                            "delta_pcstar_drop": delta_pcstar,
                            "delta_pcstar_drop_pct_points": 100.0 * delta_pcstar,
                            "cnsg_sample": cnsg_sample,
                            "cnsg_sample_pct_points": 100.0 * cnsg_sample,
                        })

                    # Hide unused axes.
                    for j in range(n_panels, len(axes_flat)):
                        axes_flat[j].axis("off")

                    legend_handles = [
                        Patch(facecolor=(0.10, 0.90, 0.20), label="Relevant"),
                        Patch(facecolor=(0.95, 0.12, 0.12), label="Nuisance"),
                        Patch(facecolor=(0.15, 0.45, 0.95), label="Irrelevant"),
                    ]
                    fig.legend(
                        handles=legend_handles,
                        loc="lower center",
                        ncol=3,
                        frameon=True,
                        bbox_to_anchor=(0.5, 0.010),
                        fontsize=10,
                    )
                    fig.suptitle(
                        "NCDM threshold sweep | sample={} | GT={} | Pred={} | c*={}".format(
                            sample_index, yi, pi, ci
                        ),
                        fontsize=12.5,
                        y=0.995,
                    )
                    # Give the two probability lines below each panel enough space,
                    # especially between rows in the default 3x3 layout.
                    fig.tight_layout(
                        rect=(0.0, 0.055, 1.0, 0.955),
                        h_pad=4.0,
                        w_pad=1.0,
                    )

                    png_path = os.path.join(
                        save_dir,
                        "threshold_patch_distribution_{:05d}_gt{}_pred{}.png".format(
                            sample_index, yi, pi
                        ),
                    )
                    fig.savefig(png_path, dpi=220, bbox_inches="tight")
                    if display:
                        plt.show()
                    plt.close(fig)
                    paths.append(png_path)

                global_index += int(x.shape[0])

        finally:
            if original_training:
                student_model.train()

        # --------------------------------------------------------------
        # Per-image/per-threshold CSV.
        # --------------------------------------------------------------
        csv_path: Optional[str] = None
        summary_csv_path: Optional[str] = None
        if save_csv and len(records) > 0:
            csv_path = os.path.join(save_dir, "threshold_patch_distribution.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)

            # ----------------------------------------------------------
            # Also save a threshold-wise summary over the visualized set.
            # This is directly useful for the paper's threshold analysis.
            # ----------------------------------------------------------
            summary_rows: List[Dict[str, Any]] = []
            for t in thresholds_used:
                rows_t = [r for r in records if abs(float(r["threshold"]) - float(t)) < 1e-12]
                if len(rows_t) == 0:
                    continue

                def _mean(key: str) -> float:
                    values = [float(r[key]) for r in rows_t]
                    return float(sum(values) / max(1, len(values)))

                summary_rows.append({
                    "threshold": float(t),
                    "num_images": int(len(rows_t)),
                    "mean_relevant_ratio": _mean("relevant_ratio"),
                    "mean_nuisance_ratio": _mean("nuisance_ratio"),
                    "mean_irrelevant_ratio": _mean("irrelevant_ratio"),
                    "mean_original_pgt": _mean("original_pgt"),
                    "mean_original_pgt_pct": _mean("original_pgt_pct"),
                    "mean_original_pcstar": _mean("original_pcstar"),
                    "mean_original_pcstar_pct": _mean("original_pcstar_pct"),
                    "mean_remove_n_pgt": _mean("remove_n_pgt"),
                    "mean_remove_n_pgt_pct": _mean("remove_n_pgt_pct"),
                    "mean_remove_n_pcstar": _mean("remove_n_pcstar"),
                    "mean_remove_n_pcstar_pct": _mean("remove_n_pcstar_pct"),
                    "mean_delta_pgt_drop_pct_points": _mean("delta_pgt_drop_pct_points"),
                    "mean_delta_pcstar_drop_pct_points": _mean("delta_pcstar_drop_pct_points"),
                    "mean_cnsg_pct_points": _mean("cnsg_sample_pct_points"),
                })

            if len(summary_rows) > 0:
                summary_csv_path = os.path.join(
                    save_dir,
                    "threshold_patch_distribution_summary.csv",
                )
                with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=list(summary_rows[0].keys()),
                    )
                    writer.writeheader()
                    writer.writerows(summary_rows)

        print(
            "[Threshold patch distribution + Remove-N probabilities] "
            "saved {} image(s) to: {}".format(
                len(paths), os.path.abspath(save_dir)
            )
        )
        print("  thresholds={}".format([round(t, 3) for t in thresholds_used]))
        print("  colors: Relevant=green, Nuisance=red, Irrelevant=blue")
        if csv_path is not None:
            print("  per-image CSV={}".format(os.path.abspath(csv_path)))
        if summary_csv_path is not None:
            print("  summary CSV={}".format(os.path.abspath(summary_csv_path)))

        return {
            "paths": paths,
            "csv_path": csv_path,
            "summary_csv_path": summary_csv_path,
            "thresholds": thresholds_used,
            "num_images": len(paths),
            "partition_source": (
                "Stage-I normalized GT-vs-strongest-non-GT decision margin"
            ),
            "intervention": "remove 100% nuisance patches at layer4 for every threshold",
        }

    # Paper-friendly alias.
    visualize_ncdm_threshold_sweep = visualize_threshold_patch_distribution

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save_discovery(self, path: str) -> None:
        self._check_discovered()
        payload = {
            "version": "triple_bank_rni_relative_contribution_wy_metric_select_v5",
            "clustering_metric": self.clustering_metric,
            "num_classes": self.num_classes,
            "prototype_k_factors": self.prototype_k_factors,
            "decision_threshold": self.decision_threshold,
            "temperature": self.temperature,
            "class_counts": self.class_counts,
            "class_prior": self.class_prior,
            "relevant_medoids_raw": self.relevant_medoids_raw,
            "relevant_medoids_norm": self.relevant_medoids_norm,
            "nuisance_medoids_raw": self.nuisance_medoids_raw,
            "nuisance_medoids_norm": self.nuisance_medoids_norm,
            "irrelevant_medoids_raw": self.irrelevant_medoids_raw,
            "irrelevant_medoids_norm": self.irrelevant_medoids_norm,
            "relevant_best_k": self.relevant_best_k,
            "nuisance_best_k": self.nuisance_best_k,
            "irrelevant_best_k": self.irrelevant_best_k,
            "discovery_result": self.discovery_result,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(payload, path)

    def load_discovery(
        self,
        path: str,
        model_for_assignment: nn.Module,
        device: Union[str, torch.device],
    ) -> None:
        payload = torch.load(path, map_location="cpu")
        if int(payload["num_classes"]) != self.num_classes:
            raise ValueError(
                "Saved num_classes={} but current num_classes={}.".format(
                    payload["num_classes"], self.num_classes
                )
            )
        required = [
            "relevant_medoids_raw", "relevant_medoids_norm",
            "nuisance_medoids_raw", "nuisance_medoids_norm",
            "irrelevant_medoids_raw", "irrelevant_medoids_norm",
            "relevant_best_k", "nuisance_best_k", "irrelevant_best_k",
        ]
        missing = [k for k in required if k not in payload]
        if missing:
            raise ValueError(
                "Saved discovery does not contain the required THREE banks (missing {}). "
                "Re-run discover() on dataloaders['val'].".format(missing)
            )

        self.relevant_medoids_raw = payload["relevant_medoids_raw"].float().cpu()
        self.relevant_medoids_norm = payload["relevant_medoids_norm"].float().cpu()
        self.nuisance_medoids_raw = payload["nuisance_medoids_raw"].float().cpu()
        self.nuisance_medoids_norm = payload["nuisance_medoids_norm"].float().cpu()
        self.irrelevant_medoids_raw = payload["irrelevant_medoids_raw"].float().cpu()
        self.irrelevant_medoids_norm = payload["irrelevant_medoids_norm"].float().cpu()
        self.relevant_best_k = int(payload["relevant_best_k"])
        self.nuisance_best_k = int(payload["nuisance_best_k"])
        self.irrelevant_best_k = int(payload["irrelevant_best_k"])
        self.discovery_result = payload.get("discovery_result", None)

        # Banks must be matched with the same metric used to discover them.
        saved_metric = str(payload.get("clustering_metric", "cosine")).strip().lower()
        if saved_metric in ("distance", "l2"):
            saved_metric = "euclidean"
        if saved_metric not in ("cosine", "euclidean"):
            raise ValueError(
                "Unsupported clustering_metric in saved discovery: {!r}.".format(
                    saved_metric
                )
            )
        self.clustering_metric = saved_metric

        if "class_counts" in payload:
            self.set_class_counts(payload["class_counts"])
        elif "class_prior" in payload:
            prior = payload["class_prior"].float().flatten()
            if prior.numel() == self.num_classes:
                self.class_prior = prior / prior.sum().clamp_min(self.eps)
        if "decision_threshold" in payload:
            self.decision_threshold = float(payload["decision_threshold"])

        device = torch.device(device)
        self._assignment_model = copy.deepcopy(model_for_assignment).to(device)
        self._assignment_model.eval()
        for parameter in self._assignment_model.parameters():
            parameter.requires_grad_(False)
        self._assignment_device = device


# Compatibility aliases
RelevantNuisancePatternResNet = DualPatternResNet
TriplePatternResNet = DualPatternResNet


# =============================================================================
# Minimal runnable smoke test
# =============================================================================


def _smoke_test() -> None:
    from torch.utils.data import DataLoader, TensorDataset

    class TinyResNet(nn.Module):
        def __init__(self, num_classes: int = 3):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=False),
                nn.AvgPool2d(2),
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(16, 4, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=False),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.fc = nn.Linear(4, num_classes, bias=False)

        def forward(self, x):
            x = self.stem(x)
            f = self.layer4(x)
            z = f.mean(dim=(2, 3))
            return self.fc(z)

    torch.manual_seed(7)
    random.seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    model = TinyResNet(num_classes=num_classes).to(device)
    # Make the tiny synthetic classifier weights positive so C_GT and C_MIS are
    # usually on comparable scales; this creates R/N/I examples for the smoke test.
    # This is only test scaffolding and does not affect user models.
    with torch.no_grad():
        model.fc.weight.copy_(model.fc.weight.abs() + 0.1)

    # More samples make R/N/I pools sufficiently large for candidate K values.
    images = torch.rand(180, 3, 32, 32)
    with torch.no_grad():
        pred = model(images.to(device)).argmax(dim=1).cpu()
    labels = pred.clone()
    # Half wrong, distributed across classes.
    wrong_ids = torch.arange(labels.numel()) % 2 == 1
    labels[wrong_ids] = (labels[wrong_ids] + 1) % num_classes

    loader = DataLoader(TensorDataset(images, labels), batch_size=18, shuffle=False)
    method = DualPatternResNet(
        num_classes=num_classes,
        prototype_k_factors=[1, 2],
        decision_threshold=0.10,
        temperature=0.1,
        class_counts=[90, 60, 30],
        lambda_global=1.0,
        lambda_relevant=0.1,
        lambda_nuisance=0.1,
        kmedoids_iterations=4,
        max_candidates_per_class=500,
        silhouette_sample_size=180,
        assignment_chunk_size=1024,
        random_seed=11,
    )
    result = method.discover(model, loader, device=device, verbose=False)

    model.train()
    x = images[:12].to(device)
    y = labels[:12].to(device)
    out = method(model, x, y)
    loss = method.total_loss(out)
    model.zero_grad(set_to_none=True)
    loss.backward()

    assert torch.isfinite(loss).item()
    assert out.irrelevant_mask is not None
    full = out.relevant_mask | out.nuisance_mask | out.irrelevant_mask
    assert bool(full.all())
    assert not bool((out.relevant_mask & out.nuisance_mask).any())
    assert not bool((out.relevant_mask & out.irrelevant_mask).any())
    assert not bool((out.nuisance_mask & out.irrelevant_mask).any())
    assert result.irrelevant_best_k >= 2

    print("Three-bank R/N/I smoke test passed.")
    print({
        "best_K_R": result.relevant_best_k,
        "best_K_N": result.nuisance_best_k,
        "best_K_I": result.irrelevant_best_k,
        "loss_global": float(out.loss_global.detach().cpu()),
        "loss_relevant": float(out.loss_relevant.detach().cpu()),
        "loss_nuisance": float(out.loss_nuisance.detach().cpu()),
        "num_R": out.num_relevant_regions,
        "num_N": out.num_nuisance_regions,
        "num_I": out.num_irrelevant_regions,
    })


if __name__ == "__main__":
    _smoke_test()


