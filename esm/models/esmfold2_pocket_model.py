"""Repo-owned experimental ESMFold2 pocket-conditioning forward hook."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _match_cpu_lm_projection_dtype(model: Any) -> None:
    try:
        device = model.device
    except AttributeError:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            return
    if device.type != "cpu":
        return

    esmc = getattr(model, "_esmc", None)
    language_model = getattr(model, "language_model", None)
    if esmc is None or language_model is None:
        return

    try:
        esmc_dtype = next(esmc.parameters()).dtype
        language_model_dtype = next(language_model.parameters()).dtype
    except StopIteration:
        return

    if esmc_dtype == torch.bfloat16 and language_model_dtype == torch.float32:
        language_model.to(dtype=torch.bfloat16)


def _as_batched_pair_mask(mask: Tensor, *, device: torch.device) -> Tensor:
    mask = mask.to(device=device).bool()
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    return mask


def _as_batched_token_mask(tokens: Tensor, *, device: torch.device) -> Tensor:
    tokens = tokens.to(device=device).bool()
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    return tokens


def _soft_pocket_pair_update(
    model: Any,
    pair_repr: Tensor,
    token_attention_mask: Tensor,
    pocket_feature: Tensor | None,
    disto_cond: Tensor | None,
    disto_cond_mask: Tensor | None,
) -> Tensor:
    """Return a learned soft pair update for explicit residue-ligand hints."""
    if disto_cond_mask is None:
        return pair_repr

    hint_mask = _as_batched_pair_mask(disto_cond_mask, device=pair_repr.device)
    valid_pair_mask = token_attention_mask.to(device=pair_repr.device).bool()
    valid_pair_mask = valid_pair_mask[:, :, None] & valid_pair_mask[:, None, :]
    hint_mask = hint_mask & valid_pair_mask
    if not hint_mask.any():
        return pair_repr

    max_bin = model.distogram_head.out_features - 1
    if disto_cond is None:
        target_bins = hint_mask.new_zeros(hint_mask.shape, dtype=torch.long)
    else:
        target_bins = disto_cond.to(device=pair_repr.device).long()
        if target_bins.dim() == 2:
            target_bins = target_bins.unsqueeze(0)
        target_bins = target_bins.clamp(min=0, max=max_bin)

    distogram_weight = model.distogram_head.weight.to(
        device=pair_repr.device, dtype=pair_repr.dtype
    )
    target_pair_update = F.embedding(target_bins, distogram_weight)

    hint_strength = hint_mask.to(dtype=pair_repr.dtype)
    if pocket_feature is not None:
        pocket_tokens = _as_batched_token_mask(pocket_feature, device=pair_repr.device)
        pocket_token_strength = pocket_tokens.to(dtype=pair_repr.dtype)
        hint_strength = hint_strength * (
            1.0
            + 0.25
            * (
                pocket_token_strength[:, :, None]
                + pocket_token_strength[:, None, :]
            )
        )

    return pair_repr + 0.05 * hint_strength.unsqueeze(-1) * target_pair_update


class ESMFold2PocketConditionedModel(nn.Module):
    """Repo-owned wrapper that adds experimental soft pocket hints to ESMFold2."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

        return cls(ESMFold2Model.from_pretrained(*args, **kwargs))

    @property
    def device(self) -> torch.device:
        return self.model.device  # type: ignore[no-any-return]

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            model = super().__getattr__("model")
            return getattr(model, name)

    @torch.inference_mode()
    def forward(
        self,
        token_index: Tensor,
        residue_index: Tensor,
        asym_id: Tensor,
        sym_id: Tensor,
        entity_id: Tensor,
        mol_type: Tensor,
        res_type: Tensor,
        token_bonds: Tensor,
        token_attention_mask: Tensor,
        ref_pos: Tensor,
        ref_element: Tensor,
        ref_charge: Tensor,
        ref_atom_name_chars: Tensor,
        ref_space_uid: Tensor,
        atom_attention_mask: Tensor,
        atom_to_token: Tensor,
        distogram_atom_idx: Tensor,
        deletion_mean: Tensor | None = None,
        msa: Tensor | None = None,
        has_deletion: Tensor | None = None,
        deletion_value: Tensor | None = None,
        msa_attention_mask: Tensor | None = None,
        input_ids: Tensor | None = None,
        lm_hidden_states: Tensor | None = None,
        pocket_feature: Tensor | None = None,
        disto_cond: Tensor | None = None,
        disto_cond_mask: Tensor | None = None,
        num_loops: int | None = None,
        num_diffusion_samples: int | None = None,
        num_sampling_steps: int | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        _match_cpu_lm_projection_dtype(self.model)

        hook_handle = None
        if disto_cond_mask is not None:

            def add_pocket_pair_update(_module, inputs):
                (pair_repr,) = inputs
                return (
                    _soft_pocket_pair_update(
                        self.model,
                        pair_repr,
                        token_attention_mask,
                        pocket_feature,
                        disto_cond,
                        disto_cond_mask,
                    ),
                )

            hook_handle = self.model.parcae_input_norm.register_forward_pre_hook(
                add_pocket_pair_update
            )

        try:
            return self.model(
                token_index=token_index,
                residue_index=residue_index,
                asym_id=asym_id,
                sym_id=sym_id,
                entity_id=entity_id,
                mol_type=mol_type,
                res_type=res_type,
                token_bonds=token_bonds,
                token_attention_mask=token_attention_mask,
                ref_pos=ref_pos,
                ref_element=ref_element,
                ref_charge=ref_charge,
                ref_atom_name_chars=ref_atom_name_chars,
                ref_space_uid=ref_space_uid,
                atom_attention_mask=atom_attention_mask,
                atom_to_token=atom_to_token,
                distogram_atom_idx=distogram_atom_idx,
                deletion_mean=deletion_mean,
                msa=msa,
                has_deletion=has_deletion,
                deletion_value=deletion_value,
                msa_attention_mask=msa_attention_mask,
                input_ids=input_ids,
                lm_hidden_states=lm_hidden_states,
                num_loops=num_loops,
                num_diffusion_samples=num_diffusion_samples,
                num_sampling_steps=num_sampling_steps,
                **kwargs,
            )
        finally:
            if hook_handle is not None:
                hook_handle.remove()
