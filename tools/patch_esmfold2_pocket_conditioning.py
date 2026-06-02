#!/usr/bin/env python
"""Apply the local experimental ESMFold2 pocket-conditioning model patch.

The cached Biohub ESMFold2 model used by the local tests imports
``transformers.models.esmfold2.modeling_esmfold2`` from the prepared Python
environment. This script makes the site-packages edit reproducible instead of
leaving an invisible manual dependency patch behind.
"""

from __future__ import annotations

import sys
import sysconfig
from pathlib import Path


MODEL_FILE = Path(sysconfig.get_paths()["purelib"]) / (
    "transformers/models/esmfold2/modeling_esmfold2.py"
)

SIGNATURE_NEEDLE = """\
        input_ids: Tensor | None = None,
        lm_hidden_states: Tensor | None = None,
        num_loops: int | None = None,
"""

SIGNATURE_REPLACEMENT = """\
        input_ids: Tensor | None = None,
        lm_hidden_states: Tensor | None = None,
        pocket_feature: Tensor | None = None,
        disto_cond: Tensor | None = None,
        disto_cond_mask: Tensor | None = None,
        num_loops: int | None = None,
"""

PAIR_INJECTION_NEEDLE = """\
            token_bonds_encoding = self.token_bonds(token_bonds.float())
            z_init = z_init + relative_position_encoding + token_bonds_encoding

            if (
"""

PAIR_INJECTION_REPLACEMENT = """\
            token_bonds_encoding = self.token_bonds(token_bonds.float())
            z_init = z_init + relative_position_encoding + token_bonds_encoding

            if pocket_feature is not None or disto_cond_mask is not None:
                pocket_pair_mask: Tensor | None = None
                if disto_cond_mask is not None:
                    pocket_pair_mask = disto_cond_mask.to(device=z_init.device).bool()
                    if pocket_pair_mask.dim() == 2:
                        pocket_pair_mask = pocket_pair_mask.unsqueeze(0)

                pocket_token_mask: Tensor | None = None
                if pocket_feature is not None:
                    pocket_token_mask = pocket_feature.to(device=z_init.device).bool()
                    if pocket_token_mask.dim() == 1:
                        pocket_token_mask = pocket_token_mask.unsqueeze(0)
                    token_pair_mask = (
                        pocket_token_mask[:, :, None] | pocket_token_mask[:, None, :]
                    )
                    pocket_pair_mask = (
                        token_pair_mask
                        if pocket_pair_mask is None
                        else pocket_pair_mask | token_pair_mask
                    )

                if pocket_pair_mask is not None and pocket_pair_mask.any():
                    valid_pair_mask = tok_mask[:, :, None].bool() & tok_mask[
                        :, None, :
                    ].bool()
                    pocket_pair_mask = pocket_pair_mask & valid_pair_mask
                    if pocket_pair_mask.any():
                        disto_values = torch.zeros_like(
                            pocket_pair_mask, dtype=z_init.dtype
                        )
                        if disto_cond is not None:
                            disto_values = disto_cond.to(
                                device=z_init.device, dtype=z_init.dtype
                            )
                            if disto_values.dim() == 2:
                                disto_values = disto_values.unsqueeze(0)
                            disto_values = (
                                disto_values.clamp(min=0, max=63) + 1.0
                            ) / 64.0

                        pocket_token_values = torch.zeros_like(
                            pocket_pair_mask, dtype=z_init.dtype
                        )
                        if pocket_token_mask is not None:
                            pocket_token_f = pocket_token_mask.to(dtype=z_init.dtype)
                            pocket_token_values = (
                                pocket_token_f[:, :, None]
                                + pocket_token_f[:, None, :]
                            )

                        pocket_strength = pocket_pair_mask.to(dtype=z_init.dtype) * (
                            1.0 + 0.25 * disto_values + 0.5 * pocket_token_values
                        )
                        pocket_basis = torch.linspace(
                            -1.0, 1.0, z_init.shape[-1], device=z_init.device
                        ).to(dtype=z_init.dtype)
                        pocket_pair_bias = (
                            0.05
                            * pocket_strength.unsqueeze(-1)
                            * pocket_basis.view(1, 1, 1, -1)
                        )
                        z_init = z_init + pocket_pair_bias

            if (
"""

LM_DTYPE_NEEDLE = """\
            if lm_hidden_states is not None:
                lm_z = self.language_model(lm_hidden_states.detach())
            del lm_hidden_states
"""

LM_DTYPE_REPLACEMENT = """\
            if lm_hidden_states is not None:
                if (
                    lm_hidden_states.device.type == "cpu"
                    and lm_hidden_states.dtype == torch.bfloat16
                ):
                    try:
                        language_model_dtype = next(
                            self.language_model.parameters()
                        ).dtype
                    except StopIteration:
                        language_model_dtype = lm_hidden_states.dtype
                    if language_model_dtype == torch.float32:
                        self.language_model.to(dtype=torch.bfloat16)
                lm_z = self.language_model(lm_hidden_states.detach())
            del lm_hidden_states
"""


def replace_once(source: str, needle: str, replacement: str) -> str:
    if replacement in source:
        return source
    if needle not in source:
        raise RuntimeError(f"Patch anchor not found:\n{needle}")
    return source.replace(needle, replacement, 1)


def main() -> int:
    if not MODEL_FILE.exists():
        print(f"Missing ESMFold2 model file: {MODEL_FILE}", file=sys.stderr)
        return 1

    source = MODEL_FILE.read_text()
    patched = replace_once(source, SIGNATURE_NEEDLE, SIGNATURE_REPLACEMENT)
    patched = replace_once(patched, PAIR_INJECTION_NEEDLE, PAIR_INJECTION_REPLACEMENT)
    patched = replace_once(patched, LM_DTYPE_NEEDLE, LM_DTYPE_REPLACEMENT)

    if patched == source:
        print(f"ESMFold2 pocket-conditioning model patch already applied: {MODEL_FILE}")
        return 0

    MODEL_FILE.write_text(patched)
    print(f"Applied ESMFold2 pocket-conditioning model patch: {MODEL_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
