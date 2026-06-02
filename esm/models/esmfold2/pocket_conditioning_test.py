"""Target tests for experimental ESMFold2 ligand pocket conditioning.

These tests intentionally avoid loading model weights. They verify that
PocketConditioning is a real local input-preparation feature, not just a
serializable field.
"""

from __future__ import annotations

import importlib
import ast
import inspect
import json
import os
import textwrap
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from esm.models.esmfold2.prepare_input import prepare_esmfold2_input
from esm.models.esmfold2.processor import clean_esmfold2_input
from esm.utils.structure.input_builder import (
    LigandInput,
    PocketConditioning,
    ProteinInput,
    StructurePredictionInput,
    deserialize_structure_prediction_input,
    serialize_structure_prediction_input,
)

FIXTURE_DIR = Path(__file__).with_name("testdata")
INPUT_FIXTURE = FIXTURE_DIR / "fkbp12_fk506_pocket_input.json"
EXPECTED_FIXTURE = FIXTURE_DIR / "fkbp12_fk506_expected_output.json"


def _load_fixture() -> tuple[StructurePredictionInput, dict]:
    raw_input = json.loads(INPUT_FIXTURE.read_text())
    expected = json.loads(EXPECTED_FIXTURE.read_text())
    spi = deserialize_structure_prediction_input(raw_input["structure_prediction_input"])
    return spi, expected


@pytest.fixture
def fkbp12_fk506_input() -> tuple[StructurePredictionInput, dict]:
    return _load_fixture()


@pytest.fixture(autouse=True)
def offline_ccd(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Keep tests offline by stubbing CCD lookups used during tokenization."""
    if request.node.get_closest_marker("esmfold2_inference"):
        return

    import esm.models.esmfold2.prepare_input as prep

    ligand_atoms = [
        ("C1", "C", 0),
        ("C2", "C", 0),
        ("O1", "O", 0),
    ]

    def fake_get_ligand_ccd_atoms_with_charges(comp_id: str):
        if comp_id != "FK5":
            return None
        return ligand_atoms

    def fake_get_ligand_idealized_atom_pos(comp_id: str, atom_name: str):
        positions = {
            "C1": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "C2": np.array([1.5, 0.0, 0.0], dtype=np.float32),
            "O1": np.array([0.0, 1.2, 0.0], dtype=np.float32),
        }
        return positions.get(atom_name)

    monkeypatch.setattr(prep, "get_idealized_atom_pos", lambda *_: None)
    monkeypatch.setattr(
        prep,
        "get_ligand_ccd_atoms_with_charges",
        fake_get_ligand_ccd_atoms_with_charges,
    )
    monkeypatch.setattr(
        prep,
        "get_ligand_idealized_atom_pos",
        fake_get_ligand_idealized_atom_pos,
    )
    monkeypatch.setattr(prep, "get_ccd_leaving_atoms", lambda *_: set())
    monkeypatch.setattr(
        prep, "get_ligand_ccd_bonds", lambda comp_id: [("C1", "C2"), ("C1", "O1")]
    )


def test_pocket_conditioning_is_public_esmfold2_api() -> None:
    esmfold2 = importlib.import_module("esm.models.esmfold2")
    assert hasattr(esmfold2, "PocketConditioning")


def test_fkbp12_fk506_pocket_fixture_round_trips(
    fkbp12_fk506_input: tuple[StructurePredictionInput, dict],
) -> None:
    spi, expected = fkbp12_fk506_input
    assert spi.pocket is not None
    assert spi.pocket.binder_chain_id == expected["ligand_chain_id"]
    assert [idx for _, idx in spi.pocket.contacts] == expected[
        "contact_residue_indices_zero_based"
    ]

    restored = deserialize_structure_prediction_input(
        json.loads(json.dumps(serialize_structure_prediction_input(spi)))
    )

    assert restored.pocket is not None
    assert restored.pocket.binder_chain_id == "L"
    assert restored.pocket.contacts == spi.pocket.contacts


def test_clean_esmfold2_input_preserves_pocket_conditioning(
    fkbp12_fk506_input: tuple[StructurePredictionInput, dict],
) -> None:
    spi, _ = fkbp12_fk506_input
    cleaned = clean_esmfold2_input(spi)

    assert cleaned.pocket is not None
    assert cleaned.pocket.binder_chain_id == spi.pocket.binder_chain_id
    assert cleaned.pocket.contacts == spi.pocket.contacts


def test_prepare_input_materializes_pocket_conditioning_features(
    fkbp12_fk506_input: tuple[StructurePredictionInput, dict],
) -> None:
    spi, expected = fkbp12_fk506_input
    features, chains = prepare_esmfold2_input(spi)
    chain_by_id = {chain.chain_id: chain for chain in chains}

    assert expected["protein_chain_id"] in chain_by_id
    assert expected["ligand_chain_id"] in chain_by_id
    for key in expected["expected_feature_behavior"]["required_feature_keys"]:
        assert key in features

    protein_asym = chain_by_id[expected["protein_chain_id"]].asym_id
    ligand_asym = chain_by_id[expected["ligand_chain_id"]].asym_id
    contact_residue_indices = set(expected["contact_residue_indices_zero_based"])

    target_token_indices = [
        i
        for i, (asym_id, residue_index) in enumerate(
            zip(features["asym_id"].tolist(), features["residue_index"].tolist())
        )
        if asym_id == protein_asym and residue_index in contact_residue_indices
    ]
    ligand_token_indices = [
        i for i, asym_id in enumerate(features["asym_id"].tolist()) if asym_id == ligand_asym
    ]

    assert len(target_token_indices) == len(contact_residue_indices)
    assert ligand_token_indices

    pocket_feature = features["pocket_feature"]
    assert pocket_feature[target_token_indices].sum().item() == len(target_token_indices)

    cross_mask = features["disto_cond_mask"][target_token_indices][:, ligand_token_indices]
    assert cross_mask.any().item()


def test_invalid_pocket_binder_chain_id_raises(
    fkbp12_fk506_input: tuple[StructurePredictionInput, dict],
) -> None:
    spi, _ = fkbp12_fk506_input
    bad = replace(
        spi,
        pocket=PocketConditioning(binder_chain_id="Z", contacts=spi.pocket.contacts),
    )

    with pytest.raises(ValueError, match="binder_chain_id.*Z|Z.*binder_chain_id"):
        prepare_esmfold2_input(bad)


def test_invalid_pocket_contact_chain_id_raises(
    fkbp12_fk506_input: tuple[StructurePredictionInput, dict],
) -> None:
    spi, _ = fkbp12_fk506_input
    bad = replace(
        spi,
        pocket=PocketConditioning(binder_chain_id="L", contacts=[("Z", 25)]),
    )

    with pytest.raises(ValueError, match="contact.*Z|Z.*contact|chain.*Z|Z.*chain"):
        prepare_esmfold2_input(bad)


def test_input_without_pocket_keeps_empty_pocket_features(
    fkbp12_fk506_input: tuple[StructurePredictionInput, dict],
) -> None:
    spi, _ = fkbp12_fk506_input
    no_pocket = StructurePredictionInput(sequences=spi.sequences)
    features, _ = prepare_esmfold2_input(no_pocket)

    assert features["pocket_feature"].sum().item() == 0
    assert features["disto_cond_mask"].sum().item() == 0


def test_esmfold2_forward_explicitly_accepts_pocket_conditioning_inputs() -> None:
    """The model must not silently swallow conditioning tensors via **kwargs."""
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    signature = inspect.signature(ESMFold2Model.forward)
    for name in ("pocket_feature", "disto_cond", "disto_cond_mask"):
        assert name in signature.parameters, (
            f"ESMFold2Model.forward must explicitly accept {name!r}; "
            "otherwise local pocket conditioning can be silently ignored."
        )
        assert signature.parameters[name].kind is not inspect.Parameter.VAR_KEYWORD


def test_esmfold2_forward_source_uses_pocket_conditioning_inputs() -> None:
    """Static guardrail: explicit inputs must be read in the forward body."""
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    tree = ast.parse(textwrap.dedent(inspect.getsource(ESMFold2Model.forward)))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    body_names = {
        node.id
        for statement in function.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }

    for name in ("pocket_feature", "disto_cond", "disto_cond_mask"):
        assert name in body_names, (
            f"ESMFold2Model.forward declares {name!r} but does not read it "
            "in the executable body."
        )


@pytest.mark.esmfold2_inference
@pytest.mark.skipif(
    os.environ.get("ESMFOLD2_RUN_INFERENCE") != "1",
    reason="Set ESMFOLD2_RUN_INFERENCE=1 to run the local ESMFold2 inference test.",
)
def test_local_esmfold2_inference_with_fkbp12_fk506_pocket(
    tmp_path: Path,
    fkbp12_fk506_input: tuple[StructurePredictionInput, dict],
) -> None:
    """Run a real local ESMFold2 inference smoke test with pocket conditioning."""
    import torch
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    from esm.models.esmfold2 import ESMFold2InputBuilder

    spi, expected = fkbp12_fk506_input
    model_id = os.environ.get("ESMFOLD2_MODEL_ID", "biohub/ESMFold2")
    device_name = os.environ.get(
        "ESMFOLD2_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_name)
    if device.type == "cuda":
        assert torch.cuda.is_available(), "ESMFOLD2_DEVICE=cuda but CUDA is unavailable"

    model = ESMFold2Model.from_pretrained(model_id).to(device).eval()
    builder = ESMFold2InputBuilder(ccd_cache=os.environ.get("ESMFOLD2_CCD_CACHE"))

    result = builder.fold(
        model,
        spi,
        num_loops=int(os.environ.get("ESMFOLD2_TEST_NUM_LOOPS", "1")),
        num_sampling_steps=int(os.environ.get("ESMFOLD2_TEST_NUM_SAMPLING_STEPS", "4")),
        num_diffusion_samples=1,
        seed=0,
        complex_id="1FKJ_FK5_pocket_test",
    )

    mmcif = result.complex.to_mmcif()
    out_path = tmp_path / "fkbp12_fk506_pocket.cif"
    out_path.write_text(mmcif)

    assert "_atom_site." in mmcif
    assert expected["expected_inference_behavior"]["mmcif_must_contain_ligand_ccd"] in mmcif
    assert len(mmcif) > 1000


@pytest.mark.esmfold2_inference
@pytest.mark.skipif(
    os.environ.get("ESMFOLD2_RUN_INFERENCE") != "1",
    reason="Set ESMFOLD2_RUN_INFERENCE=1 to run the local ESMFold2 inference test.",
)
def test_local_esmfold2_inference_changes_when_pocket_conditioning_changes(
    fkbp12_fk506_input: tuple[StructurePredictionInput, dict],
) -> None:
    """Same seed/input should differ when pocket conditioning is removed."""
    import torch
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    from esm.models.esmfold2 import ESMFold2InputBuilder

    spi, _ = fkbp12_fk506_input
    no_pocket = StructurePredictionInput(sequences=spi.sequences)

    model_id = os.environ.get("ESMFOLD2_MODEL_ID", "biohub/ESMFold2")
    device_name = os.environ.get(
        "ESMFOLD2_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_name)
    if device.type == "cuda":
        assert torch.cuda.is_available(), "ESMFOLD2_DEVICE=cuda but CUDA is unavailable"

    model = ESMFold2Model.from_pretrained(model_id).to(device).eval()
    builder = ESMFold2InputBuilder(ccd_cache=os.environ.get("ESMFOLD2_CCD_CACHE"))
    pocket_features, _ = builder.prepare_input(spi, seed=0, device=device)
    no_pocket_features, _ = builder.prepare_input(no_pocket, seed=0, device=device)

    assert pocket_features["disto_cond_mask"].any()
    assert not no_pocket_features["disto_cond_mask"].any()

    forward_kwargs = dict(
        num_loops=int(os.environ.get("ESMFOLD2_TEST_NUM_LOOPS", "1")),
        num_sampling_steps=int(os.environ.get("ESMFOLD2_TEST_NUM_SAMPLING_STEPS", "4")),
        num_diffusion_samples=1,
        early_exit=False,
    )

    with torch.inference_mode():
        torch.manual_seed(123)
        pocket_output = model(**pocket_features, **forward_kwargs)
        torch.manual_seed(123)
        no_pocket_output = model(**no_pocket_features, **forward_kwargs)

    changed = False
    for key in ("distogram_logits", "sample_atom_coords"):
        assert key in pocket_output
        assert key in no_pocket_output
        changed = changed or not torch.allclose(
            pocket_output[key], no_pocket_output[key], atol=1e-5, rtol=1e-5
        )

    assert changed, (
        "Pocket conditioning had no numerical effect on local ESMFold2 inference. "
        "This usually means the tensors were prepared but not consumed by the model."
    )
