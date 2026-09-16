from pathlib import Path
import sys
import types

import torch

from crecerelle.cell import TABULA_MURIS_CELL_TYPES
from crecerelle.models import FCLayers, create_set_zero_subsets
from crecerelle.training_utils import auxiliary_noise

umap_stub = types.ModuleType("umap")
umap_stub.UMAP = object
sys.modules.setdefault("umap", umap_stub)
sys.modules.setdefault("scanpy", types.ModuleType("scanpy"))

from crecerelle.utils import setup_crecerelle


def test_create_set_zero_subsets_public_api():
    assert create_set_zero_subsets(2) == []
    assert create_set_zero_subsets(4) == [
        (0,),
        (1,),
        (2,),
        (3,),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]


def test_fc_layers_forward_public_api():
    layer = FCLayers(
        input_dim=3,
        output_dim=2,
        num_hidden_layers=1,
        dropout_rate=0.0,
        use_batch_norm=False,
    )
    output = layer(torch.ones(5, 3))
    assert tuple(output.shape) == (5, 2)


def test_auxiliary_noise_public_api():
    noise = auxiliary_noise((2, 3))
    assert tuple(noise.shape) == (2, 3)


def test_setup_crecerelle_public_api(tmp_path):
    setup_crecerelle(str(tmp_path), dataset_name="tabulaMuris")
    results = tmp_path / "crecerelle_results"
    expected_dirs = [
        results / "data" / "tabulaMuris",
        results / "figures" / "tabulaMuris",
        results / "models" / "scVI",
        results / "models" / "tuVI",
        results / "models" / "TRVI",
    ]
    assert all(Path(path).is_dir() for path in expected_dirs)


def test_cell_type_constants_public_api():
    assert isinstance(TABULA_MURIS_CELL_TYPES, (list, tuple))
    assert TABULA_MURIS_CELL_TYPES
