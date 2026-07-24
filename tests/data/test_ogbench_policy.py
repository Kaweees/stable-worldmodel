"""Tests for OGBench Isaac Lab policy HDF5 streaming reader + SWM conversion."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from stable_worldmodel.data.format import detect_format, get_format
from stable_worldmodel.data.formats.ogbench_policy import (
    OgbenchPolicyHDF5Dataset,
    _episode_lengths_and_offsets,
    convert_policy_hdf5_to_swm,
)


def _write_policy_hdf5(path: Path, *, n_eps: int = 3, ep_len: int = 8) -> Path:
    """Minimal policy-schema HDF5 matching ogbench_isaaclab layout."""
    total = n_eps * ep_len
    pixels = np.random.randint(0, 256, size=(total, 3, 224, 224), dtype=np.uint8)
    # Distinct patterns per episode for spot checks
    for ep in range(n_eps):
        pixels[ep * ep_len : (ep + 1) * ep_len, 0, 0, 0] = ep + 1
    actions = np.linspace(-1, 1, total * 5 * 5 * 5, dtype=np.float32).reshape(
        total, 5, 5, 5
    )
    sub_ids = np.repeat(np.arange(n_eps, dtype=np.int64), ep_len)

    with h5py.File(path, 'w') as f:
        f.create_dataset('pixels', data=pixels)
        f.create_dataset('actions', data=actions)
        f.create_dataset('subtrajectory_ids', data=sub_ids)
        f.create_dataset(
            'goals',
            data=np.zeros((n_eps, 3, 224, 224), dtype=np.uint8),
        )
        f.create_dataset(
            'goal_subtrajectory_ids', data=np.arange(n_eps, dtype=np.int64)
        )
        f.attrs['image_normalization'] = 'none'
        f.attrs['recommended_image_normalization'] = 'imagenet'
        f.attrs['image_layout'] = 'CHW'
        f.attrs['image_dtype'] = 'uint8'
    return path


def test_episode_lengths_and_offsets_contiguous():
    ids = np.array([10, 10, 10, 12, 12, 7, 7, 7, 7], dtype=np.int64)
    lengths, offsets = _episode_lengths_and_offsets(ids)
    np.testing.assert_array_equal(lengths, [3, 2, 4])
    np.testing.assert_array_equal(offsets, [0, 3, 5])


def test_ogbench_policy_detect_and_load(tmp_path):
    path = _write_policy_hdf5(tmp_path / 'train.hdf5')
    fmt = detect_format(path)
    assert fmt is not None
    assert fmt.name == 'ogbench_policy'

    ds = get_format('ogbench_policy').open_reader(
        path,
        num_steps=4,
        frameskip=1,
        keys_to_load=['pixels', 'action'],
    )
    assert isinstance(ds, OgbenchPolicyHDF5Dataset)
    assert len(ds.lengths) == 3
    assert ds.get_dim('action') == 25  # 5 block steps × 5 dims
    assert len(ds) > 0

    sample = ds[0]
    assert sample['pixels'].shape == (4, 3, 224, 224)
    assert sample['pixels'].dtype == torch.uint8
    assert sample['action'].shape == (4, 25)
    assert torch.isfinite(sample['action']).all()
    # first sample of ep 0 has red-channel marker 1
    assert int(sample['pixels'][0, 0, 0, 0]) == 1


def test_action_is_first_horizon_block_flattened(tmp_path):
    path = _write_policy_hdf5(tmp_path / 'train.hdf5', n_eps=1, ep_len=4)
    ds = OgbenchPolicyHDF5Dataset(
        path=path, num_steps=2, frameskip=1, keys_to_load=['pixels', 'action']
    )
    with h5py.File(path, 'r') as f:
        expected = f['actions'][0, 0, :, :].reshape(-1)
    sample = ds[0]
    np.testing.assert_allclose(
        sample['action'][0].numpy(), expected, rtol=1e-5, atol=1e-5
    )


def test_convert_policy_hdf5_to_swm(tmp_path):
    src = _write_policy_hdf5(tmp_path / 'policy.hdf5', n_eps=2, ep_len=6)
    dest = tmp_path / 'swm.h5'
    convert_policy_hdf5_to_swm(src, dest, max_episodes=2)

    from stable_worldmodel.data.formats.hdf5 import HDF5Dataset

    ds = HDF5Dataset(
        path=dest,
        num_steps=3,
        frameskip=1,
        keys_to_load=['pixels', 'action'],
    )
    assert len(ds.lengths) == 2
    assert ds.get_dim('action') == 25
    batch = ds[0]
    assert batch['pixels'].dtype == torch.uint8
    assert batch['pixels'].shape[1:] == (3, 224, 224)


def test_load_dataset_path_with_format(tmp_path):
    path = _write_policy_hdf5(tmp_path / 'cube.hdf5')
    import stable_worldmodel as swm

    ds = swm.data.load_dataset(
        str(path),
        format='ogbench_policy',
        num_steps=4,
        frameskip=1,
        keys_to_load=['pixels', 'action'],
    )
    assert len(ds) > 0
    assert ds.get_dim('action') == 25


def test_lewm_style_imagenet_batch(tmp_path):
    """Drive the same ImageNet pixel path used by scripts/train/lewm.py."""
    pytest.importorskip('stable_pretraining')
    from stable_pretraining import data as dt
    from stable_worldmodel.data import column_normalizer as get_column_normalizer

    path = _write_policy_hdf5(tmp_path / 'cube.hdf5', n_eps=4, ep_len=10)
    ds = OgbenchPolicyHDF5Dataset(
        path=path,
        num_steps=4,
        frameskip=1,
        keys_to_load=['pixels', 'action'],
        keys_to_cache=['action'],
    )
    # Mirror lewm.py get_img_preprocessor + per-column action normalizer
    imagenet_stats = dt.dataset_stats.ImageNet
    transforms = [
        dt.transforms.Compose(
            dt.transforms.ToImage(
                **imagenet_stats, source='pixels', target='pixels'
            ),
            dt.transforms.Resize(224, source='pixels', target='pixels'),
        ),
        get_column_normalizer(ds, 'action', 'action'),
    ]
    ds.transform = dt.transforms.Compose(*transforms)

    # action_encoder.input_dim = frameskip * get_dim('action') as in lewm.py
    action_input_dim = 1 * ds.get_dim('action')
    assert action_input_dim == 25

    batch = torch.utils.data.default_collate([ds[i] for i in range(2)])
    pix, act = batch['pixels'], batch['action']
    assert pix.dtype.is_floating_point
    assert pix.shape == (2, 4, 3, 224, 224)
    # ImageNet-normalized: not raw 0–255
    assert float(pix.max()) < 50.0
    assert all(abs(float(pix[:, :, c].mean())) < 5.0 for c in range(3))
    assert act.shape == (2, 4, 25)
    assert torch.isfinite(act).all()


@pytest.mark.skipif(
    not Path(
        '/root/ogbench_isaaclab/data/trajectory-generations/'
        'train-episodes-40000/train.hdf5'
    ).is_file(),
    reason='40k policy corpus not present',
)
def test_real_corpus_image_contract_smoke():
    path = Path(
        '/root/ogbench_isaaclab/data/trajectory-generations/'
        'train-episodes-40000/train.hdf5'
    )
    with h5py.File(path, 'r') as f:
        assert f['pixels'].dtype == np.uint8
        assert f['pixels'].shape[-3:] == (3, 224, 224)
        assert f.attrs['image_normalization'] in ('none', b'none')
        rec = f.attrs['recommended_image_normalization']
        if isinstance(rec, bytes):
            rec = rec.decode()
        assert rec == 'imagenet'
        n_goals = f['goals'].shape[0]
        n_ids = len(np.unique(f['subtrajectory_ids'][:]))
        assert n_goals == 40_000
        assert n_ids == 40_000

    ds = OgbenchPolicyHDF5Dataset(
        path=path,
        num_steps=4,
        frameskip=1,
        keys_to_load=['pixels', 'action'],
    )
    assert len(ds.lengths) == 40_000
    assert ds.get_dim('action') == 25
    sample = ds[0]
    assert sample['pixels'].dtype == torch.uint8
    assert 0 <= int(sample['pixels'].min()) <= int(sample['pixels'].max()) <= 255
