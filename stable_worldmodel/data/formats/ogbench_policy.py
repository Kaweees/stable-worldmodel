"""Streaming reader for OGBench Isaac Lab *policy* HDF5 trajectories.

Policy corpora (e.g. ``train-episodes-40000/train.hdf5``) store decision-rate
samples keyed by ``subtrajectory_ids`` rather than SWM ``ep_len``/``ep_offset``.
Bird-view ``pixels`` are raw ``uint8`` CHW; ImageNet normalization is applied
at train time (see file attrs ``image_normalization=none``,
``recommended_image_normalization=imagenet``).

For world-model training we expose sequential clips of:

- ``pixels``: bird-view only (wrist cameras dropped)
- ``action``: first executed receding-horizon block ``actions[:, 0, :, :]``
  flattened to ``(action_block * action_dim,)`` per decision step

This keeps the full multi-camera / action-chunk schema on disk while letting
``scripts/train/lewm.py`` consume the data via ``load_dataset``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import torch

from stable_worldmodel.data.dataset import Dataset
from stable_worldmodel.data.format import Format, register_format

logger = logging.getLogger(__name__)

# Policy schema layout: actions[N, receding_horizon, action_block, action_dim]
_DEFAULT_ACTION_HORIZON_IDX = 0  # first executed action block


def _episode_lengths_and_offsets(
    subtrajectory_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build contiguous episode lengths/offsets from sample-level ids.

    Samples for one subtrajectory must be stored contiguously (the writer
    appends whole subtrajectories). Gaps between id values are allowed.
    """
    if subtrajectory_ids.ndim != 1:
        raise ValueError(
            f'subtrajectory_ids must be 1-D, got shape {subtrajectory_ids.shape}'
        )
    if len(subtrajectory_ids) == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64)

    changes = np.flatnonzero(np.diff(subtrajectory_ids) != 0) + 1
    starts = np.concatenate([[0], changes]).astype(np.int64, copy=False)
    ends = np.concatenate([changes, [len(subtrajectory_ids)]]).astype(
        np.int64, copy=False
    )
    lengths = (ends - starts).astype(np.int32, copy=False)
    return lengths, starts


def _is_policy_hdf5(path: Path) -> bool:
    if not path.is_file() or path.suffix not in ('.h5', '.hdf5'):
        return False
    try:
        with h5py.File(path, 'r') as f:
            return (
                'subtrajectory_ids' in f
                and 'pixels' in f
                and 'actions' in f
                and 'ep_len' not in f
            )
    except OSError:
        return False


class OgbenchPolicyHDF5Dataset(Dataset):
    """Episode dataset over an OGBench Isaac Lab policy HDF5 file.

    Args:
        path: Path to the policy ``.hdf5`` / ``.h5`` file.
        frameskip: Stride between observation samples (default 1 — samples
            are already at decision rate).
        num_steps: Observation steps per training clip.
        transform: Optional sample transform.
        keys_to_load: Columns to expose. ``action`` is derived from the
            policy ``actions`` chunk; ``pixels`` is bird-view only.
        keys_to_cache: Columns to load fully into memory (e.g. ``action``).
        action_horizon_idx: Which receding-horizon slot to use as the
            transition action (default 0 = first executed block).
        name: Ignored; accepted for API symmetry with other readers.
        cache_dir: Ignored; path is absolute/explicit.
        keys_to_merge: Not supported (policy schema has no proprio merge).
    """

    def __init__(
        self,
        path: str | Path | None = None,
        name: str | None = None,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
        keys_to_load: list[str] | None = None,
        keys_to_cache: list[str] | None = None,
        keys_to_merge: dict | None = None,
        cache_dir: str | Path | None = None,
        action_horizon_idx: int = _DEFAULT_ACTION_HORIZON_IDX,
        **_ignored,
    ) -> None:
        del name, cache_dir  # path is authoritative for this format
        if path is None:
            raise TypeError('OgbenchPolicyHDF5Dataset requires `path`')
        if keys_to_merge:
            raise ValueError(
                'OgbenchPolicyHDF5Dataset does not support keys_to_merge; '
                'load pixels + action only for the world-model path.'
            )

        self.h5_path = Path(path)
        if not self.h5_path.is_file():
            raise FileNotFoundError(f'policy HDF5 not found: {self.h5_path}')

        self.action_horizon_idx = int(action_horizon_idx)
        self.h5_file: h5py.File | None = None
        self._cache: dict[str, np.ndarray] = {}
        self._action_dim: int | None = None

        with self._open_h5() as f:
            if 'subtrajectory_ids' not in f or 'pixels' not in f:
                raise ValueError(
                    f'{self.h5_path} is not an OGBench policy HDF5 '
                    '(missing subtrajectory_ids / pixels)'
                )
            lengths, offsets = _episode_lengths_and_offsets(
                f['subtrajectory_ids'][:]
            )
            actions = f['actions']
            if actions.ndim != 4:
                raise ValueError(
                    f'expected actions with shape (N, H, B, D), got {actions.shape}'
                )
            h, block, dim = actions.shape[1:]
            if not 0 <= self.action_horizon_idx < h:
                raise ValueError(
                    f'action_horizon_idx={self.action_horizon_idx} out of range '
                    f'for horizon {h}'
                )
            self._action_dim = int(block * dim)
            self._pixels_shape = tuple(f['pixels'].shape[1:])

            requested = keys_to_load or ['pixels', 'action']
            self._keys: list[str] = list(requested)
            for key in self._keys:
                if key == 'action':
                    continue
                if key not in f:
                    raise KeyError(
                        f"key {key!r} not in policy HDF5 {self.h5_path}; "
                        f'available={sorted(f.keys())}'
                    )

            for key in keys_to_cache or []:
                self._cache[key] = self._load_column(f, key)
                logger.info("Cached '%s' from '%s'", key, self.h5_path)

        super().__init__(lengths, offsets, frameskip, num_steps, transform)
        logger.info(
            'OgbenchPolicyHDF5Dataset: %s episodes, %s samples, action_dim=%s '
            'from %s',
            len(lengths),
            int(lengths.sum()) if len(lengths) else 0,
            self._action_dim,
            self.h5_path,
        )

    @property
    def column_names(self) -> list[str]:
        return list(self._keys)

    def _open_h5(self) -> h5py.File:
        return h5py.File(
            self.h5_path, 'r', swmr=True, rdcc_nbytes=256 * 1024 * 1024
        )

    def _open(self) -> None:
        if self.h5_file is None:
            self.h5_file = self._open_h5()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state['h5_file'] = None
        return state

    def _load_column(self, f: h5py.File, col: str) -> np.ndarray:
        if col == 'action':
            # First executed block: (N, action_block, action_dim) -> (N, B*D)
            block = f['actions'][:, self.action_horizon_idx, :, :]
            return np.asarray(block, dtype=np.float32).reshape(
                block.shape[0], -1
            )
        return f[col][:]

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        self._open()
        g_start = int(self.offsets[ep_idx] + start)
        g_end = int(self.offsets[ep_idx] + end)
        steps: dict = {}

        for col in self._keys:
            if col in self._cache:
                data = self._cache[col][g_start:g_end]
            elif col == 'action':
                block = self.h5_file['actions'][
                    g_start:g_end, self.action_horizon_idx, :, :
                ]
                data = np.asarray(block, dtype=np.float32).reshape(
                    block.shape[0], -1
                )
            else:
                data = self.h5_file[col][g_start:g_end]

            if col != 'action':
                data = data[:: self.frameskip]

            if data.dtype == np.object_ or data.dtype.kind in ('S', 'U'):
                val = data[0] if len(data) > 0 else b''
                steps[col] = val.decode() if isinstance(val, bytes) else val
            else:
                # Policy pixels are already CHW uint8; do not NHWC-permute.
                steps[col] = torch.from_numpy(np.ascontiguousarray(data))

        return self.transform(steps) if self.transform else steps

    def get_col_data(self, col: str) -> np.ndarray:
        if col in self._cache:
            return self._cache[col]
        self._open()
        return self._load_column(self.h5_file, col)

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        self._open()
        out = {}
        for col in self._keys:
            if col == 'action':
                arr = np.asarray(
                    self.h5_file['actions'][
                        row_idx, self.action_horizon_idx, :, :
                    ],
                    dtype=np.float32,
                )
                # Scalar index → (B, D); fancy index → (N, B, D)
                out[col] = arr.reshape(*arr.shape[:-2], -1)
            else:
                out[col] = self.h5_file[col][row_idx]
        return out

    def get_dim(self, col: str) -> int:
        if col == 'action' and self._action_dim is not None:
            return self._action_dim
        data = self.get_col_data(col)
        return int(np.prod(data.shape[1:])) if data.ndim > 1 else 1

    def merge_col(
        self,
        source: list[str] | str,
        target: str,
        dim: int = -1,
    ) -> None:
        raise NotImplementedError(
            'OgbenchPolicyHDF5Dataset does not support merge_col'
        )


def convert_policy_hdf5_to_swm(
    source: str | Path,
    dest: str | Path,
    *,
    action_horizon_idx: int = _DEFAULT_ACTION_HORIZON_IDX,
    max_episodes: int | None = None,
    mode: str = 'overwrite',
) -> Path:
    """Materialize a compact SWM HDF5 with only ``pixels`` + transition ``action``.

    Useful for small smoke subsets; full 40k conversion rewrites ~340 GB of
    pixels — prefer the streaming :class:`OgbenchPolicyHDF5Dataset` for the
    full corpus.
    """
    from stable_worldmodel.data.formats.hdf5 import HDF5Writer

    source = Path(source)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source, 'r') as f:
        lengths, offsets = _episode_lengths_and_offsets(
            f['subtrajectory_ids'][:]
        )
        n_eps = len(lengths)
        if max_episodes is not None:
            n_eps = min(n_eps, int(max_episodes))

        with HDF5Writer(dest, mode=mode) as writer:
            for ep in range(n_eps):
                start = int(offsets[ep])
                end = start + int(lengths[ep])
                pixels = f['pixels'][start:end]
                block = f['actions'][start:end, action_horizon_idx, :, :]
                action = np.asarray(block, dtype=np.float32).reshape(
                    block.shape[0], -1
                )
                writer.write_episode({'pixels': pixels, 'action': action})

    return dest


@register_format
class OgbenchPolicy(Format):
    name = 'ogbench_policy'

    @classmethod
    def detect(cls, path) -> bool:
        p = Path(path)
        if p.is_file():
            return _is_policy_hdf5(p)
        if p.is_dir():
            for cand in list(p.glob('*.hdf5')) + list(p.glob('*.h5')):
                if _is_policy_hdf5(cand):
                    return True
        return False

    @classmethod
    def open_reader(cls, path, **kwargs) -> OgbenchPolicyHDF5Dataset:
        p = Path(path)
        if p.is_dir():
            cands = [
                c
                for c in list(p.glob('*.hdf5')) + list(p.glob('*.h5'))
                if _is_policy_hdf5(c)
            ]
            if not cands:
                raise FileNotFoundError(
                    f'no OGBench policy HDF5 under {p}'
                )
            # Prefer train.hdf5 when present
            cands.sort(key=lambda c: (0 if c.name == 'train.hdf5' else 1, c.name))
            p = cands[0]
        return OgbenchPolicyHDF5Dataset(path=p, **kwargs)
