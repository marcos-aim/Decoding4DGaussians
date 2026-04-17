"""AriaCachedDataset: consumes precompute.py outputs at training time.

Matches the interface our trainer expects from CachedSceneDataset:
  - get_tokens_mean()
  - get_tokens_frame(frame_idx)
  - get_points_map_frame(frame_idx)
  - load_frame_image(cam_name, frame_idx)   (but indexed by frame_idx only)
  - num_frames, num_patches, token_dim
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from aria_dataset import AriaSequenceDataset


class AriaCachedDataset:
    def __init__(self, cache_dir: str, seq_root: str, target_resolution):
        self.cache_dir = cache_dir
        self.seq_root = seq_root
        self.target_w, self.target_h = target_resolution

        self.tokens = torch.load(os.path.join(cache_dir, "tokens.pt"),
                                  weights_only=True).float()
        self.points_map = torch.load(os.path.join(cache_dir, "points_map.pt"),
                                      weights_only=True).float()
        self.unc_metric = torch.load(os.path.join(cache_dir, "unc_metric.pt"),
                                      weights_only=True).float()

        self.frame_indices = torch.load(os.path.join(cache_dir, "frame_indices.pt"),
                                         weights_only=True).tolist()
        self.aria_c2w = torch.load(os.path.join(cache_dir, "aria_c2w.pt"),
                                    weights_only=True).float()
        self.aria_K = torch.load(os.path.join(cache_dir, "aria_K.pt"),
                                  weights_only=True).float()

        self.num_frames = self.tokens.shape[0]
        self.num_patches = self.tokens.shape[1]
        self.token_dim = self.tokens.shape[2]
        self.tokens_mean = self.tokens.mean(dim=0)

        # Image backend (on-demand RGB loads)
        self._seq = AriaSequenceDataset(seq_root, target_resolution=target_resolution,
                                         frame_indices=self.frame_indices)

        if torch.cuda.is_available():
            self.tokens = self.tokens.cuda()
            self.tokens_mean = self.tokens_mean.cuda()
            self.points_map = self.points_map.cuda()

        print(f"[AriaCachedDataset] {self.num_frames} frames, {self.num_patches} patches")
        print(f"  tokens:      {self.tokens.shape}")
        print(f"  points_map:  {self.points_map.shape}")

    def get_tokens_mean(self) -> torch.Tensor:
        return self.tokens_mean

    def get_tokens_frame(self, frame_idx: int) -> torch.Tensor:
        return self.tokens[frame_idx]

    def get_points_map_frame(self, frame_idx: int) -> torch.Tensor:
        return self.points_map[frame_idx]

    def load_frame_image(self, frame_idx: int) -> torch.Tensor:
        """Return [3, H, W] float in [0,1] on CUDA."""
        img = self._seq.get_frame_image(frame_idx)
        if torch.cuda.is_available():
            img = img.cuda()
        return img
