import numpy as np
import torch

class Keyframe:
    def __init__(self, timestamp, c2w, depth, color, fx=None, fy=None, cx=None, cy=None, frame_id=None):
        # c2w
        if isinstance(c2w, torch.Tensor):
            self._c2w = c2w.detach().cpu()  # 保留为 tensor
        else:
            self._c2w = torch.tensor(c2w, dtype=torch.float32)

        # depth
        if isinstance(depth, torch.Tensor):
            self._depth = depth.detach().cpu()
        else:
            self._depth = torch.tensor(depth, dtype=torch.float32)

        # color
        if isinstance(color, torch.Tensor):
            self._color = color.detach().cpu()
        else:
            self._color = torch.tensor(color, dtype=torch.float32)

        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._frame_id = frame_id
        self._timestamp = timestamp

    @property
    def c2w(self):
        return self._c2w.clone()

    @property
    def w2c(self):
        return torch.inverse(self._c2w)

    @property
    def depth(self):
        return self._depth.clone()

    @property
    def color(self):
        return self._color.clone()

    @property
    def fx(self):
        return self._fx

    @property
    def fy(self):
        return self._fy

    @property
    def cx(self):
        return self._cx

    @property
    def cy(self):
        return self._cy

    @property
    def frame_id(self):
        return self._frame_id

    @property
    def timestamp(self):
        return self._timestamp
