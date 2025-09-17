import numpy as np

class Keyframe:
    def __init__(self, timestamp, c2w, depth, color, fx=None, fy=None, cx=None, cy=None, frame_id=None):
        self._c2w = np.array(c2w, dtype=np.float32)
        self._depth = np.array(depth, dtype=np.float32)
        self._color = np.array(color, dtype=np.float32)
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._frame_id = frame_id
        self._timestamp = timestamp

    @property
    def c2w(self):
        """相机位姿 (camera -> world)"""
        return self._c2w.copy()  # 返回拷贝，避免外部修改内部状态

    @property
    def w2c(self):
        """世界 -> 相机"""
        return np.linalg.inv(self._c2w)

    @property
    def depth(self):
        return self._depth.copy()

    @property
    def color(self):
        return self._color.copy()

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

