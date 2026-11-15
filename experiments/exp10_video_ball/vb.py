"""Shared pieces of the video ball detector: video reader, homography, classical blob search, the net.

The detector is trained on the provided match without any human label (RULES.md, "pseudo-labels
produced by an algorithm"): where the automatic tracking has the ball, its position is projected
into the image with the homography and snapped onto the ball actually visible there by a classical
white-blob search. Those pixel positions are the training targets.
"""

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
ROOT = os.path.dirname(EXPERIMENTS)
for _p in (os.path.join(EXPERIMENTS, "lib"), os.path.join(ROOT, "starter")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import common  # noqa: E402
from games import asset_paths  # noqa: E402
from load_homography import load as load_homography  # noqa: E402

CACHE = os.path.join(common.CACHE_DIR, "exp10")
MODELS = os.path.join(HERE, "models")
PITCH = np.array(common.PITCH, dtype=np.float64)
CROP = 128            # training crop, pixels
GAP = 2               # a sample is the frames t-2, t, t+2: the ball moves, the lines do not
CHANNELS = 5          # colour image at t (B, G, R) + grey images at t-GAP and t+GAP
SNAP_RADIUS = 25      # px: the visible ball must be this close to the tracked position
SNAP_MARGIN = 8       # px: and no other blob within this margin of being as close


# -- geometry -------------------------------------------------------------------------------------

def project(matrix, xy_m):
    """Pitch metres -> pixels. matrix (3, 3), xy_m (..., 2)."""
    xy = np.asarray(xy_m, dtype=np.float64) / PITCH
    den = matrix[2, 0] * xy[..., 0] + matrix[2, 1] * xy[..., 1] + matrix[2, 2]
    return np.stack([(matrix[0, 0] * xy[..., 0] + matrix[0, 1] * xy[..., 1] + matrix[0, 2]) / den,
                     (matrix[1, 0] * xy[..., 0] + matrix[1, 1] * xy[..., 1] + matrix[1, 2]) / den], axis=-1)


def unproject(matrix, uv):
    """Pixels -> pitch metres, assuming the point is on the ground. uv (..., 2)."""
    inv = np.linalg.inv(matrix)
    uv = np.asarray(uv, dtype=np.float64)
    den = inv[2, 0] * uv[..., 0] + inv[2, 1] * uv[..., 1] + inv[2, 2]
    return np.stack([(inv[0, 0] * uv[..., 0] + inv[0, 1] * uv[..., 1] + inv[0, 2]) / den,
                     (inv[1, 0] * uv[..., 0] + inv[1, 1] * uv[..., 1] + inv[1, 2]) / den], axis=-1) * PITCH


# -- video ----------------------------------------------------------------------------------------

class FrameReader:
    """Frames of the tactical video by index. Meant for increasing indices (a short look-back is
    kept in memory); anything else falls back to a seek."""

    def __init__(self, path=None, keep=10):
        self.cap = cv2.VideoCapture(path or asset_paths()["video"])
        if not self.cap.isOpened():
            raise IOError("cannot open the tactical video")
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.pos, self.buffer, self.keep = 0, {}, keep

    def get(self, index):
        if index in self.buffer:
            return self.buffer[index]
        if index < 0 or index >= self.n_frames:
            return None
        if index < self.pos or index - self.pos > 75:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.pos = index
        while self.pos <= index:
            if self.pos < index - 6:                        # far behind: skip without decoding to an array
                if not self.cap.grab():
                    return None
            else:
                ok, image = self.cap.read()
                if not ok:
                    return None
                self.buffer[self.pos] = image
                if len(self.buffer) > self.keep:
                    del self.buffer[min(self.buffer)]
            self.pos += 1
        return self.buffer.get(index)

    def sample(self, index):
        """(5, H, W) uint8 input of the net for frame `index`, or None at the ends of the video."""
        before, centre, after = self.get(index - GAP), self.get(index), self.get(index + GAP)
        if before is None or centre is None or after is None:
            return None
        return np.concatenate([centre.transpose(2, 0, 1),
                               cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)[None],
                               cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)[None]])


# -- classical search, used only to turn tracked positions into pixel-accurate targets --------------

def blob_candidates(image, u, v, radius=40):
    """Small, bright, unsaturated, compact blobs near pixel (u, v): [(distance, x, y, area)], nearest first."""
    height, width = image.shape[:2]
    u, v = int(round(u)), int(round(v))
    x0, y0, x1, y1 = max(0, u - radius), max(0, v - radius), min(width, u + radius + 1), min(height, v + radius + 1)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return []
    hsv = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    value = hsv[..., 2].astype(np.int16)
    mask = ((value - cv2.blur(hsv[..., 2], (15, 15)).astype(np.int16) > 35) & (hsv[..., 1] < 110)).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    found = []
    for i in range(1, count):
        _, _, w, h, area = stats[i]
        if 4 <= area <= 120 and 2 <= min(w, h) and max(w, h) <= 14 and max(w, h) <= 2.2 * min(w, h):
            x, y = centroids[i][0] + x0, centroids[i][1] + y0
            found.append((float(np.hypot(x - u, y - v)), float(x), float(y), int(area)))
    return sorted(found)


def snap(image, u, v):
    """The visible ball near the tracked pixel position, or None when absent or ambiguous."""
    found = blob_candidates(image, u, v)
    if not found or found[0][0] > SNAP_RADIUS:
        return None
    if len(found) > 1 and found[1][0] <= found[0][0] + SNAP_MARGIN:
        return None
    return found[0][1], found[0][2]


# -- the net --------------------------------------------------------------------------------------

def build_net():
    """A small U-Net: (B, 5, H, W) in [0, 1] -> (B, 1, H, W) ball heat-map logits. H, W multiples of 4."""
    import torch
    from torch import nn

    def block(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                             nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

    class BallNet(nn.Module):
        def __init__(self, width=24):
            super().__init__()
            self.enc1, self.enc2, self.enc3 = block(CHANNELS, width), block(width, 2 * width), block(2 * width, 4 * width)
            self.dec2, self.dec1 = block(6 * width, 2 * width), block(3 * width, width)
            self.head = nn.Conv2d(width, 1, 1)
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            nn.init.constant_(self.head.bias, -4.0)        # start from "no ball anywhere"

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            d2 = self.dec2(torch.cat([self.up(e3), e2], dim=1))
            return self.head(self.dec1(torch.cat([self.up(d2), e1], dim=1)))

    return BallNet()


def pick_device(name="auto"):
    import torch
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def load_net(device, path=None):
    """(net in eval mode, score threshold chosen at training time)."""
    import json
    import torch
    path = path or os.path.join(MODELS, "ballnet.pt")
    net = build_net()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.to(device).eval()
    with open(os.path.splitext(path)[0] + ".json") as handle:
        meta = json.load(handle)
    return net, float(meta["threshold"])
