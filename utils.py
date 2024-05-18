
import cv2
import matplotlib.pyplot as plt

import numpy as np
from dataclasses import dataclass
from matplotlib.axes import Axes

import h5py


@dataclass
class ImagePair:

    img1: np.ndarray
    img2: np.ndarray

    H: np.ndarray
    src_pts: np.ndarray
    dst_pts: np.ndarray

    i: int = None
    j: int = None


class ImageMatcher:

    def __init__(self, images, min_match_count=300, ransac_reproj_thresh=3.0, percent=0.2):

        FLANN_INDEX_KDTREE = 1

        self.images = images
        self.min_match_count = min_match_count
        self.ransac_reproj_thresh = ransac_reproj_thresh
        self.percent = percent

        # To use ORB features, the flannMatcher needs to be modified. See:
        # https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        self.detector = cv2.SIFT_create()

        self.flann = cv2.FlannBasedMatcher(
            indexParams={'algorithm': FLANN_INDEX_KDTREE, 'trees': 5},
            searchParams={'checks': 50},
        )

        self.kds = [self.detector.detectAndCompute(i, None) for i in self.images]

    def match(self, i: int, j: int) -> ImagePair | None:

        kp1, des1 = self.kds[i]
        kp2, des2 = self.kds[j]

        matches = self.flann.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) < self.min_match_count:
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_thresh,
        )

        if H is None:
            return None

        mask = mask.astype(bool).ravel()
        src_pts = src_pts[mask][::int(1 / self.percent)]
        dst_pts = dst_pts[mask][::int(1 / self.percent)]

        return ImagePair(self.images[i], self.images[j], H, src_pts, dst_pts, i, j)


def load_video(path: str, grayscale: bool = True, n_frames: int = 99999) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if grayscale else frame)
    return frames


def project_points(H, pts):

    assert H.shape == (3, 3)
    assert len(pts.shape) == 2 and pts.shape[1] == 2

    o = np.ones([pts.shape[0], 1])
    pts_h = np.concatenate([pts, o], axis=1)
    pts_p = pts_h @ H.T
    return pts_p[:, :2] / pts_p[:, 2:3]


def reprojection_error(H, src_pts, dst_pts, plot=False):

    proj_pts = project_points(H, src_pts)

    if plot:
        plt.scatter(proj_pts[:, 0], proj_pts[:, 1], marker='x', c='r', lw=0.5)
        plt.scatter(dst_pts[:, 0], dst_pts[:, 1], marker='+', c='g', lw=1)
        plt.show()

    return np.sqrt(np.square(proj_pts - dst_pts).sum(axis=1)).mean()


def visualize_matches(pair: ImagePair):

    axs: tuple[Axes]
    fig, axs = plt.subplots(1, 2, dpi=150)
    a1, a2 = axs

    for a in axs:
        a.set_axis_off()

    a1.imshow(pair.img1, 'gray')
    a1.scatter(*pair.src_pts.T, c='r', marker='.', s=2, lw=1)

    a2.imshow(pair.img2, 'gray')
    a2.scatter(*pair.dst_pts.T, c='r', marker='.', s=2, lw=1)

    fig.tight_layout()
    plt.show()


def export_image_pairs(path: str, ps: list[ImagePair]):
    with h5py.File(path, 'w') as f:
        for idx, p in enumerate(ps):
            g = f.create_group(f'pair_{idx}')
            g.create_dataset('H', data=p.H)
            g.create_dataset('src_pts', data=p.src_pts)
            g.create_dataset('dst_pts', data=p.dst_pts)
            g.attrs['i'] = p.i
            g.attrs['j'] = p.j

        indices = {}
        for p in ps:
            if p.i not in indices:
                indices[p.i] = 1
            if p.j not in indices:
                indices[p.j] = 1

        f.attrs['n_pairs'] = len(ps)
        f.create_dataset('cam_indices', data=sorted(indices.keys()))


def import_optimized_cam_params(path: str) -> dict[np.ndarray]:
    cam_params = {}
    with h5py.File(path, 'r') as f:
        for d in f.values():
            cam_idx = d.attrs['cam_idx']
            cam_params[cam_idx] = np.array(d)
    return cam_params
