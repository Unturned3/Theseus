
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as Rot
import cv2
import utils

import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass

lk_params = dict(
    winSize  = (15, 15),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS |
                cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

feature_params = dict(
    maxCorners   = 2000,
    qualityLevel = 0.3,
    minDistance  = 7,
    blockSize    = 7,
)

def shortest_axis_angle_rotation(U, V):
    # Normalize the input vectors
    U = U / np.linalg.norm(U)
    V = V / np.linalg.norm(V)

    # Compute the cross product and dot product
    cross_product = np.cross(U, V)
    dot_product = np.dot(U, V)

    # Compute the sine and cosine of the angle
    sin_angle = np.linalg.norm(cross_product)
    cos_angle = dot_product

    # Compute the angle using atan2 for better numerical stability
    angle = np.arctan2(sin_angle, cos_angle)

    # Normalize the axis of rotation
    if sin_angle != 0:
        axis = cross_product / sin_angle
    else:
        # If sin_angle is zero, the vectors are parallel or anti-parallel
        # Handle the special cases:
        if cos_angle > 0:
            # Vectors are parallel, any axis will do
            axis = np.array([1, 0, 0])
        else:
            # Vectors are anti-parallel
            # Find a vector orthogonal to U to use as the axis
            orthogonal_vector = np.array([1, 0, 0]) if abs(U[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(U, orthogonal_vector)
            axis = axis / np.linalg.norm(axis)

    return axis, angle

def fit_and_plot_plane(aas):
    # Fit a plane to the points
    # Plane equation: ax + by + cz + d = 0
    # We can write it as Ax = B where A is a matrix of the coordinates and x is the coefficients [a, b, c, d]
    A = np.c_[aas[:,0], aas[:,1], np.ones(aas.shape[0])]
    B = aas[:, 2]

    # Solve for [a, b, d] (c is set to -1)
    coefficients, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, d = coefficients
    c = -1

    print(f'Plane equation: {a:.5f}x + {b:.5f}y - z + {d:.5f} = 0')

    # Create a grid of points covering the range of the data
    x_range = np.linspace(aas[:, 0].min(), aas[:, 0].max(), 10)
    y_range = np.linspace(aas[:, 1].min(), aas[:, 1].max(), 10)
    x, y = np.meshgrid(x_range, y_range)
    z = a * x + b * y + d

    return x, y, z


@dataclass
class Track:
    uid: int
    start_frame_idx: int
    pts: list[np.ndarray]
    aas: list[np.ndarray]


class LKTracker:
    def __init__(self, frames, gt_traj=None, interactive=False):

        assert frames is not None
        assert len(frames) > 0

        self.interactive = interactive
        self.vis_len = 10
        self.detect_interval = 5
        self.active_tracks: list[Track] = []
        self.track_history: list[Track] = []
        self.frames = frames
        self.orb_detector = cv2.ORB_create(nfeatures=500)
        self.track_cnt = 0
        self.vis_scale = 2

        # We only support 640x480 video for now
        self.vid_h, self.vid_w = self.frames[0].shape[:2]
        assert self.vid_w == 640 and self.vid_h == 480

        self.is_color = self.frames[0].shape[2] == 3

        self.frame_idx = 0

        self.traj = gt_traj
        self.R = []
        self.K = []

        if self.traj is not None:
            for p in self.traj:
                self.R.append(Rot.from_euler('YXZ', p[:3], degrees=True).as_matrix())
                f = 0.5 * self.vid_w / np.tan(np.radians(0.5 * p[3]))
                K = np.array([
                    [-f, 0, 0.5 * self.vid_w],
                    [0, f, 0.5 * self.vid_h],
                    [0, 0, 1],
                ])
                self.K.append(K.copy())

    def cvtColor(self, frame):
        if self.is_color:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def run(self) -> None | list[Track]:

        for frame_idx in range(len(self.frames)):

            frame = self.frames[frame_idx]
            assert frame is not None

            frame_gray = self.cvtColor(frame)
            if self.interactive:
                vis = cv2.resize(frame.copy(),
                                 (self.vid_w * self.vis_scale,
                                 self.vid_h * self.vis_scale))

            if len(self.active_tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray

                # Select points that have consistent forward and backward flows
                p0 = np.float32([t.pts[-1] for t in self.active_tracks]
                                ).reshape(-1, 1, 2)
                p1, *_ = cv2.calcOpticalFlowPyrLK(img0, img1, p0,
                                                  None, **lk_params)
                p0r, *_ = cv2.calcOpticalFlowPyrLK(img1, img0, p1,
                                                   None, **lk_params)
                d = abs(p0 - p0r).reshape(-1, 2).max(axis=1)
                good_mask = d < 1

                fi = frame_idx

                if self.traj is not None:
                    f1_w = self.K[fi] @ inv(self.R[fi])     # world to frame 1
                    w_f0 = self.R[fi-1] @ inv(self.K[fi-1]) # frame 0 to world

                    # Tracked points from frame 0 to world via GT homography
                    w_f0_ps = utils.project_points(w_f0, p0.reshape(-1, 2), keep_z=True)

                    # Predicted coordinates of the tracked points in frame 1
                    pp1s = utils.project_points(f1_w, w_f0_ps)

                # Observed coordinates of the tracked points in frame 1
                op1s = p1.reshape(-1, 2)

                if self.traj is not None:
                    # Tracked points from frame 1 to world via GT homography
                    w_f1_ps = utils.project_points(inv(f1_w), op1s, keep_z=True)

                filtered_tracks = []

                for ti, tr in enumerate(self.active_tracks):

                    if not good_mask[ti]:
                        self.track_history.append(tr)
                        continue

                    # Observed and predicted (via GT homographies) p1
                    op1 = op1s[ti]
                    if self.traj is not None:
                        pp1 = pp1s[ti]

                    # TODO: determine run or not based on self.traj
                    wU, wV = w_f0_ps[ti], w_f1_ps[ti]
                    axis, angle = shortest_axis_angle_rotation(wU, wV)
                    aa = axis * angle

                    #residual_flow = (op1 - pp1)
                    tr.aas.append(aa)
                    tr.pts.append(op1)

                    filtered_tracks.append(tr)

                    if self.interactive:
                        cv2.circle(vis, np.intc(op1 * self.vis_scale),
                                2, (0, 255, 0), -1)
                        cv2.circle(vis, np.intc(pp1 * self.vis_scale),
                                2, (0, 0, 255), -1)
                        cv2.putText(vis, str(tr.uid), np.intc(op1 * self.vis_scale),
                                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,
                                    color=(255, 255, 255), thickness=1,
                                    lineType=cv2.LINE_AA)

                self.active_tracks = filtered_tracks

                if self.interactive:
                    cv2.putText(vis, f'Active tracks: {len(self.active_tracks)}',
                                (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5, color=(255, 255, 255),
                                thickness=1, lineType=cv2.LINE_AA)
                    cv2.putText(vis, f'Historical tracks: {len(self.track_history)}',
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5, color=(255, 255, 255),
                                thickness=1, lineType=cv2.LINE_AA)
                    cv2.putText(vis, f'Frame number: {frame_idx}',
                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5, color=(255, 255, 255),
                                thickness=1, lineType=cv2.LINE_AA)

            if self.frame_idx % self.detect_interval == 0:

                mask = np.full_like(frame_gray, 255)

                # Mask out currently tracked points
                for p in [np.intc(t.pts[-1]) for t in self.active_tracks]:
                    cv2.circle(mask, p, radius=5, color=0, thickness=-1)

                points = cv2.goodFeaturesToTrack(frame_gray, mask=mask,
                                                 **feature_params)
                #points = self.orb_detector.detect(frame_gray, mask=mask)
                #points = [p.pt for p in points]

                if points is not None:
                    for p in np.float32(points).reshape(-1, 2):
                        self.active_tracks.append(
                            Track(self.track_cnt, frame_idx, [p], []))
                        self.track_cnt += 1

            self.frame_idx += 1
            self.prev_gray = frame_gray

            if self.interactive:
                cv2.imshow('lk_track', vis)
                ch = cv2.waitKey(0)
                if ch == 27:
                    break

        while self.interactive:
            uid = input('Enter track id: ')
            # Search for the track id in active and historical tracks

            for tr in self.active_tracks + self.track_history:
                if tr.uid == int(uid):
                    aas = np.array(tr.aas)
                    magnitudes = np.linalg.norm(aas, axis=1)

                    fig = plt.figure()
                    ax: Axes3D = fig.add_subplot(111, projection='3d')
                    ax.set_proj_type('ortho')
                    ax.plot(aas[:, 0], aas[:, 1], aas[:, 2])
                    sc = ax.scatter(aas[:, 0], aas[:, 1], aas[:, 2], c=magnitudes, cmap='viridis')

                    x, y, z = fit_and_plot_plane(aas)
                    ax.plot_surface(x, y, z, color='r', alpha=0.3)
                    fig.colorbar(sc)
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_aspect('equal')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_zticks([])
                    #for i in tr[3]:
                    #    print(np.linalg.norm(i))
                    #break
                    plt.show()
                    plt.close()
                    break
            else:
                print('Track not found')

        return self.track_history + self.active_tracks

def main():
    cap = cv2.VideoCapture(sys.argv[1])
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    traj = np.load(sys.argv[2])
    LKTracker(frames, gt_traj=traj, interactive=True).run()


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
