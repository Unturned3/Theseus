#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as Rot
import cv2
import utils

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

def fla(head, offsets):
    return head + np.cumsum(offsets[::-1], axis=0)

class App:
    def __init__(self, video_src):
        self.vis_len = 10
        self.track_len = 4
        self.detect_interval = 5
        self.active_tracks = []
        self.track_history = []
        self.cam = cv2.VideoCapture(video_src)

        self.vis_scale = 2

        # We only support 640x480 video for now
        self.vid_w = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vid_h = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert self.vid_w == 640 and self.vid_h == 480

        self.frame_idx = 0

        # Load ground truth poses
        self.trajs = np.load('/Users/Richard/Desktop/Dataset/t000.npy')
        self.R = []
        self.K = []

        for p in self.trajs:
            self.R.append(Rot.from_euler('YXZ', p[:3], degrees=True).as_matrix())
            f = 0.5 * 640 / np.tan(np.radians(0.5 * p[3]))
            K = np.array([
                [-f, 0, 0.5 * 640],
                [0, f, 0.5 * 480],
                [0, 0, 1],
            ])
            self.K.append(K.copy())

    def run(self):

        n_frames = int(self.cam.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_idx in range(n_frames):

            _, frame = self.cam.read()
            if frame is None:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = cv2.resize(frame.copy(),
                             (self.vid_w * self.vis_scale,
                              self.vid_h * self.vis_scale))

            if len(self.active_tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray

                # Select points that have consistent forward and backward flows
                p0 = np.float32([p for (_, p, _) in self.active_tracks]
                                ).reshape(-1, 1, 2)
                p1, *_ = cv2.calcOpticalFlowPyrLK(img0, img1, p0,
                                                  None, **lk_params)
                p0r, *_ = cv2.calcOpticalFlowPyrLK(img1, img0, p1,
                                                   None, **lk_params)
                d = abs(p0 - p0r).reshape(-1, 2).max(axis=1)
                good_mask = d < 1

                src_pts = p0.reshape(-1, 2)
                H = self.K[frame_idx-1] @ inv(self.R[frame_idx-1]) @ self.R[frame_idx] @ inv(self.K[frame_idx])
                proj_pts = utils.project_points(inv(H), src_pts)

                filtered_tracks = []
                for tr, xy, pxy, good in zip(self.active_tracks, p1.reshape(-1, 2), proj_pts, good_mask):
                    if not good:
                        self.track_history.append(tr)
                        continue

                    residual_flow = (xy - pxy)
                    tr[2].append(residual_flow)
                    tr[1] = xy

                    #if len(tr[2]) > self.track_len:
                    #    del tr[2][0]

                    filtered_tracks.append(tr)
                    cv2.circle(vis, np.intc(xy * self.vis_scale),
                               2, (0, 255, 0), -1)
                    cv2.circle(vis, np.intc(pxy * self.vis_scale),
                               2, (0, 0, 255), -1)

                self.active_tracks = filtered_tracks

                lines = []
                for _, head, offsets in self.active_tracks:
                    o = np.array(offsets) * 10
                    l = [head, head + o[-1]]
                    lines.append(l)

                cv2.polylines(
                    vis, np.intc(lines) * 2, False, (0, 255, 0), lineType=cv2.LINE_AA)

                # Draw dot for prev tracked points
                #for tr in self.tracks:
                #    for x, y in tr[:-1]:
                #        cv2.circle(vis, (int(x*2), int(y*2)), 1, (0, 0, 127), -1)

                cv2.putText(vis, f'Active tracks: {len(self.active_tracks)}',
                            (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(255, 255, 255),
                            thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(vis, f'Historical tracks: {len(self.track_history)}',
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(255, 255, 255),
                            thickness=1, lineType=cv2.LINE_AA)

            if self.frame_idx % self.detect_interval == 0:

                mask = np.full_like(frame_gray, 255)

                # Mask out currently tracked points
                for p in [np.intc(p) for _, p ,_ in self.active_tracks]:
                    cv2.circle(mask, p, radius=5, color=0, thickness=-1)

                points = cv2.goodFeaturesToTrack(frame_gray, mask=mask,
                                                 **feature_params)

                if points is not None:
                    for p in np.float32(points).reshape(-1, 2):
                        self.active_tracks.append([frame_idx, p, []])

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)

            ch = cv2.waitKey(0)
            if ch == 27:
                break

def main():
    import sys
    video_src = sys.argv[1]
    App(video_src).run()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.destroyAllWindows()
