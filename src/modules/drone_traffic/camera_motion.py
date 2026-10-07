"""Conservative ORB/RANSAC frame-to-reference motion compensation."""
from __future__ import annotations

import cv2
import numpy as np


class CameraMotionCompensator:
    def __init__(self, config):
        self.enabled = bool(config.get("enabled", False))
        self.min_matches = int(config.get("minimum_matches", 18))
        self.max_reprojection_error = float(config.get("maximum_reprojection_error", 4.0))
        self.estimate_every = max(1, int(config.get("estimate_every_n_frames", 3)))
        self.processing_width = int(config.get("processing_width", 640))
        self.frame_count = 0
        self.last_reliable = False
        self.previous_gray = None
        self.previous_kp = None
        self.previous_des = None
        self.cumulative = np.eye(3, dtype=float)
        self.orb = cv2.ORB_create(nfeatures=int(config.get("features", 1800)))
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.status = "disabled" if not self.enabled else "initializing"

    def transform(self, frame):
        self.frame_count += 1
        if (self.enabled and self.previous_des is not None and
                self.frame_count % self.estimate_every != 0):
            return self.cumulative.copy(), self.last_reliable, self.status
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, self.processing_width / float(gray_full.shape[1]))
        if scale < 1.0:
            gray = cv2.resize(gray_full, (int(round(gray_full.shape[1] * scale)),
                                          int(round(gray_full.shape[0] * scale))),
                              interpolation=cv2.INTER_AREA)
        else:
            gray = gray_full
        kp, des = self.orb.detectAndCompute(gray, None)
        reliable = not self.enabled
        if self.enabled and self.previous_des is not None and des is not None:
            pairs = self.matcher.knnMatch(self.previous_des, des, k=2)
            good = [m for m, n in pairs if m.distance < 0.72 * n.distance]
            if len(good) >= self.min_matches:
                src = np.float32([self.previous_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC,
                                             self.max_reprojection_error * scale)
                inliers = int(mask.sum()) if mask is not None else 0
                if H is not None and inliers >= self.min_matches and np.isfinite(H).all():
                    # Keypoints are measured in the downscaled image. Convert
                    # the mapping back to full-resolution pixel coordinates.
                    S = np.diag([scale, scale, 1.0])
                    H = np.linalg.inv(S) @ H @ S
                    self.cumulative = self.cumulative @ np.linalg.inv(H)
                    reliable = True
                    self.status = f"compensated ({inliers} inliers)"
                else:
                    self.status = "unreliable: homography"
            else:
                self.status = f"unreliable: {len(good)} feature matches"
        elif self.enabled:
            self.status = "initializing"
        self.previous_gray, self.previous_kp, self.previous_des = gray, kp, des
        self.last_reliable = reliable
        return self.cumulative.copy(), reliable, self.status

    @staticmethod
    def map_point(matrix, point):
        p = np.asarray([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        return cv2.perspectiveTransform(p, matrix.astype(np.float64)).reshape(2).tolist()
