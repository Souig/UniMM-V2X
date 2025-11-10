import cv2
import numpy as np
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class UndistortMultiViewImage:
    def __init__(self, K, D):
        """
        Args:
            K (list or np.ndarray): Camera intrinsic matrix [3x3].
            D (list or np.ndarray): Distortion coefficients [5x1] or [4x1].
        """
        self.K = np.array(K)
        self.D = np.array(D)

    def __call__(self, results):
        imgs = results['img']  # [N, H, W, 3] or list of images
        undistorted_imgs = []
        h, w = imgs[0].shape[:2]
        new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w, h), 1, (w, h))
        for img in imgs:
            img = cv2.undistort(img, self.K, self.D, None, new_K)
            undistorted_imgs.append(img)
        results['img'] = undistorted_imgs
        return results