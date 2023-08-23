"""
@File    :   lofter_trackr.py
@Time    :   2023/08/23 20:50:35
@Author  :   XinLi
@Contact :   lixin.1997.lixin@gmail.com
@Desc    :   None
"""


from .base_tracker import BaseTracker, ImageFrame
import numpy as np
import cv2
import kornia.feature as KF
import torch
import kornia as K


class LoFTRTracker(BaseTracker):
    def __init__(self, config):
        super(LoFTRTracker, self).__init__(config)

        self.LoFTR = KF.LoFTR(pretrained="indoor_new")

    def process_image(
        self,
        image: np.ndarray,
        timestamp: int,
    ):
        self.current_image = ImageFrame(image, timestamp)

        if self.last_pub_image is None:
            self.fill_keypoints(self.current_image)
            self.last_pub_image = self.current_image
            print("first image")
            return False

        if self.check_pub_time(timestamp):
            self.track_keypoints(self.current_image, self.last_pub_image)
            self.current_image.filter_track_keypoints(self.last_pub_image)
            self.fill_keypoints(self.current_image)
            self.plot_tracking_result()
            self.last_pub_image = self.current_image
            return True

        else:
            return False

    def track_keypoints(
        self,
        current_image: "ImageFrame",
        last_image: "ImageFrame",
    ):
        current_image.keypoints = []
        current_image.keypoints_id = []

        last_image_keypoints = last_image.get_keypoints()
        last_image_keypoints_id = last_image.get_keypoints_id()
        if len(last_image_keypoints) == 0:
            return

        img1 = K.utils.image_to_tensor(last_image.get_image())[None] / 255.0
        img2 = K.utils.image_to_tensor(current_image.get_image())[None] / 255.0
        input_dict = {
            "image0": img1,
            "image1": img2,
        }

        with torch.no_grad():
            correspondences = self.LoFTR(input_dict)

        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()

        current_image.keypoints = []
        current_image.keypoints_id = []
        for i in range(len(last_image_keypoints)):
            for pts0, pts1 in zip(mkpts0, mkpts1):
                if np.linalg.norm(pts0 - last_image_keypoints[i]) < 5:
                    current_image.keypoints.append(pts1)
                    current_image.keypoints_id.append(last_image_keypoints_id[i])
                    break

    def fill_keypoints(
        self,
        current_image: "ImageFrame",
    ):
        cur_extracted_keypoints = current_image.max_extracted_keypoints - len(
            current_image.keypoints
        )

        if cur_extracted_keypoints <= 10:
            return

        mask = np.ones_like(current_image.gray_image)
        for keypoint in current_image.keypoints:
            cv2.circle(
                mask,
                [int(keypoint[0]), int(keypoint[1])],
                current_image.min_keypoint_distance,
                0,
                -1,
            )

        points = cv2.goodFeaturesToTrack(
            current_image.gray_image,
            maxCorners=cur_extracted_keypoints,
            qualityLevel=0.01,
            minDistance=current_image.min_keypoint_distance,
            mask=mask,
        )

        # cv2.imshow("mask", mask * 255)
        if points is None:
            return
        for point in points:
            current_image.keypoints.append(point[0])

            current_image.keypoints_id.append(self.feature_point_id)
            self.feature_point_id += 1
