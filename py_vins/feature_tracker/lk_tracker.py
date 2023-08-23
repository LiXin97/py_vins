"""
@File    :   lk_tracker.py
@Time    :   2023/08/23 20:38:47
@Author  :   XinLi
@Contact :   lixin.1997.lixin@gmail.com
@Desc    :   None
"""


from .base_tracker import BaseTracker, ImageFrame
import numpy as np
import cv2


class XinOpticalFlowPyrLK(object):
    def __init__(self):
        self.win_size = (11, 11)
        self.max_level = 2
        self.criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            100,
            0.01,
        )

    def __call__(
        self,
        last_image: np.ndarray,
        current_image: np.ndarray,
        last_image_keypoints: np.ndarray,
        current_image_keypoints: np.ndarray,
        win_size: tuple = None,
        max_level: int = None,
    ):
        if win_size is None:
            win_size = self.win_size
        if max_level is None:
            max_level = self.max_level

        tracked_keypoints = []
        tracked_status = []

        image_set_pyramid = []
        for i in range(max_level):
            image_set_pyramid.append(
                [
                    cv2.resize(last_image, (0, 0), fx=1 / (2**i), fy=1 / (2**i)),
                    cv2.resize(current_image, (0, 0), fx=1 / (2**i), fy=1 / (2**i)),
                    (2**i),
                ]
            )

        image_set_pyramid = image_set_pyramid[::-1]

        for last_image_keypoint in last_image_keypoints:
            cur_tracked_keypoint = last_image_keypoint
            status = True
            if current_image_keypoints is not None:
                cur_tracked_keypoint = current_image_keypoints[0]

            for image_set in image_set_pyramid:
                if status is False:
                    break
                last_image_pyramid = image_set[0]
                current_image_pyramid = image_set[1]
                cur_pyramid_level = image_set[2]

                half_win_size = (win_size[0] // 2, win_size[1] // 2)

                last_cost = 1e10
                for _ in range(100):
                    last_image_keypoint_pyramid = (
                        last_image_keypoint / cur_pyramid_level
                    )
                    cur_tracked_keypoint_pyramid = (
                        cur_tracked_keypoint / cur_pyramid_level
                    )
                    cur_point_move_pyrmid = (
                        cur_tracked_keypoint_pyramid - last_image_keypoint_pyramid
                    )
                    H = np.zeros((2, 2))
                    b = np.zeros((2, 1))
                    J = np.zeros((2, 1))
                    cost = 0.0
                    for i in range(-half_win_size[0], half_win_size[0]):
                        for j in range(-half_win_size[1], half_win_size[1]):

                            def get_pixel_value(image, x, y) -> float:
                                if (
                                    x < 0
                                    or x >= image.shape[1]
                                    or y < 0
                                    or y >= image.shape[0]
                                ):
                                    return 0.0
                                return float(image[int(y), int(x)])

                            J = -1 * np.array(
                                [
                                    [
                                        (
                                            get_pixel_value(
                                                last_image_pyramid,
                                                last_image_keypoint_pyramid[0] + j + 1,
                                                last_image_keypoint_pyramid[1] + i,
                                            )
                                            - get_pixel_value(
                                                last_image_pyramid,
                                                last_image_keypoint_pyramid[0] + j - 1,
                                                last_image_keypoint_pyramid[1] + i,
                                            )
                                        )
                                        / 2,
                                        (
                                            get_pixel_value(
                                                last_image_pyramid,
                                                last_image_keypoint_pyramid[0] + j,
                                                last_image_keypoint_pyramid[1] + i + 1,
                                            )
                                            - get_pixel_value(
                                                last_image_pyramid,
                                                last_image_keypoint_pyramid[0] + j,
                                                last_image_keypoint_pyramid[1] + i - 1,
                                            )
                                        )
                                        / 2,
                                    ]
                                ]
                            )

                            error = get_pixel_value(
                                last_image_pyramid,
                                last_image_keypoint_pyramid[0] + j,
                                last_image_keypoint_pyramid[1] + i,
                            ) - get_pixel_value(
                                current_image_pyramid,
                                last_image_keypoint_pyramid[0]
                                + j
                                + cur_point_move_pyrmid[0],
                                last_image_keypoint_pyramid[1]
                                + i
                                + cur_point_move_pyrmid[1],
                            )

                            b = b - J.T * error
                            cost = cost + float(error) ** 2
                            H = H + J.T @ J

                    # update = inv(H) @ b
                    # using ldlt decomposition
                    # np.linalg.cholesky(H)

                    if cost > last_cost:
                        break
                    last_cost = cost

                    if np.linalg.det(H) == 0:
                        status = False
                        break
                    update = np.linalg.inv(H) @ b

                    cur_tracked_keypoint_pyramid = (
                        cur_tracked_keypoint_pyramid + update.transpose()
                    )

                    cur_tracked_keypoint = (
                        cur_tracked_keypoint_pyramid * cur_pyramid_level
                    )[0]

                    if (
                        cur_tracked_keypoint[0] < 0
                        or cur_tracked_keypoint[0] >= current_image.shape[1]
                        or cur_tracked_keypoint[1] < 0
                        or cur_tracked_keypoint[1] >= current_image.shape[0]
                    ):
                        status = False
                        break

                    if np.linalg.norm(update) < 0.1:
                        break
            tracked_keypoints.append(cur_tracked_keypoint)
            tracked_status.append(status)

        return tracked_keypoints, tracked_status


class LKTracker(BaseTracker):
    def __init__(self, config):
        super(LKTracker, self).__init__(config)

    def process_image(
        self,
        image: np.ndarray,
        timestamp: int,
    ):
        self.current_image = ImageFrame(image, timestamp)

        if self.last_image is None:
            self.fill_keypoints(self.current_image)
            self.last_image = self.current_image
            self.last_pub_image = self.current_image
            return False

        self.track_keypoints(self.current_image, self.last_image)

        if self.check_pub_time(timestamp):
            self.current_image.filter_track_keypoints(self.last_pub_image)
            self.fill_keypoints(self.current_image)
            self.plot_tracking_result()
            self.last_image = self.current_image
            self.last_pub_image = self.current_image
            return True

        else:
            self.last_image = self.current_image
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

        points, status = XinOpticalFlowPyrLK()(
            last_image.get_gray_image(),
            current_image.gray_image,
            last_image_keypoints,
            None,
        )
        # points, status, error = cv2.calcOpticalFlowPyrLK(
        #     last_image.get_gray_image(),
        #     current_image.image,
        #     np.array(last_image_keypoints, dtype=np.float32),
        #     None,
        #     winSize=(21, 21),
        #     maxLevel=8,
        #     criteria=(
        #         cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        #         100,
        #         0.01,
        #     ),
        # )

        status_inv = status.copy()

        # # inverse check
        # points_inv, status_inv, error_inv = cv2.calcOpticalFlowPyrLK(
        #     current_image.image,
        #     last_image.get_gray_image(),
        #     points,
        #     None,
        #     winSize=(21, 21),
        #     maxLevel=8,
        #     criteria=(
        #         cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        #         100,
        #         0.01,
        #     ),
        # )

        for i in range(len(points)):
            if status[i] == 0 or status_inv[i] == 0:
                continue
            current_image.keypoints.append(points[i])
            current_image.keypoints_id.append(last_image_keypoints_id[i])

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
