"""
@File    :   base_tracker.py
@Time    :   2023/08/20 21:22:28
@Author  :   XinLi
@Contact :   lixin.1997.lixin@gmail.com
@Desc    :   None
"""
import numpy as np
import cv2


class ImageFrame(object):
    def __init__(self, image: np.ndarray, timestamp_ns: int):
        self.image = image
        equalize = False
        if equalize:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            self.gray_image = clahe.apply(image)
        else:
            self.gray_image = image.copy()
        self.timestamp_ns = timestamp_ns
        self.keypoints = []
        self.keypoints_id = []

        self.max_extracted_keypoints = 150
        self.min_keypoint_distance = 30

    def get_timestamp_ns(self):
        return self.timestamp_ns

    def get_image(self):
        return self.image

    def get_gray_image(self):
        return self.gray_image

    def get_keypoints(self):
        return self.keypoints

    def get_keypoints_id(self):
        return self.keypoints_id

    def get_matching_index(
        self,
        other_image: "ImageFrame",
    ):
        id_map = {}
        for i in range(len(self.keypoints_id)):
            id_map[self.keypoints_id[i]] = i

        matching_pairs = []
        matching_pairs_id = []

        other_image_keypoints_id = other_image.get_keypoints_id()
        for i in range(len(other_image_keypoints_id)):
            other_image_keypoint_id = other_image_keypoints_id[i]
            if other_image_keypoint_id in id_map.keys():
                matching_pairs.append([id_map[other_image_keypoints_id[i]], i])
                matching_pairs_id.append(other_image_keypoints_id[i])

        return matching_pairs, matching_pairs_id

    def filter_track_keypoints(
        self,
        last_image: "ImageFrame",
    ):
        last_image_keypoints = last_image.get_keypoints()
        if len(last_image_keypoints) == 0:
            return

        matching_pairs, matching_pairs_id = self.get_matching_index(last_image)

        matching_cur_keypoints = []
        matching_last_keypoints = []
        for pair in matching_pairs:
            matching_cur_keypoints.append(self.keypoints[pair[0]])
            matching_last_keypoints.append(last_image_keypoints[pair[1]])

        if len(matching_cur_keypoints) < 8:
            self.keypoints = []
            self.keypoints_id = []
            return

        F, mask = cv2.findFundamentalMat(
            np.array(matching_cur_keypoints),
            np.array(matching_last_keypoints),
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=4.0,
        )
        # print("tracking inliers: ", np.sum(mask))
        # print("keypoints before filter: ", len(self.keypoints))
        self.keypoints = []
        self.keypoints_id = []
        for i in range(len(mask)):
            if mask[i] != 0:
                self.keypoints.append(matching_cur_keypoints[i])
                self.keypoints_id.append(matching_pairs_id[i])
        # print("keypoints after filter: ", len(self.keypoints))


class BaseTracker(object):
    def __init__(self, config):
        self.__config = config

        self.current_image = None
        self.last_image = None
        self.last_pub_image = None

        self.tracking_result = None

        self.feature_point_id = 0

    def process_image(
        self,
        image: np.ndarray,
        timestamp: int,
    ):
        pass

    def check_pub_time(self, timestamp):
        delta_time = timestamp - self.last_pub_image.get_timestamp_ns()
        process_time_pub = 1.0 / 10
        if delta_time / 1e9 >= process_time_pub:
            return True
        else:
            return False

    def plot_tracking_result(self):
        show_image = self.current_image.get_image().copy()
        show_image = cv2.cvtColor(show_image, cv2.COLOR_GRAY2BGR)
        current_keypoints = self.current_image.get_keypoints()
        # current_keypoints_id = self.current_image.get_keypoints_id()
        # cur_id_key_map = {}

        # for i in range(len(current_keypoints)):
        #     cv2.circle(
        #         show_image,
        #         [int(current_keypoints[i][0]), int(current_keypoints[i][1])],
        #         3,
        #         (0, 0, 255),
        #         -1,
        #     )
        #     cur_id_key_map[i] = current_keypoints_id[i]

        matching_pairs, matching_pairs_id = self.current_image.get_matching_index(
            self.last_pub_image
        )
        last_keypoints = self.last_pub_image.get_keypoints()
        for pair in matching_pairs:
            cv2.circle(
                show_image,
                [
                    int(current_keypoints[pair[0]][0]),
                    int(current_keypoints[pair[0]][1]),
                ],
                3,
                (0, 0, 255),
                -1,
            )
            cv2.line(
                show_image,
                [
                    int(current_keypoints[pair[0]][0]),
                    int(current_keypoints[pair[0]][1]),
                ],
                [
                    int(last_keypoints[pair[1]][0]),
                    int(last_keypoints[pair[1]][1]),
                ],
                (255, 0, 0),
                1,
            )
            # cv2.putText(
            #     show_image,
            #     "{}".format(cur_id_key_map[pair[0]]),
            #     (
            #         int(current_keypoints[pair[0]][0]),
            #         int(current_keypoints[pair[0]][1]),
            #     ),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 255, 0),
            #     1,
            # )
        tracking_num = len(matching_pairs)

        cv2.putText(
            show_image,
            "tracking_num: {}".format(tracking_num),
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

        self.tracking_result = show_image

    def show_current_frame(self):
        return self.tracking_result
