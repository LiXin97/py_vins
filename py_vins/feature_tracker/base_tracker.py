"""
@File    :   base_tracker.py
@Time    :   2023/08/20 21:22:28
@Author  :   XinLi
@Contact :   lixin.1997.lixin@gmail.com
@Desc    :   None
"""
import numpy as np
import cv2

feature_point_id = 0


class ImageFrame(object):
    def __init__(self, image: np.ndarray, timestamp_ns: int):
        self.__image = image
        equalize = False
        if equalize:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            self.__gray_image = clahe.apply(image)
        else:
            self.__gray_image = image.copy()
        self.__timestamp_ns = timestamp_ns
        self.__keypoints = []
        self.__keypoints_id = []

        self.__max_extracted_keypoints = 300
        self.__min_keypoint_distance = 30

    def get_timestamp_ns(self):
        return self.__timestamp_ns

    def get_image(self):
        return self.__image

    def get_gray_image(self):
        return self.__gray_image

    def get_keypoints(self):
        return self.__keypoints

    def get_keypoints_id(self):
        return self.__keypoints_id

    def fill_keypoints(self):
        cur_extracted_keypoints = self.__max_extracted_keypoints - len(self.__keypoints)

        if cur_extracted_keypoints <= 10:
            return

        mask = np.ones_like(self.__gray_image)
        for keypoint in self.__keypoints:
            cv2.circle(
                mask,
                [int(keypoint[0]), int(keypoint[1])],
                self.__min_keypoint_distance,
                0,
                -1,
            )

        points = cv2.goodFeaturesToTrack(
            self.__gray_image,
            maxCorners=cur_extracted_keypoints,
            qualityLevel=0.01,
            minDistance=self.__min_keypoint_distance,
            mask=mask,
        )

        # cv2.imshow("mask", mask * 255)
        if points is None:
            return
        for point in points:
            self.__keypoints.append(point[0])
            global feature_point_id
            self.__keypoints_id.append(feature_point_id)
            feature_point_id += 1

    def get_matching_index(
        self,
        other_image: "ImageFrame",
    ):
        id_map = {}
        for i in range(len(self.__keypoints_id)):
            id_map[self.__keypoints_id[i]] = i

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
            matching_cur_keypoints.append(self.__keypoints[pair[0]])
            matching_last_keypoints.append(last_image_keypoints[pair[1]])

        if len(matching_cur_keypoints) < 8:
            self.__keypoints = []
            self.__keypoints_id = []
            return

        F, mask = cv2.findFundamentalMat(
            np.array(matching_cur_keypoints),
            np.array(matching_last_keypoints),
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=4.0,
        )
        # print("tracking inliers: ", np.sum(mask))
        # print("keypoints before filter: ", len(self.__keypoints))
        self.__keypoints = []
        self.__keypoints_id = []
        for i in range(len(mask)):
            if mask[i] != 0:
                self.__keypoints.append(matching_cur_keypoints[i])
                self.__keypoints_id.append(matching_pairs_id[i])
        # print("keypoints after filter: ", len(self.__keypoints))

    def track_keypoints(
        self,
        last_image: "ImageFrame",
    ):
        self.__keypoints = []
        self.__keypoints_id = []

        last_image_keypoints = last_image.get_keypoints()
        last_image_keypoints_id = last_image.get_keypoints_id()
        if len(last_image_keypoints) == 0:
            return

        points, status, error = cv2.calcOpticalFlowPyrLK(
            last_image.get_gray_image(),
            self.__image,
            np.array(last_image_keypoints, dtype=np.float32),
            None,
            winSize=(21, 21),
            maxLevel=8,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                100,
                0.01,
            ),
        )

        for i in range(len(points)):
            if status[i] == 0:
                continue
            self.__keypoints.append(points[i])
            self.__keypoints_id.append(last_image_keypoints_id[i])


class BaseTracker(object):
    def __init__(self, config):
        self.__config = config

        self.__current_image = None
        self.__last_image = None
        self.__last_pub_image = None

        self.__tracking_result = None

    def process_image(
        self,
        image: np.ndarray,
        timestamp: int,
    ):
        self.__current_image = ImageFrame(image, timestamp)

        if self.__last_image is None:
            self.__current_image.fill_keypoints()
            self.__last_image = self.__current_image
            self.__last_pub_image = self.__current_image
            return False

        self.__current_image.track_keypoints(self.__last_image)

        if self.check_pub_time(timestamp):
            self.__current_image.filter_track_keypoints(self.__last_pub_image)
            self.__current_image.fill_keypoints()
            self.plot_tracking_result()
            self.__last_image = self.__current_image
            self.__last_pub_image = self.__current_image
            return True

        else:
            self.__last_image = self.__current_image
            return False

    def check_pub_time(self, timestamp):
        delta_time = timestamp - self.__last_pub_image.get_timestamp_ns()
        # print("delta_time: ", delta_time / 1e9)
        process_time_pub = 1.0 / 10
        if delta_time / 1e9 >= process_time_pub:
            return True
        else:
            return False

    def plot_tracking_result(self):
        show_image = self.__current_image.get_image().copy()
        show_image = cv2.cvtColor(show_image, cv2.COLOR_GRAY2BGR)
        current_keypoints = self.__current_image.get_keypoints()
        current_keypoints_id = self.__current_image.get_keypoints_id()
        cur_id_key_map = {}

        for i in range(len(current_keypoints)):
            cv2.circle(
                show_image,
                [int(current_keypoints[i][0]), int(current_keypoints[i][1])],
                3,
                (0, 0, 255),
                -1,
            )
            cur_id_key_map[i] = current_keypoints_id[i]

        matching_pairs, matching_pairs_id = self.__current_image.get_matching_index(
            self.__last_pub_image
        )
        last_keypoints = self.__last_pub_image.get_keypoints()
        for pair in matching_pairs:
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
            cv2.putText(
                show_image,
                "{}".format(cur_id_key_map[pair[0]]),
                (
                    int(current_keypoints[pair[0]][0]),
                    int(current_keypoints[pair[0]][1]),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
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

        self.__tracking_result = show_image

    def show_current_frame(self):
        return self.__tracking_result


class LKTracker(BaseTracker):
    def __init__(self, config):
        super(LKTracker, self).__init__(config)
        pass
