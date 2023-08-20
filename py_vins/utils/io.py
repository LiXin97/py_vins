"""
@File    :   io.py
@Time    :   2023/08/20 21:28:19
@Author  :   XinLi
@Contact :   lixin.1997.lixin@gmail.com
@Desc    :   None
"""


import os
import cv2


class Dataset(object):
    def __init__(
        self,
        dataset_dir: str,
        seq_name: str,
    ):
        self.dataset_dir = dataset_dir
        self.seq_name = seq_name


class EuRoC(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        seq_name: str,
    ):
        super(EuRoC, self).__init__(dataset_dir, seq_name)
        gt_file = os.path.join(
            self.dataset_dir,
            self.seq_name,
            "mav0",
            "state_groundtruth_estimate0",
            "data.csv",
        )
        imu_file = os.path.join(
            self.dataset_dir, self.seq_name, "mav0", "imu0", "data.csv"
        )
        image_timestamps_file = os.path.join(
            self.dataset_dir, self.seq_name, "mav0", "cam0", "data.csv"
        )

        self.image_timestamps, self.image_names = self.read_image_timestamps(
            image_timestamps_file
        )
        self.imu_datas = self.read_imu(imu_file)
        self.gt_datas = self.read_gt(gt_file)

        self.image_path = {}
        image_dir = os.path.join(
            self.dataset_dir, self.seq_name, "mav0", "cam0", "data"
        )
        for image_name, image_timestamp in zip(self.image_names, self.image_timestamps):
            self.image_path[image_timestamp] = os.path.join(image_dir, image_name)

    def read_imu(self, imu_file):
        with open(imu_file, "r") as f:
            lines = f.readlines()
        imu_datas = []
        # timestamp, ax, ay, az, gx, gy, gz
        for line in lines:
            if line[0] == "#":
                continue
            line = line.strip().split(",")
            imu_datas.append(
                [
                    int(line[0]),
                    float(line[4]),
                    float(line[5]),
                    float(line[6]),
                    float(line[1]),
                    float(line[2]),
                    float(line[3]),
                ]
            )
        return imu_datas

    def read_gt(self, gt_file):
        with open(gt_file, "r") as f:
            lines = f.readlines()
        gt_datas = []
        # timestamp, px, py, pz, qw, qx, qy, qz
        for line in lines:
            if line[0] == "#":
                continue
            line = line.strip().split(",")
            gt_datas.append(
                [
                    int(line[0]),
                    float(line[1]),
                    float(line[2]),
                    float(line[3]),
                    float(line[4]),
                    float(line[5]),
                    float(line[6]),
                    float(line[7]),
                ]
            )
        return gt_datas

    def read_image_from_timestamp(
        self,
        timestamp: int,
    ):
        return self.read_image(self.image_path[timestamp])

    def read_image_timestamps(
        self,
        timestamp_file_path: str,
    ):
        # timestamp [ns],filename
        with open(timestamp_file_path, "r") as f:
            lines = f.readlines()
        timestamps = []
        image_names = []
        for line in lines:
            if line[0] == "#":
                continue
            line = line.strip().split(",")
            timestamps.append(int(line[0]))
            image_names.append(line[1])
        return timestamps, image_names

    def read_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        return image


if __name__ == "__main__":
    dataset_dir = "/home/xin/Documents/py_vins/data/EuRoc"
    seq_name = "V1_01_easy"
    dataset = EuRoC(dataset_dir, seq_name)
    print("image length: ", len(dataset.image_names))
    print("imu length: ", len(dataset.imu_datas))
    print("gt length: ", len(dataset.gt_datas))

    print("first image name: ", dataset.image_names[0])
    print("first image timestamp: ", dataset.image_timestamps[0])
    print("first imu data: ", dataset.imu_datas[0])
    print("first gt data: ", dataset.gt_datas[0])
    print("first image path: ", dataset.image_path[dataset.image_timestamps[0]])

    image0 = dataset.read_image(dataset.image_path[dataset.image_timestamps[0]])
    print("image0 shape: ", image0.shape)
