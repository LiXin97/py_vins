from py_vins.utils import io
from py_vins.utils.config import Config
from py_vins.feature_tracker.base_tracker import LKTracker
import cv2


if __name__ == "__main__":
    config = Config()
    lk_tracker = LKTracker(config)
    dataset = io.EuRoC(config.dataset_dir, config.seq_name)

    for timestamp in dataset.image_timestamps[1500:]:
        image = dataset.read_image_from_timestamp(timestamp)
        if lk_tracker.process_image(image, timestamp):
            show_image = lk_tracker.show_current_frame()
            cv2.imshow("show_image", show_image)
            cv2.waitKey(1)

    pass


# ffmpeg -i input.webm -pix_fmt rgb24 output.gif