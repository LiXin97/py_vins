from py_vins.utils import io
from py_vins.utils.config import Config
from py_vins.feature_tracker.lk_tracker import LKTracker
from py_vins.feature_tracker.loftr_trackr import LoFTRTracker
import cv2


if __name__ == "__main__":
    config = Config()
    lk_tracker = LoFTRTracker(config)
    lk_tracker = LKTracker(config)
    dataset = io.EuRoC(config.dataset_dir, config.seq_name)

    print("start tracking")
    save_i = 0
    for timestamp in dataset.image_timestamps[1500:]:
        image = dataset.read_image_from_timestamp(timestamp)
        # image = cv2.resize(image, (320, 240))
        if lk_tracker.process_image(image, timestamp):
            show_image = lk_tracker.show_current_frame()
            # cv2.imshow("show_image", show_image)
            # cv2.waitKey(1)
            
            import os
            cv2.imwrite(
                os.path.join(
                    "/home/xin/Documents/py_vins/test_save", str(timestamp) + ".jpg"
                ),
                show_image,
            )
            save_i += 1
        if save_i > 3:
            break

    pass
