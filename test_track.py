from ultralytics import YOLO
import cv2
import numpy as np
from boxmot import BotSort, ByteTrack, StrongSort
from pathlib import Path
detector = YOLO("yolo11n.pt")
tracker = ByteTrack(
    track_thresh=0.1
    # reid_weights=Path("osnet_x0_25_msmt17.pt"),
    # device="cpu",
    # half=False,
    # per_class=True,

)
image = cv2.imread("test_track.jpg")
res = detector.predict(image)
boxes_for_tracker = []
for box in res[0].boxes:
    xyxy = box.xyxy.cpu().numpy().tolist()[0]
    xyxy.append(box.conf.item())
    xyxy.append(int(box.cls.item()))
    
    boxes_for_tracker.append(xyxy)
boxes_for_tracker = np.array(boxes_for_tracker)
print(len(boxes_for_tracker))
tracked = tracker.update(boxes_for_tracker, image)
# tracker.plot_results(image, show_trajectories=True)

print(len(tracked))
# cv2.imshow("test", image)
# cv2.waitKey(0)
    
# print(res[0].boxes)