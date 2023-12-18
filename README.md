# YOLOv8-Inference-Project

## Introduce
This is a project that inference YOLOv8 detection, segmentation, and pose estimate. <br>
The tracker I use in the detection is the SORT method (Kalman Filter + Hungarian Algorithm).
## Requirement
1. python3.8
2. numpy
3. opencv
4. filterpy
5. ultralytics
6. argparse

## Run YOLO in RGB webcam
To run this project you need to main.py as following:
```markdown
python main.py
```
--mode: Select [det, seg, pose] mode. <br>
--cam_device: The webcam id. <br>
--video_path: If you want to use video put the video path here. <br>

### Result
1. Object Detection <br>
![image](https://github.com/ycchen218/YOLOv8-Inference-Project/git_img/yolo_rgb_det.png)
2. Object Segmentation <br>
![image](https://github.com/ycchen218/YOLOv8-Inference-Project/git_img/yolo_rgb_seg.png)
3. Pose Estimate <br>
![image](https://github.com/ycchen218/YOLOv8-Inference-Project/git_img/yolo_rgb_pos.png)
