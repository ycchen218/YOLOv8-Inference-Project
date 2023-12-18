import numpy as np
import cv2
from ultralytics import YOLO
from sort import Sort
from yolo_segmentation import YOLOSegmentation
def detect_objects(webcam_img,model,tracker):
    results = model(webcam_img)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    confs = np.array(result.boxes.conf.cpu(), dtype="int")

    dets_rgb = []
    for xy, cls,conf in zip(bboxes, classes, confs):
        if cls == 0:
            (x1, y1, x2, y2) = xy
            dets_rgb.append([x1, y1, x2, y2,conf])
            # cv2.rectangle(webcam_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    dets_rgb = np.array(dets_rgb)
    boxes_ids = tracker.update(dets_rgb)
    for box_id in boxes_ids:
        x, y, x2, y2, id = map(int, box_id)
        cv2.putText(webcam_img, f"ID: {id}", (x, y -10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 3)
        cv2.rectangle(webcam_img, (x, y), (x2, y2), (0, 255, 0), 2)

    return webcam_img

def seg_objects(webcam_img,model,alpha=0.4):
    seg_rgb = []
    overlay = webcam_img.copy()
    bboxes, classes, segmentations, scores,names = model.detect(webcam_img)
    for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
        (x, y, x2, y2) = bbox
        if class_id == 0:
            cv2.rectangle(webcam_img, (x, y), (x2, y2), (255, 0, 0), 2)
            cv2.polylines(webcam_img, [seg], True, (0, 0, 255), 4)
            cv2.fillPoly(overlay, [seg], (0, 0, 255))
            cv2.putText(webcam_img, names[class_id], (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 3)
            seg_rgb.append(seg)
    webcam_img = cv2.addWeighted(overlay, alpha, webcam_img, 1 - alpha, 0)
    return webcam_img

def pos_objects(webcam_img,model, kpt_color,skeleton,limb_color):
    results = model(webcam_img)
    result = results[0]
    keypoints = result.keypoints.xy.cpu().numpy()
    for kpt in reversed(keypoints):
        for i, k in enumerate(kpt):
            color_k = [int(x) for x in kpt_color[i]]
            x_coord, y_coord = k[0], k[1]
            if x_coord % webcam_img.shape[1] != 0 and y_coord % webcam_img.shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < 0.5:
                        continue
                cv2.circle(webcam_img,(int(x_coord), int(y_coord)),5, color_k, -1, lineType=cv2.LINE_AA)
        if kpt is not None:
            if kpt.shape[0]!=0:
                for i, sk in enumerate(skeleton):
                    pos1 = (int(kpt[(sk[0] - 1), 0]), int(kpt[(sk[0] - 1), 1]))
                    pos2 = (int(kpt[(sk[1] - 1), 0]), int(kpt[(sk[1] - 1), 1]))
                    if kpt.shape[-1] == 3:
                        conf1 = kpt[(sk[0] - 1), 2]
                        conf2 = kpt[(sk[1] - 1), 2]
                        if conf1 < 0.5 or conf2 < 0.5:
                            continue
                    if pos1[0] % webcam_img.shape[1] == 0 or pos1[1] % webcam_img.shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                        continue
                    if pos2[0] % webcam_img.shape[1] == 0 or pos2[1] % webcam_img.shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                        continue
                    cv2.line(webcam_img,
                             pos1, pos2,
                             [int(x) for x in limb_color[i]],
                             thickness=2, lineType=cv2.LINE_AA)
    return webcam_img

def run_RGB_det(in_info):

    model = YOLO("yolov8n.pt")
    tracker = Sort()
    cap = cv2.VideoCapture(in_info)

    while True:
        ret, webcam_img = cap.read()
        webcam_img = detect_objects(webcam_img,model,tracker)
        cv2.imshow('IR Image', webcam_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def run_RGB_seg(in_info):
    model = YOLOSegmentation("yolov8n-seg.pt")
    cap = cv2.VideoCapture(in_info)

    while True:

        ret, webcam_img = cap.read()


        webcam_img = seg_objects(webcam_img,model)

        cv2.imshow('IR Image', webcam_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def run_RGB_pose(in_info):

    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                           [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                           [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                           [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                          dtype=np.uint8)
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(in_info)

    while True:

        ret, webcam_img = cap.read()

        webcam_img = pos_objects(webcam_img, model,kpt_color,skeleton,limb_color)

        cv2.imshow('IR Image', webcam_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


