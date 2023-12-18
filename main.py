import argparse
from YOLOv8_all import run_RGB_det,run_RGB_seg,run_RGB_pose

def parse_args():
    description = "Input the mode and device"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mode", default=".det", type=str, help='Select [det, seg, pose] mode.')
    parser.add_argument("--cam_device", default=0, type=int, help='Camera id.')
    parser.add_argument("--video_path", type=str, help='Video path.')

    args = parser.parse_args()
    return args

def run(args):
    if args.video_path==None:
        in_info = args.cam_device
    else:
        in_info = args.video_path
    if args.mode == 'det':
        run_RGB_det(in_info)
    elif args.mode == 'seg':
        run_RGB_seg(in_info)
    elif args.mode == 'pose':
        run_RGB_pose(in_info)
    else:
        print(f"Invalid mode: {args.mode}")

if __name__ == '__main__':

    args = parse_args()
