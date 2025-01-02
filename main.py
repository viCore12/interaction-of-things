import os
import sys
sys.path.append("upt")
sys.path.append("upt/detr")
import os.path as osp
import torch
import pocket
import warnings
import cv2
import copy
import argparse
from argparse import Namespace
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as peff
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import DataFactory
from upt import build_detector

# Points of leaving zone
pts = np.array([
    [205, 183], [385, 145], [433, 185], [255, 254]
], np.int32)

contours = np.array([pts])
# Set stores ids that have interaction with staffs
interact_tracking = set()

def main(args):
    dataset = DataFactory(name="vcoco", partition="test", data_root="upt/vcoco")
    conversion = dataset.dataset.object_to_verb if args.dataset == 'hicodet' \
        else list(dataset.dataset.object_to_action.values())
    args.num_classes = 117 if args.dataset == 'hicodet' else 24
    args.num_classes = 2
    actions = dataset.dataset.verbs if args.dataset == 'hicodet' else \
        dataset.dataset.actions
    # Load UPT model
    upt = build_detector(args, conversion)
    print(f"=> Continue from saved checkpoint {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu')
    upt.load_state_dict(checkpoint['model_state_dict'])
    upt.eval()
    # Video capture
    cap = cv2.VideoCapture(args.video_path)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ori_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ori_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter("demo.mp4", cv2.VideoWriter_fourcc(*"MP4V"), 10, (ori_width, ori_height))
    with tqdm(total=total_frame, desc="Inferencing") as bar:

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Read image
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                image_tensor, _ = dataset.transforms(pil_image, None)
                non_normalized_image_tensor, _ = dataset.transforms_without_normalize(pil_image, None)
                # Convert to ndarray
                non_normalized_image_tensor = np.array(non_normalized_image_tensor)
                # Convert to BGR
                non_normalized_image_tensor = cv2.cvtColor(non_normalized_image_tensor, cv2.COLOR_RGB2BGR)
                detections ,upt_output = upt.custom_inference([non_normalized_image_tensor], [image_tensor])        
                height_from_upt, width_from_upt = upt_output[0]["size"]
                # Scale factor to convert from HOI output size to original size
                scale_fct = torch.as_tensor([
                    ori_width / width_from_upt, ori_height / height_from_upt, ori_width / width_from_upt, ori_height / height_from_upt

                ]).unsqueeze(0)
                detections = detections[0]
                detections["boxes"] = detections["boxes"] * scale_fct
                cv2.polylines(frame, [pts], True, (255, 0, 0), 2, cv2.LINE_AA)

                for box, sc, lb, id in zip(detections["boxes"], detections["scores"], detections["labels"], detections["ids"]): 
                    box = box.cpu().numpy().tolist()
                    box = [int(x) for x in box]
                    id = id.item()
                    # If id has interacted with staff
                    if id in interact_tracking:
                        center_box = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                        isLeave = cv2.pointPolygonTest(contours, center_box, False)
                        # If id is leaving
                        if isLeave > 0:
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 4, cv2.LINE_AA)
                            print(f"Person {id} is leaving")
                            cv2.putText(frame, str(id) + " leaving after interaction", (box[0] + 15, box[1] + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (120, 12, 56), 1, cv2.LINE_AA)
                    cv2.putText(frame, str(id), (box[0] + 15, box[1] + 15), cv2.FONT_HERSHEY_PLAIN, 2, (120, 116, 56), 2, cv2.LINE_AA)
                hoi_output = detect_interaction(pil_image, upt_output[0], actions, args.action_score_thresh)
                if len(hoi_output) > 0:
                    for hoi in hoi_output:
                        boxes_h = hoi["boxes_h"].cpu().numpy()
                        boxes_o = hoi["boxes_o"].cpu().numpy()
                        actions_ = hoi["actions"]
                        ids = hoi["ids"].cpu().numpy().flatten()
                        # Add id to tracking after inteaction
                        if actions_ == 1:
                            interact_tracking.add(ids[0])
                        for i in range(len(boxes_h)):
                            box_h = boxes_h[i]
                            box_o = boxes_o[i]
                            box_h = [int(x) for x in box_h]
                            box_o = [int(x) for x in box_o]
                            center_h = (int((box_h[0] + box_h[2]) / 2), int((box_h[1] + box_h[3]) / 2))
                            center_o = (int((box_o[0] + box_o[2]) / 2), int((box_o[1] + box_o[3]) / 2))
                            cv2.rectangle(frame, (box_h[0], box_h[1]), (box_h[2], box_h[3]), (0, 0, 0), 4, cv2.LINE_AA)
                            cv2.rectangle(frame, (box_o[0], box_o[1]), (box_o[2], box_o[3]), (12, 255, 93), 4, cv2.LINE_AA)
                            cv2.line(frame, center_h, center_o, (5, 5, 5), 2, cv2.LINE_AA)
                            cv2.putText(frame, str(actions[actions_]), center_o, cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                # cv2.imshow("test", frame)
                # cv2.waitKey(1)
                video_writer.write(frame)
                bar.update(1)
            else:
                print("Done!")
                break
        
def detect_interaction(image, output, actions, thresh=0.98):
    ow, oh = image.size
    h, w = output['size']
    scale_fct = torch.as_tensor([
        ow / w, oh / h, ow / w, oh / h
    ]).unsqueeze(0)
    boxes = output['boxes'] * scale_fct
    # Find the number of human and object instances
    nh = len(output['pairing'][0].unique()); no = len(boxes)
    scores = output['scores']
    pred = output['labels']
    ids = output["ids"]
    hoi_output = []    
    for action in [1]:
        keep = torch.nonzero(torch.logical_and(scores >= thresh, pred == action)).squeeze(1)        
        id = ids[output["pairing"][0][keep]]
        bx_h, bx_o = boxes[output['pairing']].unbind(0)
        class_ = actions[action]
        if len(bx_h[keep]) > 0:
            temp_box = {
                "boxes_h": bx_h[keep],
                "boxes_o": bx_o[keep],
                "ids": ids[output["pairing"][0][keep]],
                "actions": action
            }
            hoi_output.append(temp_box)
    return hoi_output


if __name__ == "__main__":
    default_args=Namespace(backbone='resnet50', dilation=False, position_embedding='sine', repr_dim=512, hidden_dim=256, enc_layers=6, dec_layers=6, dim_feedforward=2048, dropout=0.1, nheads=8, num_queries=100, pre_norm=False, aux_loss=True, set_cost_class=1, set_cost_bbox=5, set_cost_giou=2, bbox_loss_coef=5, giou_loss_coef=2, eos_coef=0.1, alpha=0.5, gamma=0.2, dataset='vcoco', partition='test', data_root='upt/vcoco', pretrained='', box_score_thresh=0.2, fg_iou_thresh=0.5, min_instances=3, max_instances=15,  index=0, action=None)
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default="upt.pt", type=str, help="UPT checkpoint path")
    parser.add_argument('--device', default="cpu", type=str, help="device to inference on")
    parser.add_argument('--human-idx', default=1, type=int, help="human index of detector")
    parser.add_argument('--action-score-thresh', default=0.95, type=float, help="threshold for action")
    parser.add_argument('--num-classes', default=2, type=int, 
    help="number of actions")
    parser.add_argument('--video-path', default="merged.mp4", type=str, help="video path to demo")
    customized_args = parser.parse_args()
    upt_args = Namespace(**vars(default_args), **vars(customized_args))
    main(upt_args)

