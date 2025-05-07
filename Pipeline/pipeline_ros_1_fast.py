#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import torch
import numpy as np
import argparse
from pathlib import Path
from lib.models import model_factory
from configs import cfg_factory
import lib.transform_cv2 as T
import yaml
from datetime import datetime

class BiSeNetV2SegNode:
    def __init__(self, model_path: str, output_dir: str, ros_publish: bool):
        rospy.init_node('bisenetv2_seg_node', anonymous=True)
        self.bridge = CvBridge()

        # Load ontology
        with open('/media/df/data/RELLIS/ontology.yaml', 'r') as stream:
            ontology = yaml.safe_load(stream)
            self.labels = ontology[0]
            pal_dict = ontology[1]
        
        self.pal = np.zeros((max(pal_dict.keys()) + 1, 3), dtype=np.uint8)
        for i, color in pal_dict.items():
            self.pal[i] = color

        self.to_tensor = T.ToTensor(
            mean=(0.3257, 0.3690, 0.3223),
            std=(0.2112, 0.2148, 0.2115),
        )

        # Load segmentation model
        cfg = cfg_factory['bisenetv2']
        self.net = model_factory 
        self.net.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.net.eval()
        self.net.cuda()
        self.net.half()
        torch.set_grad_enabled(False)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ros_publish = ros_publish
        
        self.subscription = rospy.Subscriber(
            '/zed/zed_node/rgb/image_rect_color',
            Image,
            self.listener_callback,
            queue_size=10
        )

        if self.ros_publish:
            rospy.loginfo("Publishing segmented images to ROS topic")
            self.publisher = rospy.Publisher('/bisenetv2/segmented_image', Image, queue_size=10)
        else:
            self.publisher = None

        rospy.loginfo(f"BiSeNetV2 segmentation node initialized. Saving to {self.output_dir}")
        self.start_time = 0
        self.frame_count = 0

    def listener_callback(self, msg: Image):
        try:
            if self.start_time == 0:
                self.start_time = datetime.now()
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            img_rgb = img_bgr[:, :, ::-1]  # BGR to RGB
            im_tensor = self.to_tensor(dict(im=img_rgb, lb=None))['im'].unsqueeze(0).cuda().half()

            pred = self.net(im_tensor)[0].argmax(dim=1).squeeze().cpu().numpy()
            color_pred = self.pal[pred]

            if self.publisher:
                mask_msg = self.bridge.cv2_to_imgmsg(color_pred, encoding='bgr8')
                mask_msg.header.stamp = msg.header.stamp
                self.publisher.publish(mask_msg)
                fps = self.frame_count / (datetime.now() - self.start_time).total_seconds()
                rospy.loginfo(f"Published segmented image to ROS topic -- Average FPS: {fps:.2f}")
            else:
                ros_time = msg.header.stamp
                timestamp = f"{ros_time.secs}_{ros_time.nsecs:09d}"
                save_path = self.output_dir / f"{timestamp}_seg.png"
                cv2.imwrite(str(save_path), color_pred)
                rospy.loginfo(f"Saved segmented frame: {save_path} -- Average FPS: {self.frame_count / (datetime.now() - self.start_time).total_seconds():.2f}")
            self.frame_count += 1
        except Exception as e:
            rospy.logerr(f"Failed to process image: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='/home/df/OFFSEG-AGX/Models/BiSeNet_RUGD/model_final.pth', help='Path to BiSeNetV2 .pth model file')
    parser.add_argument('--run-number', required=True, help='Run number for saving images')
    parser.add_argument('--output-root', default='/media/df/data/', help='Directory to save output images')
    parser.add_argument('--ros-publish', action='store_true', help='Publish segmented images to ROS topic')
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_root, args.run_number)

    node = BiSeNetV2SegNode(args.model_path, args.output_dir, args.ros_publish)
    rospy.spin()

if __name__ == '__main__':
    main()

