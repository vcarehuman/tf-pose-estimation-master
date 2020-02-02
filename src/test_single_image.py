from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize

from model import yolov3

import argparse
import logging
import time
import ast

import common
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose
import matplotlib.pyplot as plt

class HumanSkeleton:
    partinfo = [{}]
    def __init__(self, skeletonId=None):
        self.skeletonId = skeletonId
        
        
        

def get_heatMapPoints(humans,x0, y0, x1, y1,width_ori, height_ori):
    points = []
    for human in humans:
#        print (hu0man.body_parts[1].x)
        for i in human.body_parts:
            part = human.body_parts[i]
            x = int( part.x * width_ori)
            y = int(part.y * height_ori )
            center = (x , y , part.get_part_name())
#            print('width = '+ str(width_ori) + ' height = '+ str(height_ori) + 'x0 ='+ str(x0) + 'y0 ='+ str(y0) + 'x1 = '+ str(x1) + 'y1 = '+str(y1) + 'x = '+str(x) + ', y ='+  str(y) +' body part='+ str(part.get_part_name()))
            points.append (center)
    return points


parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("--input_image", type=str, default="team3.jpg",
                    help="The path of the input image.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=False,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
                    help="The path of the weights to restore.")

parser.add_argument('--resolution', type=str, default='416x416', help='network input resolution. default=416x416')
parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
args = parser.parse_args()
scales = ast.literal_eval(args.scales)



args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

img_ori = cv2.imread(args.input_image)
img_ori = cv2.resize(img_ori, (416, 416))   
height_ori, width_ori = img_ori.shape[:2]
#if args.letterbox_resize:
#    img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
#else:
#    img = cv2.resize(img_ori, tuple(args.new_size))

img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
img = np.asarray(img, np.float32)
img = img[np.newaxis, :] / 255.

    
    
w, h = model_wh(args.resolution)
e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

# estimate human poses from a single image !
image = common.read_imgfile(args.input_image, None, None)
# image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
t = time.time()
humans = e.inference(image, scales=scales)
print(len(humans))



#print("print body")
#print(str(body_part.get_part_name()))
#print(str(body_part.x))
#print(str(body_part.y))

                


elapsed = time.time() - t

#logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

tf.reset_default_graph()
with tf.Session() as sess:
    input_data = tf.placeholder(tf.float32, [1,416, 416, 3], name='input_data')
    yolo_model = yolov3(args.num_class, args.anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)

    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)

    boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

    # rescale the coordinates to the original image
    if args.letterbox_resize:
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
    else:
        boxes_[:, [0, 2]] *= (width_ori/float(args.new_size[0]))
        boxes_[:, [1, 3]] *= (height_ori/float(args.new_size[1]))

    print("box coords:")
    print(boxes_)
    print('*' * 30)
    print("scores:")
    print(scores_)
    print('*' * 30)
    print("labels:")
    print(labels_)
    humanSkeletons = []
    count = 0
    for i in range(len(boxes_)):
        print(args.classes[labels_[i]])
        if args.classes[labels_[i]] != 'person':
            continue
        height_ori, width_ori = img_ori.shape[:2]
        x0, y0, x1, y1 = boxes_[i]
        plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=(255, 255, 0))
        humanSkeletons.append(HumanSkeleton(i))
        points = get_heatMapPoints(humans,x0, y0, x1, y1, width_ori, height_ori)
        for point in points:
            center =  (point[0],point[1])
            if (int(point[0]) >= x0) and (int(point[0]) <= x1) and (int(point[1]) >= int(y0)) and (int(point[1]) <= int(y1)):
                cv2.circle(img_ori, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)
                humanSkeletons[count].partinfo.append( (point[2],center))
        count = count+1        
                
#                
#            cv2.putText(img_ori , 'x0='+str(x0)+',y0='+str(y0)  , ( x0, y0) ,  cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 1, cv2.LINE_AA) 
#            cv2.putText(img_ori , 'x1='+str(x1)+',y1='+str(y1)  , ( x1, y1) ,  cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 1, cv2.LINE_AA) 
#            cv2.putText(img_ori , 'x='+str(point[0])+',y='+str(point[1])  , ( point[0], point[1]) ,  cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 1, cv2.LINE_AA) 
            
            
        
#        body_part = humans[1].body_parts[1]
#        print(body_part.get_part_name())
#        center = (int(body_part.x * width_ori), int(body_part.y * height_ori ))
#        cv2.circle(img_ori, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)
#        print(center)
##        cv2.putText(img_ori, body_part.get_part_name(), center,cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1 )
#        body_part = humans[1].body_parts[2]
#        print(body_part.get_part_name())
#        center = (int(body_part.x * width_ori), int(body_part.y * height_ori ))
#        cv2.circle(img_ori, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)
#        
#        print(center)
    print('total person by yolo' + str(count))
    print('total by part aff' +  str(len(humans)))
#    print(humanSkeletons[0].partinfo)                
    winname = 'Detection result'
    cv2.namedWindow(winname)            
    cv2.moveWindow(winname, 40,30) 
    cv2.imshow(winname, img_ori)
    cv2.imwrite('detection_result.jpg', img_ori)
    cv2.waitKey(0)
