import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import numpy as np
import caffe, cv2, os, time
import sys
import argparse

"""
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
"""
CLASSES = ('__background__', 'car')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
        'detrac': ('DETRAC',
                    'detrac_faster_rcnn_final.caffemodel')}



def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.putText(im, class_name, (bbox[0], bbox[1]), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

def demo(net, im):
    scores, boxes = im_detect(net, im)
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind = cls_ind + 1
        cls_boxes = boxes[:, 4 * cls_ind: 4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    parser.add_argument('--inputDir', type=str, help='Video input dir', required=True)
    parser.add_argument('--outputDir', type=str, help='Video ouput dir', required=True)

    args = parser.parse_args()

    return args



if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True
    args = parse_args()
    #prototxt = os.path.join('model', NETS[args.demo_net][0],
                    #        'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    #caffemodel = os.path.join(, 'faster_rcnn_models',
                     #         NETS[args.demo_net][1])
    prototxt = '/home/dattt/SELab/py-faster-rcnn/models/detrac/faster_rcnn_alt_opt/faster_rcnn_test.pt'
    caffemodel = '/home/dattt/SELab/py-faster-rcnn/data/DETRAC/detrac_faster_rcnn_final.caffemodel'
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    files = os.listdir(args.inputDir)

    for file_name in files:
        video = cv2.VideoCapture('%s/%s' % (args.inputDir, file_name))
        out = cv2.VideoWriter('%s/result_%s' % (args.outputDir, file_name), cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (1920, 1080))
        print 'Process Video %s/%s' % (args.inputDir, file_name)
        duration = time.time()
        while (video.isOpened()):
            ret, im = video.read()
            if (ret == False):
                break

            demo(net, im)
            out.write(im)

        duration = time.time() - duration

        out.release()
        video.release()
        print 'Finish Video %s/%s takes' % (args.inputDir, file_name), duration

