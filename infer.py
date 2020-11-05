from lib.config import cfg
from lib.predictor import COCODemo
import cv2
import argparse
import distutils.util
import csv
import os
import shutil
import matplotlib.path as mplPath
import torch
import numpy as np
import datetime
import json


def parse_args():
    """
    Parse command line arguments.

    Args:
    """
    parser = argparse.ArgumentParser(description='evaluation on zensors data')

    parser.add_argument(
        '-cfg', '--config_file', help='the config file of the model',
        default='configs/infer.yaml')

    parser.add_argument(
        '-ut', '--url_txt',
        help='text file containing url of an image each line to be processed', default='')
    parser.add_argument(
        '-ul', '--url_list', nargs='+',
        help='list of urls of images to be processed', default=[])
    parser.add_argument(
        '-ir', '--image_dir',
        help='directory to load images to infer')
    parser.add_argument(
        '-min_size', '--min_test_size',
        help='the minimum size of the test images (default: 800)', type=int, default=800)
    parser.add_argument(
        '-th', '--confidence_threshold',
        help='the confidence threshold of the bounding boxes (default: 0.3)',
        type=float, default=0.6)
    parser.add_argument(
        '-t', '--target',
        help='the objects want to detect, support car and person', default='car'
    )
    parser.add_argument(
        '-v', '--visualize', type=distutils.util.strtobool, default=False)
    parser.add_argument(
        '-vis_color', default='rainbow')


    args = parser.parse_args()
    return args


def main():
    """ main function """
    args = parse_args()

    config_file = args.config_file
    assert config_file

    assert args.url_list or args.url_txt or args.image_dir
    if len(args.url_list) > 0:
        url_list = args.url_list
    elif args.url_txt:
        url_list = list(np.loadtxt(args.url_txt, dtype=str))
    else:
        image_dir = args.image_dir
        url_list = [os.path.join(image_dir, item) for item in os.listdir(image_dir)]
    save_image = True if args.visualize else False

    target = args.target
    vis_color = args.vis_color

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    coco_demo = COCODemo(
        cfg,
        min_image_size=args.min_test_size,
        confidence_threshold=args.confidence_threshold,
    )
    if target == 'person':
        coco_demo.CATEGORIES = ["__background", "person"]

    output_dir = cfg.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(output_dir, now+'.json')

    record_dict = {'model': cfg.MODEL.WEIGHT,
                   'time': now,
                   'results': []}

    for url in url_list:
        if not os.path.exists(url):
            print('Image {} does not exist!'.format(url))
            continue
        img = cv2.imread(url)

        #predictions = coco_demo.compute_prediction(img)
        #top_predictions = coco_demo.select_top_predictions(predictions)
        #print(top_predictions.get_field("keypoints").Keypoints[0])
        try:
        #if 2>1:
            predictions = coco_demo.compute_prediction(img)
            top_predictions = coco_demo.select_top_predictions(predictions)

            scores = top_predictions.get_field("scores")
            labels = top_predictions.get_field("labels")
            boxes = predictions.bbox

            infer_result = {'url': url,
                            'boxes': [],
                            'scores': [],
                            'labels': []}
            for box, score, label in zip(boxes, scores, labels):
                boxpoints = [item for item in box.tolist()]
                infer_result['boxes'].append(boxpoints)
                infer_result['scores'].append(score.item())
                infer_result['labels'].append(label.item())
            record_dict['results'].append(infer_result)
            # visualize the results
            if save_image:
                result = np.copy(img)
                #result = coco_demo.overlay_boxes(result, top_predictions)
                #result = coco_demo.overlay_class_names(result, top_predictions)
                if cfg.MODEL.KEYPOINT_ON:
                     if target == 'person':
                        result = coco_demo.overlay_keypoints_graph(result, top_predictions, target='person')
                     if target == 'car':
                        result = coco_demo.overlay_keypoints_graph(result, top_predictions,vis_color , target='car')
                cv2.imwrite(os.path.join(output_dir, url.split('/')[-1]), result)
                print(os.path.join(output_dir, url.split('/')[-1]))
        except:
            print('Fail to infer for image {}. Skipped.'.format(url))
            continue
    print(now)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(now)
   

    with open(output_path, 'w') as f:
        json.dump(record_dict, f)


if __name__ == '__main__':
    main()
