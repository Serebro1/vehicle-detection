import datetime
import argparse
import os
from PIL import Image

import cv2 as cv
import numpy as np

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--mode',
                        help='Mode (\'image\', \'video\', \'img-pack\')',
                        type=str,
                        dest='mode',
                        default='image')
    parser.add_argument('-l', '--layout',
                        help='Path to a dir of layout',
                        type=str,
                        dest='dir_layout_path',
                        default=None)
    parser.add_argument('--model',
                        help='Path to a model',
                        type=str,
                        dest='model_path',
                        default=None)

    parser.add_argument('-v', '--video',
                        help='Path to a video file',
                        type=str,
                        dest='video_path',
                        default='samplevideo.mp4')
    parser.add_argument('-ip', '--image-pack',
                        help='Path to a dir of images',
                        type=str,
                        dest='dir_img_path',
                        default="../DLMini/data/imgs_MOV03478/")
    parser.add_argument('-i', '--image',
                        help='Path to an image',
                        type=str,
                        dest='image_path',
                        default="ILSVRC2012_val_00000023.JPEG")
    args = parser.parse_args()
    return args

def image_show(image_path):
    if image_path is None:
        raise ValueError('Empty path to the image')

   
    image = cv.imread(image_path)
    height, width, nchannels = image.shape

    # возможная обработка изображения

    # Отображение изображения
    cv.imshow(f'Image: {image_path}', image)
    cv.waitKey(0)
    # Освобождение ресурсов для последующей работы    
    cv.destroyAllWindows()


def video_processing(video_path, model_path, layout_dir):
    if video_path is not None:
        capt = cv.VideoCapture(video_path)

    if not capt.isOpened():
        raise ValueError('Path of the video file is incorrect '
                         'or camera is unavailable')
    #возможная обработка видео
    while capt.isOpened():
        ret, frame = capt.read()
        if ret:    
            cv.imshow("Video", frame)
            # Press Q on the keyboard to exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    
    capt.release()
    cv.destroyAllWindows()

def image_pack_processing(image_dir, model_path, layout_dir):
    image_files = [file for file in os.listdir(image_dir) if file.lower().endswith((".jpg", ".jpeg", ".png"))]
    num_of_images = len(image_files)
    if num_of_images == 0:
        raise ValueError("No images found in the specified directory.")
    # возможная обработка изображений
    for img in image_files:
        image = cv.imread(image_dir + img)
        #
        cv.imshow(f'Image pack', image) 
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

def main():
    args = cli_argument_parser()
    
    if args.mode == 'image':
        image_show(args.image_path, args.model_path)
    elif args.mode == 'video':
        video_processing(args.video_path, args.model_path, args.dir_layout_path)
    elif args.mode == "img-pack":
        image_pack_processing(args.dir_img_path, args.model_path, args.dir_layout_path)
    else:
        raise 'Unsupported \'mode\' value'

if __name__ == "__main__":
    main()