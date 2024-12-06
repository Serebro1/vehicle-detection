import datetime
import argparse

import cv2 as cv
import numpy as np

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode',
                        help='Mode (\'image\', \'video\', \'imgproc\')',
                        type=str,
                        dest='mode',
                        default='image')
    parser.add_argument('-i', '--image',
                        help='Path to an image',
                        type=str,
                        dest='image_path')
    parser.add_argument('-oi', '--output-img',
                        help='Output image name',
                        type=str,
                        dest='output_image',
                        default='output.jpg')
    parser.add_argument('-v', '--video',
                        help='Path to a video file',
                        type=str,
                        dest='video_path')
    parser.add_argument('-ov', '--output-video',
                        help='Output video name',
                        type=str,
                        dest='output_video',
                        default=f"test{datetime.date.today()}.mp4"
                        )
    args = parser.parse_args()
    return args

def highgui_image_samples(image_path, output_image):
    if image_path is None:
        raise ValueError('Empty path to the image')

    # Загрузка изображения
    image = cv.imread(image_path)
    height, width, nchannels = image.shape

    # Отрисовка примитивов на изображении
    cv.line(image, (0, 0), (width-1, height-1), (0, 255, 0), 5)
    cv.rectangle(image, (379, 216), (584, 423), (255, 0, 0), 5) # BGR
    cv.circle(image, (279, 292), 100, (0, 0, 255), 5) # BGR
    cv.putText(image, 'APPLE', (384, 235), cv.FONT_HERSHEY_COMPLEX,
               0.5, (255, 0, 0), 1, cv.LINE_AA)

    # Выделение и изменение ROI
    roi_x = 238
    roi_y = 238
    roi_width = roi_height = 100
    image_roi = image[roi_x:roi_x+roi_width, roi_y:roi_y+roi_height]
    image_roi_gray = np.zeros((roi_height, roi_width), np.uint8)
    cv.cvtColor(image_roi, cv.COLOR_BGR2GRAY, image_roi_gray)
    image[roi_x:roi_x+roi_width, roi_y:roi_y+roi_height, 0] = image_roi_gray
    image[roi_x:roi_x+roi_width, roi_y:roi_y+roi_height, 1] = image_roi_gray
    image[roi_x:roi_x+roi_width, roi_y:roi_y+roi_height, 2] = image_roi_gray

    # Отображение изображения
    cv.imshow('Init image', image)
    cv.waitKey(0)
    
    # Сохранение изображения
    cv.imwrite(output_image, image)

    # Освобождение ресурсов для последующей работы    
    cv.destroyAllWindows()


def highgui_video_samples(video_path, output_video):
    if video_path is not None:
        # Создание объекта для работы с видео
        capt = cv.VideoCapture(video_path)
        # Modify the frame resolution
        frame_width = int(capt.get(3))
        frame_height = int(capt.get(4))
        # Create a video file in a given directory
        out = cv.VideoWriter(f'{output_video}', cv.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (frame_width,frame_height))
    else:
        # Создание объекта для захвата видео с камеры
        capt = cv.VideoCapture(0)

    # Проверка, отрыт ли видеопоток
    if not capt.isOpened():
        raise ValueError('Path of the video file is incorrect '
                         'or camera is unavailable')

    while True:
    # Capture each frame of the video
        ret, frame = capt.read()
        if ret:    
            img_bgr = np.copy(frame)
            # Apply a gaussian blur to the frames
            frame = cv.GaussianBlur(frame, (5, 5), 0)
            
            # Apply dilation to the frames
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            # Apply laplacian edge detection to convert the frames to binary format
            frame = cv.Laplacian(src=frame, ddepth=cv.CV_8U, ksize=3)

            # Load the pre-trained model caar.xml into the classifier
            car_cascade = cv.CascadeClassifier('car.xml')
            cars = car_cascade.detectMultiScale(gray, 1.1, 1)
            # Read the list of rectangles to draw rectangle boundaries 
            # around the cars in each frame
            for (x, y, w, h) in cars:
                cv.rectangle(img_bgr, (x,y), (x+w,y+h), (0,0,255), 2)


            # Display video 
            
            cv.imshow("frame", img_bgr)

            # Write video frames to the file
            out.write(img_bgr)

            # Press Q on the keyboard to exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break
    
    capt.release()
    # Освобождение ресурсов для последующей работы
    cv.destroyAllWindows()


def imgproc_samples(image_path):
    if image_path is None:
        raise ValueError('Empty path to the image')
    # Загрузка изображения
    src_image = cv.imread(image_path)
    
    # Преобразование в другие цветовые пространства
    gray_dst_image = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray image', gray_dst_image)
    cv.waitKey(0)
    
    # Размытие с ядром kSize - усреднение по окрестности
    blurred_image = cv.blur(src_image, (5, 5))
    cv.imshow('Blurred image', blurred_image)
    cv.waitKey(0)
    
    # Бинаризация изображения
    t = 190
    max_value = 255
    _, thresh_image = cv.threshold(gray_dst_image, t, max_value, cv.THRESH_BINARY)
    cv.imshow('Thresholded image', thresh_image)
    cv.waitKey(0)
    
    # Дилатация от бинарного изображения
    dilatation_shape = cv.MORPH_RECT
    dilatation_size = 1
    kernel = cv.getStructuringElement(dilatation_shape,
                                      (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                      (dilatation_size, dilatation_size))
    dilate_dst_image = cv.dilate(thresh_image, kernel)
    cv.imshow('Dilatation output', dilate_dst_image)
    cv.waitKey(0)
    
    # Детектирование краев объектов с помощью детектора Канни
    edges = cv.Canny(image=blurred_image, threshold1=10, threshold2=190)
    cv.imshow('Canny edge detection', edges)
    cv.waitKey(0)
    
    # Освобождение ресурсов для последующей работы
    cv.destroyAllWindows()

def main():
    args = cli_argument_parser()
    
    if args.mode == 'image':
        highgui_image_samples(args.image_path, args.output_image)
    elif args.mode == 'video':
        highgui_video_samples(args.video_path, args.output_video)
    elif args.mode == 'imgproc':
        imgproc_samples(args.image_path)
    else:
        raise 'Unsupported \'mode\' value'

if __name__ == "__main__":
    main()