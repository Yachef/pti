import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import glob

def detect_img(yolo):
    """folder = "my_data/images"
    images = glob.glob(folder+"/*.png")
    fail=0;
    counter = 0;
    for img in images:
        if counter <538:
    #        img = input('Input image filename:')
            try:
                image = Image.open(img)
                row_image = Image.open(img)
                image_name = image.filename.replace(folder+"\\","")
            except:
                print('Open Error! Try again!')
                continue
            else:
                try:
                    image_and_box = yolo.detect_image(image)
                    r_image = image_and_box['image']
                    r_box = image_and_box['box']
                    print("CECI EST BOX {}".format(r_box))
                    r_image_cropped = row_image.crop((r_box[1],r_box[0],r_box[3],r_box[2]))
                    #r_image_cropped = image.crop((r_box[1], r_box[0], r_box[3], r_box[2]))
                    r_image_cropped_square = make_square(r_image_cropped)
                    r_image.save("results_images/results_"+image_name)
                    r_image_cropped_square.save("cropped_images/"+image_name)
                except :
                    print('Box error !')
                    fail = fail+1
        counter = counter + 1
    yolo.close_session()
    print(fail);"""
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
            row_image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
            image_and_box = yolo.detect_image(image)
            r_image = image_and_box['image']
            r_box = image_and_box['box']
            print("CECI EST BOX {}".format(r_box))
            r_image_cropped = row_image.crop((r_box[1], r_box[0], r_box[3], r_box[2]))
            r_image_cropped_square = make_square(r_image_cropped)
            r_image.save("results_images/results_res.png")
            r_image_cropped_square.save("cropped_images/res.png")
    yolo.close_session()
FLAGS = None

def make_square(im, min_size=256, fill_color=(0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
