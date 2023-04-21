from mtcnn import MTCNN
from PIL import Image
import os, shutil
from os import listdir
from os.path import isdir
from numpy import asarray
import random

detector = MTCNN()

folder = './dataset'

your_name = input("Your name is: ")
img_counter = int(input("How many photos do you want to take?: "))

try:
    face_path = "./dataset/train/" + your_name + "/"
    photo_path = "./dataset/photos/" + your_name + "/"
    test_path = "./dataset/test/" + your_name + "/"
    os.makedirs(face_path)
    os.makedirs(photo_path)
    os.makedirs(test_path)

except:
    print("Some of the directories already exist. You need to delete dataset directories")


def extract_face(file, size=(160, 160)):
    # load a image
    img = Image.open(file)
    img = img.convert('RGB')
    array = asarray(img)

    result = detector.detect_faces(array)

    # load the position from face
    x1, y1, width, height = result[0]['box']
    x2, y2 = x1 + width, y1 + height

    # get face from point position
    face = array[y1:y2, x1:x2]

    img_face = Image.fromarray(face)
    img_face = img_face.resize(size)

    return img_face


def flip_image_FLIP_LEFT_RIGHT(img):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def flip_image_FLIP_TOP_BOTTOM(img):
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


import cv2

camera = cv2.VideoCapture(0)
k = 0
while True:
    ret, frame = camera.read()
    cv2.imshow("test", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    elif k == img_counter:
        break

    img_name = "opencv_frame_{}.png".format(k)
    cv2.imwrite(photo_path + img_name, frame)

    try:
        face = extract_face(photo_path + img_name)

        flip1 = flip_image_FLIP_TOP_BOTTOM(face)
        flip2 = flip_image_FLIP_LEFT_RIGHT(face)

        face.save(face_path + img_name, "JPEG", quality=100, optimize=True, progressive=True)
        flip1.save(face_path + "flip-1" + img_name, "JPEG", quality=100, optimize=True, progressive=True)
        flip2.save(face_path + "flip-2" + img_name, "JPEG", quality=100, optimize=True, progressive=True)
        k += 1
    except:
        print("No faces were identified")

camera.release()
cv2.destroyAllWindows()



train_folder = os.path.join('dataset', 'train', your_name)
test_folder = os.path.join('dataset', 'test', your_name)

# Get the list of image filenames in the train folder
image_filenames = os.listdir(train_folder)

# Calculate the number of images to move to the test folder
num_images_to_move = int(len(image_filenames) * 0.2)

# Randomly select which images to move
images_to_move = random.sample(image_filenames, num_images_to_move)

# Move the selected images from the train folder to the test folder
for image_filename in images_to_move:
    src_path = os.path.join(train_folder, image_filename)
    dst_path = os.path.join(test_folder, image_filename)
    shutil.move(src_path, dst_path)
