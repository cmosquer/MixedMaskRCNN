from keras import backend as K
import tensorflow as tf
import os, pickle
import keras_retinanet.models as retinanet_models
from keras.models import load_model, model_from_json
import numpy as np
import sys
import keras
import logging
import cv2
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)

from keras import backend as K
import numpy as np
import matplotlib.cm as cm
from skimage.transform import resize

LINE_SIZE = 5
LIMIT_PERCENTAGE = 0.1
NORMAL = 0
ABNORMAL = 1
LIMIT = 1  # TODO definir si el limite triggerea o no


def addReferenceLines(image, linePoints, scale):
    heart_left_contour, heart_right_contour, heart_y, \
    left_lung_right_contour, right_lung_left_contour, lung_y = linePoints

    if scale:
        H_LEFT_X = int(heart_left_contour[0] * scale[0])
        H_RIGHT_X = int(heart_right_contour[0] * scale[0])
        H_MIDDLE_Y = int(heart_y * scale[1])

        TORAX_LEFT_X = int(left_lung_right_contour[0] * scale[0])
        TORAX_RIGHT_X = int(right_lung_left_contour[0] * scale[0])
        TORAX_MIDDLE_Y = int(lung_y * scale[1])


    else:
        H_LEFT_X = heart_left_contour[0]
        H_MIDDLE_Y = heart_y
        H_RIGHT_X = heart_right_contour[0]
        TORAX_LEFT_X = left_lung_right_contour[0]
        TORAX_RIGHT_X = right_lung_left_contour[0]

    image = make_line(  # Linea horizontal corazon
        (H_LEFT_X, H_MIDDLE_Y),  # Point 1
        (H_RIGHT_X, H_MIDDLE_Y),  # Point 2
        image, lw=5)

    image = make_line(  # Linea horizontal torax
        (TORAX_LEFT_X, TORAX_MIDDLE_Y),  # Point 1
        (TORAX_RIGHT_X, TORAX_MIDDLE_Y),  # Point 2
        image, lw=5)

    image = make_line(  # Pequeña linea vertical de limite corazon
        (H_RIGHT_X, H_MIDDLE_Y - LINE_SIZE),  # Point 1
        (H_RIGHT_X, H_MIDDLE_Y + LINE_SIZE),  # Point 2
        image, lw=8)

    image = make_line(  # Pequeña linea vertical de limite corazon
        (H_LEFT_X, H_MIDDLE_Y - LINE_SIZE),  # Point 1
        (H_LEFT_X, H_MIDDLE_Y + LINE_SIZE),  # Point 2
        image, lw=8)

    image = make_line(  # Pequeña linea vertical de limite torax
        (TORAX_LEFT_X, TORAX_MIDDLE_Y - LINE_SIZE),  # Point 1
        (TORAX_LEFT_X, TORAX_MIDDLE_Y + LINE_SIZE),  # Point 2
        image, lw=8)

    image = make_line(  # Pequeña linea vertical de limite torax
        (TORAX_RIGHT_X, TORAX_MIDDLE_Y - LINE_SIZE),  # Point 1
        (TORAX_RIGHT_X, TORAX_MIDDLE_Y + LINE_SIZE),  # Point 2
        image, lw=8)

    return image


def make_line(pt1, pt2, img, lw=5):
    cv2.line(img, pt1, pt2, (255, 0, 0), lw)
    return img


def overlap_images(heart, left_lung, right_lung):
    heart = change_color(heart, (204, 51, 0))
    left_lung = change_color(left_lung, (51, 153, 102))
    right_lung = change_color(right_lung, (51, 153, 102))

    heart = smooth_image(heart)
    left_lung = smooth_image(left_lung)
    right_lung = smooth_image(right_lung)

    overlap = cv2.addWeighted(heart, 1, left_lung, 1, 0)
    overlap = cv2.addWeighted(overlap, 1, right_lung, 1, 0)

    return overlap


def smooth_image(image):
    kernel = np.ones((5, 5), np.float32) / 25
    image = cv2.filter2D(image, -1, kernel)
    return image


def change_color(image, color):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 168])
    upper_white = np.array([172, 111, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    image[mask > 0] = color
    return image


def clear_mask(image):
    # Converting image to a binary image
    if len(image.shape) > 3:
        image = np.squeeze(image, axis=0)

    if len(image.shape) == 3:
        if image.shape[2] == 1:
            gray = np.float32(np.squeeze(255 * image, axis=-1))

        else:
            gray = cv2.cvtColor(np.float32(255 * image), cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)

    connectivity = 4

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(threshold), connectivity,
                                                                               cv2.CV_32S)

    max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])

    img2 = np.zeros(gray.shape, np.uint8)
    img2[output == max_label] = 128

    return img2


def get_unet_prediction(model, weights_path, radiograph):
    model.load_weights(weights_path)
    return model.predict(radiograph)


def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        gray = image
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    if alpha > 2 or beta < -100:
        auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    else:
        auto_result = image

    return auto_result


def get_contours(image):
    image = clear_mask(image)
    # Detecting external contours in image.
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    c = max(contours, key=cv2.contourArea)
    # Obtain outer coordinates
    left = tuple(c[c[:, :, 0].argmin()][0])
    right = tuple(c[c[:, :, 0].argmax()][0])

    bottom = tuple(c[c[:, :, 1].argmin()][0])
    top = tuple(c[c[:, :, 1].argmax()][0])

    return left, right, bottom, top


def get_cropping_limits(left, right, bottom, top, height, width, delta=0.08):
    xmin = int(left - delta * width)
    xmax = int(right + delta * width)
    ymin = int(bottom - delta * height)
    ymax = int(top + delta * height)
    return (xmin, xmax), (ymin, ymax)


def get_ctr(predicted_img, orig_img, heart, left_lung, right_lung):
    scale = (orig_img.shape[1] / predicted_img.shape[1], orig_img.shape[0] / predicted_img.shape[0])
    heart_left_contour, heart_right_contour, _, _ = get_contours(heart)
    right_lung_left_contour, _, right_lung_bottom_contour, right_lung_top_contour = get_contours(right_lung)
    _, left_lung_right_contour, left_lung_bottom_contour, left_lung_top_contour = get_contours(left_lung)

    ctr = (heart_right_contour[0] - heart_left_contour[0]) / (
            left_lung_right_contour[0] - right_lung_left_contour[0])

    # img = automatic_brightness_and_contrast(img)

    height, width = predicted_img.shape[0], predicted_img.shape[1]  # deberian ser 128 y 128

    cropping_limits = get_cropping_limits(right_lung_left_contour[0],  # limite izq crudo
                                          left_lung_right_contour[0],  # limite derecho crudo
                                          min(right_lung_bottom_contour[1], left_lung_bottom_contour[1]),
                                          # limite inferior crudo
                                          max(right_lung_top_contour[1], left_lung_top_contour[1]),
                                          # limite superior crudo
                                          height, width
                                          )
    heart_y = int((heart_left_contour[1] + heart_right_contour[1]) / 2)
    lung_y = int((left_lung_right_contour[1] + right_lung_left_contour[1]) / 2)

    linePoints = [heart_left_contour, heart_right_contour, heart_y,
                  left_lung_right_contour,
                  right_lung_left_contour, lung_y]

    xmin = int(cropping_limits[0][0] * scale[0])
    xmax = int(cropping_limits[0][1] * scale[0])
    ymin = int(cropping_limits[1][0] * scale[1])
    ymax = int(cropping_limits[1][1] * scale[1])

    scaled_cropping_limits = (xmin, xmax, ymin, ymax)

    heatmap = addReferenceLines(orig_img.copy(), linePoints, scale)
    """
    else:
        # Converting image to a binary image
        if len(heart.shape) > 3:
            heart = np.squeeze(heart,axis=0)
        if len(left_lung.shape) > 3:
            left_lung = np.squeeze(left_lung,axis=0)
        if len(right_lung.shape) > 3:
            right_lung = np.squeeze(right_lung,axis=0)
        if len(heart.shape) == 3:
            if heart.shape[2]==1:
                heart = cv2.cvtColor(np.float32(np.squeeze(255*heart,axis=-1)),cv2.COLOR_GRAY2RGB)
        if len(left_lung.shape) == 3:
            if left_lung.shape[2]==1:
                left_lung = cv2.cvtColor(np.float32(np.squeeze(255*left_lung,axis=-1)),cv2.COLOR_GRAY2RGB)
        if len(right_lung.shape) == 3:
            if right_lung.shape[2]==1:
                right_lung = cv2.cvtColor(np.float32(np.squeeze(255*right_lung,axis=-1)),cv2.COLOR_GRAY2RGB)    

        heatmap = overlap_images(heart, left_lung, right_lung) """

    return ctr, heatmap, scaled_cropping_limits


def get_cpu_session():
    config = tf.ConfigProto(device_count={'CPU':1,'GPU': 0})
    #config.gpu_options.per_process_gpu_memory_fraction = 0.9
    #config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess

def get_gpu_session():
    config = tf.ConfigProto(device_count={'CPU':1,'GPU': 1})
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess

def save_heatmap(savingpath,ready_heatmap,orig_img):
    #print('\n')
    #print(ready_heatmap.shape[0], orig_img.shape[0])
    #print(ready_heatmap.shape[1], orig_img.shape[1])
    assert ready_heatmap.shape[0] == orig_img.shape[0]
    assert ready_heatmap.shape[1] == orig_img.shape[1]
    """
    finalShape = (orig_img.shape[0], orig_img.shape[1], 3)
    newBuild = np.zeros(finalShape, dtype='uint8')
    newBuild[:, :, 0] = orig_img[:, :, 0] + ready_heatmap[:,:,0]
    newBuild[:, :, 1] = orig_img[:, :, 1] + ready_heatmap[:,:,1]
    newBuild[:, :, 2] = orig_img[:, :, 2] + ready_heatmap[:,:,2]
    Image.fromarray(newBuild).save(savingpath)"""
    ##if not retinanet:
    fig, ax = plt.subplots(1, 1, figsize=(40, 40))
    ax.imshow(orig_img, cmap='gray')
    ax.imshow(ready_heatmap, cmap='jet')
    ax.set_axis_off()
    plt.savefig(savingpath, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    ##else:
    ##    cv2.imwrite(savingpath, ready_heatmap)

def set_keras_device(device):
    if device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        sess = get_gpu_session()
    else:
        sess = get_cpu_session()
    reinit_variables(sess)
    return sess

def clear_mask(image):
    # Converting image to a binary image
    if len(image.shape) > 3:
        image = np.squeeze(image,axis=0)

    if len(image.shape) == 3:
        if image.shape[2]==1:
            gray = np.float32(np.squeeze(255*image,axis=-1))

        else:
            gray = cv2.cvtColor(np.float32(255*image), cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)

    connectivity = 4

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(threshold), connectivity, cv2.CV_32S)
    img2 = np.zeros(gray.shape, np.uint8)

    try:
        max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])

        img2[output == max_label] = 128
    except ValueError as e:
        print('empty mask', e)
    return img2

def load_keras_model(modelpath,sess):

    K.tensorflow_backend.set_session(sess)
    with open(modelpath + '/model_config.json') as json_file:
        json_config = json_file.read()
    model = model_from_json(json_config)
    return model

def inference_image_mask(model,img, modelpath,orig_img=None):
    img = automatic_brightness_and_contrast(img) / 255.0
    # Add a layer to let the network to read the radiograph
    img = np.expand_dims(img, axis=0)
    if len(img.shape) < 4:
        img = np.expand_dims(img, axis=-1)
    # Get the prediction of the hearth
    heart_result = get_unet_prediction(model, modelpath + '/Heart-Weights.h5', img)
    # Get the prediction of the left lung
    left_lung_result = get_unet_prediction(model, modelpath + '/Left-Lung-Weights.h5', img)
    # Get the prediction of the right lung
    right_lung_result = get_unet_prediction(model, modelpath + '/Right-Lung-Weights.h5', img)
    # Clean the images

    heart_result = clear_mask(heart_result > 0.5)
    left_lung_result = clear_mask(left_lung_result > 0.5)
    right_lung_result = clear_mask(right_lung_result > 0.5)

    dim = (orig_img.shape[1],orig_img.shape[0])
#    left_lung_result = smooth_image(change_color(resize_image(left_lung_result,dim),(255, 255, 255)))
#    heart_result = smooth_image(change_color(resize_image(heart_result,dim)),(255, 255, 255))
#    right_lung_result = smooth_image(change_color(resize_image(right_lung_result,dim),(255, 255, 255)))
    left_lung_result = smooth_image(resize_image(left_lung_result,dim))
    heart_result = smooth_image(resize_image(heart_result,dim))
    right_lung_result = smooth_image(resize_image(right_lung_result,dim))

    return left_lung_result,heart_result,right_lung_result

def reinit_variables(sess):
    sess.run(tf.global_variables_initializer())


def load_image_keras(imagepath,dim=(256,256),retinanet=None):
    ext = os.path.splitext(imagepath)[-1]
    #assert ext in ['.dcm','.jpg','.png']
    assert os.path.exists(imagepath)

    channels = 3
    imgtype = cv2.IMREAD_COLOR

    if ext in ['.jpg','.png']:
        orig_img = cv2.imread(imagepath, imgtype)

    img = cv2.resize(orig_img, dim)       #
    #img = np.array(img, dtype="float64")
    return img, orig_img
def smooth_image(image,k=5):
    kernel = np.ones((k, k), np.float32) / (k*k)
    image = cv2.filter2D(image, -1, kernel)
    return image

def resize_image(img, dim):
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img


def change_color(image, color):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 168])
    upper_white = np.array([172, 111, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    image[mask > 0] = color
    return image

save_as_jpg = True
overlap_grad_cam = True
base_dir = "//lxestudios.hospitalitaliano.net/pacs/T-Rx/TRx-v2/Datasets/"
images = pd.read_csv(base_dir+"Opacidades/TX-RX-ds-20210423-00_ubuntu.csv")
images_chexpert = images[images.image_source=='chexpert']
file_names = [f.replace('/run/user/1000/gvfs/smb-share:server=lxestudios.hospitalitaliano.net,share=pacs/','//lxestudios.hospitalitaliano.net/pacs/') for f in images_chexpert.file_name.values]
modelpath = "C:/ImageAnalytics/Torax/trxapi-desa/ai_models/ict_models"
session = set_keras_device('cpu')
model = load_keras_model(modelpath,session)
input_shape = model.layers[0].input.shape
raw_heatmaps = {}
dim = (int(input_shape[1]), int(input_shape[2]))
for imagepath in tqdm(file_names[15000:]):
    img, orig_img = load_image_keras(imagepath, dim)
    print('orig',orig_img.shape)
    maskpath = "{}/Chexpert_gradcams/OP_thorax_masks/npy/{}.npy".format(base_dir,os.path.basename(imagepath.replace('.jpg','')))


    left_lung_result, heart_result, right_lung_result = inference_image_mask(model, img,
                                                                   modelpath=modelpath,
                                                                   orig_img=orig_img
                                                                   )

    #plt.imsave(maskpath.replace('.npy', '_ll.png'), left_lung_result)
    #plt.imsave(maskpath.replace('.npy', '_h.png'), heart_result)
    #plt.imsave(maskpath.replace('.npy', '_rl.png'), right_lung_result)

    all_mask = cv2.addWeighted(left_lung_result,1,heart_result,1,0)
    all_mask = cv2.addWeighted(all_mask,1,right_lung_result,1,0)

    with open(maskpath, 'wb') as f:
        pickle.dump(all_mask, f)

    if save_as_jpg:
        cv2.imwrite(maskpath.replace('npy','jpg'),all_mask)

    gradcampath = maskpath.replace('OP_thorax_masks','OP_gradcams')
    print(gradcampath,'npy')
    with open(gradcampath,'rb') as f:
        gradcam = pickle.load(f)

    thorax_gradcam = cv2.bitwise_or(gradcam, gradcam, mask=all_mask)

    thorax_gradcampath = gradcampath.replace('OP_gradcams','OP_thorax_gradcams')
    print(thorax_gradcampath,'npy')
    with open(thorax_gradcampath,'wb') as f:
        pickle.dump(thorax_gradcam, f)

    if save_as_jpg:
        cv2.imwrite(thorax_gradcampath.replace('npy','jpg'),thorax_gradcam)

    del all_mask, img, orig_img, thorax_gradcam, gradcam
