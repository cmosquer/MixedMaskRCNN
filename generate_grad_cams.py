from keras import backend as K
import tensorflow as tf
import os, pickle
import keras_retinanet.models as retinanet_models
from keras.models import load_model
import numpy as np
import sys
import keras
import logging
import cv2
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)
import vis.visualization.saliency as vis
from vis.utils import utils as utils
from vis.losses import ActivationMaximization
from vis.optimizer import Optimizer
from keras import backend as K
import numpy as np
import matplotlib.cm as cm
from skimage.transform import resize

def visualize_cam_init(model, layer_idx, filter_indices):

    penultimate_layer = vis._find_penultimate_layer(model, layer_idx, None)
    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), -1)
    ]
    penultimate_output = penultimate_layer.output
    opt = Optimizer(model.input, losses, wrt_tensor=penultimate_output, norm_grads=False)
    return opt

def visualize_cam_run(model, layer_idx, opt,seed_input):
    input_tensor = model.input
    penultimate_layer = vis._find_penultimate_layer(model, layer_idx, None)
    penultimate_output = penultimate_layer.output

    _, grads, penultimate_output_value = opt.minimize(seed_input, max_iter=1, grad_modifier=None, verbose=False)
    #opt.minimize(seed_input, max_iter=1, grad_modifier=grad_modifier, verbose=False)

    # For numerical stability. Very small grad values along with small penultimate_output_value can cause
    # w * penultimate_output_value to zero out, even for reasonable fp precision of float32.
    grads = grads / (np.max(grads) + K.epsilon())
    # Average pooling across all feature maps.
    # This captures the importance of feature map (channel) idx to the output.
    channel_idx = 1 if K.image_data_format() == 'channels_first' else -1
    other_axis = np.delete(np.arange(len(grads.shape)), channel_idx)
    weights = np.mean(grads, axis=tuple(other_axis))

    # Generate heatmap by computing weight * output over feature maps
    output_dims = utils.get_img_shape(penultimate_output)[2:]
    heatmap = np.zeros(shape=output_dims, dtype=K.floatx())

    for i, w in enumerate(weights):
        if channel_idx == -1:
            heatmap += w * penultimate_output_value[0, ..., i]
        else:
            heatmap += w * penultimate_output_value[0, i, ...]

    # ReLU thresholding to exclude pattern mismatch information (negative gradients).
    heatmap = np.maximum(heatmap, 0)

    # The penultimate feature map size is definitely smaller than input image.
    input_dims = utils.get_img_shape(input_tensor)[2:]
    heatmap = resize(heatmap, input_dims)#, order='3')

    # Normalize and create heatmap.
    heatmap = utils.normalize(heatmap)
    return np.uint8(cm.jet(heatmap)[..., :3] * 255)

class ActivationMap():
    def __init__(self,model,filter_index):
        self.model = model
        self.isCamInit = False
        self.CamOpt = None
        self.filter_indices = filter_index
        self.layer_idx =  len(self.model.layers) - 1

    def init_cam(self):
        if not self.isCamInit:
            self.isCamInit = True
            del self.CamOpt
            # for opt in self.CamOpt:
            # print('self.labels=',self.labels)
            opt = visualize_cam_init(self.model, self.layer_idx, filter_indices=self.filter_indices)
            self.CamOpt = opt

    def Get_Activation_map(self, image):
        if not self.isCamInit:
            self.init_cam()
        grads = visualize_cam_run(self.model, self.layer_idx,  self.CamOpt, seed_input=image)
        return grads

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

def dice_coef_continuos(y_true, y_pred):
    smooth = 1.
    ignored_label=-3
    negative_label=0.01

    zeros = tf.zeros(shape=tf.shape(y_true))
    ones = tf.ones(shape=tf.shape(y_true))
    unc = tf.where(tf.logical_or(tf.equal(y_true, ignored_label),tf.equal(y_true,negative_label)), zeros, ones)

    y_true_f = K.flatten(unc)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def AUC_continuos(y_true,y_pred):
    ignored_label=-3
    negative_label=0.01

    zeros = tf.zeros(shape=tf.shape(y_true))
    ones = tf.ones(shape=tf.shape(y_true))
    unc = tf.where(tf.logical_or(tf.equal(y_true, ignored_label),tf.equal(y_true,negative_label)), zeros, ones)
    value, update_op = tf.metrics.auc(unc, y_pred)

    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
    return value

def set_keras_device(device):
    if device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        sess = get_gpu_session()
    else:
        sess = get_cpu_session()
    reinit_variables(sess)
    return sess

def reinit_variables(sess):
    sess.run(tf.global_variables_initializer())

def load_keras_model(modelpath,sess):

    K.tensorflow_backend.set_session(sess)
    logger.info('keras version: '.format(keras.__version__))
    custom_metrics = {
        'dice_coef_continuos': dice_coef_continuos,
        'AUC_continuos': AUC_continuos
    }

    model = load_model(modelpath, custom_objects=custom_metrics)
    #model._make_predict_function()
    #reinit_variables(sess)
    activation_maps = ActivationMap(model,1)

    #global graph
    #graph = tf.get_default_graph()
    return model, activation_maps

def inference_image_keras(model,img,activation_map=None):

    img = np.array(img, dtype="float64")
    #X = np.empty((1, img.shape[0], img.shape[1], 1))

    if len(img.shape) == 2:
        oneChanneled = np.zeros((img.shape[0],img.shape[1],1),dtype="float64")
        oneChanneled[:,:,0] = img
        #threeChanneled[:,:,1] = img
        #threeChanneled[:,:,2] = img
        img = oneChanneled

    img = np.expand_dims(img, axis=0)
    pred = model.predict_on_batch(img)
    pred = pred[0][1]
    if activation_map:
        heatmap = activation_map.Get_Activation_map(img)
    else:
        heatmap=None
    return pred,heatmap

def load_image_keras(imagepath,dim=(256,256),retinanet=None):
    ext = os.path.splitext(imagepath)[-1]
    #assert ext in ['.dcm','.jpg','.png']
    assert os.path.exists(imagepath)

    channels = 1
    imgtype = cv2.IMREAD_GRAYSCALE

    if ext in ['.jpg','.png']:
        orig_img = cv2.imread(imagepath, imgtype)

    img = cv2.resize(orig_img, dim)       #
    #img = np.array(img, dtype="float64")
    return img, orig_img

save_jpg_colormap = True
save_jpg_overlapped = False
base_dir = "//lxestudios.hospitalitaliano.net/pacs/T-Rx/TRx-v2/Datasets/"
images = pd.read_csv(base_dir+"Opacidades/TX-RX-ds-20210423-00_ubuntu.csv")
images_chexpert = images[images.image_source=='chexpert']
file_names = [f.replace('/run/user/1000/gvfs/smb-share:server=lxestudios.hospitalitaliano.net,share=pacs/','//lxestudios.hospitalitaliano.net/pacs/') for f in images_chexpert.file_name.values]
modelpath = "C:/ImageAnalytics/Torax/trxapi-desa/ai_models/model-Epoch12.hdf5"
session = set_keras_device('cpu')
model,activation_maps = load_keras_model(modelpath,session)
input_shape = model.layers[0].input.shape
#raw_heatmaps = {}
dim = (int(input_shape[1]), int(input_shape[2]))
for imagepath in tqdm(file_names[15000:]):
    img, orig_img = load_image_keras(imagepath, dim)
    raw_pred, raw_heatmap = inference_image_keras(model,img,activation_map=activation_maps)
    raw_heatmap = cv2.resize(raw_heatmap,(orig_img.shape[1],orig_img.shape[0]), interpolation=cv2.INTER_AREA)
    #raw_heatmaps[os.path.basename(imagepath)] = raw_heatmap
    gradcampath = "{}/Chexpert_gradcams/OP_gradcams/npy/{}.npy".format(base_dir,os.path.basename(imagepath.replace('.jpg','')))
    raw_heatmap = cv2.cvtColor(raw_heatmap,cv2.COLOR_BGR2RGB)

    TXTPATH = "{}/Chexpert_gradcams/OP_gradcams/predictions.txt".format(base_dir)
    with open(TXTPATH,'a') as f:
        f.write("{},{:.4f}\n".format(os.path.basename(imagepath),raw_pred))
    with open(gradcampath, 'wb') as f:
        pickle.dump(raw_heatmap, f)
    #print(img.shape,orig_img.shape,raw_heatmap.shape)
    if save_jpg_colormap:
        cv2.imwrite(gradcampath.replace('npy','jpg'),raw_heatmap)


    if save_jpg_overlapped:
        overlappath=base_dir+"Chexpert_gradcams/OP_raw_heatmaps_jpg/"+os.path.basename(imagepath).replace('jpg','png')

        alpha = np.mean(raw_heatmap, axis=2) / 4
        ready_heatmap = cv2.cvtColor(raw_heatmap, cv2.COLOR_RGB2RGBA)
        ready_heatmap[:, :, 3] = alpha
        print(overlappath)
        save_heatmap(overlappath,ready_heatmap,orig_img)
    del raw_heatmap, img, orig_img

