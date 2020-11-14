import cv2
import numpy as np

from Sketcher import Sketcher
from libs.util import MaskGenerator, ImageChunker
from libs.pconv_model import PConvUnet

import sys
from copy import deepcopy

print('load model...')
model = PConvUnet(vgg_weights=None, inference_only=True)
model.load('pconv_imagenet.h5', train_bn=False)
# model.summary()r

img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)

img_masked = img.copy()
mask = np.zeros(img.shape[:2], np.uint8)

sketcher = Sketcher('image', [img_masked, mask], lambda : ((255, 255, 255), 255))
chunker = ImageChunker(512, 512, 30)

while True:
  key = cv2.waitKey()

  if key == ord('q'): # quit
    break
  if key == ord('r'): # reset
    print('reset')
    img_masked[:] = img
    mask[:] = 0
    sketcher.show()

  if key == ord('c'): #color
    proto = 'models/colorization_deploy_v2.prototxt'
    weights = 'models/colorization_release_v2.caffemodel'

    net = cv2.dnn.readNetFromCaffe(proto, weights)

    pts_in_hull = np.load('models/pts_in_hull.npy')
    pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1).astype(np.float32)
    net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull]

    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full((1, 313), 2.606, np.float32)]

    input_img = img_masked.copy()
    h, w, c = input_img.shape
    input_img = input_img.astype('float32') / 255.
    img_lab = cv2.cvtColor(input_img, cv2.COLOR_BGR2Lab)
    img_l = img_lab[:, :, 0:1]

    blob = cv2.dnn.blobFromImage(img_l, size=(224, 224), mean=[50, 50, 50])

    net.setInput(blob)
    output = net.forward()
    print('processing...')

    output = output.squeeze().transpose((1, 2, 0))
    output_resized = cv2.resize(output, (w, h))
    output_lab = np.concatenate([img_l, output_resized], axis=2)

    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_Lab2BGR)
    output_bgr = output_bgr * 255
    output_bgr = np.clip(output_bgr, 0, 255)
    output_bgr = output_bgr.astype('uint8')

    print('completed!')

    cv2.imshow('img', output_bgr)
    if key == 's':
      cv2.imwrite('graytoRGB.jpg',output_bgr)
      print("saved")

  if key == 32: # hit spacebar to run inpainting
    input_img = img_masked.copy()
    input_img = input_img.astype(np.float32) / 255.

    input_mask = cv2.bitwise_not(mask)
    input_mask = input_mask.astype(np.float32) / 255.
    input_mask = np.repeat(np.expand_dims(input_mask, axis=-1), repeats=3, axis=-1)

    # cv2.imshow('input_img', input_img)
    # cv2.imshow('input_mask', input_mask)

    print('processing...')

    chunked_imgs = chunker.dimension_preprocess(deepcopy(input_img))
    chunked_masks = chunker.dimension_preprocess(deepcopy(input_mask))

    # for i, im in enumerate(chunked_imgs):
    #   cv2.imshow('im %s' % i, im)
    #   cv2.imshow('mk %s' % i, chunked_masks[i])

    pred_imgs = model.predict([chunked_imgs, chunked_masks])
    result_img = chunker.dimension_postprocess(pred_imgs, input_img)

    print('completed!')

    cv2.imshow('result', result_img)

cv2.destroyAllWindows()
