import cv2
import numpy as np
class model:
    proto = 'models/colorization_deploy_v2.prototxt'
    weights = 'models/colorization_release_v2.caffemodel'

    net = cv2.dnn.readNetFromCaffe(proto, weights)

    pts_in_hull = np.load('models/pts_in_hull.npy')
    pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1).astype(np.float32)
    net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull]

    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full((1, 313), 2.606, np.float32)]


net.setInput(blob)#딥러닝 모델에 삽입
output = net.forward()#결과값 예상

## 이미지 후처리
output = output.squeeze().transpose((1, 2, 0))

output_resized = cv2.resize(output, (w, h))

output_lab = np.concatenate([img_l, output_resized], axis=2)

output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_Lab2BGR)
output_bgr = output_bgr * 255
output_bgr = np.clip(output_bgr, 0, 255)
output_bgr = output_bgr.astype('uint8')

## 출력
cv2.imshow('img', img_input)
cv2.imshow('result', output_bgr)
cv2.waitKey(0)