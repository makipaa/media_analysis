import numpy as np
import scipy.io as sio
import cv2

file = sio.loadmat('data/sat-4-full.mat')

x_train = np.array(file['train_x'])
y_train = np.array(file['train_y'])

x_test = np.array(file['test_x'])
y_test = np.array(file['test_y'])

# Annotations contains hot-encoded vectors for the classes
annotations = file['annotations']

x_train = x_train[:,:,:3,:]


n_of_images = 10
#n_of_images = np.size(x_train, 3)

for i in range(n_of_images):
    img = x_train[:,:,:,i]
    img = cv2.resize(img, (224,224), interpolation=cv2.INTER_CUBIC)
    class_id = np.argmax(y_train[:,i])
    path = f'Swin-Transformer/data/imagenet/train/{class_id+1}/img{i+1}.jpeg'
    cv2.imwrite(path, img)
