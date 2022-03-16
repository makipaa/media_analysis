import numpy as np
import scipy.io as sio

def main():
    file = sio.loadmat('data/sat-4-full.mat')

    x_train = np.array(file['train_x'])
    y_train = np.array(file['train_y'])

    x_test = np.array(file['test_x'])
    y_test = np.array(file['test_y'])

    # Annotations contains hot-encoded vectors for the classes
    annotations = file['annotations']

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)





if __name__ == '__main__':
    main()