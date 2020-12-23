import keras
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

uploaded_image = str(sys.argv[1])
# clean_data_filename = str(sys.argv[1])
# model_filename = str(sys.argv[2])
# clean_data_filename = clean_validation_data.h5
# model = sunglasses_bd_net.h5

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255

def main():
    # x_test, y_test = data_loader(clean_data_filename)
    x_test = data_preprocess(x_test)
    img = mpimg.imread(uploaded_image)
    print('Shape of output image is:', img.shape)


    # bd_model = keras.models.load_model(model_filename)

    clean_label_p = np.argmax(bd_model.predict(x_test), axis=1)
    class_accu = np.mean(np.equal(clean_label_p, y_test))*100
    print('Classification accuracy:', class_accu)

if __name__ == '__main__':
    main()
