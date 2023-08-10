#%load py_src/reshape_train_data.py
#%%file
def reshape_train_data(_x_train,  _x_test):
    # reshaping the data
    # reshaping pixels in a 28x28px image with greyscale, canal = 1. This is needed for the Keras API
    _x_train = _x_train.reshape(-1,28,28,1)
    _x_test = _x_test.reshape(-1,28,28,1)

    # normalizing the data
    # each pixel has a value between 0-255. Here we divide by 255, to get values from 0-1
    _x_train = _x_train / 255
    _x_test = _x_test / 255

    return _x_train, _x_test

