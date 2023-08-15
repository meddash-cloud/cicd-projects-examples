#%load py_src/print_data.py
# check shape of the data

def print_data(_x_train, _y_train, _x_test, _y_test):
    x_train, y_train, x_test, y_test =_x_train, _y_train, _x_test, _y_test
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")
print_data(x_train, y_train, x_test, y_test)
