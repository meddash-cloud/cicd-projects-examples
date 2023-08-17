#%load py_src/compile_model.py
#%%file
#compile the model - we want to have a multiple outcome
def compile_model(_model):
    _model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
