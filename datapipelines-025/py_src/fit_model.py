#%load py_src/fit_model.py
#%%file
def fit_model(_model, x_train, y_train):
    #fit the model and return the history while training
    history = _model.fit(
      x=x_train,
      y=y_train,
      epochs=1
    )
    return history
