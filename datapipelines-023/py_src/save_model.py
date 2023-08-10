#%load py_src/save_model.py
#%%file
saved_model_name="models/detect-digits"

def save_model(_model, model_path):
    keras.models.save_model(_model,model_path)
