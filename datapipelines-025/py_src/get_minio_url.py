#%load py_src/get_minio_url.py
#%%file
import os
def get_minio_url():
    minio_host, minio_port = os.environ["MINIO_SERVICE_SERVICE_HOST"], os.environ["MINIO_SERVICE_SERVICE_PORT_HTTP"]
    minio_url= "{}:{}".format(minio_host, minio_port)
    return minio_url

