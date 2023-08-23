# %load py_src/put_public_mnist_to_local_minio.py


# comment it to ease save/load
# %%writefile py_src/put_public_mnist_to_local_minio.py
def put_public_mnist_to_local_minio(args:dict) :

    import tempfile
    import os
    from minio import Minio
    import numpy as np
    import uuid
    from torchvision import datasets, transforms
    import glob
    
    
    bucket_name = args.get("bucket_name", None)
    batch_size = args.get("batch_size", 64)
    test_batch_size = args.get("batch_size", 1000)
    
    
    
    def upload_local_directory_to_minio(minio_client, local_path, bucket_name, minio_path):
        # assert os.path.isdir(local_path)

        for local_file in glob.glob(local_path + '/**'):
            local_file = local_file.replace(os.sep, "/") # Replace \ with / on Windows
            if not os.path.isfile(local_file):
                upload_local_directory_to_minio(
                    minio_client, local_file, bucket_name, minio_path + "/" + os.path.basename(local_file))
            else:
                remote_path = os.path.join(
                    minio_path, local_file[1 + len(local_path):])
                remote_path = remote_path.replace(
                    os.sep, "/")  # Replace \ with / on Windows
                minio_client.fput_object(bucket_name, remote_path, local_file)

    def get_minio_url():
        minio_host, minio_port = os.environ["MINIO_SERVICE_SERVICE_HOST"], os.environ["MINIO_SERVICE_SERVICE_PORT_HTTP"]
        minio_url= "{}:{}".format(minio_host, minio_port)
        return minio_url
    minio_url = get_minio_url()
    print("minio url:", minio_url)

    config = {"endpoint": minio_url,
        "access_key": "minio",
        "secret_key": "minio123",
        "secure": False}
    minio_client = Minio(**config)

    print("try to find bucket {}".format(bucket_name))
    found = minio_client.bucket_exists(bucket_name)
    print("found", found)
    if not found:
        minio_client.make_bucket(bucket_name)
    else:
        print("Bucket '{}' already exists".format(bucket_name))
    

    mnist_data_dirpath="/tmp/"+str(uuid.uuid4())
    

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST(mnist_data_dirpath, train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST(mnist_data_dirpath, train=False,
                       transform=transform)
    
    upload_local_directory_to_minio(minio_client, mnist_data_dirpath, bucket_name, "data/original")
                                   
    from shutil import rmtree
    rmtree(mnist_data_dirpath)
    
