# %load py_src/train_model.py


# comment it to ease save/load
# %%writefile py_src/train_model.py
def train_model(args:dict) :
    import os
    from minio import Minio
    import numpy as np
    import uuid
    import glob
    import json
    import shutil
    import argparse
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.optim.lr_scheduler import StepLR

    
    bucket_name = args.get("bucket_name", None)
    device_name = args.get("device_name", "cpu")
    epochs = args.get("epochs", 1)
    optimizer = args.get("optimizer", "adam")
    model_save_prefix = args.get("model_save_prefix", "models/trained/detect-digits")
    version = args.get("version", "1")
    lr = args.get("lr", 0.03)
    version = args.get("version", "1")
    gamma = args.get("gamma", 0.7)
    batch_size = args.get("batch_size", 64)
    test_batch_size = args.get("test_batch_size", 1000)
    log_interval = args.get("log_interval", 100)
    take_nth_in_subset = args.get("take_nth_in_subset", 10)
    
    
    
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

    def train(log_interval,model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    
        
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
    
    config = {"endpoint": minio_url,
        "access_key": "minio",
        "secret_key": "minio123",
        "secure": False}
    minio_client = Minio(**config)
    
    random_prefix=str(uuid.uuid4())
    def download_path(filename):
        return "/tmp/{}_{}.npy".format(random_prefix,filename)


    
    print("downlaod training data from the bucket:", bucket_name)
    train_data_saved_path="/tmp"
    os.makedirs(train_data_saved_path, exist_ok=True)
    model_data_remote_path="data/original"
    for bucket in minio_client.list_buckets():
        if bucket.name!=bucket_name:
            continue
        for item in minio_client.list_objects(bucket.name,model_data_remote_path,recursive=True):
            print("remote name:",item.object_name)
            print("local name:", train_data_saved_path+"/"+item.object_name)
            minio_client.fget_object(bucket.name,item.object_name, train_data_saved_path+"/"+item.object_name)

    
    train_kwargs = {'batch_size':  batch_size}
    test_kwargs = {'batch_size': test_batch_size}

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])



    local_mnist_data_rootdir=train_data_saved_path+"/"+model_data_remote_path
    print("local_mnist_data_rootdir:", local_mnist_data_rootdir)
    os.makedirs(local_mnist_data_rootdir, exist_ok=True)
    files = os.listdir(local_mnist_data_rootdir)
    print("local_mnist_data_rootdir:",files)
    
    


    dataset1 = datasets.MNIST(local_mnist_data_rootdir, train=True, download=False,
                       transform=transform)
    dataset2 = datasets.MNIST(local_mnist_data_rootdir, train=False, download=False,
                       transform=transform)

    
    #take susbset of the training set by 1/5
    nth = list(range(0, len(dataset1), take_nth_in_subset))
    dataset1 = torch.utils.data.Subset(dataset1, nth)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    device = torch.device(device_name)
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(log_interval, model, device, train_loader, optimizer, epoch)
        #test(model, device, test_loader)
        scheduler.step()
    
    
    
    model_remote_path="{}/{}".format(model_save_prefix, version)
    model_save_dir="/tmp/model_{}/{}".format(model_save_prefix, version)
    os.makedirs(model_save_dir, exist_ok=True)


    model_save_path="{}/{}".format(model_save_dir, "mnist.pt")
    torch.save(model.state_dict(), model_save_path)

    
    model_script_save_path="{}/{}".format(model_save_dir, "model_scripted.pt")
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(model_script_save_path) # Save
    
    
    
    
    upload_local_directory_to_minio(minio_client, model_save_dir,bucket_name,model_remote_path) 
    
    
    

