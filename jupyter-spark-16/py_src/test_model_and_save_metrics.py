# %load py_src/test_model_and_save_metrics.py


# comment it to ease save/load
# %%writefile py_src/test_model_and_save_metrics.py

from typing import NamedTuple

def test_model_and_save_metrics(args:dict) -> NamedTuple('Output', [('mlpipeline_ui_metadata', 'UI_metadata'),('mlpipeline_metrics', 'Metrics')]) :
    from minio import Minio
    import numpy as np
    import uuid
    import glob
    import pandas as pd
    import json
    import shutil
    import argparse
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.optim.lr_scheduler import StepLR
    import os
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    from torch.nn.modules.module import _addindent
    
    bucket_name = args.get("bucket_name", None)
    device_name = args.get("device_name", "cpu")
    test_batch_size = args.get("test_batch_size", 1000)
    model_save_prefix = args.get("model_save_prefix", "models/trained/detect-digits")
    version = args.get("version", "1")
    
    
    
    def torch_summarize(model, show_weights=True, show_parameters=True):
        """Summarizes torch model by showing trainable parameters and weights."""
        tmpstr = model.__class__.__name__ + ' (\n'
        for key, module in model._modules.items():
            # if it contains layers let call it recursively to get params and weights
            if type(module) in [
                torch.nn.modules.container.Container,
                torch.nn.modules.container.Sequential
            ]:
                modstr = torch_summarize(module)
            else:
                modstr = module.__repr__()
            modstr = _addindent(modstr, 2)

            params = sum([np.prod(p.size()) for p in module.parameters()])
            weights = tuple([tuple(p.size()) for p in module.parameters()])

            tmpstr += '  (' + key + '): ' + modstr 
            if show_weights:
                tmpstr += ', weights={}'.format(weights)
            if show_parameters:
                tmpstr +=  ', parameters={}'.format(params)
            tmpstr += '\n'   

        tmpstr = tmpstr + ')'
        return tmpstr
    
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
        model_accuracy = 100. * correct / len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), model_accuracy))
        return {"loss":test_loss, "accuracy":model_accuracy}
    # generate confusion matrix csv
    def gen_cm_csv(y_test=None,test_predictions=None):
        confusion_matrix = sk_confusion_matrix(y_test, test_predictions)
        vocab = list(np.unique(y_test))
        data = []
        for target_index, target_row in enumerate(confusion_matrix):
            for predicted_index, count in enumerate(target_row):
                data.append((vocab[target_index], vocab[predicted_index], count))

        df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])
        cm_csv = df_cm.to_csv(header=False, index=False)
        return cm_csv
    
    
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


    model_remote_path="{}/{}".format(model_save_prefix, version)
    model_saved_path="/tmp/{}/{}".format(model_save_prefix, version)
    model_script_remote_path="{}/model_scripted.pt".format(model_remote_path)
    model_script_save_path="{}/model_scripted.pt".format(model_saved_path)

    print(bucket_name,model_script_remote_path,model_script_save_path)
    minio_client.fget_object(bucket_name,model_script_remote_path,model_script_save_path)
    #load model without class prototype
    model = torch.jit.load(model_script_save_path)
    model.eval()
    
    
    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
    
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
    local_mnist_data_rootdir=train_data_saved_path+"/"+model_data_remote_path
    test_data = datasets.MNIST(local_mnist_data_rootdir, train=False, download=False,transform=transform)

    test_kwargs = {'batch_size': test_batch_size}
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)
    device = torch.device(device_name)
    
    #{"loss":test_loss, "accuracy":model_accuracy}
    model_test_result = test(model, device, test_loader)
    model_loss = model_test_result["loss"]
    model_accuracy = model_test_result["accuracy"]
    
    
    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in test_loader:
        output = model(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth
        
        
    
    cm_csv = gen_cm_csv(y_test=y_true,test_predictions=y_pred)
    
    metric_model_summary = torch_summarize(model)
    
    output_confussion_matrix = {
                "type": "confusion_matrix",
                "format": "csv",
                "schema": [
                    {'name': 'target', 'type': 'CATEGORY'},
                    {'name': 'predicted', 'type': 'CATEGORY'},
                    {'name': 'count', 'type': 'NUMBER'},
                  ],
                "target_col" : "actual",
                "predicted_col" : "predicted",
                "source": cm_csv,
                "storage": "inline",
                "labels": list(np.arange(10)) #0..9 labels
            }
    output_model_summary = {
                'type': 'markdown',
                'storage': 'inline',
                'source': f'''# Model Overview
## Model Summary

```
{metric_model_summary}
```

## Model Performance

**Accuracy**: {model_accuracy}
**Loss**: {model_loss}

'''
            }
    
    metadata = {"outputs": [output_confussion_matrix, output_model_summary]}
    metrics = {
      'metrics': [{
          'name': 'model_accuracy',
          'numberValue':  float(model_accuracy),
          'format' : "PERCENTAGE"
        },{
          'name': 'model_loss',
          'numberValue':  float(model_loss),
          'format' : "PERCENTAGE"
        }]}
    
    
    class NpJsonEncoder(json.JSONEncoder):
        """Serializes numpy objects as json."""

        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.floating):
                if np.isnan(obj):
                    return None  # Serialized as JSON null.
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super().default(obj)
        
    from collections import namedtuple
    output = namedtuple('Output', ['mlpipeline_ui_metadata', 'mlpipeline_metrics'])
    return output(json.dumps(metadata, cls=NpJsonEncoder),json.dumps(metrics, cls=NpJsonEncoder))
