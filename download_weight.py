# This code is used to download the weights of vit (cnn) models and load them into the model.
import requests
from pathlib import Path
from tqdm import tqdm
import timm
import yaml
import torchvision.models as models

cnn_model = ['resnet18', 'resnet101', 'resnext50_32x4d', 'densenet121',
             'vgg19', 'resnet18', 'resnet101',
             'resnext50_32x4d', 'densenet121', 'mobilenet_v2']
vit_model = ['vit_base_patch16_224', 'pit_b_224','visformer_small', 'swin_tiny_patch4_window7_224',
             'cait_s24_224', 'deit_base_distilled_patch16_224', 'tnt_s_patch16_224', 'levit_256', 'convit_base']

def vit_weight_download(model_name):
    with open(Path(__file__).parent / "vit_weights_config.yaml") as file:
        _models_config = yaml.load(file, Loader=yaml.FullLoader)
    model_config = _models_config[model_name]
    weights_dir = Path(model_config.pop("weights_dir"))
    weights_dir.mkdir(parents=True, exist_ok=True)
    url = model_config.pop("url")
    file_name = Path(url).name
    weights_path = weights_dir / file_name
    if not weights_path.is_file():
        # download weights for vit models
        print("Downloading weights {} to {}".format(url, weights_path))
        with requests.get(url, stream=True, allow_redirects=True) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(weights_path, "wb") as file, tqdm(
                desc="Downloading",
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
                leave=False,
            ) as bar:
                for data in resp.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
    return weights_path

def load_pretrained_model(cnn_model=[], vit_model=[]):
    for model_name in cnn_model:
        yield model_name, models.__dict__[model_name](weights="DEFAULT")
        # yield model_name, models.__dict__[model_name](weights="IMAGENET1K_V1")
    for model_name in vit_model:
        if model_name == 'tnt_s_patch16_224':
            yield model_name, timm.create_model(model_name, pretrained=True)
        else:
            saved_weight = vit_weight_download(model_name)
            yield model_name, timm.create_model(model_name=model_name, num_classes=1000, pretrained=True,pretrained_cfg_overlay=dict(file=saved_weight))
if __name__ == '__main__':
    for model_name, model in load_pretrained_model(cnn_model, vit_model):
        print(model_name)