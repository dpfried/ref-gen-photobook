import os
import torch
import glob
import pickle
import json

from PIL import Image
from torchvision import models, datasets, transforms as T
from torch import nn

BATCH_SIZE = 128

def create_image_vectors(image_dir):
    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    vectors = {}
    resnet = models.resnet152(pretrained=True)
    resnet.eval()
    resnet.cuda()

    resnet.fc = nn.Identity()

    img_ids = []
    img_tensors = []

    files = glob.glob(os.path.join(image_dir, "*.jpg"))
    print(f"{len(files)} files")

    for img_file in files:
        img_id = str(int(os.path.basename(img_file).split('.')[0].split('_')[2]))
        img = Image.open(img_file)
        img_tensor = transform(img)
        img_ids.append(img_id)
        img_tensors.append(img_tensor)

    img_features = []
    for ix in range(0, len(img_tensors), BATCH_SIZE):
        batch = img_tensors[ix:ix+BATCH_SIZE]
        batch = torch.stack(batch, dim=0)
        batch = batch.cuda()
        with torch.no_grad():
            features = resnet(batch)
        img_features.extend(features.cpu())

    assert len(img_features) == len(img_ids)

    vectors = {
        img_id: feature.tolist()
        for img_id, feature in zip(img_ids, img_features)
    }
    return vectors

if __name__ == "__main__":
    vectors = create_image_vectors('photobook_coco_images/images')
    with open('../dataset/v1/vectors.pickle', 'wb') as f:
        pickle.dump(vectors, f)
    with open('../dataset/v1/vectors.json', 'w') as f:
        json.dump(vectors, f)
