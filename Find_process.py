# obj multi-view images에서 region propsal을 찾고
# label 예측
import os
import glob
import sys
import torch.nn as nn
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import selectivesearch
import torch.utils.data as data
from PIL import Image


def make_datapath_list(folder, n):
    rootpath = 'D:\\Data'
    # cls = ['B', 'D', 'E']
    cls = ['E']

    path_list = []
    for c in cls:
        p = os.path.join(rootpath, folder, c)
        target_path = os.path.join(p + '\\*.png')
        pathes = sorted(glob.glob(target_path))
        for i in range(0,len(pathes),n):
            path_list.append(pathes[i:i+n])
    return path_list

def find_region(images):
    size = 224
    mean = [0.567, 0.570, 0.571]
    std = [0.102, 0.103, 0.103]
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)
    ])
    region_con = np.zeros((1, 3, 224, 224))
    for i, img_path in enumerate(images):
        img_bgr = cv2.imread(img_path)
        img_bgr = cv2.resize(img_bgr, (1000,1000))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        _, regions = selectivesearch.selective_search(img_rgb, scale=100, min_size=1000)
        cand_rects = [box['rect'] for box in regions]
        img_rgb_copy = img_rgb.copy()
        # print(f'Find {len(cand_rects)} region IN {i}_image')
        for rect in cand_rects:
            left = rect[0]
            top = rect[1]
            right = left + rect[2]
            bottom = top + rect[3]
            region_img = img_rgb_copy[top:bottom, left:right]
            trans_img = transform(Image.fromarray(region_img)).unsqueeze(dim=0)
            region_con = np.concatenate((region_con, trans_img), axis=0)

    return region_con

folder_dict = {'6view':['6view_dataset_528_2000',6], '20view':['20view_dataset_528_2000',20]}

# create data list
images_6 = make_datapath_list(folder_dict['6view'][0], folder_dict['6view'][1])
images_20 = make_datapath_list(folder_dict['20view'][0], folder_dict['20view'][1])

# create model and load parameters
model = models.vgg16(pretrained=True).cuda(0)
model.classifier[6] = nn.Linear(4096, 3).cuda(0)

pt_path = 'C:\\Users\\user\\PycharmProjects\\Wire_detection\\model_save\\model.pt'
model.load_state_dict(torch.load(pt_path))
model.eval()

for idx, image_set in enumerate(images_6):
    obj_name = image_set[0].split('\\')[-1][:-8]
    obj_class = image_set[0].split('\\')[-2]
    print(f'=== Create {idx} OBJ {obj_name} ===')
    inputs = find_region(image_set)
    pred_dict = {'0':0, '1':0, '2':0}
    print()
    for i, region in enumerate(inputs):
        input = torch.from_numpy(region).float().cuda()
        outputs = model(input.unsqueeze(dim=0))
        pred_val = torch.max(outputs, 1)[0]
        pred_idx = torch.max(outputs, 1)[1]
        if pred_val.item() > 0.5:
            pred_dict[str(pred_idx.item())] += 1

    result = list(pred_dict.values())
    print(f'class:{obj_class} || 고속가공:{result[0]} 연삭:{result[1]} 와이어:{result[2]}')
    print()