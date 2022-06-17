# obj multi-view images에서 region propsal을 찾고
# label 예측
import os
import glob
import shutil
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
    cls = ['D','E']

    path_list = []
    for c in cls:
        p = os.path.join(rootpath, folder, c)
        target_path = os.path.join(p + '\\*.png')
        pathes = sorted(glob.glob(target_path))[:12]
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
    view_region = []
    region_info = []
    for i, img_path in enumerate(images):
        region_con = np.zeros((1, 3, 224, 224))
        info = []
        image = cv2.imread(img_path)
        image = cv2.resize(image, (500,500))
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, regions = selectivesearch.selective_search(img_rgb, scale=100, min_size=3000)
        # cand_rects = [box['rect'] for box in regions]
        cand_rects = []
        for cand in regions[1:]:
            width = cand['rect'][2]
            height = cand['rect'][3]
            if max(width, height) // min(width, height) < 5:
                cand_rects.append(cand['rect'])

        img_rgb_copy = img_rgb.copy()
        # print(f'Find {len(cand_rects)} region IN {i}_image')
        for rect in cand_rects[1:]:
            left = rect[0]
            top = rect[1]
            right = left + rect[2]
            bottom = top + rect[3]
            region_img = img_rgb_copy[top:bottom, left:right]
            trans_img = transform(Image.fromarray(region_img)).unsqueeze(dim=0)
            region_con = np.concatenate((region_con, trans_img), axis=0)
            info.append([left,top,right,bottom])

        view_region.append(region_con)
        region_info.append(info)

    return view_region, region_info

def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

green_rgb = (125, 255, 51)
folder_dict = {'6view':['6view_dataset_528_2000', 6],
               '20view':['20view_dataset_528_2000', 20]}

# create data list
images_6 = make_datapath_list(folder_dict['6view'][0], folder_dict['6view'][1])
images_20 = make_datapath_list(folder_dict['20view'][0], folder_dict['20view'][1])

num_view = 6
# create model and load parameters
model = models.vgg16(pretrained=True).cuda(0)
model.classifier[6] = nn.Linear(4096, 3).cuda(0)
softmax = nn.Softmax(dim=1)

pt_path = 'C:\\Users\\user\\PycharmProjects\\Wire_detection\\model_save\\model_1048.pt'
model.load_state_dict(torch.load(pt_path))
model.eval()

save_path = 'C:\\Users\\user\\PycharmProjects\\Wire_detection\\test_output'

for idx, image_set in enumerate(images_6):
    obj_name = image_set[0].split('\\')[-1][:-8]
    obj_class = image_set[0].split('\\')[-2]
    dir_path = os.path.join(save_path, obj_name)
    create_folder(dir_path)
    # print(f'=== Create {idx} OBJ {obj_name} ===')
    inputs, inputs_info = find_region(image_set)
    pred_dict = {'0':0, '1':0, '2':0}
    except_cnt = 0
    for v in range(num_view):
        v_inputs = inputs[v]
        v_info = inputs_info[v]
        img_path = image_set[v]
        image = cv2.imread(img_path)
        image = cv2.resize(image, (500,500))
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_rgb_copy = img_rgb.copy()
        save = False
        for i, (region, pos) in enumerate(zip(v_inputs, v_info)):
            input = torch.from_numpy(region).float().cuda()
            output = model(input.unsqueeze(dim=0))
            outputs = softmax(output)
            pred_val = torch.max(outputs, 1)[0]
            pred_idx = torch.max(outputs, 1)[1]
            if pred_val.item() > 0.7:
                pred_dict[str(pred_idx.item())] += 1
                if pred_idx.item() == 2:
                    save = True
                    img_rgb_copy = cv2.rectangle(img_rgb_copy, (pos[0], pos[1]), (pos[2], pos[3]), color=green_rgb, thickness=2)
            else:
                except_cnt += 1

        if save:
            result_img = os.path.join(dir_path, img_path.split('\\')[-1])
            cv2.imwrite(result_img, img_rgb_copy)

    result = list(pred_dict.values())
    print(f'{obj_name} || {obj_class} || 고속가공:{result[0]} 연삭:{result[1]} 와이어:{result[2]} || Except:{except_cnt}')
    print()