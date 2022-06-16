import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import glob
import torch.utils.data as data

from PIL import Image

# random seed
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="model_save")
parser.add_argument("-example", "--ex_path", type=str, help="path of the example_file", default="D:\\Data\\crop_dataset\\train\\E2\\7133.png")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=128)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0)
parser.add_argument("-epochs", type=int, help="number of epoch", default=10)
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vgg16")
parser.set_defaults(train=False)

print("쿠다 가능 :{}".format(torch.cuda.is_available()))
print("현재 디바이스 :{}".format(torch.cuda.current_device()))
print("디바이스 갯수 :{}".format(torch.cuda.device_count()))

class ImageTransform() :
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ]),
            'show': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)

class HymenoperaDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):

        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        '''화상 개수 반환'''
        return len(self.file_list)

    def __getitem__(self, index):
        '''전처리한 화상의 텐서 형식 데이터와 라벨 획득'''

        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_transformed = self.transform(img)

        if self.phase == 'train':
            label = img_path.split('\\')[-2]
        elif self.phase == 'val':
            label = img_path.split('\\')[-2]
        elif self.phase == 'show':
            return img_transformed
        # 라벨을 숫자로
        if label == 'B':
            label = 0
        elif label == 'D':
            label = 1
        elif label == 'E' or label == 'E2':
            label = 2
        else:
            print(label)
        return img_transformed, label

def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

def get_mean_std(dataset):
    meanRGB = [np.mean(image.numpy(), axis=(1, 2)) for image in dataset]
    stdRGB = [np.std(image.numpy(), axis=(1, 2)) for image in dataset]

    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])

    stdR = np.std([s[0] for s in stdRGB])
    stdG = np.std([s[1] for s in stdRGB])
    stdB = np.std([s[2] for s in stdRGB])

    mean = [meanR, meanG, meanB]
    std = [stdR, stdG, stdB]

    return mean, std

def show_example(path):
    image_file_path = path
    img = Image.open(image_file_path)

    plt.imshow(img)
    plt.show()

    size = 224
    mean = (0.567, 0.570, 0.571)
    std = (0.102, 0.103, 0.103)

    transform = ImageTransform(size, mean, std)
    img_transformed = transform(img, phase='train')

    img_transformed = img_transformed.numpy().transpose((1, 2, 0))
    img_transformed = np.clip(img_transformed, 0, 1)
    plt.imshow(img_transformed)
    plt.show()

def make_datapath_list(phase):
    rootpath = 'D:\\Data\\crop_dataset'
    cls = ['B', 'D', 'E', 'E2']
    path_list = []
    for c in cls:
        p = os.path.join(rootpath, phase, c)
        target_path = os.path.join(p + '\\*.png')

        for path in glob.glob(target_path):
            path_list.append(path)

    return path_list

def train(net, dataloaders_dict, criterion, optimizer, num_epochs, log_dir):
    best_acc = 0.0
    for epoch in range(0, num_epochs):
        print(f'Epoch {epoch + 1}/ {num_epochs}')
        print('*' * 30)
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # 훈련 모드
            else:
                net.eval()  # 검증 모드

            all_correct_points = 0
            all_points = 0

            wrong_class = np.zeros(3)
            samples_class = np.zeros(3)

            epoch_loss = 0.0
            epoch_corrects = 0

            if (epoch == 0) and (phase == 'train'):
                continue

            for idx, (inputs, labels) in enumerate(dataloaders_dict[phase]):
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                    outputs = net(inputs)
                    preds = torch.max(outputs, 1)[1]
                    loss = criterion(outputs, labels)
                    epoch_loss += loss.item()

                    results = preds == labels

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    else:
                        for i in range(results.size()[0]):
                            if not bool(results[i]):
                                wrong_class[labels.cpu().data.numpy().astype('int')[i]] += 1
                            samples_class[labels.cpu().data.numpy().astype('int')[i]] += 1
                        correct_points = sum(results.long())

                        all_correct_points += correct_points
                        all_points += results.size()[0]

                    epoch_corrects += torch.sum(preds == labels.data)
                    iter_acc = torch.sum(preds == labels.data).float() / labels.size()[0]

                    if phase == 'train':
                        print('epoch %d, step [%d/%d]: train_loss %.3f; train_acc %.3f'
                              % (epoch + 1, idx + 1, len(dataloaders_dict[phase]), loss, iter_acc))

            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print(f'{phase} Loss {epoch_loss:.4f} Acc : {epoch_acc:.4f}')
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(net.state_dict(), log_dir+'\\model.pt')


def main():
    args = parser.parse_args()

    # show_example(args.ex_path)

    log_dir = args.name
    create_folder(args.name)
    config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()

    train_list = make_datapath_list(phase='train')
    val_list = make_datapath_list(phase='val')

    size = 224
    mean = (0.567, 0.570, 0.571)
    std = (0.102, 0.103, 0.103)

    train_dataset = HymenoperaDataset(file_list=train_list, transform=ImageTransform(size, mean, std), phase='show')
    val_dataset = HymenoperaDataset(file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')

    # mean, std = get_mean_std(train_dataset)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False)

    dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

    index = 0
    print(train_dataset.__getitem__(index)[0].size())
    print(train_dataset.__getitem__(index)[1])

    net = models.vgg16(pretrained=True).cuda(0)
    net.classifier[6] = nn.Linear(4096, 3).cuda(0)

    net.train()
    params_to_update = []
    update_param_names = []
    update_param_names.extend([f'classifier.{i}.weight' for i in range(7)])
    update_param_names.extend([f'classifier.{i}.bias' for i in range(7)])

    # updata 파라미터 외에 파라미터 fix
    for name, param in net.named_parameters():
        if name in update_param_names:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)

    train(net, dataloaders_dict, criterion, optimizer, args.epochs, log_dir)

if __name__ == '__main__':
    main()
