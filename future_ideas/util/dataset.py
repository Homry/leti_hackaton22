from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import random


def controller(img, brightness=40, contrast=127):
    p = random.randint(1, 3)
    if p == 2:
        return img
    p = random.randint(1, 3)
    if p == 1:
        brightness = 40
    elif p == 2:
        brightness = 60
    else:
        brightness = 80
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            max = 255
        else:
            shadow = 0
            max = 255 + brightness
        al_pha = (max - shadow) / 255
        ga_mma = shadow
        cal = cv2.addWeighted(img, al_pha,
                              img, 0, ga_mma)
    else:
        cal = img
    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)
        cal = cv2.addWeighted(cal, Alpha,
                              cal, 0, Gamma)
    return cal


class LFW(Dataset):
    def __init__(self, path_to_data, transform=None):
        self.transform = transform
        data_same_person = []
        data_dif_person = []
        with open(path_to_data, 'r') as f:
            line = f.read()
            line = line.split('\n')
            line = [i.split('\t') for i in line]
            [data_same_person.append(i) if len(i) == 3 else data_dif_person.append(i) for i in line]
        data_same_person_res = []
        for i in data_same_person:
            if len(i[1]) == 1:
                inp = f'./lfw/{i[0]}/{i[0]}_000{i[1]}.jpg'
            elif len(i[1]) == 2:
                inp = f'./lfw/{i[0]}/{i[0]}_00{i[1]}.jpg'
            else:
                inp = f'./lfw/{i[0]}/{i[0]}_0{i[1]}.jpg'
            if len(i[2]) == 1:
                tar = f'./lfw/{i[0]}/{i[0]}_000{i[2]}.jpg'
            elif len(i[2]) == 2:
                tar = f'./lfw/{i[0]}/{i[0]}_00{i[2]}.jpg'
            else:
                tar = f'./lfw/{i[0]}/{i[0]}_0{i[2]}.jpg'
            data_same_person_res.append([inp, tar, 1.0])

        del (data_dif_person[-1])
        data_dif_person_res = []
        for i in data_dif_person:
            if len(i[1]) == 1:
                inp = f'./lfw/{i[0]}/{i[0]}_000{i[1]}.jpg'
            elif len(i[1]) == 2:
                inp = f'./lfw/{i[0]}/{i[0]}_00{i[1]}.jpg'
            else:
                inp = f'./lfw/{i[0]}/{i[0]}_0{i[1]}.jpg'
            if len(i[3]) == 1:
                tar = f'./lfw/{i[2]}/{i[2]}_000{i[3]}.jpg'
            elif len(i[3]) == 2:
                tar = f'./lfw/{i[2]}/{i[2]}_00{i[3]}.jpg'
            else:
                tar = f'./lfw/{i[2]}/{i[2]}_0{i[3]}.jpg'
            data_dif_person_res.append([inp, tar, 0.0])

        self.data = data_same_person_res + data_dif_person_res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_inp = cv2.imread(self.data[index][0])
        img = img_inp.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        image = np.resize(gray, (32, 32))

        tar = cv2.imread(self.data[index][1])
        img = tar.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_target = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        img_target = np.resize(img_target, (32, 32))
        label = float(self.data[index][2])

        return image.flatten(), img_target.flatten(), label.flatten()


class LFW_Train(Dataset):
    def __init__(self, path_to_data, transform=None):
        self.transform = transform
        data_same_person = []
        data_dif_person = []
        with open(path_to_data, 'r') as f:
            line = f.read()
            line = line.split('\n')
            line = [i.split('\t') for i in line]
            [data_same_person.append(i) if len(i) == 3 else data_dif_person.append(i) for i in line]
        res = []
        for i in data_same_person:
            for j in data_dif_person:
                if i[0] in j:
                    index = j.index(i[0])
                    if index == 0:
                        index = 2
                    else:
                        index = 0
                    tmp = i.copy()
                    tmp.append(j[index])
                    tmp.append(j[index + 1])
                    res.append(tmp)
        data_res = []
        for i in res:
            if len(i[1]) == 1:
                anchor = f'./lfw/{i[0]}/{i[0]}_000{i[1]}.jpg'
            elif len(i[1]) == 2:
                anchor = f'./lfw/{i[0]}/{i[0]}_00{i[1]}.jpg'
            else:
                anchor = f'./lfw/{i[0]}/{i[0]}_0{i[1]}.jpg'
            if len(i[2]) == 1:
                pos = f'./lfw/{i[0]}/{i[0]}_000{i[2]}.jpg'
            elif len(i[2]) == 2:
                pos = f'./lfw/{i[0]}/{i[0]}_00{i[2]}.jpg'
            else:
                pos = f'./lfw/{i[0]}/{i[0]}_0{i[2]}.jpg'
            if len(i[4]) == 1:
                neg = f'./lfw/{i[3]}/{i[3]}_000{i[4]}.jpg'
            elif len(i[4]) == 2:
                neg = f'./lfw/{i[3]}/{i[3]}_00{i[4]}.jpg'
            else:
                neg = f'./lfw/{i[3]}/{i[3]}_0{i[4]}.jpg'
            data_res.append([anchor, pos, neg])
        self.data = data_res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img_inp = cv2.imread(self.data[index][0])
        img = img_inp.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        anchor = gray

        img_inp = cv2.imread(self.data[index][1])
        img = img_inp.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        pos = gray
        pos = np.reshape(pos, (32, 32))

        img_inp = cv2.imread(self.data[index][2])
        img = img_inp.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        neg = gray
        neg = np.reshape(neg, (32, 32))

        return anchor.flatten(), pos.flatten(), neg.flatten()
