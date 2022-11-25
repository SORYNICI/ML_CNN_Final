# Reference
# https://yceffort.kr/2019/01/30/pytorch-3-convolutional-neural-network(2)

import os
import json
from PIL import Image
import numpy as np

image_path = './ml/image/'
label_path = './ml/label/'

def get_image():
    data_list = []
    label_list = []

    label_files = os.listdir(label_path)
    label_count = 0
    for file_path in label_files:
        with open(label_path + file_path, 'r') as file:
            data = json.load(file)
            json_str = data["FILE"][0]
            file_name = json_str["FILE_NAME"]
            items = json_str["ITEMS"][0]
            package = items["PACKAGE"]
            if label_count == 0:
                print(file_name)
                print(package)
            
            # 이미지
            img = Image.open(image_path + file_name, 'r')
            resize_img = img.resize((128, 128))

            # 이미지를 RGB 컬러로 각각 쪼갠다.
            # https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.split 참조
            r, g, b = resize_img.split()
            # 각 쪼갠 이미지를 255로 나눠서 0~1 사이의 값이 나오도록 정규화 한다.
            r_resize_img = np.asarray(np.float32(r) / 255.0)
            b_resize_img = np.asarray(np.float32(g) / 255.0)
            g_resize_img = np.asarray(np.float32(b) / 255.0)

            rgb_resize_img = np.asarray([r_resize_img, b_resize_img, g_resize_img])
            # 이렇게 가공한 이미지를 추가한다.
            data_list.append(rgb_resize_img)

            # Label 1은 불법, 0은 정상
            if package == "불법차량":
                label_list.append(1)
            else:
                label_list.append(0)
            label_count += 1

    print('count: ', label_count)
    print('data_list: ', len(data_list))
    print('label_list: ', len(label_list))
    # print(data_list)
    print(label_list)

    return data_list, label_list