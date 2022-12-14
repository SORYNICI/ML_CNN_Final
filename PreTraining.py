# Reference
# https://yceffort.kr/2019/01/30/pytorch-3-convolutional-neural-network(2)

import os
import json
from PIL import Image
import numpy as np

image_path = 'image/'
label_path = 'label/'
image_path_s = 'image_s/'
label_path_s = 'label_s/'

def get_image(type = 3):

    if type == 0:
        label_files = os.listdir(label_path)
        label_list, data_list, label_count = get_file_image(label_files, label_path, image_path, type)
    elif type == 1:
        label_files_m = os.listdir(label_path)
        label_list, data_list, label_count = get_file_image(label_files_m, label_path, image_path, type)
    elif type == 2:
        label_files_s = os.listdir(label_path_s)
        label_list, data_list, label_count = get_file_image(label_files_s, label_path_s, image_path_s, type)
    else: # type == 3
        label_files = os.listdir(label_path)
        label_list_l, data_list_l, label_count_l = get_file_image(label_files, label_path, image_path, type)
        label_files_s = os.listdir(label_path_s)
        label_list_s, data_list_s, label_count_s = get_file_image(label_files_s, label_path_s, image_path_s, type)
        label_list = label_list_l + label_list_s
        data_list = data_list_l + data_list_s
        label_count = label_count_l + label_count_s

    print('count: ', label_count)
    print('data_list: ', len(data_list))
    print('label_list: ', len(label_list))
    # print(data_list)
    # print(label_list)

    return data_list, label_list, label_count

def get_file_image(label_files, label_path, image_path, type):
    print('get_file_image() - type: ', type)
    data_list = []
    label_list = []
    label_count = 0
    for file_path in label_files:
        try:
            with open(label_path + file_path, 'r', encoding="UTF-8") as file:
                data = json.load(file)
                json_str = data["FILE"][0]
                file_name = json_str["FILE_NAME"]
                if not file_name.endswith('1.jpg'):
                    continue
                if type == 0 and not file_name.startswith('A01'):
                    continue
                if type == 1 and not file_name.startswith('A02'):
                    continue

                items = json_str["ITEMS"][0]
                package = items["PACKAGE"]
                if label_count % 100 == 0:
                    print(file_name)
                    print(package)
                
                # ?????????
                img = Image.open(image_path + file_name, 'r')
                resize_img = img.resize((128, 128))

                # ???????????? RGB ????????? ?????? ?????????.
                # https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.split ??????
                r, g, b = resize_img.split()
                # ??? ?????? ???????????? 255??? ????????? 0~1 ????????? ?????? ???????????? ????????? ??????.
                r_resize_img = np.asarray(np.float32(r) / 255.0)
                b_resize_img = np.asarray(np.float32(g) / 255.0)
                g_resize_img = np.asarray(np.float32(b) / 255.0)

                rgb_resize_img = np.asarray([r_resize_img, b_resize_img, g_resize_img])
                # ????????? ????????? ???????????? ????????????.
                data_list.append(rgb_resize_img)

                # Label 1??? ??????, 0??? ??????
                if package == "????????????":
                    label_list.append(1)
                else:
                    label_list.append(0)
                label_count += 1
        except FileNotFoundError:
            print(file_path)
    
    return label_list, data_list, label_count