import os
import json
import shutil
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

print('MoveFiles')

image_train_path = 'image_train/' # 'darknet/yolov5/image_train/'
image_valid_path = 'image_valid/' # 'darknet/yolov5/image_valid/'
label_train_path = 'label_train/'
label_valid_path = 'label_valid/'

label_train_files = os.listdir(label_train_path)
label_valid_files = os.listdir(label_valid_path)

scaler = MinMaxScaler()
mix_x = 0
min_y = 0
max_x = 1920
max_y = 1080

index = 0
for file_path in os.listdir(image_train_path):
    if file_path.endswith('jpg'):
        img = Image.open(image_train_path + file_path).convert("RGB")
        new_img = img.resize((128, 128), Image.ANTIALIAS)
        new_img.save(image_train_path + file_path, format='jpeg', quality=100)
        if index <= 10:
            print(index)
    index += 1

# index = 0
# for file_path in os.listdir(image_train_path):
#     if file_path.endswith('jpg') and not os.path.isfile(image_train_path + file_path.replace('jpg', 'txt')):
#         os.remove(image_train_path + file_path)
#         continue
#     if file_path.endswith('txt') and not os.path.isfile(image_train_path + file_path.replace('txt', 'jpg')):
#         os.remove(image_train_path + file_path)
#         continue
# for file_path in os.listdir(image_valid_path):
#     if file_path.endswith('jpg') and not os.path.isfile(image_valid_path + file_path.replace('jpg', 'txt')):
#         os.remove(image_valid_path + file_path)
#         continue
#     if file_path.endswith('txt') and not os.path.isfile(image_valid_path + file_path.replace('txt', 'jpg')):
#         os.remove(image_valid_path + file_path)
#         continue

# 노말라이즈
# index = 0
# for file_path in os.listdir(image_valid_path):
#     if file_path.endswith('.jpg'):
#         continue
#     with open(image_valid_path + file_path, 'r+') as file:
#         data = file.readline()
#         data = data.split(' ')
#         if index == 0:
#             print(data)
#         # print(file)
#         if data[1] == '':
#             os.remove(image_valid_path + file_path)
#             os.remove(image_valid_path + file_path.replace('txt', 'jpg'))
#             continue
#         data[1] = float(data[1]) / max_x
#         data[2] = float(data[2]) / max_y
#         data[3] = float(data[3]) / max_x
#         data[4] = float(data[4]) / max_y
#         file.close()
#     os.remove(image_valid_path + file_path)
#     with open(image_valid_path + file_path, 'w') as file:    
#         file.write(data[0] + ' ' + str(data[1]) + ' ' + str(data[2]) + ' ' + str(data[3]) + ' ' + str(data[4]))
#         # if index == 0:
#         print(data[0] + ' ' + str(data[1]) + ' ' + str(data[2]) + ' ' + str(data[3]) + ' ' + str(data[4]))
#     index += 1


# index = 0
# for file_path in label_train_files:
#     with open(label_train_path + file_path, 'r') as file:
#         data = json.load(file)
#         json_str = data["FILE"][0]
#         file_name = json_str["FILE_NAME"]
#         items = json_str["ITEMS"][0]
#         package = items["PACKAGE"]
#         if package == "불법차량":
#             package = '1'
#         else:
#             package = '0'
#         box = items["BOX"]
#         box = box.replace(',', ' ')
#         text = package + ' ' + box
#         if (index == 0):
#             print(text)
#         f = open(image_train_path + file_path.replace('json', 'txt'), 'w')
#         f.write(text)
#         if (index == 0):
#             print(f)
#         f.close()
#         index += 1

# txt 만들기
# index = 0
# for file_path in label_train_files:
#     with open(label_train_path + file_path, 'r') as file:
#         data = json.load(file)
#         json_str = data["FILE"][0]
#         file_name = json_str["FILE_NAME"]
#         items = json_str["ITEMS"][0]
#         package = items["PACKAGE"]
#         if package == "불법차량":
#             package = '1'
#         else:
#             package = '0'
#         if items["CLASS"] != "적재불량":
#             if os.path.isfile('img/' + file_path.replace('json', 'jpg')):
#                 os.remove('img/' + file_path.replace('json', 'jpg'))
#             continue
#         box = items["BOX"]
#         box = box.split(',')
#         resolution = json_str["RESOLUTION"]
#         resolution = resolution.split('*')
#         max_x = float(resolution[0])
#         max_y = float(resolution[1])
#         if max_x != 1920:
#             if os.path.isfile('img/' + file_path.replace('json', 'jpg')):
#                 os.remove('img/' + file_path.replace('json', 'jpg'))
#             continue
#         print(file_path)
#         if box[0] == '':
#             if os.path.isfile('img/' + file_path.replace('json', 'jpg')):
#                 os.remove('img/' + file_path.replace('json', 'jpg'))
#             continue
#         box[0] = float(box[0]) / max_x
#         box[1] = float(box[1]) / max_y
#         box[2] = float(box[2]) / max_x
#         box[3] = float(box[3]) / max_y
#         text = package + ' ' + str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3])
#         if (index == 0):
#             print(text)
#         file.close()
#         if os.path.isfile(image_train_path + file_path.replace('json', 'txt')):
#             os.remove(image_train_path + file_path.replace('json', 'txt'))
#         f = open(image_train_path + file_path.replace('json', 'txt'), 'w')
#         f.write(text)
#         if (index == 0):
#             print(f)
#         f.close()
#         index += 1

# 파일 이동
# image_path = 'img/'
# label_path = 'txt/'
# image_files = os.listdir(image_path)
# index = 0
# for file_name in image_files:
#     if index % 10 < 3:
#         if (index == 0):
#             print('0', file_name)
#         shutil.move(image_path + file_name, 'image_valid/' + file_name)
#         # shutil.move(label_path + file_name.replace('.jpg', '.txt'), 'image_valid/' + file_name.replace('.jpg', '.txt'))
#     elif index % 10 >= 3: 
#         if (index == 4):
#             print('4', file_name)
#         shutil.move(image_path + file_name, 'image_train/' + file_name)
#         # shutil.move(label_path + file_name.replace('.jpg', '.txt'), 'image_train/' + file_name.replace('.jpg', '.txt'))
#     index += 1

# label_path = './ml/txt/'
# label_files = os.listdir(label_path)
# index = 0
# for file_name in label_files:
#     if index % 10 < 3:
#         if (index == 0):
#             print('0', file_name)
#         shutil.move(label_path + file_name, './ml/image_valid/' + file_name)
#     elif index % 10 >= 3: 
#         if (index == 4):
#             print('4', file_name)
#         shutil.move(label_path + file_name, './ml/image_train/' + file_name)
#     index += 1

# 파일 롤백
# train_path = 'image_train/'
# train_files = os.listdir(train_path)
# index = 0
# for file_name in train_files:
#     if file_name.endswith('.jpg'):
#         shutil.move(train_path + file_name, 'img/' + file_name)
#     elif file_name.endswith('.txt'):
#         shutil.move(train_path + file_name, 'txt/' + file_name)
#     index += 1
# valid_path = 'image_valid/'
# valid_files = os.listdir(valid_path)
# index = 0
# for file_name in valid_files:
#     if file_name.endswith('.jpg'):
#         shutil.move(valid_path + file_name, 'img/' + file_name)
#     elif file_name.endswith('.txt'):
#         shutil.move(valid_path + file_name, 'txt/' + file_name)
#     index += 1


# 폴더 삭제 및 이동
# os.rmdir('./ml/image_s')
# os.rmdir('./ml/image')

# shutil.rmtree('darknet/yolov5/image_train/')
# shutil.rmtree('darknet/yolov5/image_valid/')

# shutil.copytree('image_train/', 'darknet/yolov5/image_train/')
# shutil.copytree('image_valid/', 'darknet/yolov5/image_valid/')