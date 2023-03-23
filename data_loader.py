import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


def loader(pic_free_path, pic_charge_path):
    # /* *********************处理免费字体************************* */
    files_free = os.listdir(pic_free_path)
    data = []

    for free_file in files_free:
        label = free_file.split('-')[0]
        img_path = pic_free_path + '/' + free_file
        image = Image.open(img_path).convert('L')
        img = np.array(image)
        data.append((img, label))

    # /* *********************处理收费字体************************* */
    files_charge = os.listdir(pic_charge_path)

    for charge_file in files_charge:
        label = free_file.split('-')[0]
        img_path = pic_charge_path + '/' + charge_file
        image = Image.open(img_path).convert('L')
        img = np.array(image)
        data.append((img, label))

    train_lst, test_lst = train_test_split(data, test_size=0.1, random_state=86)

    a = []
    b = []
    c = []
    d = []

    for i in train_lst:
        a.append(i[0])
        b.append(i[1])
    for i in test_lst:
        c.append(i[0])
        d.append(i[1])

    x_train = np.array(a)
    y_train = np.array(b)
    x_test = np.array(c)
    y_test = np.array(d)
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    data_charge_path = "D:/data_charge"
    data_free_path = "D:/data_free"
    print(loader(data_free_path, data_charge_path))
