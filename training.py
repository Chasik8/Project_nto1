import time
import cv2
import torch
import torch.nn as nn
from Model import Net
from dop_functions import *


# ------------------------------------------------------------------------------------
class Launch_training:
    def __init__(self):
        f = open('config.txt', 'r')
        self.epoch = int(list(f.readline().strip().split())[2])
        self.size_w = int(list(f.readline().strip().split())[2])
        self.size_h = int(list(f.readline().strip().split())[2])
        self.net = Net(self.epoch, self.cnn_rech(self.size_w) * self.cnn_rech(self.size_w) * 90)
        self.password = int(list(f.readline().strip().split())[1])
        self.kol_bit_password = int(list(f.readline().strip().split())[7])
        self.path_in_model = list(f.readline().strip().split())[4]
        self.path_save_the_best = list(f.readline().strip().split())[5]
        self.path_save_last = list(f.readline().strip().split())[5]
        self.loading = bool(list(f.readline().strip().split())[5])
        self.loading_best_num = int(list(f.readline().strip().split())[3])
        self.loading_not_num = int(list(f.readline().strip().split())[3])
        self.path_in_train_x = list(f.readline().strip().split())[7]
        self.path_in_train_y = list(f.readline().strip().split())[7]
        self.Train()
        f.close()
    def cnn_rech(self, n):
        f = 5
        p = 2
        s = 1
        fm = 6
        pm = 0
        sm = 3
        for i in range(2):
            n = int((n - f + (2 * p)) / s + 1)
            n = int((n - fm + (2 * pm)) / sm + 1)
        return n
    def Run(self):
        if self.loading:
            self.net = self.in_model()
        self.net.cuda()
        criterion = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.net.learning_rate)
        otw_min = torch.tensor(10000, device='cuda:0')
        x_train, y_train = self.Train()
        # ----------------------------------------------------------------------
        for epoch in range(self.net.num_epochs):
            loss = 0
            for i in range(len(x_train)):
                optimizer.zero_grad()
                outputs = self.net(x_train[i])
                loss = criterion(outputs,
                                 y_train[i])
                loss.backward()
                optimizer.step()
            if epoch == 0:
                otw_min = loss.data
            else:
                if otw_min >= loss.data:
                    otw_min = loss.data
                    torch.save(self.net.state_dict(), self.path_save_the_best)
            print(epoch, loss.data)
        torch.save(self.net.state_dict(), self.path_save_last)
        print(otw_min)

    def in_model(self):
        model = Net(self.epoch,self.cnn_rech(self.size_w) * self.cnn_rech(self.size_w) * 90)
        model.load_state_dict(torch.load(self.path_in_model))
        model.eval()
        return model

    def bin(self, num):
        newNum = ''
        base = 2
        l = self.kol_bit_password
        while num > 0:
            newNum = str(num % base) + newNum
            num //= base
        out = ''
        for i in range(l - len(newNum)):
            out += '0'
        out += newNum
        out = list(out)
        for i in range(l):
            out[i] = int(out[i])
            if out[i] != 0:
                out[i] = 10
        return out

    def Train(self):
        img = np.asarray(cv2.cvtColor(cv2.imread(f'{self.path_in_train_x}\\test_new0.jpg'), cv2.COLOR_BGR2RGB))
        img = resixe_3d(img)
        img = np.expand_dims(img, 0)
        img = np.expand_dims(img, 0)
        img_train_x = img
        img_train_y_i = np.array(self.bin(self.password))
        img_train_y_i_not = [0] * self.kol_bit_password
        img_train_y = img_train_y_i
        for i in range(1, self.loading_best_num):
            img = np.asarray(cv2.cvtColor(cv2.imread(f'{self.path_in_train_x}\\test_new{i}.jpg'), cv2.COLOR_BGR2RGB))
            img = resixe_3d(img)
            img = np.expand_dims(img, 0)
            img = np.expand_dims(img, 0)
            img_train_x = np.vstack((img_train_x, img))
            img_train_y = np.vstack((img_train_y, img_train_y_i))
        for i in range(self.loading_not_num):
            img = np.asarray(cv2.cvtColor(cv2.imread(f'{self.path_in_train_y}\\{i}.jpg'), cv2.COLOR_BGR2RGB))
            img = resixe_3d(img)
            img = np.expand_dims(img, 0)
            img = np.expand_dims(img, 0)
            img_train_x = np.vstack((img_train_x, img))
            img_train_y = np.vstack((img_train_y, img_train_y_i_not))
        img_train_x = img_train_x.astype(np.float32)
        img_train_y = img_train_y.astype(np.float32)
        img_train_x = torch.from_numpy(img_train_x)
        img_train_y = torch.from_numpy(img_train_y)
        img_train_x = img_train_x.to('cuda:0')
        img_train_y = img_train_y.to('cuda:0')
        return img_train_x, img_train_y


def print_hi(name):
    tim = time.time()
    Launch_training().Run()
    print(time.time() - tim)
if __name__ == '__main__':
    print_hi('PyCharm')

