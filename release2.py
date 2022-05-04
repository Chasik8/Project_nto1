import torch
from Model import Net
from dop_functions import *
import cv2


# ------------------------------------------------------------------------------------
class Launch_release:
    def __init__(self):
        f = open('config.txt', 'r')
        self.epoch = int(list(f.readline().strip().split())[2])
        self.size_w = int(list(f.readline().strip().split())[2])
        self.size_h = int(list(f.readline().strip().split())[2])
        self.net = Net(self.epoch, self.cnn_rech(self.size_w) * self.cnn_rech(self.size_w) * 90)
        self.password = int(list(f.readline().strip().split())[1])
        self.kol_bit_password = int(list(f.readline().strip().split())[7])
        self.path_in_model = list(f.readline().strip().split())[4]
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

    def Run_in_model(self):
        self.net = self.in_model()
        self.net.cuda()
        return self.net(self.Train())

    def in_model(self):
        model = Net(self.epoch, self.cnn_rech(self.size_w) * self.cnn_rech(self.size_w) * 90)
        model.load_state_dict(torch.load(self.path_in_model))
        model.eval()
        return model

    def Train(self):
        img = self.save_face()
        img = resixe_3d(img)
        img = np.expand_dims(img, 0)
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        img = img.to('cuda:0')
        return img

    def save_face(self):
        cap = cv2.VideoCapture(0)
        size = [self.size_w, self.size_h]
        h = True
        img = 0
        while h:
            ret, frame = cap.read()
            rech_new(frame, size)
            cv2.imshow('video feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                img = frame
                h = False
        cap.release()
        cv2.destroyAllWindows()
        return img

    def iint(self, a):
        if abs(a) >= 5:
            return 1
        else:
            return 0

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

    def main(self):
        out = self.Run_in_model()
        out = out[0].cpu().data.numpy()
        otw = ''
        for i in out:
            otw += str(self.iint(i))
        if int(otw, 2) == self.password:
            print("Yes")
        else:
            print("No")


if __name__ == '__main__':
    Launch_release().main()
