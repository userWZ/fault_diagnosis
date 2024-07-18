from base_dataset import BaseDataset
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import scipy
import numpy as np

class CWRU(BaseDataset):
    def load_data(self, data_dir):
        for class_idx, class_folder in enumerate(os.listdir(data_dir)):
            class_dir = os.path.join(data_dir, class_folder)
            # print(class_dir)

            mat  = scipy.io.loadmat(class_dir)
            key_name = list(mat.keys())[3]
            # print(key_name)
            
            DE_data = mat.get(key_name)
            DE_data = DE_data.squeeze(-1)
            
            samples = []
            num_segments = ((DE_data.shape[0] - self.data_length) // self.stride) if self.num_samples is None else self.num_samples
            if self.num_samples is not None:
                assert (num_segments * self.data_length) > DE_data.shape[0], "数据长度不足以每个文件生成{}个样本".format(
                    self.num_samples)
            for i in  range(num_segments):
                sample = DE_data[i * self.stride: (i * self.stride + self.data_length)]
                samples.append(sample)
            self.data.extend(samples)
            self.label.extend([class_idx] * len(samples))
            # print(len(self.data))
                    
if __name__ == '__main__':
    data_dir = ['data\\cwru\\0HP','data\\cwru\\1HP']
    data_length = 1024
    stride = 200
    dataset = CWRU(data_dir, data_length, stride, num_samples=600)
    
    
    