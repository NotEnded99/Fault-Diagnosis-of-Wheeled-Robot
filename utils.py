import torch
import numpy as np
from torch.utils.data import Dataset

class LoadData(Dataset):
    def __init__(self, data_dir, model):
        
        self.model = model
        self.graph = get_adjacent_matrix() 

        if self.model == 'train':
            self.train_data_x = np.load(data_dir+'/x_train.npy') # x_train.npy  [N, 18, 100]
            self.train_data_y = np.load(data_dir+'/y_train.npy') # y_train.npy  [M, 18, 100]
        elif self.model == 'test':
            self.test_data_x = np.load(data_dir+'/x_test.npy')   # x_test.npy  [N]
            self.test_data_y = np.load(data_dir+'/y_test.npy')   # y_test.npy  [M]

    def __getitem__(self, index): 

        if self.model == 'train':
            data_x = LoadData.to_float_tensor(self.train_data_x[index,:,:])  
            data_x = data_x.unsqueeze(2)                                   
            data_y = LoadData.to_long_tensor(self.train_data_y[index])      
        elif self.model == 'test':
            data_x = LoadData.to_float_tensor(self.test_data_x[index,:,:])  
            data_x = data_x.unsqueeze(2)                                   
            data_y = LoadData.to_long_tensor(self.test_data_y[index])       

        return {"graph": LoadData.to_float_tensor(self.graph), "sensor_data": data_x, "label": data_y}
    
    def __len__(self):
        if self.model == 'train':
            dataset_size = len(self.train_data_x)
        elif self.model == 'test':
            dataset_size = len(self.test_data_x)
        else:
            raise ValueError("The parameter of model must be train or test")
        return dataset_size
    
    @staticmethod
    def to_float_tensor(data):
        return torch.tensor(data, dtype=torch.float)
    
    @staticmethod
    def to_long_tensor(data):
        return torch.tensor(data, dtype=torch.long)

def get_adjacent_matrix():
    edge_inf = np.zeros((18, 18))   
    edge_inf[0, 0] += 1; edge_inf[0, 1] += 1; edge_inf[0, 2] += 1; edge_inf[0, 3] += 1; edge_inf[0, 4] += 1; edge_inf[0, 17] += 1
    edge_inf[1, 0] += 1; edge_inf[1, 1] += 1; edge_inf[1, 2] += 1; edge_inf[1, 3] += 1; edge_inf[1, 5] += 1; edge_inf[1, 17] += 1
    edge_inf[2, 0] += 1; edge_inf[2, 1] += 1; edge_inf[2, 2] += 1; edge_inf[2, 3] += 1; edge_inf[2, 6] += 1; edge_inf[2, 17] += 1
    edge_inf[3, 0] += 1; edge_inf[3, 1] += 1; edge_inf[3, 2] += 1; edge_inf[3, 3] += 1; edge_inf[3, 7] += 1; edge_inf[3, 17] += 1
    edge_inf[4, 0] += 1; edge_inf[4, 4] += 1; 
    edge_inf[5, 1] += 1; edge_inf[5, 5] += 1; 
    edge_inf[6, 2] += 1; edge_inf[6, 6] += 1; 
    edge_inf[7, 3] += 1; edge_inf[7, 7] += 1;
    edge_inf[8, 8] += 1; edge_inf[8, 12] += 1; edge_inf[8, 13] += 1; edge_inf[8, 17] += 1
    edge_inf[9, 9] += 1; edge_inf[9, 12] += 1; edge_inf[9, 13] += 1; edge_inf[9, 17] += 1
    edge_inf[10, 10] += 1; edge_inf[10, 12] += 1; edge_inf[10, 13] += 1; edge_inf[10, 17] += 1
    edge_inf[11, 11] += 1; edge_inf[11, 12] += 1; edge_inf[11, 13] += 1; edge_inf[11, 17] += 1
    edge_inf[12, 8] += 1; edge_inf[12, 9] += 1; edge_inf[12, 10] += 1; edge_inf[12, 11] += 1;edge_inf[12, 12] += 1
    edge_inf[13, 8] += 1; edge_inf[13, 9] += 1; edge_inf[13, 10] += 1; edge_inf[13, 11] += 1;edge_inf[13, 13] += 1
    edge_inf[14, 14] += 1
    edge_inf[15, 15] += 1
    edge_inf[16, 16] += 1
    edge_inf[17, 0] += 1; edge_inf[17, 1] += 1; edge_inf[17, 2] += 1; edge_inf[17, 3] += 1    
    edge_inf[17, 8] += 1; edge_inf[17, 9] += 1; edge_inf[17, 10] += 1; edge_inf[17, 11] += 1; edge_inf[17, 17] += 1    
    
    return edge_inf

if __name__ == '__main__':
    pass
 