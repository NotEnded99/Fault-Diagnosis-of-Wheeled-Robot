import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from utils import LoadData
from STDGCN import STDGCN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="/home/users/****", help='The directory of the data')
parser.add_argument('--save_dir', type=str, default="/home/users/****", help='The directory to save the model')
parser.add_argument('--batch_size', type=int, default=64, help='The batch size')
parser.add_argument('--num_trial', type=int, default=5, help='The number of trials')
parser.add_argument('--epoch', type=int, default=200, help='The number of epoch')
parser.add_argument('--lr', type=float, default=1e-3, help='The learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='The weight decay')
parser.add_argument('--difference', type=int, default=4, choices=[0, 1, 2, 3, 4], help='The parameter for the difference layer')
parser.add_argument('--node_features', type=int, default=5, help='The dimension of node features after feature enhancement')
parser.add_argument('--num_class', type=int, default=6, help='The number of fault classes')
parser.add_argument('--num_nodes', type=int, default=18, help='The number of nodes')
args = parser.parse_args()

def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Loading Model
    STDGCN_model = STDGCN(num_nodes = args.num_nodes, node_features = args.node_features, num_class = args.num_class )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    STDGCN_model = STDGCN_model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params = STDGCN_model.parameters(), lr = args.lr, weight_decay = args.weight_decay) 
    
    history_best_accuracy = 0

    result = []
    
    # Train model
    STDGCN_model.train()
    
    for epoch in range(args.epoch):
        
        for data in train_loader: 

            STDGCN_model.zero_grad()

            predict_value = STDGCN_model(data, device, args.difference).to(torch.device("cpu"))  

            loss = criterion(predict_value, data["label"])

            loss.backward()

            optimizer.step()
    
        STDGCN_model.eval()

        with torch.no_grad():

            correct = 0.

            for data in test_loader:
                predict_value = STDGCN_model(data, device, args.difference).to(torch.device("cpu"))  
                pred = predict_value.max(dim=1)[1]
                correct += pred.eq(data["label"]).sum().item()

            accuracy = 100*correct/len(test_loader.dataset)

            result.append(accuracy)

            print("Epoch: {:d} | Test Accuracy: {:02.4f}".format(epoch, accuracy))
            
            if accuracy > history_best_accuracy:
                    history_best_accuracy = accuracy
                    torch.save(STDGCN_model, args.save_dir+'/STDGCN_model_{i}.pth'.format(i = i))

    return result

if __name__ == '__main__':
    
    # Loading Dataset
    train_data = LoadData(args.data_dir, model = 'train')
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle=True)
    
    test_data = LoadData(args.data_dir, model = 'test')
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle=False)
    
    total_result = np.zeros((args.num_trial, args.epoch)) 
    
    for i in range(args.num_trial):  
        set_random_seed(i)
        result = main()
        total_result[i,:] = np.array(result)

    max_accuracy = np.max(np.amax(total_result, axis = 1))
    mean_accuracy = np.sum(np.amax(total_result, axis = 1))/args.num_trial
    min_accuracy = np.min(np.amax(total_result, axis = 1))
    std = np.std(np.amax(total_result, axis = 1))
    
    print( "\nFinal Mean Accuracy: {:02.4f}".format(mean_accuracy))
    print( "\nFinal max Accuracy: {:02.4f}".format(max_accuracy))
    print( "\nFinal min Accuracy: {:02.4f}".format(min_accuracy))
    print( "\nFinal std : {:02.4f}".format(std))
