from torch.utils.data import Dataset
import sys
import torch
import os
import re
import numpy as np

class DropPercentDataset(Dataset):
    def __init__(self, folder) -> None:
        super().__init__()
        file_path = []
        for (dirpath, dirnames, filenames) in os.walk(folder):
            x_path = None
            y_path = None
            for f in filenames:
                if f.endswith('x.npy'):
                    x_path = dirpath + '/' + f
                if f.endswith('y.npy'):
                    y_path = dirpath + '/' + f
            if x_path != None and y_path != None:
                file_path.append((x_path, y_path)) 

        self.xs = []
        self.yhats = []
        self.lens = []
        self.len = 0
        for p in file_path:
            x = np.load(p[0])
            self.xs.append(x)
            yhat = np.load(p[1])
            self.yhats.append(yhat)
            self.lens.append(len(yhat))
            self.len = self.len + len(yhat)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        offset = 0
        for l in self.lens:
            if idx >= l:
                offset = offset + 1
                idx = idx - l
        return {'x': self.xs[offset][idx], 'y': self.yhats[offset][idx]}

if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    folder = "/home/rui/work/drop-percentage-model/data/train/"
    dataset = DropPercentDataset(folder)
    print(dataset[100])
    # print(len(train_dataset))
    # train_dataloader  = torch.utils.data.DataLoader(train_dataset,
                                                    # batch_size=16,
                                                    # collate_fn=my_collate,
                                                    # shuffle=True,
                                                    # drop_last=True,
                                                    # num_workers=5)
    # for batch_idx, data in enumerate((train_dataloader)):
        # pass
    # print(DS[575])
    # print(DS[2504])
    # print(DS[19999])