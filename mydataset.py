from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, X, C, Y):
        self.X = X
        self.C = C
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.C[idx], self.Y[idx]