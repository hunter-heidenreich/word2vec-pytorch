from datasets import load_dataset
from torch.utils.data import Dataset


class WindowedDatasets(Dataset):
    
    def __init__(
        self, 
        dataset_name: str = 'wikitext',
        subset: str = 'wikitext-2-raw-v1',
        split: str = 'train',
        window_size: int = 2,
    ):
        self.dataset = load_dataset(dataset_name, subset, split=split)
        self.window_size = window_size
        
        import code 
        code.interact(local=locals())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target = self.data[idx]
        context = self.data[max(0, idx - self.window_size):idx] + self.data[idx + 1:idx + self.window_size + 1]
        return context, target


if __name__ == '__main__':
    dataset = WindowedDatasets()
    print(len(dataset))
    print(dataset[0])
