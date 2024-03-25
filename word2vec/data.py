from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from word2vec.tokenizer import get_tokenizer, add_spaces_around_foreign_characters


class WindowedDatasets(Dataset):
    
    def __init__(
        self, 
        dataset_name: str = 'wikitext',
        subset: str = 'wikitext-2-raw-v1',
        split: str = 'train',
        window_size: int = 5,
    ):
        self.window_size = window_size
        self.dataset = load_dataset(dataset_name, subset, split=split)['text']
        self.dataset = [add_spaces_around_foreign_characters(text) for text in self.dataset if text.strip()]
        
        # get tokenizer
        self.tokenizer = get_tokenizer(data=self.dataset, dataset_name=dataset_name, subset=subset)
        
        # encode dataset
        self.centers = []
        self.contexts = []
        for text in tqdm(self.dataset, desc='Encoding dataset'):
            encoding = self.tokenizer.encode(text)
            datum = encoding.ids
            # pre- and post-pad with <pad> token
            datum = [self.tokenizer.token_to_id('<pad>')] * self.window_size + datum + [self.tokenizer.token_to_id('<pad>')] * self.window_size
            for i in range(self.window_size, len(datum) - self.window_size):
                center = datum[i]
                context = datum[i - self.window_size:i] + datum[i + 1:i + self.window_size + 1]
                self.centers.append(center)
                self.contexts.append(context)
            
    def __len__(self):
        return len(self.centers)
    
    def __getitem__(self, idx):
        return self.centers[idx], self.contexts[idx]


if __name__ == '__main__':
    dataset = WindowedDatasets()
    print(len(dataset))
    print(dataset[0])
