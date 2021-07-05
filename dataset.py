import torch
import spacy
import cv2
import pandas as pd
import numpy as np
import albumentations as A

import config


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(config.SIZE, config.SIZE, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(config.SIZE, config.SIZE, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )


class FlickerDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, vocabulary, transforms, mode="train"):
        """
        vocabulary: the one which is fed to train dataset is not fit before, but the one fed to valid and test must be fit on train set
        """
        self.vocab = vocabulary
        self.transforms = transforms
        self.img_paths = [
            f"{config.DATA_PATH}/Images/{value}" for value in dataframe["image"].values
        ]
        self.orig_captions = dataframe["caption"].values
        if mode == "train":
            self.vocab.fit(self.orig_captions)
        self.encoded_captions = [
            self.vocab.encode(caption) for caption in self.orig_captions
        ]

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        encoded_caption = self.encoded_captions[idx]

        return (
            torch.tensor(img).permute(2, 0, 1).float(),
            torch.tensor(encoded_caption).long(),
            torch.tensor(len(encoded_caption)).long(),
        )

    def __len__(self):
        return len(self.img_paths)


class Collate:
    def __init__(self, pad_index, batch_first=True):
        self.pad_index = pad_index
        self.batch_first = batch_first

    def __call__(self, batch):
        images, encoded_captions, caption_lengths = list(zip(*batch))
        encoded_captions = torch.nn.utils.rnn.pad_sequence(
            list(encoded_captions),
            batch_first=self.batch_first,
            padding_value=self.pad_index,
        )
        images = torch.stack(list(images), dim=0)
        caption_lengths = torch.stack(list(caption_lengths), dim=0)
        return {
            "images": images,
            "encoded_captions": encoded_captions,
            "caption_lengths": caption_lengths,
        }


# if __name__ == "__main__":
    # df = pd.read_csv(f"{config.DATA_PATH}/captions.txt")
    # from utils import Vocabulary

    # vocab = Vocabulary(freq_threshold=5)
    # dataset = FlickerDataset(
    #     dataframe=df.sample(1000), vocabulary=vocab, transforms=get_transforms("train")
    # )
    # valid_dataset = FlickerDataset(
    #     dataframe=df.sample(100),
    #     vocabulary=dataset.vocab,
    #     transforms=get_transforms("valid"),
    # )
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=8,
    #     collate_fn=Collate(batch_first=True, pad_index=dataset.vocab.stoi["<PAD>"]),
    # )
    # batch = next(iter(dataloader))
    # print("")
