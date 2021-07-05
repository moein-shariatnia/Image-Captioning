import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

import config
from utils import Vocabulary, show_results
from dataset import FlickerDataset, Collate, get_transforms
from encoder import Encoder
from decoder import DecoderWithAttention
from engine import train_one_epoch, predict


def make_loaders(**kwargs):
    dataset = FlickerDataset(**kwargs)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True, # Putting it True for both loaders to get diversity in predict function
        collate_fn=Collate(pad_index=dataset.vocab.stoi["<PAD>"], batch_first=True),
        num_workers=config.NUM_WORKERS,
    )
    return dataloader


def make_train_valid_dfs():
    dataframe = pd.read_csv(f"{config.DF_PATH}/captions.csv")
    image_ids = np.arange(0, 10000)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def build_model(vocab_size):
    encoder = Encoder(model_name=config.MODEL_NAME, pretrained=config.PRETRAINED).to(
        config.DEVICE
    )
    decoder = DecoderWithAttention(
        attention_dim=config.ATTENTION_DIM,
        decoder_dim=config.DECODER_DIM,
        embed_dim=config.EMBED_DIM,
        vocab_size=vocab_size,
        device=config.DEVICE,
        encoder_dim=config.ENCODER_DIM,
        dropout=config.DROPOUT,
    ).to(config.DEVICE)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=config.ENCODER_LR)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=config.DECODER_LR)

    return encoder, decoder, encoder_optimizer, decoder_optimizer


def main():

    train_dataframe, valid_dataframe = make_train_valid_dfs()
    train_loader = make_loaders(
        dataframe=train_dataframe,
        vocabulary=Vocabulary(freq_threshold=config.FREQ_THRESHOLD),
        transforms=get_transforms(mode="train"),
        mode="train",
    )
    vocab = train_loader.dataset.vocab
    valid_loader = make_loaders(
        dataframe=valid_dataframe,
        vocabulary=vocab,
        transforms=get_transforms(mode="valid"),
        mode="valid",
    )
    encoder, decoder, encoder_optimizer, decoder_optimizer = build_model(
        vocab_size=vocab.vocab_size
    )
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        encoder_optimizer, factor=config.FACTOR, patience=config.PATIENCE, verbose=True
    )
    decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        decoder_optimizer, factor=config.FACTOR, patience=config.PATIENCE, verbose=True
    )

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(
            train_loader,
            encoder,
            decoder,
            criterion,
            encoder_optimizer,
            decoder_optimizer,
            config.DEVICE,
        )

        # encoder_scheduler.step(valid_loss.avg)
        # decoder_scheduler.step(valid_loss.avg)

        predict(valid_loader, encoder, decoder, config.DEVICE)

if __name__ == "__main__":
    main()
