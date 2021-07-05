import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

import config
from dataset import get_transforms
from utils import Vocabulary, AvgMeter, get_lr, remove_normalization
from encoder import Encoder
from transformer import CaptioningTransformer
from transformer_beta import TransformerCaptioning
from main import make_loaders, make_train_valid_dfs


def one_epoch(
    model,
    criterion,
    loader,
    device,
    optimizer=None,
    lr_scheduler=None,
    mode="train",
    step="batch",
):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(loader, total=len(loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = model(batch)  # shape: (N, T, d_model)
        caption_lengths, sort_indices = batch['caption_lengths'].sort(dim=0, descending=True)
        caption_lengths = (caption_lengths - 1).tolist()
        targets = batch["encoded_captions"][sort_indices, 1:]
        targets = pack_padded_sequence(targets, caption_lengths, batch_first=True).data
        preds = pack_padded_sequence(preds, caption_lengths, batch_first=True).data
        # vocab_size = preds.size(-1)
        # loss = criterion(preds.reshape(-1, vocab_size), targets.reshape(-1))
        loss = criterion(preds, targets)
        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step == "batch":
                lr_scheduler.step()

        count = batch["images"].size(0)
        loss_meter.update(loss.item(), count)

        if mode == "train":
            tqdm_object.set_postfix(loss=loss_meter.avg, lr=get_lr(optimizer))
        else:
            tqdm_object.set_postfix(loss=loss_meter.avg)

    return loss_meter

def evaluate(model, loader):
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        images = batch['images'].to(config.DEVICE)
        output = torch.tensor([1]).long().expand(images.size(0), 1).to(config.DEVICE)
        for _ in range(config.MAX_LEN_TRANFORMER - 1):
            preds = model.predict(images, output)
            preds = preds[:, -1:, :].argmax(dim=-1).long()
            output = torch.cat([output, preds], dim=1)
    
    return images, output, batch['encoded_captions']

# def show_results(model, batch, vocab):
#     with torch.no_grad():
#         preds = model(batch).argmax(dim=-1)
    
#     _, sort_indices = batch['caption_lengths'].sort(dim=0, descending=True)
#     targets = batch["encoded_captions"][sort_indices, 1:]


#     _, axes = plt.subplots(2, 2, figsize=(10, 10))
#     for i, (ax, img, target_caption, pred_caption) in enumerate(
#         zip(
#             axes.flatten(),
#             batch["images"][sort_indices].cpu(),
#             targets.cpu(),
#             preds.cpu(),
#         )
#     ):
#         img = remove_normalization(img)
#         ax.imshow(img.permute(1, 2, 0))
#         ax.set_title(f"Image {i + 1}")
#         target_caption = " ".join(
#             [token for token in vocab.decode(target_caption.tolist()) if token != "<PAD>"]
#         )
#         pred_caption = " ".join(
#             [token for token in vocab.decode(pred_caption.tolist()) if token != "<PAD>"]
#         )
        
#         print(f"Image {i + 1}")
#         print(f"Target: {target_caption}")
#         print(f"Predicted: {pred_caption}")
#         print("**************************")
#     plt.show()

def show_results(images, preds, targets, vocab):
    _, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, (ax, img, target_caption, pred_caption) in enumerate(
        zip(
            axes.flatten(),
            images.cpu(),
            targets.cpu(),
            preds.cpu(),
        )
    ):
        img = remove_normalization(img)
        ax.imshow(img.permute(1, 2, 0))
        ax.set_title(f"Image {i + 1}")
        target_caption = " ".join(
            [token for token in vocab.decode(target_caption.tolist()) if token not in ("<PAD>", "<EOS>")]
        )
        pred_caption = " ".join(
            [token for token in vocab.decode(pred_caption.tolist()) if token not in ("<PAD>", "<EOS>")]
        )
        
        print(f"Image {i + 1}")
        print(f"Target: {target_caption}")
        print(f"Predicted: {pred_caption}")
        print("**************************")
    plt.show()

def train_eval(
    epochs,
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    device,
    config,
    lr_scheduler=None,
):

    best_loss = float("inf")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")

        model.train()
        train_loss = one_epoch(
            model,
            criterion,
            train_loader,
            device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            mode="train",
            step=config.STEP,
        )

        images, preds, targets = evaluate(model, valid_loader)
        show_results(images, preds, targets, valid_loader.dataset.vocab)

        # model.eval()
        # with torch.no_grad():
        #     valid_loss = one_epoch(
        #         model,
        #         criterion,
        #         valid_loader,
        #         device,
        #         optimizer=None,
        #         lr_scheduler=None,
        #         mode="valid",
        #     )
        #     batch = next(iter(valid_loader))
        #     batch = {k: v.to(device) for k, v in batch.items()}
        #     show_results(model, batch, valid_loader.dataset.vocab)

        # if valid_loss.avg < best_loss:
        #     best_loss = valid_loss.avg
        #     torch.save(
        #         model.state_dict(), f"{config.MODEL_PATH}/{config.MODEL_SAVE_NAME}"
        #     )
        #     print("Saved best model!")

        # # or you could do: if step == "epoch":
        # if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #     lr_scheduler.step(valid_loss.avg)


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

    # model = CaptioningTransformer(vocab_size=vocab.vocab_size, d_model=config.D_MODEL).to(config.DEVICE)
    model = TransformerCaptioning(vocab_size=vocab.vocab_size).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=3
    )
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    train_eval(
        config.EPOCHS,
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        config.DEVICE,
        config,
        lr_scheduler,
    )

    # encoder_scheduler.step(valid_loss.avg)
    # decoder_scheduler.step(valid_loss.avg)

    # predict(valid_loader, encoder, decoder, config.DEVICE)


if __name__ == "__main__":
    main()
    # batch = {
    #     "images": torch.randn(8, 3, 224, 224),
    #     "encoded_captions": torch.randint(0, 2000, (8, 20)),
    # }

    # model = CaptioningTransformer()
    # out = model(batch)
    # print("")
