import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

import config
from utils import AvgMeter, show_results


def train_one_epoch(
    train_loader,
    encoder,
    decoder,
    criterion,
    encoder_optimizer,
    decoder_optimizer,
    device,
):

    loss_meter = AvgMeter()
    # switch to train mode
    encoder.train()
    decoder.train()

    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch_size = batch["images"].size(0)
        features = encoder(batch["images"])
        (
            predictions,
            encoded_captions_sorted,
            decode_lengths,
            alphas,
            sort_ind,
        ) = decoder(
            features, batch["encoded_captions"], batch["caption_lengths"].unsqueeze(1)
        )
        targets = encoded_captions_sorted[:, 1:]
        predictions = pack_padded_sequence(
            predictions, decode_lengths, batch_first=True
        ).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        loss = criterion(predictions, targets)
        # record loss
        loss_meter.update(loss.item(), batch_size)
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.MAX_GRAD_NORM)
        _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), config.MAX_GRAD_NORM)
        encoder_optimizer.step()
        decoder_optimizer.step()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        tqdm_object.set_postfix(train_loss=loss_meter.avg)

    return loss_meter


def predict(valid_loader, encoder, decoder, device, show=True):
    # switch to evaluation mode
    encoder.eval()
    decoder.eval()

    vocab = valid_loader.dataset.vocab
    batch = next(iter(valid_loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        features = encoder(batch["images"])
        predictions = decoder.predict(features, config.MAX_LEN_PRED, vocab)
        predicted_sequence = torch.argmax(predictions.cpu(), -1)
           
        captions = vocab.decode_batch(predicted_sequence)
        show_results(captions, batch, vocab)

    return captions


# if __name__ == "__main__":
#     import pandas as pd

#     import config
#     from dataset import FlickerDataset, get_transforms, Collate
#     from utils import Vocabulary, remove_normalization
#     from encoder import Encoder
#     from decoder import DecoderWithAttention

#     df = pd.read_csv(f"{config.DATA_PATH}/captions.txt")
#     vocab = Vocabulary(freq_threshold=5)
#     train_dataset = FlickerDataset(
#         dataframe=df, vocabulary=vocab, transforms=get_transforms("train"), mode="train"
#     )
#     valid_dataset = FlickerDataset(
#         dataframe=df.sample(100),
#         vocabulary=train_dataset.vocab,
#         transforms=get_transforms("valid"),
#         mode="valid",
#     )
#     train_dataloader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=8,
#         collate_fn=Collate(batch_first=True, pad_index=train_dataset.vocab.stoi["<PAD>"]),
#         shuffle=True,
#     )
#     valid_dataloader = torch.utils.data.DataLoader(
#         valid_dataset,
#         batch_size=8,
#         collate_fn=Collate(batch_first=True, pad_index=valid_dataset.vocab.stoi["<PAD>"]),
#         shuffle=False,
#     )

#     device = torch.device("cuda")
#     encoder = Encoder().to(device)
#     decoder = DecoderWithAttention(
#         attention_dim=128,
#         embed_dim=128,
#         decoder_dim=128,
#         vocab_size=3000,
#         device=device,
#     ).to(device)
#     encoder_optimizer = torch.optim.Adam(encoder.parameters())
#     decoder_optimizer = torch.optim.Adam(decoder.parameters())
    # train_one_epoch(
    #     train_datalaoder,
    #     encoder,
    #     decoder,
    #     torch.nn.CrossEntropyLoss(),
    #     encoder_optimizer,
    #     decoder_optimizer,
    #     device,
    # )
    # valid_one_epoch(valid_dataloader, encoder, decoder, torch.nn.CrossEntropyLoss(), device)

