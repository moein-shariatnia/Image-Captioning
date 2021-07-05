import math
import numpy as np

import torch
import torch.nn as nn

import config
from encoder import Encoder
from utils import positional_encoding_2d, positional_encoding_1d


class TransformerCaptioning(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=config.D_MODEL,
        encoder_dim=config.ENCODER_DIM,
        dropout=config.DROPOUT,
    ):
        super().__init__()
        self.d_model = d_model
        self.device = config.DEVICE
        self.encoder_dim = encoder_dim
        self.encoder = Encoder(config.MODEL_NAME, config.PRETRAINED)
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.encoder_embedding = nn.Linear(encoder_dim, d_model)
        self.encoder_pos_embedding = positional_encoding_2d(
            row=config.FEATURES_GRID_SIZE,
            col=config.FEATURES_GRID_SIZE,
            d_model=d_model,
        ).to(self.device)
        self.encoder_dropout = nn.Dropout(dropout)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_pos_embedding = positional_encoding_1d(
            config.MAX_LEN_TRANFORMER, d_model
        ).to(self.device)
        self.decoder_dropout = nn.Dropout(dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=config.N_HEAD,
            num_encoder_layers=config.N_ENC_LAYERS,
            num_decoder_layers=config.N_DEC_LAYERS,
            dim_feedforward=config.DIM_FF,
            dropout=dropout,
        )
        self.fc_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, vocab_size)

    @staticmethod
    def decoder_nopeek_mask(length):
        mask = torch.tril(torch.ones(length, length)) == 1
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, batch):
        """
        batch containing the following:
        1. image_features: tensor with shape(N, H * W, encoder_dim)
        2. encoded_captions: tensor with shape(N, seq_length)
        3. caption_lengths: tensor with shape(N)
        """
        # sorting
        _, sort_indices = batch['caption_lengths'].sort(dim=0, descending=True)
        
        ###########
        ## Image
        ###########

        images = batch['images'][sort_indices]
        image_features = self.encoder(images).reshape(images.size(0), -1, self.encoder_dim)
        seq_length = image_features.size(1)
        encoder_embeddings = self.encoder_embedding(image_features)
        encoder_embeddings += self.encoder_pos_embedding[:, :seq_length, :]
        encoder_embeddings = self.encoder_dropout(
            encoder_embeddings
        )  # shape: (N, seq_length, d_model)
        encoder_embeddings = encoder_embeddings.permute(
            1, 0, 2
        )  # shape: (seq_length, N, d_model)

        ###########
        ## Caption
        ###########

        encoded_captions = batch["encoded_captions"][sort_indices, :-1]  # sparing the <EOS> token
        seq_length = encoded_captions.size(1)
        tgt_padding_mask = encoded_captions == config.PAD_TOKEN_ID
        decoder_embeddings = self.decoder_embedding(encoded_captions)
        decoder_embeddings *= math.sqrt(self.d_model)
        decoder_embeddings += self.decoder_pos_embedding[:, :seq_length, :]
        decoder_embeddings = self.decoder_dropout(
            decoder_embeddings
        )  # shape: (N, seq_length, d_model)
        decoder_embeddings = decoder_embeddings.permute(
            1, 0, 2
        )  # shape: (seq_length, N, d_model)
        tgt_nopeek_mask = self.decoder_nopeek_mask(seq_length).to(self.device)

        output = self.transformer(
            src=encoder_embeddings,
            tgt=decoder_embeddings,
            tgt_key_padding_mask=tgt_padding_mask,
            tgt_mask=tgt_nopeek_mask
        ).permute(1, 0, 2)
        
        return self.fc(self.fc_dropout(output))
    
    def predict(self, images, encoded_captions):
        image_features = self.encoder(images).reshape(images.size(0), -1, self.encoder_dim)
        seq_length = image_features.size(1)
        encoder_embeddings = self.encoder_embedding(image_features)
        encoder_embeddings += self.encoder_pos_embedding[:, :seq_length, :]
        encoder_embeddings = self.encoder_dropout(
            encoder_embeddings
        )  # shape: (N, seq_length, d_model)
        encoder_embeddings = encoder_embeddings.permute(
            1, 0, 2
        )  # shape: (seq_length, N, d_model)


        seq_length = encoded_captions.size(1)
        tgt_padding_mask = encoded_captions == config.PAD_TOKEN_ID
        decoder_embeddings = self.decoder_embedding(encoded_captions)
        decoder_embeddings *= math.sqrt(self.d_model)
        decoder_embeddings += self.decoder_pos_embedding[:, :seq_length, :]
        decoder_embeddings = self.decoder_dropout(
            decoder_embeddings
        )  # shape: (N, seq_length, d_model)
        decoder_embeddings = decoder_embeddings.permute(
            1, 0, 2
        )  # shape: (seq_length, N, d_model)
        tgt_nopeek_mask = self.decoder_nopeek_mask(seq_length).to(self.device)

        output = self.transformer(
            src=encoder_embeddings,
            tgt=decoder_embeddings,
            tgt_key_padding_mask=tgt_padding_mask,
            tgt_mask=tgt_nopeek_mask
        ).permute(1, 0, 2)
        
        return self.fc(self.fc_dropout(output))

if __name__ == '__main__':
    batch = {
        'images': torch.randn(16, 3, 224, 224),
        'encoded_captions': torch.randint(0, 1999, (16, 25)),
        'caption_lengths': torch.randint(10, 25, (16,))
    }
    batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
    model = TransformerCaptioning(2000).to(config.DEVICE)
    out = model(batch)
    print("")