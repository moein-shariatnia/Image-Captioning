import math
import torch
import torch.nn as nn

import config
from encoder import Encoder

class CaptioningTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512):
        super().__init__()
        self.d_model = d_model
        self.encoder = Encoder(model_name=config.MODEL_NAME, pretrained=config.PRETRAINED)
        self.transformer = nn.Transformer(d_model=d_model)
        self.encoder_pos_embedding = nn.Embedding(7 * 7, config.ENCODER_DIM) # the encoder output is N * 7 * 7 * 512
        self.decoder_pos_embedding = nn.Embedding(config.MAX_LEN_TRANFORMER, d_model)
        # you can change the positional embeddings to sin/cos embeddings
        # make sure you try them
        self.decoder_word_embedding = nn.Embedding(vocab_size, d_model)

        self.classifier = nn.Linear(d_model, vocab_size)
        self.device = config.DEVICE

    @staticmethod    
    def positionalencoding2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return:  height * width * d_model position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.permute(1, 2, 0)

    @staticmethod
    def decoder_nopeek_mask(length):
        mask = torch.tril(torch.ones(length, length)) == 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, batch):
        images = batch['images']
        batch_size = images.size(0)
        # sorting
        _, sort_indices = batch['caption_lengths'].sort(dim=0, descending=True)
        images = images[sort_indices]

        # image_features = self.encoder(images) * math.sqrt(self.d_model)
        # image_features_embedded = image_features + self.encoder_pos_embedding.repeat(batch_size, 1, 1, 1).to(self.device) # using dropout could help
        # image_features_embedded = image_features_embedded.view(batch_size, -1, image_features.size(-1)) # shape: (N, T, encoder_dim)
        image_features = self.encoder(images)
        image_features = image_features.reshape(batch_size, -1, image_features.size(-1))
        num_patches = image_features.size(1)
        positions = torch.arange(0, num_patches).expand(batch_size, num_patches).to(self.device)
        image_pos_embeddings = self.encoder_pos_embedding(positions)
        image_features_embedded = image_features + image_pos_embeddings # shape: (N, T, encoder_dim)
        image_features_embedded = image_features_embedded.permute(1, 0, 2).contiguous() # shape: (T, N, encoder_dim)

        encoded_captions = batch['encoded_captions'][sort_indices, :-1] # shape: (N, max_batch_length - 1)
        target_pad_mask = encoded_captions == config.PAD_TOKEN_ID
        target_nopeek_mask = self.decoder_nopeek_mask(encoded_captions.size(1)).to(self.device)
        encoded_captions_embedded = self.decoder_word_embedding(encoded_captions) # * math.sqrt(self.d_model)
        seq_length = encoded_captions.size(-1)
        positions = torch.arange(0, seq_length).expand(batch_size, seq_length).to(self.device)
        encoded_captions_embedded += self.decoder_pos_embedding(positions) # using dropout could help
        encoded_captions_embedded = encoded_captions_embedded.permute(1, 0, 2).contiguous()
        

        
        output = self.transformer(src=image_features_embedded, 
                                  tgt=encoded_captions_embedded, 
                                  tgt_key_padding_mask=target_pad_mask,
                                  tgt_mask=target_nopeek_mask)
        
        output = self.classifier(output.permute(1, 0, 2).contiguous())
        
        return output