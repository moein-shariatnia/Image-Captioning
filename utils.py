from collections import Counter
import spacy
import numpy as np
import torch
import matplotlib.pyplot as plt


class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.tokenizer = spacy.load("en_core_web_sm")

    def tokenize(self, sentence):
        return [token.text.lower() for token in self.tokenizer.tokenizer(sentence)]

    def fit(self, text_list):
        tokens = []
        for sentence in text_list:
            tokens.extend(self.tokenize(sentence))

        idx = 4  # three first indices are for special tokens
        counts = Counter(tokens)
        for word, count in counts.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
        self.vocab_size = len(self.stoi)

    def encode(self, sentence):
        tokenized = self.tokenize(sentence)
        encoded = [self.stoi["<SOS>"]]
        for token in tokenized:
            if token not in list(self.stoi.keys()):
                encoded.append(self.stoi["<UNK>"])
            else:
                encoded.append(self.stoi[token])
        encoded += [self.stoi["<EOS>"]]
        return encoded

    def decode(self, encoded):
        decoded = [self.itos[token] for token in encoded]
        return decoded

    def decode_batch(self, batch):
        batch = batch.tolist()  # a list of lists (N, MAX_LEN)
        captions = []
        for item in batch:
            captions.append(self.decode(item))

        return captions


def remove_normalization(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    image tensor shape (channel, height, width)
    """
    mean, std = torch.tensor(mean), torch.tensor(std)
    mean = mean.unsqueeze(1).unsqueeze(2)
    std = std.unsqueeze(1).unsqueeze(2)
    return image * std + mean


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def show_results(predicted_captions, batch, vocab):
    _, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, (ax, img, enc_caption, pred_caption) in enumerate(
        zip(
            axes.flatten(),
            batch["images"].cpu(),
            batch["encoded_captions"].cpu(),
            predicted_captions,
        )
    ):
        img = remove_normalization(img)
        ax.imshow(img.permute(1, 2, 0))
        ax.set_title(f"Image {i + 1}")
        target_caption = " ".join(
            [token for token in vocab.decode(enc_caption.tolist()) if token != "<PAD>"]
        )

        print(f"Image {i + 1}")
        print(f"Target: {target_caption}")
        print(f"Predicted: {pred_caption}")
        print("**************************")
    plt.show()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def positional_encoding_1d(position, d_model):
   angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis, :],
                           d_model)

   angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
   angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
   pos_encoding = angle_rads[np.newaxis, ...]

   return torch.tensor(pos_encoding).float()

def positional_encoding_2d(row, col, d_model):
    assert d_model % 2 == 0
    row_pos = np.repeat(np.arange(row), col)[:, np.newaxis]
    col_pos = np.repeat(np.expand_dims(np.arange(col), 0), row, axis=0).reshape(-1, 1)

    angle_rads_row = get_angles(
        row_pos, np.arange(d_model // 2)[np.newaxis, :], d_model // 2
    )
    angle_rads_col = get_angles(
        col_pos, np.arange(d_model // 2)[np.newaxis, :], d_model // 2
    )

    angle_rads_row[:, 0::2] = np.sin(angle_rads_row[:, 0::2])
    angle_rads_row[:, 1::2] = np.cos(angle_rads_row[:, 1::2])
    angle_rads_col[:, 0::2] = np.sin(angle_rads_col[:, 0::2])
    angle_rads_col[:, 1::2] = np.cos(angle_rads_col[:, 1::2])
    pos_encoding = np.concatenate([angle_rads_row, angle_rads_col], axis=1)[
        np.newaxis, ...
    ]

    return torch.tensor(pos_encoding).float()


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


# if __name__ == "__main__":
#     import pandas as pd

#     df = pd.read_csv("C:/Moein/AI/Datasets/Flicker-8k/captions.txt")
#     vocab = Vocabulary(5)
#     vocab.fit(df["caption"].values)
#     sentence = "A girl is playing with her friend in Barcelona."
#     encoded = vocab.encode(sentence)
#     print(encoded)
#     decoded = vocab.decode(encoded)
#     print(decoded)
#     print("")
