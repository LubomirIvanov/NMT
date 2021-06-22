#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################
###
### Невронен машинен превод
###
#############################################################################

import torch
import numpy as np


class Decoder(torch.nn.Module):
    def preparePaddedBatch(self, source, word_2_ind):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[word_2_ind.get(w, self.unk_token_idx) for w in s] for s in source]
        sents_padded = [s + (m - len(s)) * [self.pad_token_idx] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))

    def save(self, fileName):
        torch.save(self.state_dict(), fileName)

    def load(self, fileName):
        self.load_state_dict(torch.load(fileName))

    def __init__(self, embed_size, hidden_size, target_word_2_ind, unk_token, pad_token, dropout):
        super(Decoder, self).__init__()

        self.target_word_2_Ind = target_word_2_ind
        self.unk_token_idx = target_word_2_ind[unk_token]
        self.pad_token_idx = target_word_2_ind[pad_token]
  

        self.hidden_size = hidden_size

        self.decoder_LSTM = torch.nn.LSTM(embed_size, 2 * hidden_size, num_layers=2, dropout=dropout)
        self.embed_decoder = torch.nn.Embedding(len(target_word_2_ind), embed_size)

    def forward(self, target, encoder_output):

        encoder_hidden_states, encoder_last_hn, encoder_last_cn = encoder_output
        Y = self.preparePaddedBatch(target, self.target_word_2_Ind)

        E2 = self.embed_decoder(Y[:-1])

        target_lengths = [len(s) - 1 if (len(s) > 1) else len(s) for s in target]
        decoder_hidden_states_packed, _ = self.decoder_LSTM(
            torch.nn.utils.rnn.pack_padded_sequence(E2, target_lengths, enforce_sorted=False),
            (encoder_last_hn, encoder_last_cn))
        decoder_hidden_states, _ = torch.nn.utils.rnn.pad_packed_sequence(decoder_hidden_states_packed)

        return decoder_hidden_states, Y
