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
import utils

class Encoder(torch.nn.Module):
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

    def __init__(self, embed_size, hidden_size, source_word_2_ind, unk_token, pad_token, dropout):
        super(Encoder, self).__init__()

        self.source_word_2_ind = source_word_2_ind

        self.unk_token_idx = source_word_2_ind[unk_token]
        self.pad_token_idx = source_word_2_ind[pad_token]

        self.hidden_size = hidden_size
        self.encoder_LSTM = torch.nn.LSTM(embed_size, hidden_size, num_layers=2, bidirectional=True, dropout=dropout)
        self.embed_encoder = torch.nn.Embedding(len(source_word_2_ind), embed_size)

    def forward(self, source):
        X = self.preparePaddedBatch(source, self.source_word_2_ind)

        E1 = self.embed_encoder(X[:-1])
        source_lengths = [len(s) - 1 if (len(s) > 1) else len(s) for s in source]
        encoder_hidden_states_packed, (hn, cn) = self.encoder_LSTM(
            torch.nn.utils.rnn.pack_padded_sequence(E1, source_lengths, enforce_sorted=False))
        encoder_hidden_states, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_hidden_states_packed)

        hn_concatenated = utils.concatenateBidirectionalLayers(hn)
        cn_concatenated = utils.concatenateBidirectionalLayers(cn)


        return encoder_hidden_states, hn_concatenated, cn_concatenated
