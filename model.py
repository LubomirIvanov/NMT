#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################
###
### Невронен машинен превод
###
#############################################################################
import encoder
import decoder
import attention
import torch
import numpy as np


class NMTmodel(torch.nn.Module):
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

    def __init__(self, embed_size, hidden_size, source_word_2_ind, target_word_2_ind, start_token, unk_token, pad_token, end_token, dropout, linear_dropout):
        super(NMTmodel, self).__init__()

        self.source_word_2_ind = source_word_2_ind
        self.target_word_2_ind = target_word_2_ind
        self.target_ind_2_word = {v: k for k, v in self.target_word_2_ind.items()}
        self.unk_token_idx = target_word_2_ind[unk_token]
        self.pad_token_idx = target_word_2_ind[pad_token]
        self.end_token_idx = target_word_2_ind[end_token]
        self.start_token = start_token
        self.end_token = end_token

        self.hidden_size = hidden_size
        self.encoder = encoder.Encoder(embed_size, hidden_size, source_word_2_ind, unk_token, pad_token, dropout)
        self.decoder = decoder.Decoder(embed_size, hidden_size, target_word_2_ind, unk_token, pad_token, dropout)
        self.attention = attention.Attention(hidden_size, target_word_2_ind)

        self.dropout = torch.nn.Dropout(dropout)
        self.linear_dropout = torch.nn.Dropout(linear_dropout)

        self.softmax = torch.nn.Softmax(dim=0)
        self.projection = torch.nn.Linear(4 * hidden_size, len(target_word_2_ind))

    def forward(self, source, target):

        encoder_hidden_states, hn_concatenated, cn_concatenated = self.encoder(source)
        decoder_hidden_states, Y = self.decoder(target, (encoder_hidden_states, hn_concatenated, cn_concatenated))
        attention_decoder_concat = self.attention(encoder_hidden_states, decoder_hidden_states)

        Z = self.projection(self.linear_dropout(attention_decoder_concat.flatten(0, 1)))
        Y_bar = Y[1:].flatten(0, 1)

        H = torch.nn.functional.cross_entropy(Z, Y_bar, ignore_index=self.pad_token_idx)
        return H

    def translateSentence(self, sentence, limit=1000):

        def greedySearch(attention_vector):
            Z = self.projection(attention_vector.flatten(0, 1))
            probabilities = self.softmax(Z)
            highest_prob_idx = torch.argmax(probabilities).item()
            predicted_word = self.target_ind_2_word[highest_prob_idx]
         
            return predicted_word
        
        def generateWord(encoder_hidden_states, decoder_hidden_state):
            encoder_hidden_states = encoder_hidden_states.squeeze(1)
            decoder_hidden_state = decoder_hidden_state.squeeze(1)

            attention_vector = self.attention(encoder_hidden_states, decoder_hidden_state)
            input_word = greedySearch(attention_vector)

            return input_word

        with torch.no_grad():
            input_word = self.start_token

            Y = self.preparePaddedBatch([[input_word]], self.target_word_2_ind)
            E1 = self.decoder.embed_decoder(Y)
            
            encoder_hidden_states, hn_concatenated, cn_concatenated = self.encoder([sentence])
            decoder_hidden_state, (decoder_hn, decoder_cn) = self.decoder.decoder_LSTM(E1, (hn_concatenated, cn_concatenated))

            input_word = generateWord(encoder_hidden_states, decoder_hidden_state)

            result = []
            result.append(input_word)
            
            sentence_length = 0
            
            while input_word != self.end_token and sentence_length <= limit:

                Y = self.preparePaddedBatch([[input_word]], self.target_word_2_ind)
                E1 = self.decoder.embed_decoder(Y)

                decoder_hidden_state, (decoder_hn, decoder_cn) = self.decoder.decoder_LSTM(E1, (decoder_hn, decoder_cn))
                input_word = generateWord(encoder_hidden_states, decoder_hidden_state)
                
                if input_word == self.end_token:
                    break

                result.append(input_word)
                sentence_length = sentence_length + 1

        return result