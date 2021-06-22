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


class Attention(torch.nn.Module):

    def save(self, fileName):
        torch.save(self.state_dict(), fileName)

    def load(self, fileName):
        self.load_state_dict(torch.load(fileName))

    def __init__(self, hidden_size, targetWord2ind):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size

        self.Ua = torch.nn.Linear(2 * hidden_size, hidden_size)
        self.Wa = torch.nn.Linear(2 * hidden_size, hidden_size)
        self.V = torch.nn.Linear(hidden_size, 1)
        
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim = 1)

    def forward(self, encoder_hidden_states, decoder_hidden_states):

        e = self.V(self.tanh(self.Ua(encoder_hidden_states) + self.Wa(decoder_hidden_states.unsqueeze(1))))
        attention_weights = self.softmax(e)
        attention_vector = torch.sum(attention_weights * encoder_hidden_states, dim=1)
        attention_decoder_concat = torch.cat((attention_vector, decoder_hidden_states), -1)

        return attention_decoder_concat
