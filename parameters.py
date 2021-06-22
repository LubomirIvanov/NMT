import torch

sourceFileName = 'en_bg_data/train.en'
targetFileName = 'en_bg_data/train.bg'
sourceDevFileName = 'en_bg_data/dev.en'
targetDevFileName = 'en_bg_data/dev.bg'

corpusDataFileName = 'corpusData'
wordsDataFileName = 'wordsData'
modelFileName = 'NMTmodel'

device = torch.device("cuda:0")
#device = torch.device("cpu")

embed_size=128
hidden_size=128
dropout = 0.2
linearDropout = 0.4
uniform_init = 0.1
learning_rate = 0.003
clip_grad = 5.0
learning_rate_decay = 0.5

batchSize = 32

maxEpochs = 30
log_every = 250
test_every = 50

max_patience = 30
max_trials = 6
