#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
##########################################################################
###
### Невронен машинен превод
###
#############################################################################

import sys
import random
import nltk
import torch

from nltk.translate.bleu_score import corpus_bleu
nltk.download('punkt')

class progressBar:
    def __init__(self ,barWidth = 50):
        self.barWidth = barWidth
        self.period = None
    def start(self, count):
        self.item=0
        self.period = int(count / self.barWidth)
        sys.stdout.write("["+(" " * self.barWidth)+"]")
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.barWidth+1))
    def tick(self):
        if self.item>0 and self.item % self.period == 0:
            sys.stdout.write("-")
            sys.stdout.flush()
        self.item += 1
    def stop(self):
        sys.stdout.write("]\n")

def readCorpus(fileName):
    ### Чете файл от изречения разделени с нов ред `\n`.
    ### fileName е името на файла, съдържащ корпуса
    ### връща списък от изречения, като всяко изречение е списък от думи
    print('Loading file:',fileName)
    return [ nltk.word_tokenize(line) for line in open(fileName) ]

def getDictionary(corpus, startToken, endToken, unkToken, padToken, wordCountThreshold = 2):
    dictionary={}
    for s in corpus:
        for w in s:
            if w in dictionary: dictionary[w] += 1
            else: dictionary[w]=1

    words = [startToken, endToken, unkToken, padToken] + [w for w in sorted(dictionary) if dictionary[w] > wordCountThreshold]
    return { w:i for i,w in enumerate(words)}


def prepareData(sourceFileName, targetFileName, sourceDevFileName, targetDevFileName, startToken, endToken, unkToken, padToken):

    sourceCorpus = readCorpus(sourceFileName)
    targetCorpus = readCorpus(targetFileName)
    sourceWord2ind = getDictionary(sourceCorpus, startToken, endToken, unkToken, padToken)
    targetWord2ind = getDictionary(targetCorpus, startToken, endToken, unkToken, padToken)

    targetCorpus = [ [startToken] + s + [endToken] for s in targetCorpus]

    sourceDev = readCorpus(sourceDevFileName)
    targetDev = readCorpus(targetDevFileName)

    targetDev = [ [startToken] + s + [endToken] for s in targetDev]

    print('Corpus loading completed.')
    return sourceCorpus, sourceWord2ind, targetCorpus, targetWord2ind, sourceDev, targetDev
def concatenateBidirectionalLayers(hn):
    firstPairfirstDirection = hn[0]
    firstPairsecondDirection = hn[1]
    hnConcatenated1 = torch.cat((firstPairfirstDirection, firstPairsecondDirection), 1).unsqueeze(0)

    secondPairfirstDirection = hn[2]
    secondPairsecondDirection = hn[3]
    hnConcatenated2 = torch.cat((secondPairfirstDirection, secondPairsecondDirection), 1).unsqueeze(0)

    concatenated = torch.cat((hnConcatenated1, hnConcatenated2), 0)
    return concatenated
   


