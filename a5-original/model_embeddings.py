#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): embedding size (dimensionality) for the output word
        @param vocab (vocabentry): vocabentry object. see vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>'] # notice that in assignment 4 vocab is of type (Vocab), not (VocabEntry) as assignment 5.
        # self.embeddings = nn.Embedding(len(vocab.src), word_embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1h

        # hyperparameters
        char_embed_size = 50
        kernel_size = 5
        padding = 1

        self.vocab = vocab
        self.word_embed_size = word_embed_size

        # layers
        self.embeddings = nn.Embedding(len(self.vocab.char2id), char_embed_size, padding_idx=self.vocab.char_pad)
        self.cnn = CNN(char_embed_size, word_embed_size, kernel_size, padding)
        self.highway = Highway(word_embed_size)
        self.dropout = nn.Dropout(p=0.3)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1h
        
        # shape: (sentenced_length, batch_size, max_word_length, char_embed_size)
        embeddings = self.embeddings(input)
        cnn_input = embeddings.permute([1, 0, 3, 2]).contiguous()
        cnn_output = self.cnn(cnn_input)
        highway_output = self.highway(cnn_output)
        output = self.dropout(highway_output)
        
        return output.permute([1, 0, 2]).contiguous()
        ### END YOUR CODE

