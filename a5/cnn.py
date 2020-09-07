#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, char_embed_size, word_embed_size, kernel_size, padding):
        super(CNN, self).__init__()

        # conv layer
        in_channels = char_embed_size
        out_channels = word_embed_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, input_t: torch.Tensor):
        """
        @param input_t tensor.Tensor: size (batch_size, max_sentence_length, char_embed_size, max_word_length)
        @return tensor.Tensor: size (batch_size, max_sentence_length, word_embed_size)
        """
        # reshpae input to (max_sentence_length * batch_size, char_embed_size, max_word_length) 
        batch_size, max_sentence_length, char_embed_size, max_word_length = input_t.shape
        conv_in = input_t.view([-1, char_embed_size, max_word_length]).contiguous()

        # output shape: (max_sentence_length * batch_size, word_embed_size, max_word_length) 
        conv_out = F.relu(self.conv(conv_in))
        
        # max pool over time
        pooled_out, _ = torch.max(conv_out, 2)

        # reshape and return
        return pooled_out.view(batch_size, max_sentence_length, -1).contiguous()      
