#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self, word_embed_size):
        super(Highway, self).__init__()
        
        self.gate_proj = nn.Linear(word_embed_size, word_embed_size)
        self.highway_proj = nn.Linear(word_embed_size, word_embed_size) 
        
    def forward(self, input_t: torch.Tensor):
        """
        Forward output of conv net through highway layer

        @param input_t torch.Tensor: shape (batch_size, max_sentence_length, embed_size) 
        @return torch.Tensor: shape (batch_size, max_sentence_length, embed_size)
        """
        # binary gate
        gate = self.gate_proj(input_t)

        # projection
        hidden_input = self.highway_proj(input_t)
        hidden_output = F.relu(hidden_input)

        # combine projection with input according to gate
        combined_output = gate * hidden_output + (1 - gate) * input_t 

        return combined_output
