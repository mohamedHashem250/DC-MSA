import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer, EncoderLayer_conv
from layers.SelfAttention_Family import FullAttention, AttentionLayer_conv
from layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.c_out = configs.c_out
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer_conv(
                    AttentionLayer_conv(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)
        else: #forecasting problem
            # Step 1: project time dimension L → H
            self.temporal_proj = nn.Linear(configs.seq_len, configs.pred_len)
            # Step 2: project feature dimension d_model → c_out
            self.feature_proj = nn.Linear( configs.d_model, configs.c_out)
            self.gelu          = nn.GELU()
            self.dropout = nn.Dropout(configs.dropout)
        #self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

            # AFTER (correct for standard encoder):
            B, L, N = x_enc.shape  # B L N
            # B: batch_size
            # L: seq_len
            # N: number of variates (enc_in)

        # Embedding
        #  (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        #  (B L E -> B L E in the vanilla Transformer)
        # 
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        #dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates
                # x: (B, L, d_model)

        # Step 1: project along time axis
        enc_out = enc_out.transpose(1, 2)          # (B, d_model, L)
        enc_out = self.gelu(self.temporal_proj(enc_out))       # (B, d_model, H)
        enc_out = self.dropout(enc_out)
        enc_out = enc_out.transpose(1, 2)          # (B, H, d_model)

        # Step 2: project along feature axis
        dec_out = self.feature_proj(enc_out)        # (B, H, C)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))[:, :, :self.c_out]
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))[:, :, :self.c_out]
        return dec_out, attns

    # def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
    #     dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
    #     if self.output_attention:
    #         return dec_out, attns
    #     else:
    #         return dec_out  # [B, L, D]
        
    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if self.output_attention:
                return dec_out, attns
            else:
                return dec_out  # [B, L, D]
            #return dec_out[:, -self.pred_len:, :]  # [B, L, D]

        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None