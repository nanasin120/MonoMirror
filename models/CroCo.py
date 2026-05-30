import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import PositionalEncoding2D, MultiHead, FeedForwardNetwork

class Encoder(nn.Module): # Inputs안의 단어는 자기만 알던 놈이었음, 이젠 남들도 아는 놈이 됨
    def __init__(self, d_model=768, h=12):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.h = h

        # nn.Embedding(단어의 개수, 임베딩할 벡터 차원)
        self.multi_head_attention = MultiHead(self.d_model, self.h) # Q단어와 K단어의 관계성에 V단어의 의미를 곱해 문장 내 모든 단어의 의미를 합친 놈이 나옴
        self.layer_norm_1 = nn.LayerNorm(self.d_model) # 정규화
        self.FFN = FeedForwardNetwork(self.d_model, self.d_model * 4) # 더 단어의 특징(의미)를 농축시킴
        self.layer_norm_2 = nn.LayerNorm(self.d_model) # 정규화

    def forward(self, Inputs, mask=None):
        norm_1 = self.layer_norm_1(Inputs)
        after_multi_head = self.multi_head_attention(norm_1, norm_1, norm_1, mask=mask)
        after_multi_head = after_multi_head + Inputs

        norm_2 = self.layer_norm_2(after_multi_head)
        after_ffn = self.FFN(norm_2)
        after_ffn = after_ffn + after_multi_head

        return after_ffn

class CroCo(nn.Module):
    def __init__(self):
        super(CroCo, self).__init__()
        self.patch_size = 16
        self.patch_embedded_dim = 768
        self.patch_embedding = nn.Conv2d(
            in_channels= 3, 
            out_channels=self.patch_embedded_dim,
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )

        self.positionalEncoding2D = PositionalEncoding2D(d_model=768, h_patches=14, w_patches=14)

        self.encoder_layers = 12
        self.encoders = nn.ModuleList([
            Encoder(d_model=self.patch_embedded_dim, h=12) for _ in range(self.encoder_layers)
        ])

    def forward(self, prev_img, curr_img, next_img):
        # image : [B, 3, H, W] [B, 3, 224, 224]

        # [B, 3, 224, 224] -> [B, 768, 14, 14] -> [B, 768, 196] -> [B, 196, 768]
        prev_p = self.patch_embedding(prev_img).flatten(2).transpose(1, 2)
        curr_p = self.patch_embedding(curr_img).flatten(2).transpose(1, 2)
        next_p = self.patch_embedding(next_img).flatten(2).transpose(1, 2)

        # --- Encoder Section --- 

        prev_features = []
        curr_features = []
        next_features = []
        extract_layers = [2, 5, 8, 11]

        prev_p = self.positionalEncoding2D(prev_p)
        curr_p = self.positionalEncoding2D(curr_p)
        next_p = self.positionalEncoding2D(next_p)
        
        for i, encoder in enumerate(self.encoders):
            prev_p = encoder(prev_p)
            curr_p = encoder(curr_p)
            next_p = encoder(next_p)

            if i in extract_layers:
                prev_features.append(prev_p)
                curr_features.append(curr_p)
                next_features.append(next_p)


        return prev_features, curr_features, next_features