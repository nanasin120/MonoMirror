import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def Attention(Q, K, V, d_k, mask): # 자신의 의미만 갖고 있던 단어를 다른 단어들의 의미를 아는 단어로 바꿈
    # [64, 8, 128, 64], [64, 8, 64, 128]
    # [64, 8, 128, 128] 단어와 단어의 관계
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k) 

    # mask가 0인 부분은 -1e9로 바꿔버림
    if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)

    # [64, 8, A, B]라 하면 A단어와 B단어가 얼마나 관계가 있을지의 확률인데 이걸 softmax해서 A단어와 가장 관계 있는 B단어를 알 수 있음
    scores = F.softmax(scores, dim=-1)

    # [64, 8, 128, 128], [64, 8, 128, 64]
    # [64, 8, 128, 64] 단어가 얼마나 중요한지에다가 단어의 의미를 곱해주는 것
    # 이렇게 되면 원래는 해당 단어의 의미만 가졌지만 이젠 다른 모든 단어들의 의미를 흡수한 놈이 됨

    output = torch.matmul(scores, V)

    return output

class MultiHead(nn.Module): # Q단어와 K단어의 관계성에 V단어의 의미를 곱해 문장 내 모든 단어의 의미를 합친 놈이 나옴
    def __init__(self, d_model, h):
        super(MultiHead, self).__init__()
        self.h = h # 8
        self.d_model = d_model # 512
        self.d_k = d_model // h # 512 / 8 = 64
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.WO = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        # Inputs : [64, 128, 512], mask : [64, 1, 1, 128]
        # Inputs, Inputs, Inputs, mask

        # 입력으로 들어오는 Q, K, V 모두 [batch_size, sequence_len, d_model]
        # 여기서 d_model을 h개로 나눠 나눈것들에 matmul을 하고 attention을 한뒤 concat하고 다시 matmul해야함
        # 이 과정을 for문을 사용하지 않고 할거임
        
        # 배치 사이즈와 문장의 길이를 사용할거임
        batch_size, sequence_len, d_model = Q.shape

        # Linear로 한번에 가중치 연산을 해준뒤 self.h로 뒷부분을 나눠주고 1번과 2번을 바꿔줌
        # 이렇게 되면 [배치 사이즈, self.h, 문장의 길이, self.d_k]로 나뉘게됨
        # 이건 마치 matmul을 한뒤 self.h로 나눈것과 같음을 알 수 있음
        # [64, 128, 512] -> [64, 128, 8, 64] -> [64, 8, 128, 64] [배치, self.h, 문장 길이, self.d_k]
        nQ = self.WQ(Q).view(batch_size, sequence_len, self.h, self.d_k).transpose(1, 2)
        nK = self.WK(K).view(batch_size, sequence_len, self.h, self.d_k).transpose(1, 2)
        nV = self.WV(V).view(batch_size, sequence_len, self.h, self.d_k).transpose(1, 2)

        # 그후 head를 구해줌, 이 head는 concat할 필요가 없음 이미 다 붙어있으니
        # [64, 8, 128, 64] 이제 모든 단어들의 의미를 조금씩 흡수한 단어가 나옴
        head = Attention(nQ, nK, nV, self.d_k, mask)

        # [배치 사이즈, self.h, 문장 길이, self.d_k]인거를
        # [배치 사이즈, 문장 길이, self.h, self.d_k]로 바꿔주고
        # [배치 사이즈, 문장 길이, self.h * self.d_k = self.d_model]로 바꿔줌
        # [64, 8, 128, 64] -> [64, 128, 8, 64] -> [64, 128, 512]
        head = head.transpose(1, 2).contiguous().view(batch_size, sequence_len, self.d_model)
        
        # 마지막으로 WO와 곱하면 끝 [64, 128, 512]
        output = self.WO(head)

        # [배치 사이즈, 문장 길이, self.d_model]으로 반환됨
        return output

class FeedForwardNetwork(nn.Module): # 더 고차원적으로 특징을 추출해서 단어에 넣어줌
    def __init__(self, d_model, d_ff): # 512, 2048
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # 512, 2048
        self.linear2 = nn.Linear(d_ff, d_model) # 2048, 512

    def forward(self, x):
        x = torch.relu(self.linear1(x))

        output = self.linear2(x)

        return output

class PositionalEncoding2D(nn.Module): # 단어의 위치를 더해줌
    def __init__(self, d_model=768, h_patches=14, w_patches=14):
        super(PositionalEncoding2D, self).__init__()

        d_model_half = d_model // 2

        y_grid, x_grid = torch.meshgrid(torch.arange(h_patches), torch.arange(w_patches), indexing='ij')
        y_grid = y_grid.flatten().float() # [196]
        x_grid = x_grid.flatten().float() # [196]

        div_term = torch.exp(torch.arange(0, d_model_half, 2).float() * (-math.log(10000.0) / d_model_half))

        pe_y = torch.zeros(h_patches * w_patches, d_model_half) # [196, 384]
        pe_y[:, 0::2] = torch.sin(y_grid.unsqueeze(1) * div_term) # 짝수
        pe_y[:, 1::2] = torch.cos(y_grid.unsqueeze(1) * div_term) # 홀수

        pe_x = torch.zeros(h_patches * w_patches, d_model_half) # [196, 384]
        pe_x[:, 0::2] = torch.sin(x_grid.unsqueeze(1) * div_term) # 짝수
        pe_x[:, 1::2] = torch.cos(x_grid.unsqueeze(1) * div_term) # 홀수

        pe = torch.cat([pe_y, pe_x], dim=1) # [196, 768]

        pe = pe.unsqueeze(0) # [1, 196, 768]

        self.register_buffer('pe', pe)

    def forward(self, x):
        
        x = x + self.pe[:, :x.shape[1], :]

        return x