import torch

from torch import nn
import torch.nn.init as init

import torch.nn.functional as F
from torch.nn.utils import weight_norm


class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()

    def forward(self, embed_reaction, embed_enzyme):
        Q1 = embed_reaction  # Q1 shape: [batch_size, 1, 256]
        K2 = embed_enzyme  # K2 shape: [batch_size, n, 256]
        V2 = embed_enzyme  # V2 shape: [batch_size, n, 256]
        embed_dim = embed_reaction.shape[-1]
        # Calculate attention
        K2_t = K2.permute(0, 2, 1)
        attn_weights_1 = F.softmax(torch.matmul(Q1, K2_t) / (embed_dim ** 0.5), dim=-1)  # [batch_size, 1, n]
        updated_V2 = torch.matmul(attn_weights_1, V2)  # Updated V2 shape: [batch_size, 1, 64]

        # Add & Norm
        updated_V2 = updated_V2 + embed_reaction  # Shape: [batch_size, 1, 64]
        updated_V2 = F.layer_norm(updated_V2, normalized_shape=[embed_dim])  # Layer Norm

        Q2 = embed_enzyme  # Q2 shape: [batch_size, n, 64]
        K1 = embed_reaction  # K1 shape: [batch_size, 1, 64]
        V1 = embed_reaction  # V1 shape: [batch_size, 1, 64]

        # Calculate attention
        K1_t = K1.permute(0, 2, 1)
        attn_weights_2 = F.softmax(torch.matmul(Q2, K1_t) / (embed_dim ** 0.5), dim=-1)  # [batch_size, n, 1]
        updated_V1 = torch.matmul(attn_weights_2, V1)  # Updated V1 shape: [batch_size, n, 64]

        # Add & Norm
        updated_V1 = updated_V1 + embed_enzyme  # Shape: [batch_size, n, 64]
        updated_V1 = F.layer_norm(updated_V1, normalized_shape=[embed_dim])  # Layer Norm

        updated_V1 = updated_V1.permute(0, 2, 1)  # Reshape to [batch_size, 64, n]
        # Use max pooling with kernel size n to reduce sequence length to 1
        updated_V1 = F.max_pool1d(updated_V1, kernel_size=updated_V1.shape[-1])

        # Reshape back to [batch_size, 1, 64]
        updated_V1 = updated_V1.permute(0, 2, 1)

        out = torch.cat((updated_V2, updated_V1), dim=2).squeeze(1)

        return out


class ModelFC(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden1 = 1024
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256
        self.convs = nn.ModuleList([nn.Conv2d(1, self.num_filters, (k, self.hidden1)) for k in self.filter_sizes])

        self.cross_attention = CrossAttention()

        self.drop1 = nn.Dropout(0.5)  # Special drop_rate  # 0.5  0.75
        self.drop2 = nn.Dropout(0.2)  # 0.2  0.5

        self.fc1 = nn.Linear(512, 256)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)

        self.fc2 = nn.Linear(256, 64)
        self.fc2.weight.data.normal_(0, 0.01)
        self.fc2.bias.data.fill_(0.0)

    def conv_and_pool(self, x, conv):  # [seq_len,1,100,20]
        # print(x.shape) #[128, 1, 100, 20])
        x = F.relu(conv(x)).squeeze(3)  # [128 100 [31 30 29]=90 ]
        return x

    def forward(self, reactions, protein):  # , crafted_feature
        # print("reactions shape:", reactions.shape)
        # print("protein shape:", protein.shape)
        # Use average pooling with kernel size n to reduce sequence length to 1, [batch_size, 1024]
        # protein_pool = torch.mean(protein, dim=1)
        # reactions_temp = reactions.squeeze(1)

        out = protein
        out = out.unsqueeze(1)  # in[128,100,20] out[128,1,100,20]
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 2)

        # Reshape to [batch_size, seq_len, 256] for subsequent network
        protein1 = out.permute(0, 2, 1)
        aggreX1 = self.cross_attention(reactions, protein1)

        # aggreX = torch.cat((protein_pool, aggreX1, reactions_temp), dim=1)

        self.interact = aggreX1

        output = self.drop1(F.leaky_relu(self.fc1(aggreX1)))
        output = self.drop2(F.leaky_relu(self.fc2(output)))

        return output


class Model_Regression(nn.Module):
    def __init__(self):
        super(Model_Regression, self).__init__()
        self.model_fc = ModelFC()
        # self.classifier_layer = nn.Linear(256, 1)
        self.classifier_layer = nn.Linear(64, 1)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, reactions, protein):
        feature = self.model_fc(reactions, protein)
        outC = self.classifier_layer(feature)

        return (outC, feature)