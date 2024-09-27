import math
import torch
import torch.nn as nn


class OverlappingLocalSelfAttention(nn.Module):
    def __init__(self, embed_size, heads, window_size):
        super(OverlappingLocalSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.window_size = window_size
        self.overlap_size = window_size // 8
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        n, value_len, key_len, query_len = queries.shape[0], values.shape[1], keys.shape[1], queries.shape[1]
        values = self.values(values.view(n, value_len, self.heads, self.head_dim))
        keys = self.keys(keys.view(n, key_len, self.heads, self.head_dim))
        queries = self.queries(queries.view(n, query_len, self.heads, self.head_dim))
        full_attention = torch.zeros(n, query_len, self.heads, self.head_dim, device=values.device)
        for i in range(0, query_len, self.window_size):
            window_end = min(i + self.window_size, query_len)
            local_q = queries[:, i:window_end, :, :]
            start_index = max(0, i - self.overlap_size)
            end_index = min(query_len, window_end + self.overlap_size)
            local_size = end_index - start_index
            local_k = keys[:, start_index:end_index, :, :]
            local_v = values[:, start_index:end_index, :, :]
            energies = torch.einsum("nqhd,nkhd->nhqk", [local_q, local_k])
            if mask is not None:
                local_mask = mask[:, i:window_end].unsqueeze(1).unsqueeze(-1)
                local_mask = (local_mask.expand(n, self.heads, -1, local_size))
                energies = energies.masked_fill(local_mask == 0, float("-1e10"))
            attention_local = torch.softmax(energies / (self.head_dim ** (1 / 2)), dim=3)
            out_local = torch.einsum("nhql,nlhd->nqhd", [attention_local, local_v])
            full_attention[:, i:min(i + self.window_size, query_len), :, :] \
                = out_local[:, :self.window_size, :, :]
        x = full_attention.reshape(n, query_len, self.heads * self.head_dim)
        x = self.fc_out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, window_size):
        super(TransformerBlock, self).__init__()
        self.attention = OverlappingLocalSelfAttention(embed_size, heads, window_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU('tanh'),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        x = self.dropout(self.norm2(forward + x))
        return x


class Transformer(nn.Module):
    def __init__(self, embed_size, num_layers, heads, device, forward_expansion, dropout, window_size):
        super(Transformer, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    window_size=window_size,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, x, x, mask)
        return x


class RoPE(nn.Module):
    def __init__(self, embed_size):
        super(RoPE, self).__init__()
        self.embed_size = embed_size
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, embed_size, 2).float() / embed_size))

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
        inv_freq = self.inv_freq.to(x.device)
        sinusoid_inp = torch.einsum('ij,j->ij', pos, inv_freq)
        pos_enc = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return x * pos_enc.unsqueeze(0)


class Backbone(nn.Module):
    def __init__(self, embed_size, num_layers, heads, device, forward_expansion, dropout, window_size, input_features):
        super(Backbone, self).__init__()
        self.device = device
        self.rope = RoPE(embed_size)
        self.transformer = Transformer(embed_size, num_layers, heads, device, forward_expansion, dropout, window_size)
        self.embedding = nn.Linear(input_features, embed_size, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.rope(x)
        x = self.dropout(x)
        x = self.transformer(x, mask)
        return x


class MaskedPredictionBranch(nn.Module):
    def __init__(self, input_feature, embed_size, forward_expansion, dropout):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(embed_size, embed_size * forward_expansion),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(embed_size * forward_expansion, input_feature)
        )

    def forward(self, x):
        x = self.predictor(x)
        return x


class ContrastiveBranch(nn.Module):
    def __init__(self, embed_size, forward_expansion, dropout):
        super(ContrastiveBranch, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(embed_size, embed_size * forward_expansion),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(embed_size * forward_expansion, embed_size)
        )

    def forward(self, x):
        x = self.projection_head(x)
        return x


class Classifier(nn.Module):
    def __init__(self, embed_size, forward_expansion, dropout, num_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_size, embed_size * forward_expansion),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(embed_size * forward_expansion, embed_size),
            nn.GELU(approximate='tanh'),
            nn.Linear(embed_size, num_classes)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class DualbranchWovenNetwork(nn.Module):
    def __init__(self, input_features, embed_size, layers, heads, device, forward_expansion, dropout, window_size,
                 num_classes):
        super(DualbranchWovenNetwork, self).__init__()
        self.classifier = Classifier(embed_size, forward_expansion, dropout, num_classes)
        self.backbone = Backbone(embed_size, layers, heads, device, forward_expansion, dropout, window_size,
                                 input_features)
        self.masked_prediction_branch = MaskedPredictionBranch(input_features, embed_size, forward_expansion, dropout)
        self.contrastive_branch = ContrastiveBranch(embed_size, forward_expansion, dropout)


    def forward(self, x, mask):
        x = self.backbone(x, mask)
        x = self.classifier(x)
        return x

    # def calculate_losses(self, predictions, target, prediction_mask, contrastive_features, mask):
    #     prediction_mask_expanded = prediction_mask.unsqueeze(-1).expand_as(predictions)
    #     masked_pred_output = predictions[prediction_mask_expanded]
    #     masked_targets = target[prediction_mask_expanded]
    #     masked_pred_loss = nn.L1Loss()(masked_pred_output, masked_targets) * 128
    #     contrastive_loss = self.contrastive_loss(contrastive_features, mask)
    #     total_loss = masked_pred_loss + contrastive_loss
    #     return masked_pred_loss, contrastive_loss, total_loss
