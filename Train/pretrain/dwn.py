import torch
import torch.nn as nn


class OverlappingLocalSelfAttention(nn.Module):
    def __init__(self, embed_size, heads, window_size):
        super(OverlappingLocalSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.window_size = window_size
        self.overlap_size = window_size // 8  # overlap size
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
        for i in range(0, query_len, self.window_size):  # local self-attention
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
            nn.GELU('tanh'),  # use 'tanh' to approximate GELU
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


class RoPE(nn.Module):  # Sine-Cosine Positional Encoding with Rotational Characteristics
    def __init__(self, embed_size):
        super(RoPE, self).__init__()
        self.embed_size = embed_size
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, embed_size, 2).float() / embed_size))

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
        inv_freq = self.inv_freq.to(x.device)
        sinusoid_inp = torch.einsum('ij,j->ij', pos, inv_freq)  # outer product
        pos_enc = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return x * pos_enc.unsqueeze(0)


class Backbone(nn.Module):  # 'f' function
    def __init__(self, embed_size, num_layers, heads, device, forward_expansion, dropout, window_size, input_features,
                 mask_rate):
        super(Backbone, self).__init__()
        self.device = device
        self.transformer = Transformer(embed_size, num_layers, heads, device, forward_expansion, dropout, window_size)
        self.rope = RoPE(embed_size)
        self.embedding = nn.Linear(input_features, embed_size, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.mask_rate = mask_rate

    def generate_prediction_mask(self, input_tensor, mask, mask_rate):
        rand_tensor = torch.rand(input_tensor.shape[:2], device=self.device)
        prediction_mask = (rand_tensor < mask_rate) & mask
        target = torch.zeros_like(input_tensor)
        target[prediction_mask.unsqueeze(-1).expand(-1, -1, input_tensor.shape[-1])] \
            = input_tensor[prediction_mask.unsqueeze(-1).expand(-1, -1, input_tensor.shape[-1])]
        return prediction_mask, target

    def apply_mask(self, input_tensor, prediction_mask):
        replace_rate = 0.3 if torch.rand(1) < 0.75 else 0  # masked batch rate
        replace_mask = torch.rand(input_tensor.shape[:2], device=self.device) < replace_rate
        extended_mask = (replace_mask.unsqueeze(-1) & prediction_mask.unsqueeze(-1)).expand_as(input_tensor)

        mask_value_select = (torch.rand(input_tensor.shape[:2], device=self.device) < 0.2).unsqueeze(-1).expand_as(
            input_tensor)
        mask_value = torch.where(mask_value_select, torch.full_like(input_tensor, 0.5),
                                 torch.tensor(-9, device=self.device, dtype=input_tensor.dtype))

        masked_input = torch.where(extended_mask, mask_value, input_tensor)
        return masked_input

    def forward(self, x, mask):
        prediction_mask, target = self.generate_prediction_mask(x, mask, self.mask_rate)
        x = self.apply_mask(x, prediction_mask)
        x = self.embedding(x)
        x = self.rope(x)
        x = self.dropout(x)
        x = self.transformer(x, mask)
        return x, target, prediction_mask


class MaskedPredictionBranch(nn.Module):  # 'g' function
    def __init__(self, input_feature, embed_size, forward_expansion, dropout):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(embed_size, embed_size * forward_expansion),
            nn.GELU('tanh'),
            nn.Dropout(dropout),
            nn.Linear(embed_size * forward_expansion, input_feature)
        )

    def forward(self, x):
        x = self.predictor(x)
        return x


class ContrastiveBranch(nn.Module):  # 'w' function
    def __init__(self, embed_size, forward_expansion, dropout):
        super(ContrastiveBranch, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(embed_size, embed_size * forward_expansion),
            nn.GELU('tanh'),
            nn.Dropout(dropout),
            nn.Linear(embed_size * forward_expansion, embed_size)
        )

    def forward(self, x):
        x = self.projection_head(x)
        return x


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature, selected_features=192):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.selected_features = selected_features

    @staticmethod
    def create_positive_mask(mask):
        batch_size, seq_length = mask.shape
        base_mask = torch.zeros(seq_length, seq_length, device=mask.device, dtype=torch.bool)
        base_mask[2:, :-2] |= torch.eye(seq_length - 2, device=mask.device, dtype=torch.bool)
        base_mask[:-2, 2:] |= torch.eye(seq_length - 2, device=mask.device, dtype=torch.bool)
        base_mask[1:, :-1] |= torch.eye(seq_length - 1, device=mask.device, dtype=torch.bool)
        base_mask[:-1, 1:] |= torch.eye(seq_length - 1, device=mask.device, dtype=torch.bool)
        base_mask = base_mask.unsqueeze(0).expand(batch_size, -1, -1)
        mask_expanded = mask.unsqueeze(1) & mask.unsqueeze(2)
        positive_mask = base_mask & mask_expanded
        return positive_mask

    @staticmethod
    def create_negative_mask(pos_mask, mask, neg_samples=12):
        batch_size, seq_length = mask.shape
        neg_samples = min(neg_samples, seq_length)
        no_self_mask = ~torch.eye(seq_length, dtype=torch.bool, device=pos_mask.device).unsqueeze(0)
        full_neg_mask = no_self_mask & ~pos_mask & mask.unsqueeze(1) & mask.unsqueeze(2)
        probabilities = full_neg_mask.float()
        sum_probabilities = probabilities.sum(dim=2, keepdim=True)
        probabilities = torch.where(sum_probabilities > 0, probabilities / sum_probabilities,
                                    torch.ones_like(probabilities) / seq_length)
        selected_indices = (torch.multinomial(probabilities.view(-1, seq_length), neg_samples, replacement=False).
                            view(batch_size, seq_length, neg_samples))
        neg_mask = torch.zeros_like(full_neg_mask, dtype=torch.bool)
        batch_indices = torch.arange(batch_size, device=mask.device).view(-1, 1, 1)
        seq_indices = torch.arange(seq_length, device=mask.device).view(1, -1, 1)
        neg_mask[batch_indices, seq_indices, selected_indices] = True
        neg_mask &= full_neg_mask
        return neg_mask

    def forward(self, x, mask):
        batch_size, seq_length, feature_dim = x.shape
        indices = torch.randperm(feature_dim)[:self.selected_features]
        x_subset = x[:, :, indices]
        x_subset = nn.functional.normalize(x_subset, p=2, dim=2)
        sim_matrix = (torch.bmm(x_subset, x_subset.transpose(1, 2))
                      / self.temperature)
        diagonal_indices = torch.arange(seq_length)
        sim_matrix[:, diagonal_indices, diagonal_indices] = float('-1e10')
        pos_mask = self.create_positive_mask(mask)
        neg_mask = self.create_negative_mask(pos_mask, mask)
        sim_matrix_exp = torch.exp(sim_matrix)
        pos_sum_exp = sim_matrix_exp.masked_fill(~pos_mask, 0).sum(dim=1)
        neg_sum_exp = sim_matrix_exp.masked_fill(~neg_mask, 0).sum(dim=1)
        total_sum_exp = (pos_sum_exp + neg_sum_exp).clamp(min=1e-6)
        info_nce_loss = -torch.log((pos_sum_exp / total_sum_exp).clamp(min=1e-6))
        valid_rows_mask = pos_mask.any(dim=1)
        valid_cols_mask = pos_mask.any(dim=2)
        valid_mask = valid_rows_mask & valid_cols_mask
        info_nce_loss = info_nce_loss[valid_mask].mean()
        return info_nce_loss


class DualbranchWovenNetwork(nn.Module):  # dual-task woven network, or DWN
    def __init__(self, input_features, embed_size, layers, heads, device, forward_expansion, dropout, window_size,
                 mask_rate, temperature):
        super(DualbranchWovenNetwork, self).__init__()
        self.device = device
        self.backbone = Backbone(embed_size, layers, heads, device, forward_expansion, dropout, window_size,
                                 input_features, mask_rate)
        self.masked_prediction_branch = MaskedPredictionBranch(input_features, embed_size, forward_expansion, dropout)
        self.contrastive_branch = ContrastiveBranch(embed_size, forward_expansion, dropout)
        self.contrastive_loss = ContrastiveLoss(temperature)

    def forward(self, x, mask):
        x, target, prediction_mask = self.backbone(x, mask)
        predictions = self.masked_prediction_branch(x)
        contrastive_features = self.contrastive_branch(x)
        return predictions, target, prediction_mask, contrastive_features

    def calculate_losses(self, predictions, target, prediction_mask, contrastive_features, mask):
        prediction_mask_expanded = prediction_mask.unsqueeze(-1).expand_as(predictions)
        masked_pred_output = predictions[prediction_mask_expanded]
        masked_targets = target[prediction_mask_expanded]
        masked_pred_loss = nn.HuberLoss(delta=0.5)(masked_pred_output, masked_targets) * 5000  # lambda = 5000
        contrastive_loss = self.contrastive_loss(contrastive_features, mask)
        total_loss = masked_pred_loss + contrastive_loss
        return masked_pred_loss, contrastive_loss, total_loss
