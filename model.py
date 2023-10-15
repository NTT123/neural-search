import torch
from transformers import AutoConfig, T5EncoderModel


class MsNet(torch.nn.Module):
    def __init__(self, base_model: str = "google/t5-v1_1-small", dim: int = 256):
        super().__init__()
        config = AutoConfig.from_pretrained(base_model)
        config.dropout_rate = 0.0
        self.encoder = T5EncoderModel.from_pretrained(base_model, config=config)
        self.projection = torch.nn.Linear(self.encoder.config.d_model, dim)

    def get_embed(self, x):
        x = self.encoder(x, attention_mask=(x > 0).long()).last_hidden_state[:, 0]
        x = self.projection(x)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x

    def forward(self, query, passage):
        if query is not None:
            query = self.get_embed(query)
        if passage is not None:
            passage = self.get_embed(passage)
        return query, passage
