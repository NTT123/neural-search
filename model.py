import torch
from transformers import AutoConfig, T5EncoderModel


class MsNet(torch.nn.Module):
    def __init__(self, base_model: str = "google/t5-v1_1-small"):
        super().__init__()
        config = AutoConfig.from_pretrained(base_model)
        config.dropout_rate = 0.0
        self.encoder = T5EncoderModel.from_pretrained(base_model, config=config)

    def get_embed(self, x):
        mask = (x > 0).long()
        x = self.encoder(x, attention_mask=mask).last_hidden_state
        mask = mask[:, :, None]
        x = (x * mask).sum(dim=1)
        x = x / mask.sum(dim=1)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x

    def forward(self, query, passage):
        if query is not None:
            query = self.get_embed(query)
        if passage is not None:
            passage = self.get_embed(passage)
        return query, passage
