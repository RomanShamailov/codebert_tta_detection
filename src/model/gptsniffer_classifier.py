import torch
from torch import nn
from transformers import AutoModelForSequenceClassification


class GPTSnifferClassifier(nn.Module):
    """
    HuggingFace sequence classifier wrapper.

    Returns both logits and a CLS embedding used by centroid-based TTA.
    """

    def __init__(self, path, num_labels=2):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path,
            num_labels=num_labels,
            output_hidden_states=True,
        )

    def forward(self, input_ids, attention_mask=None, **batch):
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "return_dict": True,
        }

        outputs = self.model(**model_inputs)
        embeddings = outputs.hidden_states[-1][:, 0]
        return {"logits": outputs.logits, "embeddings": embeddings}

    def __str__(self):
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return (
            super().__str__()
            + f"\nAll parameters: {all_parameters}"
            + f"\nTrainable parameters: {trainable_parameters}"
        )
