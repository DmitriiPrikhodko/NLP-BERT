import torch
from torch import nn
from transformers import AutoModel

class MyPersonalTinyBert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # и снова грузим
        self.bert = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
        # заморозим параметры
        for param in self.bert.parameters():
            param.requires_grad = False
        # делаем свой слой для классификации
        self.linear = nn.Sequential(
            nn.Linear(312, 256),  # начинаем с длины embedding, которые делает модель
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),  # это добавил для души))
            nn.ReLU(),
            nn.Linear(64, 1),  # заканчиваем кол-вом классов
        )

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        vector = bert_out.last_hidden_state[:, 0, :]
        classes = self.linear(vector)
        return classes
    

class BertInputs(torch.utils.data.Dataset):
    def __init__(self, encoded_text):
        super().__init__()
        self.inputs = encoded_text

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        # print(self.inputs[idx])
        return (
            torch.Tensor(self.inputs[idx]["input_ids"]).long(),
            torch.Tensor(self.inputs[idx]["attention_mask"]).long(),
        )