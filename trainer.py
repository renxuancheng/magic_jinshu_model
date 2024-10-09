import torch
import yaml
from yaml.loader import SafeLoader
import numpy as np
from transformers import AutoTokenizer
from model import GPT2Model
from data_loader import JinshuDataset
from torch.utils.data import DataLoader
from torchinfo import summary
import tqdm
import torch._dynamo

torch._dynamo.config.suppress_errors = True
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
ctx = torch.amp.autocast(device_type=device, dtype=torch.bfloat16)
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))


class Trainer:
    def __init__(self, model, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model.to(device)
        # model = torch.compile(model)
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
        self.train_data_path = config["data"]["train_path"]
        self.validate_data_path = config["data"]["validate_path"]
        self.micro_step = config["train"]["gradient_accumulation_steps"]
        self.config = config
        self.block_size = config["model"]["block_size"]
        self.batch_size = config["train"]["batch"]
        train_dataset = JinshuDataset(self.config, self.train_data_path, self.block_size + 1)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validate_dataset = JinshuDataset(self.config, self.validate_data_path, self.block_size + 1)
        self.validate_dataloader = DataLoader(validate_dataset, batch_size=self.batch_size, shuffle=True)

    def train(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["train"]["lr_rate"])
        for epoch in range(config["train"]["epoch"]):
            micro_step = 1
            for data in tqdm.tqdm(self.train_dataloader):
                x, y = (
                    data[:, 0 : self.block_size].int(),
                    data[:, 1 : self.block_size + 1].int(),
                )
                x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                    device, non_blocking=True
                )
                with ctx:
                    logits, loss = self.model(idx=x, targets=y)
                    #    print(logits)
                    loss = loss / self.micro_step
                # loss.backward()
                scaler.scale(loss).backward()
                micro_step = micro_step + 1
                if micro_step % self.micro_step == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    # optimizer.step()
                    optimizer.zero_grad()
                    print(loss)


if __name__ == "__main__":
    with open("./config/magic_jinshu_pt.yaml", "r", encoding="utf-8") as f_read:
        config = yaml.load(f_read, SafeLoader)
        model = GPT2Model(config)
        # model.to(device)
        trainer = Trainer(model, config)
        trainer.train()
