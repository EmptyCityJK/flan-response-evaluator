import torch
from transformers import T5ForConditionalGeneration
import pytorch_lightning as pl

class FlanGeneratorLightning(pl.LightningModule):
    def __init__(self, model_name, tokenizer, lr=2e-5):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])  # 不保存 tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask, labels=None):
        # 通过将 labels 转为 tensors 保证它是一个 tensor 类型
        if labels is not None and isinstance(labels, list):
            labels = torch.tensor(labels).to(input_ids.device)  # 确保标签在相同设备上
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # 确保 labels 中的 padding token 被忽略，替换为 -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        outputs = self.forward(input_ids, attention_mask, labels)
        loss = outputs.loss

        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss


    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        # 确保 labels 中的 padding token 被忽略，替换为 -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        # 计算损失
        outputs = self.forward(input_ids, attention_mask, labels)
        loss = outputs.loss

        # 生成预测
        preds = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=64)
        pred_texts = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_texts = self.tokenizer.batch_decode(
            torch.where(labels != -100, labels, self.tokenizer.pad_token_id), skip_special_tokens=True
        )

        # 逐个比对，完全匹配认为正确
        correct = sum([p.strip() == l.strip() for p, l in zip(pred_texts, label_texts)])
        total = len(label_texts)
        acc = correct / total if total > 0 else 0.0

        # 记录验证集的 loss 和 acc
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
