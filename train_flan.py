# train_flan.py
import torch
import os
from datetime import datetime
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from preprocess.flan_dataset import load_flan_dataset
from model.flan_generator import FlanGeneratorLightning
import wandb
import argparse

torch.set_float32_matmul_precision('high')

def train_flan(model_name, lr=2e-5, max_epochs=3, gpus=1, batch_size=32, num_workers=4, data_dir='./data', resume_checkpoint=None):
    # 初始化 tokenizer 和数据集
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    train_loader, val_loader = load_flan_dataset(tokenizer, data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)

    # 创建 checkpoints 目录
    os.makedirs("checkpoints", exist_ok=True)
    
    # 初始化模型
    if resume_checkpoint is not None and os.path.isfile(resume_checkpoint):
        print(f"Loading model from checkpoint: {resume_checkpoint}")
        model = FlanGeneratorLightning.load_from_checkpoint(resume_checkpoint, model_name=model_name, tokenizer=tokenizer, lr=lr)
    else:
        print("No checkpoint provided or not found. Training from scratch.")
        model = FlanGeneratorLightning(model_name=model_name, tokenizer=tokenizer, lr=lr)

    # 获取时间戳和数据集名
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset_name = os.path.basename(os.path.normpath(data_dir))

    # 定义 checkpoint 保存逻辑
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"flan-{dataset_name}-acc={{val/acc:.4f}}-{current_time}",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        save_weights_only=True,
    )

    # 初始化 Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=gpus,
        # precision="16-mixed",
        precision=32,
        logger=pl.loggers.WandbLogger(project='flan-generator'),
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLAN-T5 Model Training")
    parser.add_argument('--model_name', type=str, default="/root/flan-response/flan-t5-base")
    parser.add_argument('--data_dir', type=str, default='./data/Vanilla')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to checkpoint to resume training")
    args = parser.parse_args()

    train_flan(
        model_name=args.model_name,
        lr=args.lr,
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        resume_checkpoint=args.checkpoint
    )
