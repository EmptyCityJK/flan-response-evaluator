import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class FlanText2TextDataset(Dataset):
    def __init__(self, data_path, tokenizer: PreTrainedTokenizer, max_length=512, target_max_length=16):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_max_length = target_max_length
        self.label_keys = ["fully_response", "partially_response", "blank_response"]
        self.data = self.load_and_flatten(data_path)

    def load_and_flatten(self, path):
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        samples = []
        for item in raw_data:
            context = item["context"]
            abstract = item.get("abstract_30", "")
            qas = item["qas"]

            for key in self.label_keys:
                if key not in item:
                    continue

                student_answers = item[key]
                if not student_answers or len(student_answers) != len(qas):
                    continue

                # 拼接多个QA对为 Q：A 格式
                qa_pairs = []
                for qa, answer in zip(qas, student_answers):
                    question = qa.get("question", "No question")
                    qa_pairs.append(f"{question}：{answer}")

                qa_section = " [SEP] ".join(qa_pairs)

                # 构造 prompt（符合 Flan 风格）
                input_text = (
                    "Given the context, questions and student's answers, "
                    "classify the overall answer quality as one of: fully_response, partially_response, blank_response.\n"
                    f"{qa_section} [SEP] {abstract} [SEP] {context}"
                )

                label_text = key  # 输出标签是字符串

                if len(samples) == 0:
                    print("First Flan input_text:\n", input_text)
                    print("First Flan label_text:\n", label_text)

                samples.append({
                    "input_text": input_text,
                    "label_text": label_text
                })

        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        encoded_input = self.tokenizer(
            item["input_text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        encoded_label = self.tokenizer(
            item["label_text"],
            padding="max_length",
            truncation=True,
            max_length=self.target_max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded_input["input_ids"].squeeze(0),
            "attention_mask": encoded_input["attention_mask"].squeeze(0),
            "labels": encoded_label["input_ids"].squeeze(0)
        }


def load_flan_dataset(tokenizer, data_dir, max_length=512, target_max_length=16, batch_size=16, num_workers=4):
    from torch.utils.data import DataLoader

    train_set = FlanText2TextDataset(f"{data_dir}/train.json", tokenizer, max_length, target_max_length)
    val_set = FlanText2TextDataset(f"{data_dir}/valid.json", tokenizer, max_length, target_max_length)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
