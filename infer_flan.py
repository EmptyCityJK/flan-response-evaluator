import torch
import json
import os
from transformers import AutoTokenizer
from model.flan_generator import FlanGeneratorLightning
from tqdm import tqdm

def load_test_data(test_path):
    """Load test data which contains only one response field per item"""
    with open(test_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    test_items = []
    for item in data:
        context = item["context"]
        abstract = item.get("abstract_30", "")
        qas = item["qas"]
        student_answer = item["response"]  # Only one response field
        
        # Build QA pairs
        qa_pairs = []
        for qa, ans in zip(qas, student_answer):
            question = qa.get("question", "没有问题")
            qa_pairs.append(f"{question}：{ans}")
        qa_section = " [SEP] ".join(qa_pairs)
        
        # Build FLAN-style input text
        input_text = (
            "Given the context, questions and student's answers, "
            "classify the overall answer quality as one of: fully_response, partially_response, blank_response.\n"
            f"{qa_section} [SEP] {abstract} [SEP] {context}"
        )
        
        test_items.append({
            "text": input_text,
            "id": item.get("id", len(test_items))  # Use index as ID if not provided
        })
    
    return test_items

def predict(model, tokenizer, test_items, max_length=512, batch_size=16):
    """Perform batch prediction on test items"""
    model.eval()
    predictions = []
    
    # Process in batches
    for i in tqdm(range(0, len(test_items), batch_size), desc="Predicting"):
        batch_items = test_items[i:i+batch_size]
        
        # Tokenize batch
        texts = [item["text"] for item in batch_items]
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        
        # Generate predictions
        with torch.no_grad():
            preds = model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=16  # Short output length for our labels
            )
            pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Clean and validate predictions
        valid_labels = {"fully_response", "partially_response", "blank_response"}
        for pred in pred_texts:
            pred = pred.strip().lower()
            if pred not in valid_labels:
                # Default to partially_response if model outputs something unexpected
                pred = "partially_response"
            predictions.append(pred)
    
    return predictions

def save_predictions(predictions, output_file):
    """Save predictions to a text file"""
    with open(output_file, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(f"{pred}\n")

if __name__ == "__main__":
    # Configuration
    checkpoint_filename = "acc=0.9800-20250513-194620.ckpt"  # Replace with your checkpoint filename
    model_path = os.path.join("checkpoints/flan-narriative-acc=val", checkpoint_filename)  # Proper path construction
    model_name = "/root/flan-response/flan-t5-base"  # Same as training
    test_path = "./data/narriative/test.json"  # Path to your test file
    output_file = "flan_predictions.txt"  # Output file name
    
    # Check if checkpoint exists
    if not os.path.exists(model_path):
        print(f"Error: Checkpoint file not found at {model_path}")
        print("Available checkpoints:")
        for f in os.listdir("checkpoints"):
            print(f"  - {f}")
        exit(1)
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = FlanGeneratorLightning.load_from_checkpoint(
        model_path,
        model_name=model_name,
        tokenizer=tokenizer
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and prepare test data
    print("Loading test data...")
    test_items = load_test_data(test_path)
    
    # Perform prediction
    print("Making predictions...")
    predictions = predict(model, tokenizer, test_items)
    
    # Save results
    save_predictions(predictions, output_file)
    print(f"Predictions saved to {output_file}")