# train_model.py
from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Load external MolTextNet dataset
dataset = load_dataset("liuganghuggingface/moltextnet")

# Use T5-small for faster training, T5-base for better accuracy 
# PDF suggests starting small (GPT-2 Medium mentioned as example)
model_name = "laituan245/molt5-small" 
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(examples):
    inputs = [s for s in examples["canonical_smiles"]]
    targets = [d for d in examples["description"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_data = dataset.map(preprocess, batched=True)

model = T5ForConditionalGeneration.from_pretrained(model_name)

args = Seq2SeqTrainingArguments(
    "molt5_finetuned",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    save_strategy="epoch",
    fp16=True # crucial for GPU speed
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"] # If available, otherwise split train
)

trainer.train()
trainer.save_model("checkpoints/my_final_model")