import os
from unsloth import FastLanguageModel
from peft import PeftModel
import torch
import json
from tqdm import tqdm
import re
from huggingface_hub import login


# Hugging Faceトークン
HF_TOKEN = ""

# モデルとアダプターの情報
model_id = "llm-jp/llm-jp-3-13b"
adapter_id = ""

# ローカル保存ディレクトリの設定
MODEL_DIR = "./local_models"
ADAPTER_DIR = "./local_adapters"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ADAPTER_DIR, exist_ok=True)

# ローカルモデルパス
local_model_path = os.path.join(MODEL_DIR, model_id.replace("/", "_"))
local_adapter_path = os.path.join(ADAPTER_DIR, adapter_id.replace("/", "_"))

# unslothのFastLanguageModelで元のモデルをロード。
dtype = torch.float16 # Noneにしておけば自動で設定(gpuによってfloat16,bfloat16などを使用して)
load_in_4bit = True # 今回は13Bモデルを扱うためTrue

# モデルのダウンロードまたは読み込み
if not os.path.exists(local_model_path):
    print("Downloading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        dtype=dtype,  
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
        cache_dir=MODEL_DIR
    )
    # 保存
    model.save_pretrained(local_model_path)
    tokenizer.save_pretrained(local_model_path)
else:
    print("Loading model from local...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=local_model_path,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
    )

# アダプターのダウンロードまたは読み込み
if not os.path.exists(local_adapter_path):
    print("Downloading adapter...")
    adapter_model = PeftModel.from_pretrained(model, adapter_id, token=HF_TOKEN)
    adapter_model.save_pretrained(local_adapter_path)
else:
    print("Loading adapter from local...")
    adapter_model = PeftModel.from_pretrained(model, local_adapter_path)

# アダプター統合
model = adapter_model

# タスクデータの読み込み
datasets = []
with open("./elyza-tasks-100-TV_0.jsonl", "r") as f:
    item = ""
    for line in f:
        line = line.strip()
        item += line
        if item.endswith("}"):
            datasets.append(json.loads(item))
            item = ""

# 推論モード設定
FastLanguageModel.for_inference(model)

# 推論処理
results = []
for dt in tqdm(datasets):
    input_text = dt["input"]
    prompt = f"### 指示\n{input_text}\n### 回答\n"

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        use_cache=True,
        do_sample=False,
        repetition_penalty=1.2
    )
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).split("\n### 回答")[-1]
    results.append({"task_id": dt["task_id"], "input": input_text, "output": prediction})

# jsonlで保存
output_file = os.path.join("./", f"{adapter_id.replace('/', '_')}_output.jsonl")
with open(output_file, 'w', encoding='utf-8') as f:
    for result in results:
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')

print(f"Results saved to {output_file}")
