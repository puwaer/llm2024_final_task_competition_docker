import os
import torch
from unsloth import FastLanguageModel
import json
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from tqdm import tqdm
from datasets import load_dataset

# Hugging Face Token を指定
# https://huggingface.co/settings/tokens
HF_TOKEN = "" #@param {type:"string"}

model_id = "llm-jp/llm-jp-3-13b"       #llm-jp/llm-jp-3-13bは12gbのメモリでは乗らない
adapter_id = "llm-jp-3-13b-it-docker" #Fine-Tuningしたモデルにつけたい名前、it: Instruction Tuning

# ローカル保存ディレクトリの設定
MODEL_DIR = "./local_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ローカルモデルパス
local_model_path = os.path.join(MODEL_DIR, model_id.replace("/", "_"))

# llm-jp/llm-jp-3-13bを4bit量子化のqLoRA設定でロード。
max_seq_length = 512 # unslothではRoPEをサポートしているのでコンテキスト長は自由に設定可能
dtype = torch.float16 # Noneにしておけば自動で設定
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


# SFT用のモデルを用意
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
    max_seq_length = max_seq_length,
)

dataset = load_dataset("json", data_files="./ichikara-instruction-003-001-test.json")

# 学習時のプロンプトフォーマットの定義
prompt = """### 指示
{}
### 回答
{}"""


"""
formatting_prompts_func: 各データをプロンプトに合わせた形式に合わせる
"""
EOS_TOKEN = tokenizer.eos_token # トークナイザーのEOSトークン（文末トークン）
def formatting_prompts_func(examples):
    input = examples["text"] # 入力データ
    output = examples["output"] # 出力データ
    text = prompt.format(input, output) + EOS_TOKEN # プロンプトの作成
    return { "formatted_text" : text, } # 新しいフィールド "formatted_text" を返す
pass

# # 各データにフォーマットを適用
dataset = dataset.map(
    formatting_prompts_func,
    num_proc= 4, # 並列処理数を指定
)

dataset

# データを確認
print(dataset["train"]["formatted_text"][3])


#training_arguments: 学習の設定
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset=dataset["train"],
    max_seq_length = max_seq_length,
    dataset_text_field="formatted_text",
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        num_train_epochs = 1,
        logging_steps = 10,
        warmup_steps = 10,
        save_steps=100,
        save_total_limit=2,
        max_steps=-1,
        learning_rate = 2e-4,
        fp16 = True,
        bf16 = False,
        group_by_length=True,
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

"""
training_arguments: 学習の設定

  - output_dir:
      -トレーニング後のモデルを保存するディレクトリ

  - per_device_train_batch_size:
      - デバイスごとのトレーニングバッチサイズ

  - per_device_eval_batch_size:
      - デバイスごとの評価バッチサイズ

  - gradient_accumulation_steps:
      - 勾配を更新する前にステップを積み重ねる回数

  - optim:
      - オプティマイザの設定

  - num_train_epochs:
      - エポック数

  - eval_strategy:
      - 評価の戦略 ("no"/"steps"/"epoch")

  - eval_steps:
      - eval_strategyが"steps"のとき、評価を行うstep間隔

  - logging_strategy:
      - ログ記録の戦略

  - logging_steps:
      - ログを出力するステップ間隔

  - warmup_steps:
      - 学習率のウォームアップステップ数

  - save_steps:
      - モデルを保存するステップ間隔

  - save_total_limit:
      - 保存しておくcheckpointの数

  - max_steps:
      - トレーニングの最大ステップ数

  - learning_rate:
      - 学習率

  - fp16:
      - 16bit浮動小数点の使用設定（第8回演習を参考にすると良いです）

  - bf16:
      - BFloat16の使用設定

  - group_by_length:
      -  入力シーケンスの長さによりバッチをグループ化 (トレーニングの効率化)

  - report_to:
      - ログの送信先 ("wandb"/"tensorboard"など)
"""

#@title 現在のメモリ使用量を表示
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

#@title 学習実行
trainer_stats = trainer.train()


# ELYZA-tasks-100-TVの読み込み。事前にファイルをアップロードしてください
# データセットの読み込み。
# omnicampusの開発環境では、左にタスクのjsonlをドラッグアンドドロップしてから実行。
datasets = []
with open("./elyza-tasks-100-TV_test.jsonl", "r") as f:
    item = ""
    for line in f:
      line = line.strip()
      item += line
      if item.endswith("}"):
        datasets.append(json.loads(item))
        item = ""

# 推論するためにモデルのモードを変更
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
with open(f"{adapter_id}_output.jsonl", 'w', encoding='utf-8') as f:
    for result in results:
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')

        
# LoRAアダプタだけ保存
model.push_to_hub_merged(
    adapter_id+"_lora",
    tokenizer=tokenizer,
    save_method="lora",
    token=HF_TOKEN,
    private=True
)