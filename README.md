# llm2024_final_task_competition_dockerセットアップガイド
このプログラムは、NVIDIA GPUを使用してをllm2024の最終課題のサンプルプログラムをDocker上で動作させるためのものです

## 前提条件

- Dockerがインストールされていること
- NVIDIA GPUドライバーがインストールされていること

## イメージのビルド

プロジェクトのルートディレクトリで以下のコマンドを実行し、Dockerイメージをビルドします：

```bash
docker image build -t llm2024_competition:latest .
```

## コンテナの作成と起動

GPUを利用し、ソースコードをマウントするコンテナを作成・起動します：

```bash
docker container run -it --gpus all --name llm2024_competition -v $(pwd)/src:/app llm2024_competition:latest
```

### コマンドの解説

- `--gpus all`: すべてのGPUをコンテナで使用可能にします
- `--name llm2024_competition`: コンテナに名前を付けます
- `-v $(pwd)/src:/app`: ローカルの`src`ディレクトリをコンテナの`/app`にマウントします

## 既存のコンテナの開始

既に作成済みのコンテナを再開する場合：

```bash
docker start -i llm2024_competition
```

## lora学習と推論の実行

- ソースコードは`/app`ディレクトリにマウントされているため、そこで作業を行う
- `/src`ディレクトリに必要なスクリプトとデータセットが配置されていること

```
/src
├── LoRA_template_unsloth.py        # LoRA学習用スクリプト
├── Model_Inference_Template_unsloth.py  # 推論用スクリプト
├── elyza-tasks-100-TV_0.jsonl      # データセットファイル1
└── ichikara-instruction-003-001-1.jsonl  # データセットファイル2
```

lora学習のコマンド
```bash
python3 LoRA_template_unsloth.py
```

推論のコマンド
```bash
python3 Model_Inference_Template_unsloth.py
```
