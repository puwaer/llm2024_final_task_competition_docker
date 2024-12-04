# ベースイメージとして Ubuntu を使用
#FROM ubuntu:22.04
# ベースイメージとしてUbuntu 22.04を使用
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# 必要なツールとライブラリをインストール
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    mecab \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Pythonバージョンの確認（オプション）
RUN python3 --version

# pipをアップグレード
RUN python3 -m pip install --upgrade pip
RUN apt-get update && apt-get install -y git
RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# 必要な Python ライブラリをインストール
COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN pip install --upgrade pip
RUN pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"

# 作業ディレクトリを設定
WORKDIR /app

# スクリプトファイルをコピー
COPY /src/. /app/

# 環境変数設定（CUDA関連）
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

# 実行コマンド
#CMD ["python3", "your_script.py"]
CMD ["/bin/bash"]

