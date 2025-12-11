# 纯 Python 基础镜像（小），不带系统 CUDA
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# 仅装运行期需要的系统库（音频处理等）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# 設定CUDA環境變數
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 重新編譯CUDA kernel以獲得更好的性能

# 用 uv 管理依赖（快）
RUN pip install -U pip uv

WORKDIR /workspace

# 先同步你项目里除 torch 之外的依赖（pyproject.toml / uv.lock）
COPY pyproject.toml uv.lock* README.md ./
# 拷贝代码（只放源代码；模型/日志运行时挂载）
COPY app/ app/
COPY indextts/ indextts/

RUN uv sync --default-index "https://mirrors.aliyun.com/pypi/simple"   # 不带 deepspeed，也不需要 build-essential/nvcc

# 运行期目录 & 环境变量（推理优化开关按需）
ENV HF_HOME=/workspace/checkpoints/hf_cache \
    HF_HUB_CACHE=/workspace/checkpoints/hf_cache \
    INDEXTTS_CHECKPOINTS=/workspace/checkpoints \
    INDEXTTS_CONFIG=/workspace/checkpoints/config.yaml \
    INDEXTTS_OUTPUT_DIR=/workspace/outputs \
    INDEXTTS_LOG_DIR=/workspace/logs \
    INDEXTTS_FP16=true \
    INDEXTTS_DEEPSPEED=false \
    INDEXTTS_CUDA_KERNEL=false

# 設定CUDA環境變數
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 重新編譯CUDA kernel以獲得更好的性能
RUN pip install -e . --no-deps --no-build-isolation || echo "CUDA kernel compilation failed, will use fallback"

RUN mkdir -p /workspace/checkpoints /workspace/outputs /workspace/logs

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

EXPOSE 8000
CMD PYTHONPATH=. uv run app/main.py
