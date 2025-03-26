FROM python:3.9-slim

# 安装依赖包（包括 PyTorch, torchvision, Pillow 等）
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 拷贝代码
COPY . .

# 安装 Python 依赖
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 设置环境变量（Cloud Run 默认使用 PORT=8080）
ENV PORT=8080
ENV PYTHONUNBUFFERED=TRUE

# 容器启动时运行 gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]
