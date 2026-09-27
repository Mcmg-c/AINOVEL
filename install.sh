#!/bin/bash
# AINOVEL Linux 服务器一键安装脚本
# 用法: bash install.sh
set -e

cd "$(dirname "$0")"

PY=python3
if ! command -v python3 >/dev/null 2>&1; then
  echo "错误：未找到 python3，请先安装 Python 3.10+"
  exit 1
fi

if [ ! -d .venv ]; then
  echo "==> 创建虚拟环境"
  python3 -m venv .venv
fi
source .venv/bin/activate

echo "==> 安装依赖（优先清华镜像）"
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
  || pip install -r requirements.txt

if [ ! -f config.json ]; then
  cp config.example.json config.json
  echo "==> 已生成 config.json，请编辑并填入 DeepSeek 的 api_key（文件在服务器本机，勿提交 git）"
fi

cat > novel <<'EOF'
#!/bin/bash
# AINOVEL CLI 便捷入口
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/.venv/bin/python" "$DIR/cli.py" "$@"
EOF
chmod +x novel

echo "==> 安装完成"
echo "==> 自检：.venv/bin/python -c 'import llm_adapters' && echo OK"
.venv/bin/python -c 'import llm_adapters' && echo OK
echo "==> 用法示例："
echo "./novel from-docs --docs <服务器上的文档路径> --project <输出目录> --guidance \"题材关键词\" --workers 4"
