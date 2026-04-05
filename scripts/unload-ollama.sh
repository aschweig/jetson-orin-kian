#!/usr/bin/env bash
# Unload all Ollama models from GPU memory.
set -euo pipefail

OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"

models=$(curl -s "$OLLAMA_URL/api/ps" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for m in data.get('models', []):
    print(m['name'])
" 2>/dev/null)

if [ -z "$models" ]; then
    echo "No models loaded."
    exit 0
fi

for model in $models; do
    echo "Unloading $model ..."
    curl -s "$OLLAMA_URL/api/generate" -d "{\"model\": \"$model\", \"keep_alive\": 0}" > /dev/null
done

echo "Done."
