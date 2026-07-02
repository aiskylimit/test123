#1 +20+a
#accelerate-config
mkdir -p ~/.cache/huggingface/accelerate
sleep 2

cp resources/accelerate_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml
sleep 2

echo "=== Accelerate config copied ==="
cat ~/.cache/huggingface/accelerate/default_config.yaml
