export OPENAI_BASE_URL="https://models.sjtu.edu.cn/api/v1"
export OPENAI_API_KEY="sk-tYfudtgpHIqCKpxWisXgtA" # sjtu
export SWANLAB_API_KEY=6xX60x1CzPEWbfdso1UwJ
# python main.py -d webqsp --prompt_mode scored_100 -m deepseek-chat
export VLLM_ATTENTION_BACKEND="XFORMERS"
python main.py -d webqsp --prompt_mode scored_100 -m ~/Documents/data/Llama-3.1-8B-Instruct --tensor_parallel_size 4 --frequency_penalty 0.0