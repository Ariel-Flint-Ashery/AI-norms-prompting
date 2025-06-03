def create_prompts(options, correct, incorrect):
    """Create the system and user prompts with the specified parameters."""
    system_prompt = f"""Context: Player 1 is playing a multi-round partnership game with Player 2 for 100 rounds. At each round, Player 1 and Player 2 simultaneously pick an action from the following values: {options}. The payoff that both players get is determined by the following rule:
    1. If Players play the SAME action as each other, they will both be REWARDED with payoff +{correct} points.
    2. If Players play DIFFERENT actions to each other, they will both be PUNISHED with payoff {incorrect} points.
    The objective of each Player is to maximize their own accumulated point tally, conditional on the behavior of the other player."""

    #user_prompt = """Do you think that Player 1 is playing against a single player or a whole population?"""
    user_prompt = """Here is a description of something.
        Does this setup remind you of any existing model or theory in social science? Please name it and briefly explain the analogy. 
        Additionally, tell me if you know what the optimal move is after a success.
        Finally, do you think that you are playing against a single player or a whole population?"""
    return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]

# query the HuggingFace api (e.g., For Llama-3-70B Instruct)
from huggingface_hub import InferenceClient

HF_API_KEY = "<YOUR_API_KEY>"

client = InferenceClient(
    provider="novita",
    api_key=HF_API_KEY,
)

messages = create_prompts(options = ['Q', 'M'], correct = 100, incorrect = -50)
print(messages)
repeats = 5
completions_list = []
for r in range(repeats):
    completion = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        messages=messages,
        temperature = 0.5,
        max_tokens = 500,
    )

    print(completion.choices[0].message.content)
    completions_list.append(completion.choices[0].message.content)

# query the anthropic api
import anthropic
ANTHROPIC_API_KEY = "YOUR_API_KEY"
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

messages = create_prompts(options = ['Q', 'M'], correct = 100, incorrect = -50)
print(messages)
repeats = 5
claude_completions_list = []
for r in range(repeats):
    completion = client.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=500,
    temperature=0.5,
    system=messages[0]['content'],
    messages=[messages[1]])

    print(completion.content[0].text)
    claude_completions_list.append(completion.content[0].text)

# query a local LLM (using Huggingface Transformers, and using quantization)

print('Start', flush=True)
import sys
import torch
print(f'torch available: {torch.cuda.is_available()}', flush=True)
print(torch.version.cuda)
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch import cuda, bfloat16
print(f'torch available: {torch.cuda.is_available()}', flush=True)

print(f'python: {sys.version}', flush=True)
print(f'torch: {torch.__version__}', flush=True)


access_token = 'YOUR_TOKEN'
model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
print(f'model: {model_name}')
cache_dir = 'FOLDER_WHERE_MODEL_IS_LOCATED'

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=bfloat16)
model = AutoModelForCausalLM.from_pretrained(model_name, resume_download = True, device_map='cuda:0', quantization_config=bnb_config, cache_dir = cache_dir)

def query(messages, temperature = 0.5, max_new_tokens = 500):
    try:
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda:0")
        #print('using chat template')
    except:
        text = ''.join([x['content'] for x in messages])
        inputs = tokenizer.encode(text, return_tensors = "pt").to("cuda:0")
    #print(inputs)
    with torch.no_grad():
        if temperature == 0:
            outputs = model.generate(inputs, max_new_tokens = max_new_tokens,output_scores=True, return_dict_in_generate=True, output_hidden_states=True, do_sample = False, pad_token_id=tokenizer.eos_token_id)
        else:
            outputs = model.generate(inputs, max_new_tokens = max_new_tokens,output_scores=True, return_dict_in_generate=True, output_hidden_states=True, do_sample = True, temperature = temperature, pad_token_id=tokenizer.eos_token_id)

    generated_tokens = outputs.sequences[0]
    prompt_length = inputs.shape[1]
    generated_text = tokenizer.decode(generated_tokens[prompt_length:], skip_special_tokens=True)
    #print(f'{generated_text}', flush = True)
    return {'generated_text': generated_text}

messages = create_prompts(options = ['Q', 'M'], correct = 100, incorrect = -50)
repeats = 5
local_completions_list = []
for r in range(repeats):
    completion = query(messages = messages, temperature = 0.5, max_new_tokens = 500)
    print(f"{r}, {completion['generated_text']}", flush = True)
    local_completions_list.append(completion['generated_text'])

