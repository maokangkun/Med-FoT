from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig
from rich import print
import re

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def generate_r1_prompt(tokenizer, input, target):
    r1_prefix = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input},
        {"role": "assistant", "content": "<think>"},
    ]

    return {
        "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), 
        "target": target
    }

def format_reward_func(completions, target, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
      
      Returns:
          list[float]: Reward scores
    """
    rewards = []
 
    for completion, gt in zip(completions, target):
 
      try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion        
        # Check if the format is correct
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n?<answer>([\s\S]*?)<\/answer>$"
 
        match = re.search(regex, completion, re.DOTALL) 
        # if the format is not correct, reward is 0
        if match is None or len(match.groups()) != 2:
            rewards.append(0.0)
        else:
            rewards.append(1.0)
      except Exception:
        rewards.append(0.0)
    return rewards
 
def equation_reward_func(completions, target, **kwargs):
    rewards = []
    for completion, gt in zip(completions, target):
      try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            rewards.append(0.0)
            continue
        # Extract the "answer" part from the completion
        ans = match.group(1).strip()

        if ans == gt:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
      except Exception:
            rewards.append(0.0) 
    return rewards

def grpo():
    base_model_path = 'experiments/RL/Qwen2.5-1.5B-SFT'
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    dataset = load_dataset('json', data_files='experiments/data/sft_grpo_pipeline_v7_llama3.3_70b_d4.json', field='grpo')
    dataset = dataset.map(lambda x: generate_r1_prompt(tokenizer, x["input"], x["output"]))
    dataset = dataset.remove_columns(["input", "output"])
    print(dataset)

    model_config = ModelConfig(
        model_name_or_path=base_model_path,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
        use_peft=True,
        load_in_4bit=True,
    )
    training_args = GRPOConfig(
        output_dir="experiments/RL/Qwen2.5-1.5B-GRPO",
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        logging_steps=5,
        num_train_epochs=1,
        # max_steps=200,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        # GRPO specific parameters
        max_prompt_length=32768,
        max_completion_length=2048,
        num_generations=2,
        beta=0.001,
    )
    trainer = GRPOTrainer(
        model=model_config.model_name_or_path,
        reward_funcs=[format_reward_func, equation_reward_func],
        args=training_args,
        train_dataset=dataset['train'],
        peft_config=get_peft_config(model_config),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

def make_conversation(example):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ],
    }

def grpo_test():
    dataset = load_dataset('json', data_files='experiments/data/sft_grpo_pipeline_v7_llama3.3_70b_d4.json', field='sft')
    dataset = dataset.map(make_conversation)
    dataset = dataset.remove_columns(["input", "output"])
    print(dataset)

    dataset2 = load_dataset('json', data_files='experiments/data/sft_grpo_pipeline_v7_llama3.3_70b_d4.json', field='grpo')
    dataset2 = dataset2.map(make_conversation)
    dataset2 = dataset2.remove_columns(["input", "output"])
    print(dataset2)

    model_id = "experiments/RL/Qwen2.5-1.5B-GRPO/checkpoint-350"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    idx1 = 0
    messages = dataset['train'][idx1]['messages'][:-1]
    print('='*30)
    # print(messages)
    # print('='*30)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    # print(prompt)
    # print('='*30)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    print('='*30)
    label = dataset['train'][idx1]['messages'][-1]['content'].split("</think>")[1]
    print(f'label: {label}')

    idx2 = -1
    messages = dataset2['train'][idx2]['messages'][:-1]
    print('='*30)
    # print(messages)
    # print('='*30)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    # print(prompt)
    # print('='*30)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    print('='*30)
    label = dataset2['train'][idx2]['messages'][-1]['content']
    print(f'label: {label}')
