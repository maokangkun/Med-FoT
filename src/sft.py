from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from rich import print

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def make_conversation(example):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ],
    }

def sft():
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig)
    parser = TrlParser(dataclass_types)
    script_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)

    ################
    # Dataset
    ################
    dataset = load_dataset('json', data_files=script_args.dataset_name, field='sft')
    dataset = dataset.map(make_conversation)
    dataset = dataset.remove_columns(["pid", "input", "output"])
    print(dataset)

    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        args=training_args,
        peft_config=get_peft_config(model_args),
    )
    trainer.train()
    # Save and push to hub
    trainer.save_model(training_args.output_dir)

def sft_test():
    dataset = load_dataset('json', data_files='experiments/data/sft_grpo_pipeline_v7_llama3.3_70b_d4.json', field='sft')
    dataset = dataset.map(make_conversation)
    dataset = dataset.remove_columns(["pid", "input", "output"])
    print(dataset)

    dataset2 = load_dataset('json', data_files='experiments/data/sft_grpo_pipeline_v7_llama3.3_70b_d4.json', field='grpo')
    dataset2 = dataset2.map(make_conversation)
    dataset2 = dataset2.remove_columns(["pid", "input", "output"])
    print(dataset2)

    model_id = "experiments/RL/Qwen2.5-1.5B-SFT"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    idx1 = 0
    messages = dataset['train'][idx1]['messages'][:-1]
    print('='*30)
    print(messages)
    print('='*30)

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
    print(messages)
    print('='*30)

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
