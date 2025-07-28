import random
import collections
from .utils import *


def analysis():
    patients = jload('./experiments/exp_gpt4o_v5_pipeline/llama3.3_70b_q4_16k@20241231-211004.json')
    right = collections.defaultdict(list)
    wrong = collections.defaultdict(list)

    for p in patients:
        if 'diagnosis' in p and p['diagnosis'] and 'final_diagnosis' in p['diagnosis']:
            if p['pkl_label'] != p['diagnosis']['final_diagnosis']:
                wrong[p['pkl_label']].append(p['diagnosis']['confidence'])
            else:
                right[p['pkl_label']].append(p)
        else:
            wrong[p['pkl_label']].append(0)

    print('correct')
    for k in right:
        print(f'{k} {len(right[k])}: {sorted(set([p['diagnosis']['confidence'] for p in right[k]]))}')
    
    print('error')
    for k in wrong:
        print(f'{k} {len(wrong[k])}: {sorted(set(wrong[k]))}')
    

def sample_wrong_patient(n=2):
    patients = jload('./experiments/exp_gpt4o_v5_pipeline/llama3.2_3b_16k@20241231-015152.json')
    wrong = collections.defaultdict(list)

    for p in patients:
        include = ['present_illness', 'physical_examination', 'laboratory_tests', 'radiology']
        exclude = ['pid', 'present_illness', 'physical_examination', 'laboratory_tests', 'radiology', 'diagnosis_label', 'discharge_diagnosis', 'pkl_label']

        d = {
            'input': {i:p[i] for i in include},
            'output': {i:p[i] for i in p.keys() if i not in exclude},
        }
        if 'diagnosis' in p and 'final_diagnosis' in p['diagnosis']:
            if p['pkl_label'] != p['diagnosis']['final_diagnosis']:
                wrong[p['pkl_label']].append(d)
        else:
            wrong[p['pkl_label']].append(d)

    wrong_text = ''
    i = 1
    for k in wrong:
        print(k, len(wrong[k]))
        samples = random.sample(wrong[k], n)
        for s in samples:
            wrong_text += f'Diagnosis error case {i}\n'
            wrong_text += f'Input:\n{s["input"]}\n'
            wrong_text += f'Output:\n{s["output"]}\n'
            wrong_text += f'Correct Answer: {k}\n\n'
            i += 1
    
    return wrong_text

def improve_pipeline():
    import torch
    from transformers import pipeline

    pre_pipeline = open('./src/pipelines/gpt4o_v5_pipeline.py', 'r').read()
    bad_case = sample_wrong_patient()
    acc = 0.9

    prompt = '''[Objective]
Please create a new and improved pipeline that reduces errors and enhances accuracy. Below, we provide the details of the existing pipeline, its current accuracy, and examples of errors it produces.

[Provided Information]
Previous pipeline:
```python
{pre_pipeline}
```

Current Accuracy: {acc}

Error examples:
{bad_case}

[Task]
Based on the provided information:
- Design a new pipeline or suggest improvements to the previous pipeline.
- Focus on addressing the highlighted errors while maintaining or improving overall accuracy.
- Think step by step, and return the new pipeline in `python` format like the provided pipeline.

[Output]
'''
    prompt = prompt.replace('{pre_pipeline}', pre_pipeline).replace('{bad_case}', bad_case).replace('{acc}', str(acc))

    tokens = calc_tokens(prompt)
    print(prompt)
    print(f'tokens: {tokens}')

    pipe = pipeline(
        "text-generation",
        model=os.getenv('MODEL_PATH'),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    messages = [{"role": "user", "content": prompt}]
    outputs = pipe(
        messages,
        max_new_tokens=16384,
    )
    out = outputs[0]["generated_text"][-1]['content']

    with open('tmp_prompt.txt', 'w') as f:
        f.write(prompt)
    with open('tmp_out.txt', 'w') as f:
        f.write(out)

    # breakpoint()
    # new_pipeline = ...

def atuo_mode():
    model_name = 'llama3.2_3b_16k'

    improve_pipeline()
