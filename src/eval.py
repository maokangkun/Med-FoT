from .utils import *
from pathlib import Path
import collections

exp_dir = Path('experiments/')
show_result = [
    'exp_gpt4o_v7_pipeline', 
    'exp_gpt4o_v8_pipeline', 
    'exp_RL_v2'
]
skip_model_name = ['qwen2.5_1.5b_32k_d5']

def print_title(title):
    print(f'{"-"*(len(title)+4)}\n| {title} |\n{"-"*(len(title)+4)}')

def print_result(result):
    d4 = ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']
    name = result['name']
    max_name_len = result['max_name_len']
    cnt = result['cnt']
    acc = cnt['pred_total'] / cnt['total']
    arr = []
    for d in d4:
        a = cnt[f'pred_{d}'] / cnt[d]
        arr.append(f'{d[0].upper()}: {a:.1%}')
    d4_info = ', '.join(arr)

    print(f'{name:<{max_name_len}s}: {acc:.1%}, {d4_info}')

def eval_exp():
    test_pids = set(jload(exp_dir / 'data/sft_grpo_pipeline_v7_llama3.3_70b_d9_v3.json')['grpo_test_pids'])

    for exp_pipeline_dir in sorted(exp_dir.glob('exp_*')):
        if exp_pipeline_dir.stem not in show_result: continue

        print_title(exp_pipeline_dir.stem)

        model_list = sorted(set([p.stem.split('@')[0] for p in exp_pipeline_dir.glob('*.json')]))
        max_name_len = max(len(m) for m in model_list) + 2

        for model_name in model_list:
            if model_name in skip_model_name: continue
            result = {'name': model_name, 'max_name_len': max_name_len}
            result['cnt'] = cnt = collections.defaultdict(int)

            recent_file = sorted(exp_pipeline_dir.glob(f'{model_name}@*.json'))[-1]
            data = jload(recent_file)

            for d in data:
                label = d['pkl_label']
                cnt[label] += 1
                cnt['total'] += 1

                diagnosis = d.get('diagnosis', None)
                pred = None
                if diagnosis and type(diagnosis) is dict: pred = diagnosis.get('final_diagnosis', None)

                if str(pred).strip().lower() == label:
                    cnt['pred_total'] += 1
                    cnt[f'pred_{label}'] += 1

                if d['pid'] in test_pids:
                    cnt['grpo'] += 1

                    if str(pred).strip().lower() == label:
                        cnt['pred_grpo'] += 1
                        cnt[f'pred_grpo_{label}'] += 1

            print_result(result)
