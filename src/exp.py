import os
import re
import csv
import tqdm
import asyncio
import datetime
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from sigmaflow.manager import PipelineManager
from .utils import *

exp_dir = Path('experiments/')

def get_exp_data(data_file):
    if data_file.exists():
        patient_data = jload(data_file)
    else:
        labels = ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']
        mimic_dir = Path('/ailab/user/maokangkun/data/MIMIC-CDM')
        all_data = {}
        for pkl in mimic_dir.glob('*_hadm_info_first_diag.pkl'):
            tmp = pload(pkl)
            for t in tmp:
                all_data[t] = tmp[t] | {'pkl_label': pkl.stem.split('_')[0]}

        lab_test_mapping = {}
        with open(mimic_dir / 'lab_test_mapping.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['itemid']:
                    lab_test_mapping[row['itemid']] = row['label']

        patient_data = []
        for pid in all_data:
            data = all_data[pid]
            pih = data['Patient History']
            exam = data['Physical Examination']
            lab_tests = {lab_test_mapping[str(k)]:v for k,v in data['Laboratory Tests'].items()}

            rad = ''
            for r in data['Radiology']:
                t = ''
                for k,v in r.items(): t += k+': '+v.strip()+'\n'
                rad += t+'\n'

            discharge_diagnosis = data['Discharge Diagnosis'].lower()
            diags = sorted([(discharge_diagnosis.index(i), i) for i in labels if i in discharge_diagnosis])

            patient_data.append({
                'pid': pid,
                'present_illness': pih,
                'physical_examination': exam,
                'laboratory_tests': lab_tests,
                'radiology': rad,
                'diagnosis_label': [d for _, d in diags],
                'discharge_diagnosis': data['Discharge Diagnosis'],
                'pkl_label': data['pkl_label']
            })
        jdump(patient_data, data_file)
        jdump(patient_data[:5], data_file.with_suffix('.lite.json'))

    return patient_data

def run_exp():
    start_t = datetime.datetime.now()

    pipe_name = os.getenv('EXP_PIPELINE')
    model_name = os.getenv("EXP_MODEL_NAME")
    disease_num = int(os.getenv("EXP_DISEASE_NUM"))

    pm = PipelineManager(llm_type='lmdeploy', rag_type='json', pipes_dir='src/pipelines')
    pipeline = pm.pipes[pipe_name]

    patient_data = get_exp_data(exp_dir / f'data/{disease_num}_abdominal.json')
    candidate_diseases = ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis', 'hepatitis', 'pyelonephritis', 'cholangitis', 'peritonitis', 'gastritis', 'esophagitis', 'colitis', 'duodenitis', 'cystitis', 'enteritis', 'orchitis']

    data = [d | {'candidate_diseases': candidate_diseases[:disease_num]} for d in patient_data]
    results = pipeline.run(data, save_perf=False)#, split=50)

    out_dir = exp_dir / f'exp_{pipe_name}'
    out_dir.mkdir(parents=True, exist_ok=True)
    out = [r for r,_ in results]
    t = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    jdump(out, out_dir / f'{model_name}_d{disease_num}@{t}.json')
    end_t = datetime.datetime.now()
    print(f'cost time: {end_t-start_t}')

def replay_exp():
    exp = '/ailab/user/maokangkun/researchs/clinical_pipeline_mining/experiments/exp_gpt4o_2_pipeline/llama3.3_70b_q4_4k@20241211-175020.json'
    
    asyncio.create_task(llm_batch_processor())
    pm = PipelineManager(llm_client(is_async=True), None, pipes_dir='src/pipelines')
    pipe_name = 'gpt4o_2_pipeline'
    # model_name = os.getenv("PULSE_MODEL")
    model_name = 'llama3.1_8b_8k'
    pipeline = pm.pipes[pipe_name]

def run_exp_rl():
    from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig, ChatTemplateConfig

    # model_name = 'QwQ-32B'
    model_name = 'Qwen2.5-7B-GRPO'
    model_path = f'/mnt/petrelfs/maokangkun/code/projects/clinical_pipeline_rl/{model_name}/checkpoint-200'
    # model_path = '/mnt/petrelfs/maokangkun/models/hf/QwQ-32B'
    data_files = 'experiments/data/sft_grpo_v7_pipeline_d15_qwq_32b.json'
    partition = 'grpo_test'

    chat_temp_name = None
    session_len = 32768
    max_new_tokens = 4096
    batch_size = 4096
    pp = 1
    tp = 1

    backend_config = TurbomindEngineConfig(
                        session_len=session_len,
                        pp=pp,
                        tp=tp)
    gen_config = GenerationConfig(
                        do_sample=True,
                        # top_p=0.95,
                        # temperature=0.7,
                        max_new_tokens=max_new_tokens)
    chat_temp = ChatTemplateConfig(model_name=chat_temp_name) if chat_temp_name else None
    llm = pipeline(model_path, backend_config=backend_config, chat_template_config=chat_temp)


    data = jload(data_files)
    prompts = []
    for d in data[partition]:
        prompts.append([
            {"role": "system", "content": data['system_prompt']},
            {"role": "user", "content": d["input"]},
        ])

    start_t = datetime.datetime.now()
    outputs = llm(prompts, gen_config=gen_config)
    end_t = datetime.datetime.now()
    print(f'cost: {end_t - start_t}')

    result = []
    for d, o in zip(data[partition], outputs):
        r = {
            'pid': d['pid'],
            'out': o.text,
            'diagnosis': {
                'think': None,
                'final_diagnosis': None,
            },
            'pkl_label': d['output'],
        }
        r1 = r"<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>"
        m1 = re.search(r1, o.text, re.DOTALL)
        if m1: r['diagnosis']['think'] = m1.group()

        r2 = r"<answer>([\s\S]*?)<\/answer>"
        m2 = re.search(r2, o.text, re.DOTALL)
        if m2: r['diagnosis']['final_diagnosis'] = m2.group(1)

        result.append(r)

    t = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    jdump(result, f'experiments/exp_RL_v2/{model_name.lower()}_d15@{t}.json')
