import csv
import json
import asyncio
from sigmaflow.manager import PipelineManager
# from sigmaflow.clients.llm_openai import llm_client
# from sigmaflow.clients.llm_mlx_batch import llm_client, llm_batch_processor
from .utils import *

def mimic_cdm_example_patient():
    data = json.load(open('experiments/data/example_patient.json', 'r'))
    lab_test_mapping = {}
    with open('experiments/data/lab_test_mapping.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['itemid']:
                lab_test_mapping[row['itemid']] = row['label']

    pih = data['Patient History']
    exam = data['Physical Examination']
    lab_tests = {lab_test_mapping[k]:v for k,v in data['Laboratory Tests'].items()}

    rad = ''
    for r in data['Radiology']:
        t = ''
        for k,v in r.items(): t += k+': '+v.strip()+'\n'
        rad += t+'\n'

    # patient_info = f"Patient History: {pih}\n\nPhysical Examination: {exam}\n\nLaboratory Tests: {lab_tests}\n\nRadiology:\n{rad}"

    patient_info = {
        'present_illness': pih,
        'physical_examination': exam,
        'laboratory_tests': lab_tests,
        'radiology': rad,
    }

    return patient_info

def load_patient(i=None, d=4):
    from pathlib import Path
    exp_dir = Path('experiments/')
    patient_data = jload(exp_dir / f'data/{d}_abdominal.json')
    if i is None:
        import random
        i = random.randint(0, len(patient_data)-1)
    return patient_data[i]

def async_test():
    disease_num = int(os.getenv("EXP_DISEASE_NUM"))
    candidate_diseases = ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis', 'hepatitis', 'pyelonephritis', 'cholangitis', 'peritonitis', 'gastritis', 'esophagitis', 'colitis', 'duodenitis', 'cystitis', 'enteritis', 'orchitis']
    inp_data = load_patient(d=disease_num) | {'candidate_diseases': candidate_diseases[:disease_num]}
    test_task = {
        'gpt4o_v7_pipeline': [inp_data]
    }

    # pm = PipelineManager(llm_type='torch', rag_type='json', pipes_dir='src/pipelines', run_mode='seq')
    pm = PipelineManager(llm_type='lmdeploy', rag_type='json', pipes_dir='src/pipelines', run_mode='async')
    # pm = PipelineManager(llm_type='vllm', rag_type='json', pipes_dir='src/pipelines', run_mode='async')

    for pipe_name, data_arr in test_task.items():
        for data in data_arr:
            r, info = pm.pipes[pipe_name].run(data, save_perf=False)

async def replay_test():
    # from sigmaflow.clients.llm_vllm import llm_client, llm_batch_processor
    from sigmaflow.clients.llm_openai import llm_client
    import tqdm

    exp = 'experiments/exp_gpt4o_2_pipeline/llama3.2_3b_8k@20241213-145040.json'
    pipe_name = 'gpt4o_v2_pipeline'
    pm = PipelineManager(llm_client(is_async=True), None, pipes_dir='src/pipelines')

    data = jload(exp)
    n = 50
    parts = len(data) // n
    results = []
    run_cnt = 0
    for i in tqdm.trange(parts):
        r, cnt = await pm.pipes[pipe_name].replay('ExtractPainLocation', data[i*n:(i+1)*n])
        run_cnt += cnt
        results += r

    print(f'avg run: {run_cnt/len(results):.1f}')
    arr = [i for i in results if i['pain_location'] not in ['RLQ', 'RUQ', 'LLQ', 'epigastric']]
    print(arr)
    print(len(arr))

def calc_max_token_len():
    import tqdm
    import numpy as np
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/Users/mkk/workspace/models/Llama-3.2-3B-Instruct")
    exp = 'experiments/exp_gpt4o_v2_pipeline/llama3.2_3b_8k@20241213-145040.json'
    pipe_name = 'gpt4o_v3_pipeline'
    pm = PipelineManager(None, None, pipes_dir='src/pipelines')
    pipeline = pm.pipes[pipe_name]
    node = pipeline.pipetree.node_manager['AllInfoDiagnoseCondition']

    data = jload(exp)
    items = list(node.conf['next'].keys())
    items_text = '\n'.join([f'[#{i+1}] {t}' for i, t in enumerate(items)])
    arr = []
    for d in tqdm.tqdm(data):
        inps_t = '\n'.join(f'{k}: {d[k]}' for k in node.conf['inp'])
        prompt = node.pipe.prompt(inps_t, items_text)
        tokens = tokenizer(prompt)['input_ids']
        arr.append(len(tokens))
    
    print(f'min: {min(arr)}, max: {max(arr)}, median: {np.median(arr)}')
    arr2 = [i for i in arr if i > 8192]
    print(arr2)
    print(len(arr2))

def draw_pipeline():
    pm = PipelineManager(None, None, run_mode='normal', pipes_dir='src/pipelines')
    pipe_name = 'baseline_nocriteria_pipeline'
    pm.pipes[pipe_name].to_png('logs/baseline_nocriteria_pipeline.png')

def test():
    # draw_pipeline()
    async_test()
    # asyncio.run(replay_test())
    # calc_max_token_len()