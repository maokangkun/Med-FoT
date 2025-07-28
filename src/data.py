from .utils import *
from rich import print
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think>\n<answer> answer here </answer>"
)

inp_temp = "Diagnostic criteria:\n{diagnostic_criteria}\n\n[Patient data]\nPresent illness: {present_illness}\nPhysical examination: {physical_examination}\nLaboratory tests: {laboratory_tests}\nRadiology: {radiology}\n\n[Task]\nBased on the diagnostic criteria given above, the patient's reports and test results given below, please think carefully and give the final diagnosis results from these diseases {candidate_diseases}."

cot_temp1 = """First, I need to extract key diagnostic points for differential diagnosis from four perspectives based on the given diagnostic criteria:
- Present Illness Diagnostic Points: {phi_diag_points}
- Physical Examination Diagnostic Points: {pe_diag_points}
- Laboratory Test Diagnostic Points: {lab_diag_points}
- Radiology Diagnostic Points: {rad_diag_points}

After that, I need to match the extracted key points with the actual patient data:
- From the present illness, select descriptions that align with the key points: {phi_findings}
- From the physical exam, pinpoint findings that match the diagnostic points: {pe_findings}
- Compare laboratory data with key indicators: {lab_findings}
- Identify corresponding radiological findings: {rad_findings}

Next, I need to integrate all the information to form a preliminary diagnosis:
- Compare the candidate diseases' diagnostic criteria with the extracted findings.
- Generate one or more preliminary diagnoses, each accompanied by a rationale and a confidence score (for example, if the patient has persistent RLQ pain, rebound tenderness, and imaging shows an enlarged appendix, the preliminary diagnosis could be appendicitis with a confidence score of 0.9).
So the preliminary diagnosis is: {preliminary_diagnosis}

Then, I need to perform a comprehensive evaluation to reach a final diagnosis:
- Reassess the patient's data (present illness, physical exam, labs, and imaging) alongside the preliminary diagnosis.
- Clearly outline the diagnostic basis, reasoning, and assign a final confidence score.
- Make sure diagnosis in candidate diseases: {candidate_diseases}
So the final diagnosis is: {diagnosis}

Finally, I need to verify the final diagnosis confidence and, if necessary, initiate a recheck process:
- If the final confidence is at or above the threshold (0.95), the diagnosis is confirmed.
- If it is below the threshold, I will select a specific report (such as the physical examination or imaging report) for re-evaluation, state the purpose for rechecking, and adjust the final diagnosis based on the re-assessment.

Since the final confidence {confidence} >= 0.95, the diagnosis is confirmed.
"""

cot_temp2 = """First, I need to extract key diagnostic points for differential diagnosis from four perspectives based on the given diagnostic criteria:
- Present Illness Diagnostic Points: {phi_diag_points}
- Physical Examination Diagnostic Points: {pe_diag_points}
- Laboratory Test Diagnostic Points: {lab_diag_points}
- Radiology Diagnostic Points: {rad_diag_points}

After that, I need to match the extracted key points with the actual patient data:
- From the present illness, select descriptions that align with the key points: {phi_findings}
- From the physical exam, pinpoint findings that match the diagnostic points: {pe_findings}
- Compare laboratory data with key indicators: {lab_findings}
- Identify corresponding radiological findings: {rad_findings}

Next, I need to integrate all the information to form a preliminary diagnosis:
- Compare the candidate diseases' diagnostic criteria with the extracted findings.
- Generate one or more preliminary diagnoses, each accompanied by a rationale and a confidence score (for example, if the patient has persistent RLQ pain, rebound tenderness, and imaging shows an enlarged appendix, the preliminary diagnosis could be appendicitis with a confidence score of 0.9).
So the preliminary diagnosis is: {preliminary_diagnosis}

Then, I need to perform a comprehensive evaluation to reach a final diagnosis:
- Reassess the patient's data (present illness, physical exam, labs, and imaging) alongside the preliminary diagnosis.
- Clearly outline the diagnostic basis, reasoning, and assign a final confidence score.
- Make sure diagnosis in candidate diseases: {candidate_diseases}
So the final diagnosis is: {pre_diagnosis}

After that, I need to verify the final diagnosis confidence and, if necessary, initiate a recheck process:
- If the final confidence is at or above the threshold (0.95), the diagnosis is confirmed.
- If it is below the threshold, I will select a specific report (such as the physical examination or imaging report) for re-evaluation, state the purpose for rechecking, and adjust the final diagnosis based on the re-assessment.

Since the final confidence {pre_confidence} < 0.95, I select a specific report for re-evaluation: {recheck_item}

Finally, I need to re-evaluate and adjust the final diagnosis based on the re-assessment, so the final diagnosis is: {diagnosis}
"""

def data_process():
    model_name = 'qwq_32b'
    pipeline_name = 'v7_pipeline'
    diseases_num = 15
    in_file = sorted(Path(f'experiments/exp_gpt4o_{pipeline_name}/').glob(f'{model_name}_32k_d{diseases_num}@*.json'))[-1]

    data = jload(in_file)

    candidate_diseases = data[0]['candidate_diseases']

    sft_data_t1 = []
    sft_data_t2 = []
    grpo_data = []
    ans_arr = []
    n1 = n2 = n3 = n4 = n5 = 0
    for d in data:
        diagnosis = d.get('diagnosis', {})
        if type(diagnosis) is list:
            ans = ''
        elif diagnosis is None:
            ans = ''
        else:
            ans = diagnosis.get('final_diagnosis', '').lower()

        if ans == d['pkl_label']:
            n1 += 1
            d |= {'confidence': d['diagnosis']['confidence']}
            if d['diagnosis']['confidence'] >= 0.95:
                n3 += 1

                if 'pre_diagnosis' in d:
                    d |= {'pre_confidence': d['pre_diagnosis']['confidence']}
                    n4 += 1
                    cot = cot_temp2.format(**d)

                    sft_data_t2.append({
                        'pid': d['pid'],
                        'input': inp_temp.format(**d),
                        'output': f'<think>{cot}</think>\n<answer>{ans}</answer>',
                        'label': ans,
                    })
                else:
                    cot = cot_temp1.format(**d)

                    sft_data_t1.append({
                        'pid': d['pid'],
                        'input': inp_temp.format(**d),
                        'output': f'<think>{cot}</think>\n<answer>{ans}</answer>',
                        'label': ans,
                    })
            else:
                if 'pre_diagnosis' in d and d['diagnosis']['confidence'] >= d['pre_diagnosis']['confidence']:
                    n5 += 1
                    d |= {'pre_confidence': d['pre_diagnosis']['confidence']}
                    cot = cot_temp2.format(**d)
                    sft_data_t2.append({
                        'pid': d['pid'],
                        'input': inp_temp.format(**d),
                        'output': f'<think>{cot}</think>\n<answer>{ans}</answer>',
                        'label': ans,
                    })


        if sum([c in ans for c in candidate_diseases]) == 1:
            for c in candidate_diseases:
                if c in ans: ans = c
        if ans == d['pkl_label']:
            n2 += 1

        ans_arr.append(ans)

        grpo_data.append({
            'pid': d['pid'],
            'input': inp_temp.format(**d),
            'output': d['pkl_label'],
        })

    sft_diseases = set([d['label'] for d in sft_data_t1 + sft_data_t2])
    sft_pids = set([d['pid'] for d in sft_data_t1 + sft_data_t2])
    cot2_factor = 20
    sft_data = sft_data_t1 + sft_data_t2 * cot2_factor
    for _ in range(3): random.shuffle(sft_data)

    print('='*20)
    print(set(ans_arr))
    print(f'acc: {n1/len(data):.1%}, {n2/len(data):.1%}, c>0.95 {n3/len(data):.1%}, loop: {n4/len(data):.1%}, {n5/len(data):.1%}')
    print('='*20)
    print(f'sft: {len(sft_data)}, cot1: {len(sft_data_t1)}, cot2: {len(sft_data_t2)}')
    print(f'sft candidate diseases ({len(sft_diseases)}): {sft_diseases}')

    test_num = int(len(grpo_data) * .1)
    grpo_test = random.sample([d for d in grpo_data if d['pid'] not in sft_pids], test_num)
    test_pids = set([d['pid'] for d in grpo_test])
    grpo_train = [d for d in grpo_data if d['pid'] not in test_pids]
    print(f'grpo: {len(grpo_data)}, train: {len(grpo_train)}, test: {len(grpo_test)}')

    out_data = {
        'version': '0.2.0',
        'info': f'sft: {len(sft_data)}, cot1: {len(sft_data_t1)}, cot2: {len(sft_data_t2)}, cot2_factor: {cot2_factor}, grpo: {len(grpo_data)}, grpo_train: {len(grpo_train)}, grpo_test: {len(grpo_test)}',
        'generated_by_model': model_name,
        'generated_by_pipeline': pipeline_name,
        'diseases_num': diseases_num,
        'system_prompt': SYSTEM_PROMPT,
        'sft': sft_data,
        'grpo_train': grpo_train,
        'grpo_test': grpo_test,
        'grpo_test_pids': list(set(d['pid'] for d in grpo_test)),
    }

    jdump(out_data, f'experiments/data/sft_grpo_{pipeline_name}_d{diseases_num}_{model_name}.json')
