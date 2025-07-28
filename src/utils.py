import os
import json
import pickle
import traceback

def pload(pkl):
    with open(pkl, 'rb') as f: data = pickle.load(f)
    return data

def jload(inp):
    with open(inp, 'r', encoding='utf-8') as f:
        return json.load(f)

def jdump(obj, out):
    with open(out, 'w', encoding='utf-8') as f:
        if isinstance(obj, (dict, list)):
            json.dump(obj, f, indent=4, ensure_ascii=False)
        elif isinstance(obj, str):
            f.write(obj)
        else:
            raise ValueError(f"Unexpected type: {type(obj)}")

def check_pipeline_valid(pipeconf=None, pipefile=None):
    from sigmaflow.pipetree import PipeTree
    try:
        pt = PipeTree(None, None, None, pipeconf=pipeconf, pipefile=pipefile, is_async=True)
        return True
    except Exception as e:
        error_msg = traceback.format_exc()
        print(error_msg)
        return False

def calc_tokens(text):
    from transformers import AutoTokenizer

    model_path = os.getenv('MODEL_PATH')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    n = len(tokenizer(text)['input_ids'])
    return n

