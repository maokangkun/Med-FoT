from pathlib import Path
from .prompts.gen_base_pipeline import GEN_BASE_PIPELINE_TEMP
from llmpipeline.llmpipeline.clients.llm_openai import llm_client
from llmpipeline.llmpipeline import log
from .utils import *

def construct_prompt():
    docs = open('llmpipeline/Docs.md', 'r').read()
    cold_pipeline = open('src/pipelines/common_colds_pipeline.py', 'r').read()
    prompt = GEN_BASE_PIPELINE_TEMP.replace('{DOCS}', docs).replace('{COLDS_PIPELINE}', cold_pipeline)
    return prompt

def gen_base_pipeline(pipeline_dir=None):
    prompt = construct_prompt()
    log.debug(prompt)
    llm = llm_client()
    ret = llm(prompt)
    log.debug(ret)

    # r = check_pipeline_valid(pipefile='src/pipelines/common_colds_pipeline.py')
    # print(r)

    exit()
    out = llm(prompt)
    if check_pipe(out):
        if pipeline_dir:
            pipeline_dir = Path(pipeline_dir)
            base_file = pipeline_dir / 'g0.py'
            with open(base_file, 'w') as f: f.write(out)
        return out