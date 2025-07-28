from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import asyncio
from pydantic import BaseModel
from typing import Optional, List
import hashlib
import time
import json


app = FastAPI()

from llmpipeline.llmpipeline import PipelineManager
from llmpipeline.llmpipeline.clients.llm_torch import llm_client
from llmpipeline.llmpipeline.clients.rag_json import rag_client

# llm = llm_client(is_async=True)
# rag = rag_client(is_async=True)
llm = llm_client()
rag = rag_client()
pm = PipelineManager(llm, rag, run_mode='normal', pipes_dir='src/pipelines')
pipe_name = 'gpt4o_v10_pipeline'

async def fake_video_streamer(p_data):
    pm = PipelineManager(llm, rag, run_mode='normal', pipes_dir='src/pipelines')
    pipe_name = 'gpt4o_v9_pipeline'

    async for result in pm.pipes[pipe_name].pipetree.server_run(p_data.dict()):
        yield f"{result}\n"


class PatientData(BaseModel):
    candidate_diseases: list
    present_illness: str
    physical_examination: str
    laboratory_tests: dict
    radiology: str
    # add_rules: Optional[str] = None

@app.post("/api/diagnose")
async def diagnose(p_data: PatientData):
    return StreamingResponse(fake_video_streamer(p_data))

class ChatMessage(BaseModel):
    role: str
    content: str | dict | list

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.9
    stream: bool = False

@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    ret = {
        "id": hashlib.sha256(str(request.messages).encode()).hexdigest(),
        "object": "chat.completion" + (".chunk" if request.stream else ""),
        "model": request.model,
        "choices": []
    }
    inp = request.messages[-1].content
    if (t := type(inp)) is dict:
        if request.stream:
            from fastapi.encoders import jsonable_encoder
            async def foo():
                async for r in pm.pipes[pipe_name].pipetree.server_run(inp):
                    ret["created"] = time.time()
                    ret["choices"] = [{
                        "index": 0,
                        "delta": {"content": r},
                        "finish_reason": None,
                    }]
                    yield f"data: {json.dumps(ret)}\n\n"

            return StreamingResponse(foo())
        else:
            arr = []
            async for r in pm.pipes[pipe_name].pipetree.server_run(inp):
                arr.append(r)

            msg = {
                "message": ChatMessage(
                    role="assistant",
                    content={'result': arr}
                )
            }
            ret["choices"].append(msg)
    elif t is str:
        msg = {
            "message": ChatMessage(
                role="assistant",
                content=llm(inp)
            )
        }
        ret["choices"].append(msg)
    ret["created"] = time.time()
    return ret

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)
