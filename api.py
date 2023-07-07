from fastapi import FastAPI
from pydantic import BaseModel
import json
import time
import torch
from inferencer import Inferencer
from huggingface_hub import hf_hub_download

app = FastAPI()

class Input(BaseModel):
    content_lst: dict
    typ: str

class Response(BaseModel):
    result: dict

# Clever Flamingo
inferencer = Inferencer(
    lm_path="decapoda-research/llama-7b-hf",
    clip_vision_encoder_path="ViT-L-14-336",
    tuning_config='timdettmers/guanaco-7b',
    )

# Download pretrained checkpoint
checkpoint_path = hf_hub_download("chendelong/clever_flamingo", "clever_flamingo.pt")
inferencer.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)

log_file = 'serving/api_log.json'

@app.post("/clever_flamingo",response_model=Response)        
async def clever_flamingo(request: Input):
    time_start = time.time()
    response, full_text = inferencer(
        prompt=                 request.content_lst['prompt'],
        imgpaths=               request.content_lst['imgpaths'],
        max_new_token=          request.content_lst['args']['max_new_token'],
        num_beams=              request.content_lst['args']['num_beams'],
        temperature=            request.content_lst['args']['temperature'],
        top_k=                  request.content_lst['args']['top_k'],
        top_p=                  request.content_lst['args']['top_p'],
        do_sample=              request.content_lst['args']['do_sample'],
        length_penalty=         request.content_lst['args']['length_penalty'],
        no_repeat_ngram_size=   request.content_lst['args']['no_repeat_ngram_size'],
        response_split="### Assistant:"
    )
    res = {"response": response}
    print(f'request received: {request}')
    print(f'request processed ({round(time.time()-time_start, 2)}s): {response}')

    with open(log_file, 'a') as f:
        f.write(json.dumps({
            'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            'request': str(request),
            'response': str(response),
            'time_elapsed': round(time.time()-time_start, 2)
        }, indent=4) + '\n')

    return Response(result=res)
