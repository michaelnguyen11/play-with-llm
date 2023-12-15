import os
from dotenv import load_dotenv
from functools import partial
import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel

from constants import HF_ACCELERATE
from models import ModelDeployment
from utils import (
    ForwardRequest,
    GenerateRequest,
    TokenizeRequest,
    get_exception_response,
    get_num_tokens_to_generate,
    get_torch_dtype,
    parse_bool,
    run_and_log_time,
    setup_logging
)

load_dotenv()

class QueryID(BaseModel):
    generate_query_id: int = 0
    tokenize_query_id: int = 0
    forward_query_id: int = 0


# placeholder class for getting args. gunicorn does not allow passing args to a
# python script via ArgumentParser
class Args:
    def __init__(self) -> None:
        self.deployment_framework = os.getenv("DEPLOYMENT_FRAMEWORK", "hf_accelerate")
        self.model_name = os.getenv("MODEL_NAME", "bigscience/bloom-560m")
        self.dtype = get_torch_dtype(os.getenv("DTYPE", "fp16"))
        self.allowed_max_new_tokens = int(os.getenv("ALLOWED_MAX_NEW_TOKENS", 100))
        self.max_input_length = int(os.getenv("MAX_INPUT_LENGTH", 512))
        self.max_batch_size = int(os.getenv("MAX_BATCH_SIZE", 4))
        self.debug = parse_bool(os.getenv("DEBUG", "false"))


app = FastAPI()
router = APIRouter()
args = Args()
model = ModelDeployment(args)
query_ids = QueryID()
logger = setup_logging()


@router.get("/")
async def home():
    return {"healcheck": "KiLM Home Test service OK"}


@router.get("/query_id/", status_code=200)
def query_id():
    return query_ids.model_dump()


@router.post("/tokenize/", status_code=200)
def tokenize(token_request: TokenizeRequest):
    try:
        response, total_time_taken = run_and_log_time(
            partial(model.tokenize, request=token_request))

        response.query_id = query_ids.tokenize_query_id
        query_ids.tokenize_query_id += 1
        response.total_time_taken = "{:.2f} msecs".format(
            total_time_taken * 1000)

        return response.dict()
    except Exception as e:
        response = get_exception_response(
            query_ids.tokenize_query_id, args.debug)
        query_ids.tokenize_query_id += 1
        logger.error("Failed to tokenize : {}".format(response))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/", status_code=200)
async def generate(prompt: GenerateRequest):
    try:
        prompt.max_new_tokens = get_num_tokens_to_generate(
            prompt.max_new_tokens, args.allowed_max_new_tokens)
        response, total_time_taken = run_and_log_time(
            partial(model.generate, request=prompt))

        response.query_id = query_ids.generate_query_id
        query_ids.generate_query_id += 1
        response.total_time_taken = "{:.2f} secs".format(total_time_taken)

        return response.dict()
    except Exception as e:
        response = get_exception_response(
            query_ids.generate_query_id, args.debug)
        query_ids.generate_query_id += 1
        logger.error("Failed to generate : {}".format(response))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forward/", status_code=200)
def forward(forward_request: ForwardRequest):
    try:
        if len(forward_request.conditioning_text) != len(forward_request.response):
            raise Exception(
                "unequal number of elements in conditioning_text and response arguments")

        response, total_time_taken = run_and_log_time(
            partial(model.forward, request=forward_request))

        response.query_id = query_ids.forward_query_id
        query_ids.forward_query_id += 1
        response.total_time_taken = "{:.2f} secs".format(total_time_taken)

        return response.dict()
    except Exception as e:
        response = get_exception_response(
            query_ids.forward_query_id, args.debug)
        query_ids.forward_query_id += 1
        logger.error("Failed to forward : {}".format(response))
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, port=8000, host="0.0.0.0")
