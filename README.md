# Play With LLM models
The project is a Docker service for playing with LLM models.

The project provides an HTTP API endpoint for text generation based on input prompts. Therefore, users can play with LLM models, tuning the model by specifying parameters like `max_new_tokens`, `no_repeat_ngram_size`, etc.
Please refer to this [document](https://huggingface.co/docs/transformers/main_classes/configuration) for more parameter details.

## Dependencies:
Please ensure that you installed `docker` and `docker-compose` on your system.
- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

Note: Docker Compose supports to define GPU reservations as service containers from v1.28.0+. So if your `docker-compose --version` is lower than v1.28.0, please upgrade manually `docker-compose` to at least v1.28.0.

## Usage
1. Nagivation to the project directory
2. Build and start the Docker container:
```
docker-compuse up
```
3. When the Docker container is running, use `server_request.py` to generate text.
```
python3 server_request.py --host 0.0.0.0 --port 8008
```

You can play with Huggingface LLM models by specify parameters like `max_new_tokens`, `no_repeat_ngram_size`, etc. in the request body of `generate` function inside `server_request.py` file.

### Deployment configuration
The demo uses some environment varibales to configure the service when deployment.
1. To change model name, edit the `MODEL_NAME` in `.env` file.
2. To change deployment framework, edit `DEPLOYMENT_FRAMEWORK` in `.env` file. The demo supports 4 type of deployment :
- Hugging Face CPU inference with FP32 precision (`hf_cpu`)
- Hugging Face Accelerate inference with INT8 precision (`hf_accelerate`)
- ONNXRuntime CPU inference (`ort_cpu`)
- ONNXRuntime GPU inference (`ort_gpu`)

After editing `.evn` file, run `docker-compose up` to re-load the environment file.

## Testing
- Hardware used : Ubuntu 20.04 with Nvidia RTX2080 8G with CUDA 12
- Model tested : BLOOM 560M, BLOOM 1b7

## Known issue
If you are using Nvidia CUDA version 12 or above, to run ONNXRuntime GPU inference, you need to build from source the onnxruntime-gpu library. [Reference](https://github.com/microsoft/onnxruntime/issues/15242#issuecomment-1723807663)
