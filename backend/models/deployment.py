"""
Copyright 2022 The Microsoft DeepSpeed Team
"""
import logging
import argparse
import sys
sys.path.append("..")

from utils import (
    ForwardRequest,
    ForwardResponse,
    GenerateResponse,
    TokenizeRequest,
    TokenizeResponse,
    create_generate_request,
)

from backend.constants import LOGGER_NAME, HF_ACCELERATE, HF_CPU, ONNX_RUNTIME_GPU, ONNX_RUNTIME_CPU


class ModelDeployment:
    def __init__(self, args: argparse.Namespace):
        self.model = self.get_model_class(args.deployment_framework)(args)
        logger = logging.getLogger(LOGGER_NAME)
        logger.info("Model {} loaded with deployment framework {}".format(
            args.model_name, args.deployment_framework))

    def get_model_class(self, deployment_framework: str):
        if deployment_framework == HF_ACCELERATE:
            from .hf_inference import HFAccelerateModel

            return HFAccelerateModel
        elif deployment_framework == HF_CPU:
            from .hf_inference import HFCPUModel

            return HFCPUModel
        elif deployment_framework == ONNX_RUNTIME_GPU:
            from .onnx_inference import ONNXRuntimeGPU

            return ONNXRuntimeGPU
        elif deployment_framework == ONNX_RUNTIME_CPU:
            from .onnx_inference import ONNXRuntimeCPU

            return ONNXRuntimeCPU
        else:
            raise ValueError(
                f"Unknown deployment framework {deployment_framework}")

    def generate(self, **kwargs) -> GenerateResponse:
        if "request" in kwargs:
            request = kwargs["request"]
        else:
            request = create_generate_request(**kwargs)

        response = self.model.generate(request)

        if isinstance(response, Exception):
            raise response
        else:
            return response

    def forward(self, request: ForwardRequest) -> ForwardResponse:
        response = self.model.forward(request)

        if isinstance(response, Exception):
            raise response
        else:
            return response

    def tokenize(self, request: TokenizeRequest) -> TokenizeResponse:
        response = self.model.tokenize(request)

        return response
