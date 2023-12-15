from argparse import Namespace

from backend.models.model import Model
from optimum.onnxruntime import ORTModelForCausalLM


class ONNXRuntimeGPU(Model):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.device = 'cuda'
        self.inference_type = 'ort'

        self.model = ORTModelForCausalLM.from_pretrained(
            args.model_name, export=True, provider="CUDAExecutionProvider")

        self.post_init(args.model_name)


class ONNXRuntimeCPU(Model):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.device = 'cpu'
        self.inference_type = 'ort'

        self.model = ORTModelForCausalLM.from_pretrained(
            args.model_name, export=True, provider="CPUExecutionProvider")

        self.post_init(args.model_name)
