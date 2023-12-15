from argparse import Namespace
from backend.models.model import Model
from transformers import AutoModelForCausalLM


class HFAccelerateModel(Model):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.device = 'cuda'
        self.inference_type = 'hf'

        # this is the CUDA device for the current process. This will be used
        # later to identify the GPU on which to transfer tensors
        self.model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='auto', load_in_8bit=True)

        self.model.requires_grad_(False)
        self.model.eval()

        self.post_init(args.model_name)


class HFCPUModel(Model):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.device = 'cpu'
        self.inference_type = 'hf'

        self.model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='cpu', torch_dtype=args.dtype)
        self.model.to_bettertransformer()

        self.model.requires_grad_(False)
        self.model.eval()

        self.post_init(args.model_name)
