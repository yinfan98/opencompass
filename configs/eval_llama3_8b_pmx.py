from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM
with read_base():
    from .datasets.mmlu.mmlu_gen import mmlu_datasets

    from .opencompass.models.pmx import PMXModel

    from .summarizers.chat_OC15 import summarizer


work_dir = 'outputs/pmx/llama3_pmx_8b'
pmx_llama3_8b = PMXModel(ckpt_path='/root/model/Llama-3-pmx')

models = pmx_llama3_8b
datasets = []
datasets += mmlu_datasets