from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM
with read_base():
    from .datasets.mmlu.mmlu_gen import mmlu_datasets

    from .models.hf_llama.hf_llama3_8b import models as hf_llama3_8b

    from .summarizers.chat_OC15 import summarizer


work_dir = 'outputs/hf/llama3_hf_8b'
hf_llama3_8b = [
    dict(
        type=HuggingFaceCausalLM,
        path='/share/new_models/meta-llama/Meta-Llama-3-8B',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_seq_len=2048,
        batch_padding=False,
        # 以下参数为各类模型都有的参数，非 `HuggingFaceCausalLM` 的初始化参数
        abbr='llama3-8b',            # 模型简称，用于结果展示
        max_out_len=100,            # 最长生成 token 数
        batch_size=16,              # 批次大小
        run_cfg=dict(num_gpus=1),   # 运行配置，用于指定资源需求
    )
]

models = hf_llama3_8b
datasets = []
datasets += mmlu_datasets