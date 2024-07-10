import fire
import sys
import os
import json
from ..base import BaseModel
from pathlib import Path
from typing import List

import ppl.pmx.model_zoo.llama.modeling.Loader as Loader
from ppl.pmx.model_zoo.llama.huggingface.Tokenizer import Tokenizer
from ppl.pmx.model_zoo.ModelParams import ModelParams

class PMXModel(BaseModel):

    def __init__(self,
                 ckpt_path: str,
                 tokenizer_only: bool = False,
                 meta_template: Optional[Dict] = None,
                 **kwargs):
        self.tokenizer_path = ckpt_path
        self.ckpt_dir = ckpt_path
        self.temperature = 0.0
        self.top_p = 0.95
        self.batch = 4
        self.seqlen_scale_up = 1
        self.unaligned_batch = False
        self.max_gen_len = 256
        self.friendly_gqa = False
        self.fused_qkv = True
        self.fused_kvcache = True
        self.fused_ffn_glu = True
        self.auto_causal = True
        self.quantized_cache = True
        self.cache_layout = 0
        self.cache_mode = 0
        self.dynamic_batching = True
        self.context_chunking = True
        self.dump_tensor_path = None
        self.dump_steps = []
        ## initial
        self.tokenizer = Tokenizer(model_path=self.tokenizer_path)
        with open(Path(ckpt_dir) / "opmx_params.json", "r") as f:
            params = json.loads(f.read())
        params: ModelParams = ModelParams(**params)

        self.generator = Loader.load(
        self.ckpt_dir, self.params,
        friendly_gqa=self.friendly_gqa,
        fused_qkv=self.fused_qkv,
        fused_kvcache=self.fused_kvcache,
        fused_ffn_glu=self.fused_ffn_glu,
        fused_alibi=False,
        auto_causal=self.auto_causal,
        with_rope=True,
        with_alibi=False,
        quantized_cache=self.quantized_cache,
        cache_layout=self.cache_layout,
        cache_mode=self.cache_mode,
        dynamic_batching=self.dynamic_batching,
        attn_wqkv_bias_term=False,
        attn_wo_bias_term=False,
        ffn_linear_bias_term=False,
        load_to_cpu=False,
        rotary_dim=0,
        dump_tensor_path=self.dump_tensor_path,
        dump_steps=self.dump_steps
        )

        generator.context_chunking = context_chunking if dynamic_batching else False

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings."""
        tokenized_prompt = self.tokenizer.encode(prompt, bos=True, eos=False)
        return len(tokenized_prompt)

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """Generate results given a list of inputs. """
        prompt_tokens = [self.tokenizer.encode(t, bos=True, eos=False) for t in inputs]

        # Assuming batch size is the length of inputs
        batch = len(inputs)
        
        results = self.generator.generate(
            prompt_tokens, 
            self.tokenizer.get_eos_id(), 
            self.tokenizer.get_pad_id(),
            max_gen_len=max_out_len, 
            temperature=self.temperature, 
            top_p=self.top_p, 
            top_k=0
        )

        generated_texts = [self.tokenizer.decode(result) for result in results]
        return generated_texts

    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs."""
        all_ppls = []
        for input_text in inputs:
            tokenized_input = self.tokenizer.encode(input_text, bos=True, eos=True)
            total_log_prob = 0
            total_len = len(tokenized_input)
            
            # Loop over each token and get the prediction probability
            for i in range(1, total_len):
                context = tokenized_input[:i]
                target = tokenized_input[i]
                
                # Get the probability distribution for the next token
                log_probs = self.generator.predict(context)
                
                # Extract the log probability of the target token
                log_prob = log_probs[target]
                total_log_prob += log_prob
            
            # Calculate perplexity
            avg_log_prob = total_log_prob / total_len
            ppl = np.exp(-avg_log_prob)
            all_ppls.append(ppl)
        
        return all_ppls