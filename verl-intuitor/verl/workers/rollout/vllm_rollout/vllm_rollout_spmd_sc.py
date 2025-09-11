# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
import pickle
import socket
import threading
from contextlib import contextmanager
from copy import deepcopy
from types import MethodType
from typing import Any, Dict, List, Union

import numpy as np
import ray
import torch
import torch.distributed
import zmq
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout

#新增
from scipy.special import softmax
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = (
            {}
            if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
            else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        )
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.model.path)#加入tokenizer

        # Offload vllm model to reduce peak memory usage
        if config.free_cache_engine:
            self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    '''@GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

    
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", #     non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)'''

    import itertools

    '''@GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # 0. Ensure non_tensor_batch exists and raw_prompt_ids are set
        non_tensor_batch = prompts.non_tensor_batch or {}
        prompts.non_tensor_batch = non_tensor_batch
        idx0 = prompts.batch["input_ids"]            # [bs, prompt_len]
        bs0 = idx0.size(0)
        raw_ids = non_tensor_batch.get("raw_prompt_ids")
        if raw_ids is None or len(raw_ids) != bs0:
            raw_ids = [_pre_process_inputs(self.pad_token_id, idx0[i]) for i in range(bs0)]
            non_tensor_batch["raw_prompt_ids"] = raw_ids

        # 1. Build vLLM inputs
        if "multi_modal_data" in non_tensor_batch:
            mm = non_tensor_batch.pop("multi_modal_data")
            vllm_inputs = [
                {"prompt_token_ids": r, "multi_modal_data": m}
                for r, m in zip(raw_ids, mm)
            ]
        else:
            vllm_inputs = [{"prompt_token_ids": r} for r in raw_ids]

        # 2. Hyperparameters
        beam_size         = int(kwargs.get("step_beam_size",       self.config.step_beam_size))
        num_rollout       = int(kwargs.get("num_rollout",          self.config.num_rollout))
        num_foresight     = int(kwargs.get("num_foresight",        self.config.num_foresight))
        sigma_rate        = float(kwargs.get("sigma_rate",         self.config.sigma_rate))
        temperature       = float(kwargs.get("temperature",        self.config.temperature))
        step_response_len = int(kwargs.get("step_response_length", self.config.step_response_length))
        response_len      = int(kwargs.get("response_length",      self.config.response_length))

        # 3. Decode prompts
        raw_prompts = self.tokenizer.batch_decode(idx0, skip_special_tokens=True)

        # 4. SamplingParams for intermediate rollout
        base_sp = SamplingParams(
            max_tokens=step_response_len,
            logprobs=1,
            temperature=temperature,
            n=num_rollout,
            stop=["\n", "<end_of_reasoning>"]
        )

        # 5. Initialize beam histories
        prev_steps      = [["" for _ in range(beam_size)] for _ in range(bs0)]
        prev_values     = [[0.0 for _ in range(beam_size)] for _ in range(bs0)]
        weights_history = [[[] for _ in range(beam_size)] for _ in range(bs0)]

        # 6. Multi-step foresight
        for depth in range(num_foresight):
            step_inputs = []
            for b in range(bs0):
                for k in range(beam_size):
                    prefix = (
                        f"User: {raw_prompts[b].strip()}\n"
                        f"Reasoning so far:\n{prev_steps[b][k]}"
                    )
                    ids = self.inference_engine.llm_engine.tokenizer.encode(
                        prefix, add_special_tokens=False
                    )
                    step_inputs.append({"prompt_token_ids": ids})

            outs = self.inference_engine.generate(
                prompts=step_inputs,
                sampling_params=base_sp,
                use_tqdm=False
            )

            all_resp, all_lp, all_adv = [], [], []
            for out in outs:
                for o in out.outputs:
                    txt = o.text.strip()
                    lp  = o.cumulative_logprob / (len(o.token_ids) + 1e-8)
                    all_resp.append(txt)
                    all_lp.append(lp)

            # compute advantage per beam rollout
            for b in range(bs0):
                for k in range(beam_size):
                    start = (b * beam_size + k) * num_rollout
                    prev_v = prev_values[b][k]
                    for j in range(num_rollout):
                        all_adv.append(all_lp[start + j] - prev_v)

            new_steps = [["" for _ in range(beam_size)] for _ in range(bs0)]
            new_values = [[0.0 for _ in range(beam_size)] for _ in range(bs0)]
            new_weights = [[[] for _ in range(beam_size)] for _ in range(bs0)]

            for b in range(bs0):
                start = b * beam_size * num_rollout
                lp_slice  = np.array(all_lp[start:start + beam_size * num_rollout])
                adv_slice = np.array(all_adv[start:start + beam_size * num_rollout])
                resp_slice= all_resp[start:start + beam_size * num_rollout]

                mu, sigma = adv_slice.mean(), adv_slice.std()
                keep = [i for i,v in enumerate(adv_slice) if v > mu - sigma_rate * sigma]
                if len(keep) < beam_size:
                    wts = np.exp(adv_slice / temperature)
                    wts /= wts.sum()
                    extra = list(np.random.choice(
                        len(adv_slice), beam_size - len(keep), replace=False, p=wts
                    ))
                    keep += extra
                keep.sort()

                adv_k = adv_slice[keep]
                comb_w = softmax(adv_k / temperature)
                sel   = np.random.choice(len(keep), size=beam_size, replace=False, p=comb_w)

                for k_idx, sel_idx in enumerate(sel):
                    origin = keep[sel_idx] // num_rollout
                    resp = resp_slice[keep[sel_idx]]
                    raw_adv = float(adv_slice[keep[sel_idx]])
                    tok_ids = self.inference_engine.llm_engine.tokenizer.encode(
                        resp, add_special_tokens=False
                    )
                    #rep_adv = [raw_adv] * len(tok_ids)
                    rep_adv =  [0.0] * (len(tok_ids)-1) + [raw_adv] # last token is the response token

                    new_weights[b][k_idx] = weights_history[b][origin] + rep_adv
                    new_steps[b][k_idx]   = prev_steps[b][origin] + resp + "\n"
                    new_values[b][k_idx]  = lp_slice[keep[sel_idx]]

            prev_steps, prev_values, weights_history = new_steps, new_values, new_weights

        # 7. Final answer generation and collect raw_adv
        final_prompts, history_list, final_ws, final_raw_adv = [], [], [], []
        for b in range(bs0):
            vals = np.array(prev_values[b])
            adv  = vals - vals.mean()
            probs= np.exp(adv / temperature)
            probs/= probs.sum()
            choice = int(np.random.choice(len(probs), p=probs))

            history_list.append(prev_steps[b][choice])
            final_ws.append(weights_history[b][choice])
            final_raw_adv.append(float(adv[choice]))

            prompt_txt = (
                f"User: {raw_prompts[b].strip()}\n"
                f"Reasoning so far:\n{prev_steps[b][choice]}"
            )
            ids = self.tokenizer.encode(prompt_txt, add_special_tokens=False)
            final_prompts.append({"prompt_token_ids": ids})

        # 8. Generate final sequences
        final_sp = SamplingParams(
            max_tokens=response_len,
            logprobs=1,
            temperature=temperature,
            n=1,
            stop=["<end_of_reasoning>"]
        )
        final_outs = self.inference_engine.generate(
            prompts=final_prompts,
            sampling_params=final_sp,
            use_tqdm=False
        )

        # 9. Parse final outputs and pad with respective raw_adv
        full_texts, padded_ws = [], []
        for i, out in enumerate(final_outs):
            gen = out.outputs[0].text.strip()
            full= history_list[i] + gen
            full_texts.append(full)
            tok_ids = self.tokenizer.encode(gen, add_special_tokens=False)
            prob = final_raw_adv[i]
            segment_reward = [0.0] * (len(tok_ids) - 1) + [prob]
            #padded_ws.append(final_ws[i] + [final_raw_adv[i]] * len(tok_ids))
            padded_ws.append(final_ws[i] + segment_reward)

        full_ids = [self.tokenizer.encode(t, add_special_tokens=False) for t in full_texts]
        resp_pad= pad_2d_list_to_length(full_ids, self.pad_token_id, response_len).to(idx0.device)

        # —— 8. 构建 prm_reward 张量 ——
        # 使得 prm_reward 的每行长度与 resp_padded 的响应长度一致 (response_len)
        pr_tensors = []
        for r in padded_ws:
            # 截断或补齐到 response_len
            if len(r) >= response_len:
                row = r[:response_len]
            else:
                row = r + [0.0] * (response_len - len(r))
            pr_tensors.append(row)
        prm_reward = torch.tensor(pr_tensors, device=idx0.device)

        
        # ####neu reward r* use soft min
        # # 按公式算权重：w_i = exp(-r_i/T) / sum_j exp(-r_j/T)
        r = prm_reward
        T = 0.1 # 越小，较小的 reward 越重要 
        exp_neg = torch.exp(-r / T)           # [Bn, L]
        den = exp_neg.sum(dim=1, keepdim=True)  # [Bn, 1]
        w = exp_neg / den                       # [Bn, L]

        # 4) 最终 r*_i = w_i * r_i
        r_star = w * r                          # [Bn, L]

        # 5) 用 r_star 作为 prm_reward
        #prm_reward = r_star

        # 3) 反向累加得到 G_{i,t} = sum_{j=t}^{L-1} γ^{j-t} r*_{i,j}
        discounted = torch.zeros_like(r_star)          # [Bn, L]
        # 从最后一个位置开始
        discounted[:, -1] = r_star[:, -1]
        gamma= 1.0  # 折扣因子
        for t in range(L-2, -1, -1):
            discounted[:, t] = r_star[:, t] + gamma * discounted[:, t+1]


        # 10. Rebuild batch tensors
        Bn = resp_pad.size(0)
        repeat = Bn // bs0
        idx   = idx0.repeat_interleave(repeat, dim=0)
        mask  = prompts.batch["attention_mask"].repeat_interleave(repeat, dim=0)
        pos   = prompts.batch["position_ids"].repeat_interleave(repeat, dim=0)
        seq   = torch.cat([idx, resp_pad], dim=1)
        delta = torch.arange(1, response_len+1, device=pos.device).unsqueeze(0).expand(Bn, -1)
        last  = pos[:, -1:].expand(-1, response_len)
        pos   = torch.cat([pos, last + delta], dim=1)
        attn  = get_response_mask(resp_pad, prompts.meta_info["eos_token_id"], mask.dtype)
        mask  = torch.cat([mask, attn], dim=1)

        # 4) 只在“最后一个有效 token”上放 G_{i,0}
        #    假设当前 batch 的 attention mask 保存在 mask 里，和 resp_pad 对齐
        #    mask[b, t]==1 表示 resp_pad[b, t] 是有效 token
        Bn, Lr = resp_pad.shape
        # 计算每条序列的有效长度
        lengths   = mask.sum(dim=1).to(torch.long)   # [Bn]
        # 最后一个有效 token 的下标 = length-1
        last_idxs = lengths - 1                      # [Bn]

        # 构造新的 prm_reward，只在 last_idxs 上写入
        prm_reward = torch.zeros_like(r_star)        # [Bn, Lr]
        batch_idx  = torch.arange(Bn, device=r_star.device)
        # discounted[:, 0] 是从 t=0 开始的总折扣回报
        prm_reward[batch_idx, last_idxs] = discounted[:, 0]

        batch = TensorDict({
            "prompts":        idx,
            "responses":      resp_pad,
            "input_ids":      seq,
            "attention_mask": mask,
            "position_ids":   pos,
            "prm_reward":     prm_reward,
        }, batch_size=Bn)
        # Debugging information
        print(f"[DEBUG] resp_padded.shape: {resp_pad.shape}")####check resp_padded shape
        print(f"[DEBUG] prm_reward.shape: {prm_reward.shape}")####check prm_reward shape
        print(f"[DEBUG] prm_reward[0]: {prm_reward[0]}")####check prm_reward[0]
        # 11. Expand non_tensor_batch
        new_ntb = {}
        for k,v in non_tensor_batch.items():
            if isinstance(v, list):
                new_ntb[k] = list(itertools.chain.from_iterable([v] * repeat))
            elif isinstance(v, np.ndarray):
                new_ntb[k] = np.repeat(v, repeat, axis=0)
            elif torch.is_tensor(v):
                new_ntb[k] = v.repeat_interleave(repeat, dim=0)
            else:
                new_ntb[k] = [v] * Bn

        print(f"[DEBUG] resp_pad.shape={resp_pad.shape}, seq.shape={seq.shape}")

        return DataProto(batch=batch, non_tensor_batch=new_ntb)'''
    
    
    #sc 改进版
    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto: 
        # 0. Ensure non_tensor_batch exists and raw_prompt_ids are set
        non_tensor_batch = prompts.non_tensor_batch or {}
        prompts.non_tensor_batch = non_tensor_batch
        idx0 = prompts.batch["input_ids"]            # [bs, prompt_len]
        bs0 = idx0.size(0)
        raw_ids = non_tensor_batch.get("raw_prompt_ids")
        if raw_ids is None or len(raw_ids) != bs0:
            raw_ids = [_pre_process_inputs(self.pad_token_id, idx0[i]) for i in range(bs0)]
            non_tensor_batch["raw_prompt_ids"] = raw_ids

        # 1. Build vLLM inputs
        if "multi_modal_data" in non_tensor_batch:
            mm = non_tensor_batch.pop("multi_modal_data")
            vllm_inputs = [
                {"prompt_token_ids": r, "multi_modal_data": m}
                for r, m in zip(raw_ids, mm)
            ]
        else:
            vllm_inputs = [{"prompt_token_ids": r} for r in raw_ids]

        # 2. Hyperparameters
        beam_size         = int(kwargs.get("step_beam_size",       self.config.step_beam_size))
        num_rollout       = int(kwargs.get("num_rollout",          self.config.num_rollout))
        num_foresight     = int(kwargs.get("num_foresight",        self.config.num_foresight))
        sigma_rate        = float(kwargs.get("sigma_rate",         self.config.sigma_rate))
        temperature       = float(kwargs.get("temperature",        self.config.temperature))
        cluster_num       = int(kwargs.get("cluster_num",          self.config.cluster_num))
        step_response_len = int(kwargs.get("step_response_length", self.config.step_response_length))
        response_len      = int(kwargs.get("response_length",      self.config.response_length))
        mix_lambda        = float(kwargs.get("mix_lambda",         self.config.mix_lambda))

        # === LCF（只用于 prm_reward）参数 ===
        lcf_mode    = str(kwargs.get("lcf_mode", "mean"))   # "focal" | "sigmoid" | "mean"
        lcf_gamma   = float(kwargs.get("lcf_gamma", 1.5))
        lcf_q       = float(kwargs.get("lcf_q", 0.30))
        lcf_lambda  = float(kwargs.get("lcf_lambda", 0.30))
        lcf_fallback_zero = bool(kwargs.get("lcf_fallback_zero", True))

        # === SC 相关参数：取 top-k 候选来近似 H(U,p_t) ===
        sc_topk_req = int(kwargs.get("sc_topk", 20))   # 建议 20；部分 vLLM 版本上限为 20

        import numpy as np, itertools, torch, math

        # —— 取词表大小，用于 log(V) 与 KL(U||p) = SC - log(V) —— 
        V = int(getattr(self.inference_engine.llm_engine.tokenizer, "vocab_size",
                getattr(self.tokenizer, "vocab_size", 0)))
        assert V > 0, "vocab_size 未取到"
        LOGV = math.log(V)

        # —— 读取引擎允许的最大 logprobs 上限并夹紧（避免报错） ——
        def _max_logprobs_allowed(default=20):
            proc = getattr(self.inference_engine.llm_engine, "processor", None)
            return int(getattr(proc, "max_logprobs", default))
        sc_topk = max(1, min(sc_topk_req, _max_logprobs_allowed()))

        # 3. Decode prompts
        raw_prompts = self.tokenizer.batch_decode(idx0, skip_special_tokens=True)

        # 4. SamplingParams for intermediate rollout（SC 改动点：logprobs=sc_topk）
        base_sp = SamplingParams(
            max_tokens=step_response_len,
            logprobs=sc_topk,
            temperature=temperature,
            n=num_rollout,
            stop=["\n", "<end_of_reasoning>"]
        )

        # ========== SC 工具函数（核心） ==========
        # 把 vLLM 返回的一步 logprobs（top-k 候选）近似成 H(U,p_t)
        def _as_float_lp(x):
            if isinstance(x, (int, float, np.floating)): return float(x)
            lp = getattr(x, "logprob", None)
            if lp is not None: return float(lp)
            if isinstance(x, (list, tuple)) and len(x) >= 2:
                try: return float(x[1])
                except: pass
            return None

        def _sc_from_topk_entry(entry, V: int, eps: float = 1e-12) -> float:
            logs = []
            if isinstance(entry, dict):
                for obj in entry.values():
                    lp = _as_float_lp(obj)
                    if lp is not None: logs.append(lp)
            elif isinstance(entry, (list, tuple)):
                for obj in entry:
                    lp = _as_float_lp(obj)
                    if lp is not None: logs.append(lp)

            k = len(logs)
            if k == 0:
                return LOGV  # 无信息时返回下界 log V

            logs = np.asarray(logs, dtype=np.float64)
            m = float(logs.max())
            # 稳定求 s_top = sum p_top
            s_top = float(np.exp(logs - m).sum() * math.exp(m))
            s_top = min(1.0 - 1e-12, max(0.0, s_top))
            if V > k:
                tail_logp_each = math.log1p(-s_top) - math.log(V - k)  # log( (1-s_top)/(V-k) )
                sum_logp = float(logs.sum()) + (V - k) * tail_logp_each
            else:
                sum_logp = float(logs.sum())
            # H(U, p_t) = - (1/V) * sum_i log p_i
            return - sum_logp / float(V)

        def _avg_sc_for_output(o, V: int) -> float:
            """对一个输出段：逐 token 求 SC_t，再取段内均值。"""
            entries = getattr(o, "logprobs", None)
            if entries and len(entries) > 0:
                scs = [_sc_from_topk_entry(entries[t], V) for t in range(len(entries))]
                return float(np.mean(scs))
            # 兜底：只知道 chosen 的 logprob -> k=1 均匀尾近似
            chosen_lps = _extract_chosen_token_logprobs(o)
            if not chosen_lps:
                return LOGV
            scs = []
            for lp in chosen_lps:
                p_star = float(np.clip(np.exp(lp), 1e-12, 1.0 - 1e-12))
                tail_each = max(1e-12, (1.0 - p_star) / max(1, V - 1))
                sum_logp = math.log(p_star) + (V - 1) * math.log(tail_each)
                scs.append(- sum_logp / float(V))
            return float(np.mean(scs))

        def _sc_steps_for_output(o, V: int):
            """返回逐步 SC_t 列表；LCF 权重等需要 per-step 分数。"""
            entries = getattr(o, "logprobs", None)
            if entries and len(entries) > 0:
                return [_sc_from_topk_entry(entries[t], V) for t in range(len(entries))]
            # 兜底（k=1）
            steps = []
            chosen_lps = _extract_chosen_token_logprobs(o)
            for lp in chosen_lps:
                p_star = float(np.clip(np.exp(lp), 1e-12, 1.0 - 1e-12))
                tail_each = max(1e-12, (1.0 - p_star) / max(1, V - 1))
                sum_logp = math.log(p_star) + (V - 1) * math.log(tail_each)
                steps.append(- sum_logp / float(V))
            return steps

        # ========== 你原有的其他工具 ==========
        def _stable_softmax(x, T=1.0):
            x = np.asarray(x, dtype=np.float32)
            logits = x / max(1e-8, float(T))
            logits -= logits.max()
            p = np.exp(logits)
            s = p.sum()
            if not np.isfinite(s) or s <= 0:
                return np.full_like(p, 1.0 / len(p))
            return p / s

        def _extract_chosen_token_logprobs(one_output) -> list:
            # 与原实现一致（用于 k=1 兜底）
            import numpy as np
            token_ids = getattr(one_output, "token_ids", None) or []
            L = len(token_ids)
            if L == 0:
                return []
            tlp = getattr(one_output, "token_logprobs", None)
            if tlp is not None and len(tlp) == L:
                out = []
                for v in tlp:
                    try: out.append(float(v))
                    except: out.append(float("-10.0"))
                return out

            def _as_float_lp_inner(x):
                if isinstance(x, (int, float, np.floating)): return float(x)
                lp = getattr(x, "logprob", None)
                if lp is not None:
                    try: return float(lp)
                    except: pass
                if isinstance(x, (list, tuple)) and len(x) >= 2:
                    try: return float(x[1])
                    except: pass
                return None

            lps = getattr(one_output, "logprobs", None)
            if lps is not None and len(lps) == L:
                chosen = []
                for t in range(L):
                    entry = lps[t]
                    tok_id = token_ids[t]
                    lp_val = None
                    if isinstance(entry, dict):
                        obj = entry.get(tok_id, None)
                        if obj is not None:
                            lp_val = _as_float_lp_inner(obj)
                        if lp_val is None:
                            best_lp_val = None
                            for obj2 in entry.values():
                                cand_id = getattr(obj2, "token_id", None)
                                cand_lp = _as_float_lp_inner(obj2)
                                if cand_id == tok_id and cand_lp is not None:
                                    lp_val = cand_lp; break
                                if cand_lp is not None and (best_lp_val is None or cand_lp > best_lp_val):
                                    best_lp_val = cand_lp
                            if lp_val is None and best_lp_val is not None:
                                lp_val = best_lp_val
                    elif isinstance(entry, (list, tuple)):
                        best_lp = None
                        for obj in entry:
                            cand_id = getattr(obj, "token_id", None)
                            cand_lp = _as_float_lp_inner(obj)
                            if cand_id == tok_id and cand_lp is not None:
                                lp_val = cand_lp; break
                            if cand_lp is not None and (best_lp is None or cand_lp > best_lp):
                                best_lp = cand_lp
                        if lp_val is None and best_lp is not None:
                            lp_val = best_lp
                    if lp_val is None or not np.isfinite(lp_val):
                        lp_val = float(getattr(one_output, "cumulative_logprob", -10.0)) / (L + 1e-8)
                    chosen.append(float(lp_val))
                return chosen
            avg = float(getattr(one_output, "cumulative_logprob", -10.0)) / (L + 1e-8)
            return [avg] * L

        # === LCF：把“逐 token logprob”替换成“逐 token self-certainty（SC_t）”
        def _lcf_weighted_mean_from_scores(sc_steps: list) -> float:
            """对逐步 SC_t 做加权均值；SC_t >= log V。"""
            if not sc_steps:
                return 0.0
            x = np.asarray(sc_steps, dtype=np.float32)          # x = SC_t
            x_norm = np.maximum(0.0, x - float(LOGV))           # = KL(U||p_t) >= 0
            if lcf_mode in ("raw", "mean", "none"):
                return float(x.mean())
            if lcf_mode == "focal":
                # 单调权重：certainty 越强（KL 越大）权重越大，范围 [0,1)
                w = (1.0 - np.exp(-x_norm)) ** float(lcf_gamma)
            elif lcf_mode == "sigmoid":
                tau = float(np.quantile(x, lcf_q))
                w = 1.0 / (1.0 + np.exp((x - tau) / max(1e-6, float(lcf_lambda))))
            else:
                return float(x.mean())
            denom = float(w.sum())
            if denom <= 1e-12:
                return 0.0 if lcf_fallback_zero else float(x.mean())
            return float((w * x).sum() / denom)

        def _spread_segment_reward(total_reward: float, L: int, w: float = 1.0,
                                mode: str = "exp", gamma: float = 0.95) -> list:
            if L <= 0: return []
            R = float(total_reward) * max(float(w), 0.0)
            if mode == "last":
                return [0.0] * (L - 1) + [float(R)]
            if mode == "uniform":
                v = R / L; return [float(v)] * L
            if mode == "linear":
                coeffs = np.arange(1, L + 1, dtype=np.float32)
            else:
                coeffs = (gamma ** np.arange(L - 1, -1, -1, dtype=np.float32))
            s = float(coeffs.sum()); 
            if s <= 1e-12: return [0.0] * L
            coeffs = coeffs / s
            return list((R * coeffs).astype(np.float32))

        # 5. Initialize beam histories
        prev_steps      = [["" for _ in range(beam_size)] for _ in range(bs0)]
        prev_values     = [[0.0 for _ in range(beam_size)] for _ in range(bs0)]  # 现在表示“上一轮的段均值 SC”
        weights_history = [[[] for _ in range(beam_size)] for _ in range(bs0)]
        prev_gains      = [[0.0 for _ in range(beam_size)] for _ in range(bs0)]
        gain_history    = [[[] for _ in range(beam_size)] for _ in range(bs0)]

        # --- debug 控制 ---
        debug_prune = bool(kwargs.get("debug_prune", True))
        debug_prune_every = int(kwargs.get("debug_prune_every", 1))

        # 6. Multi-step foresight
        for depth in range(num_foresight):
            step_inputs = []
            for b in range(bs0):
                for k in range(beam_size):
                    prefix = (
                        f"User: {raw_prompts[b].strip()}\n"
                        f"Reasoning so far:\n{prev_steps[b][k]}"
                    )
                    ids = self.inference_engine.llm_engine.tokenizer.encode(
                        prefix, add_special_tokens=False
                    )
                    step_inputs.append({"prompt_token_ids": ids})

            outs = self.inference_engine.generate(
                prompts=step_inputs,
                sampling_params=base_sp,
                use_tqdm=False
            )

            # all_resp, all_sc, all_rhat = [], [], []
            # for out in outs:
            #     for o in out.outputs:
            #         txt = o.text.strip()
            #         sc  = _avg_sc_for_output(o, V)                # 段均值 SC
            #         sc_steps = _sc_steps_for_output(o, V)         # 逐步 SC_t
            #         r_hat   = _lcf_weighted_mean_from_scores(sc_steps)  # 只用于 prm_reward

            #         all_resp.append(txt)
            #         all_sc.append(sc)
            #         all_rhat.append(r_hat)
            all_resp, all_sc, all_rhat, all_sc_steps = [], [], [], []  # ← 新增 all_sc_steps
            for out in outs:
                for o in out.outputs:
                    txt = o.text.strip()
                    sc  = _avg_sc_for_output(o, V)
                    sc_steps = _sc_steps_for_output(o, V)               # ← 逐 token SC_t
                    r_hat   = _lcf_weighted_mean_from_scores(sc_steps)

                    all_resp.append(txt)
                    all_sc.append(sc)
                    all_rhat.append(r_hat)
                    all_sc_steps.append(sc_steps)                       # ← 存起来

            # === 计算 advantage；首轮用“组内均值”居中，其后用上一轮 prev_values ===
            all_adv = []
            group_baseline = {}
            for b in range(bs0):
                for k in range(beam_size):
                    start = (b * beam_size + k) * num_rollout
                    end   = start + num_rollout
                    if depth == 0:
                        baseline = float(np.mean(all_sc[start:end]))
                    else:
                        baseline = float(prev_values[b][k])
                    group_baseline[(b, k)] = baseline
                    for j in range(num_rollout):
                        all_adv.append(all_sc[start + j] - baseline)

            new_steps   = [["" for _ in range(beam_size)] for _ in range(bs0)]
            new_values  = [[0.0 for _ in range(beam_size)] for _ in range(bs0)]
            new_weights = [[[] for _ in range(beam_size)] for _ in range(bs0)]
            new_gains   = [[0.0 for _ in range(beam_size)] for _ in range(bs0)]
            new_gain_history  = [[[] for _ in range(beam_size)] for _ in range(bs0)]

            # === Debug 累计器 ===
            kept_pre_cnt = kept_post_cnt = total_cnt = supplemented_cnt = 0

            for b in range(bs0):
                start = b * beam_size * num_rollout
                end   = start + beam_size * num_rollout
                sc_slice   = np.array(all_sc[start:end],    dtype=np.float32)
                adv_slice  = np.array(all_adv[start:end],   dtype=np.float32)
                rhat_slice = np.array(all_rhat[start:end],  dtype=np.float32)
                resp_slice = all_resp[start:end]

                # low-sigma width pruning（用 sc_slice）
                mu, sigma = float(sc_slice.mean()), float(sc_slice.std())
                keep = [i for i, v in enumerate(sc_slice) if v > mu - sigma_rate * sigma]
                N = len(sc_slice)
                total_cnt += N
                kept_pre_cnt += len(keep)

                if len(keep) < beam_size:
                    pool = np.setdiff1d(np.arange(len(sc_slice)), np.array(keep), assume_unique=False)
                    if len(pool) > 0:
                        p_pool = _stable_softmax(sc_slice[pool], temperature)
                        extra = np.random.choice(pool, beam_size - len(keep), replace=False, p=p_pool).tolist()
                        keep += extra
                        supplemented_cnt += len(extra)
                kept_post_cnt += len(keep)

                # 取分数（保持原来的混合 zscore 逻辑）
                abs_k = sc_slice[keep]       # 段均值 SC
                adv_k = adv_slice[keep]      # 动态优势（SC 增量）

                def zscore(x, eps=1e-8):
                    return (x - x.mean()) / (x.std() + eps)

                z_abs  = zscore(abs_k)
                z_gain = zscore(adv_k)
                tau_abs  = float(kwargs.get("tau_abs",  temperature))
                tau_gain = float(kwargs.get("tau_gain", temperature))
                lam = float(kwargs.get("mix_lambda", 0.6))
                logits = lam * (z_abs / tau_abs) + (1 - lam) * (z_gain / tau_gain)
                combined = _stable_softmax(logits, temperature)
                sel = np.random.choice(len(keep), size=beam_size, replace=False, p=combined)

                # 更新
                for k_idx, sel_idx in enumerate(sel):
                    origin = keep[sel_idx] // num_rollout
                    resp   = resp_slice[keep[sel_idx]]

                    tok_ids = self.inference_engine.llm_engine.tokenizer.encode(
                        resp, add_special_tokens=False
                    )

                    # === prm_reward：用 LCF 的 SC 加权均值作为段奖末脉冲；如需分摊可用 _spread_segment_reward
                    # step_reward = float(rhat_slice[keep[sel_idx]])
                    # seg = [0.0] * (len(tok_ids) - 1) + [step_reward]
                    # === prm_reward：逐 token 奖励 = 对应 token 的 SC_t
                    steps_slice = all_sc_steps[start:end]                 # 和 resp_slice / sc_slice 对齐
                    seg = steps_slice[keep[sel_idx]]                      # seg: List[SC_t]，长度 = 该段 token 数

                    new_weights[b][k_idx] = weights_history[b][origin] + seg
                    new_steps[b][k_idx]   = prev_steps[b][origin] + resp + "\n"
                    new_values[b][k_idx]  = float(sc_slice[keep[sel_idx]])    # 用 SC 更新基线
                    new_gains[b][k_idx]   = float(adv_slice[keep[sel_idx]])   # 保存 SC 的增益
                    new_gain_history[b][k_idx] = gain_history[b][origin] + [float(adv_slice[keep[sel_idx]])]

            prev_steps, prev_values, weights_history = new_steps, new_values, new_weights
            prev_gains = new_gains
            gain_history = new_gain_history

            if debug_prune and (depth % max(1, debug_prune_every) == 0):
                ratio_pre  = (kept_pre_cnt  / total_cnt) if total_cnt > 0 else 0.0
                ratio_post = (kept_post_cnt / total_cnt) if total_cnt > 0 else 0.0
                print(f"[PRUNE-DEBUG] depth={depth} "
                    f"keep_pre={kept_pre_cnt}/{total_cnt} ({ratio_pre:.2%}), "
                    f"keep_post={kept_post_cnt}/{total_cnt} ({ratio_post:.2%}), "
                    f"supplemented={supplemented_cnt}, "
                    f"beam_size={beam_size}, num_rollout={num_rollout})")

        # 7. Final answer generation — 用 prev_values 的 abs+gain 计算 combined 采样 beam
        final_prompts, history_list, final_ws, final_probs = [], [], [], []
        final_gain_hists, seq_gain_list = [], [] 
        for b in range(bs0):
            L_abs = np.array(prev_values[b], dtype=np.float32)  # 段均值 SC
            G_inc = np.array(prev_gains[b],  dtype=np.float32)  # SC 增量

            def zscore(x, eps=1e-8):
                s = x.std()
                return (x - x.mean()) / (s + eps)

            zL = zscore(L_abs)
            zG = zscore(G_inc)  

            tau_abs  = float(kwargs.get("tau_abs",  temperature))
            tau_gain = float(kwargs.get("tau_gain", temperature))
            lam      = float(kwargs.get("mix_lambda", 0.6))

            logits = lam * (zL / tau_abs) + (1.0 - lam) * (zG / tau_gain)
            combined_beam = _stable_softmax(logits, temperature)

            choice   = int(np.random.choice(len(combined_beam), p=combined_beam))
            p_choice = float(combined_beam[choice])

            history_list.append(prev_steps[b][choice])
            final_ws.append(weights_history[b][choice])
            final_probs.append(p_choice)
            final_gain_hists.append(gain_history[b][choice])

            prompt_txt = (
                f"User: {raw_prompts[b].strip()}\n"
                f"Reasoning so far:\n{prev_steps[b][choice]}"
            )
            ids = self.tokenizer.encode(prompt_txt, add_special_tokens=False)
            final_prompts.append({"prompt_token_ids": ids})

        # 8. Generate final sequences（SC 改动点：logprobs=sc_topk）
        final_sp = SamplingParams(
            max_tokens=response_len,
            logprobs=sc_topk,
            temperature=temperature,
            n=1,
            stop=["<end_of_reasoning>"]
        )
        final_outs = self.inference_engine.generate(
            prompts=final_prompts,
            sampling_params=final_sp,
            use_tqdm=False
        )

        # 9. Parse final outputs；在最终段落末写入 LCF（基于 SC）
        def _seq_gain_pos_neg(gh, w_pos=1.0, w_neg=1.0, var_coef=0.10, dd_coef=0.15, second_diff_coef=0.0):
            if len(gh) == 0: return 0.0
            gh = np.asarray(gh, dtype=np.float32)
            pos = float(np.sum(np.clip(gh,  0.0, None)))
            neg = float(np.sum(np.clip(-gh, 0.0, None)))
            var_pen = float(np.var(gh)) if gh.size > 1 else 0.0
            phi = np.cumsum(gh)
            peak = np.maximum.accumulate(phi)
            drawdown = np.maximum(0.0, peak - phi)
            dd_area = float(np.sum(drawdown))
            if second_diff_coef > 0.0 and gh.size > 2:
                second_diff_var = float(np.var(np.diff(gh, n=2)))
            else:
                second_diff_var = 0.0
            return (w_pos * pos - w_neg * neg - var_coef * var_pen - dd_coef * dd_area - second_diff_coef * second_diff_var)

        full_texts, padded_ws = [], []
        for i, out in enumerate(final_outs):
            o = out.outputs[0]
            gen = o.text.strip()
            full = history_list[i] + gen
            full_texts.append(full)
            tok_ids = self.tokenizer.encode(gen, add_special_tokens=False)

            sc_final  = _avg_sc_for_output(o, V)
            baseline  = float(prev_values[i // 1][0]) if False else 0.0  # 不用：最终段增益已在 lookahead 里体现
            # 如果你需要与上一步的一致 baseline，这里也可传入相应的 final_baseline。当前版本直接用段内 LCF。

            # === LCF 段奖（最终段）：
            # sc_steps_final = _sc_steps_for_output(o, V)
            # r_hat_final = _lcf_weighted_mean_from_scores(sc_steps_final)
            # seg = [0.0] * (len(tok_ids) - 1) + [r_hat_final]
            sc_steps_final = _sc_steps_for_output(o, V)
            seg = sc_steps_final              
            padded_ws.append(final_ws[i] + seg)

            gh = final_gain_hists[i]
            sg = _seq_gain_pos_neg(
                gh,
                w_pos=1.0,
                w_neg=1.0,
                var_coef=0.10,
                dd_coef=0.15,
                second_diff_coef=0.0
            ) if len(gh) > 0 else 0.0
            seq_gain_list = []  # 若后续需要，此处可像你原版那样汇总；这里略。

        full_ids = [self.tokenizer.encode(t, add_special_tokens=False) for t in full_texts]
        resp_pad = pad_2d_list_to_length(full_ids, self.pad_token_id, response_len).to(idx0.device)

        # 10. Build prm_reward（仅此采用 LCF 段奖；其他逻辑不变）
        pr_tensors = []
        for r in padded_ws:
            if len(r) >= response_len:
                row = r[:response_len]
            else:
                row = r + [0.0] * (response_len - len(r))
            pr_tensors.append(row)

        prm_reward = torch.tensor(pr_tensors, device=idx0.device, dtype=torch.float32)
        # pr_tensors = []
        # global_sc_means = []

        # for r in padded_ws:                     # r: List[SC_t]，长度 = 该样本真实生成 token 数 L
        #     L = len(r)
        #     mu = float(np.mean(r)) if L > 0 else 0.0   # ← 全局均值
        #     global_sc_means.append(mu)

        #     # 每个真实 token 填 mu，padding 仍补 0
        #     if L >= response_len:
        #         row = [mu] * response_len
        #     else:
        #         row = [mu] * L + [0.0] * (response_len - L)
        #     pr_tensors.append(row)

        # prm_reward = torch.tensor(pr_tensors, device=idx0.device, dtype=torch.float32)
        # 如果你需要 seq_gain 张量，按需填充；此处示例置零：
        seq_gain = torch.zeros(resp_pad.size(0), device=idx0.device, dtype=torch.float32)

        # 11. Rebuild batch tensors
        Bn = resp_pad.size(0)
        repeat = Bn // bs0
        idx   = idx0.repeat_interleave(repeat, dim=0)
        mask  = prompts.batch["attention_mask"].repeat_interleave(repeat, dim=0)
        pos   = prompts.batch["position_ids"].repeat_interleave(repeat, dim=0)
        seq   = torch.cat([idx, resp_pad], dim=1)
        delta = torch.arange(1, response_len+1, device=pos.device).unsqueeze(0).expand(Bn, -1)
        last  = pos[:, -1:].expand(-1, response_len)
        pos   = torch.cat([pos, last + delta], dim=1)
        attn  = get_response_mask(resp_pad, prompts.meta_info["eos_token_id"], mask.dtype)
        mask  = torch.cat([mask, attn], dim=1)

        batch = TensorDict({
            "prompts":        idx,
            "responses":      resp_pad,
            "input_ids":      seq,
            "attention_mask": mask,
            "position_ids":   pos,
            "prm_reward":     prm_reward,   # 现为基于 SC 的 LCF 段奖
            "seq_gain":       seq_gain,
        }, batch_size=Bn)

        # 12. Expand non_tensor_batch
        new_ntb = {}
        for k, v in non_tensor_batch.items():
            if isinstance(v, list):
                new_ntb[k] = list(itertools.chain.from_iterable([v] * repeat))
            elif isinstance(v, np.ndarray):
                new_ntb[k] = np.repeat(v, repeat, axis=0)
            elif torch.is_tensor(v):
                new_ntb[k] = v.repeat_interleave(repeat, dim=0)
            else:
                new_ntb[k] = [v] * Bn

        print(f"[DEBUG] resp_pad.shape={resp_pad.shape}, seq.shape={seq.shape}, prm_reward.shape={prm_reward.shape}")
        return DataProto(batch=batch, non_tensor_batch=new_ntb)


    '''def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto: 
        # 0. Ensure non_tensor_batch exists and raw_prompt_ids are set
        non_tensor_batch = prompts.non_tensor_batch or {}
        prompts.non_tensor_batch = non_tensor_batch
        idx0 = prompts.batch["input_ids"]            # [bs, prompt_len]
        bs0 = idx0.size(0)
        raw_ids = non_tensor_batch.get("raw_prompt_ids")
        if raw_ids is None or len(raw_ids) != bs0:
            raw_ids = [_pre_process_inputs(self.pad_token_id, idx0[i]) for i in range(bs0)]
            non_tensor_batch["raw_prompt_ids"] = raw_ids

        # 1. Build vLLM inputs
        if "multi_modal_data" in non_tensor_batch:
            mm = non_tensor_batch.pop("multi_modal_data")
            vllm_inputs = [
                {"prompt_token_ids": r, "multi_modal_data": m}
                for r, m in zip(raw_ids, mm)
            ]
        else:
            vllm_inputs = [{"prompt_token_ids": r} for r in raw_ids]

        # 2. Hyperparameters
        beam_size         = int(kwargs.get("step_beam_size",       self.config.step_beam_size))
        num_rollout       = int(kwargs.get("num_rollout",          self.config.num_rollout))
        num_foresight     = int(kwargs.get("num_foresight",        self.config.num_foresight))
        sigma_rate        = float(kwargs.get("sigma_rate",         self.config.sigma_rate))
        temperature       = float(kwargs.get("temperature",        self.config.temperature))
        cluster_num       = int(kwargs.get("cluster_num",          self.config.cluster_num))
        step_response_len = int(kwargs.get("step_response_length", self.config.step_response_length))
        response_len      = int(kwargs.get("response_length",      self.config.response_length))
        mix_lambda        = float(kwargs.get("mix_lambda",         self.config.mix_lambda))

        # === PRM 使用 gain 的开关与分配策略 ===
        prm_mode = str(kwargs.get("prm_mode", "gain"))  # "gain" | "lcf"
        prm_gain_scale = float(kwargs.get("prm_gain_scale", 1.0))

        reward_spread_mode  = str(kwargs.get("reward_spread", "last"))   # "exp" | "uniform" | "linear" | last
        reward_spread_gamma = float(kwargs.get("spread_gamma", 0.95))

        # reward_spread_mode   = str(kwargs.get("reward_spread", "exp"))      # "exp" | "uniform" | "linear"
        # reward_spread_gamma  = float(kwargs.get("spread_gamma", 0.95))      # 仅对 exp 生效
        # final_spread_mode    = str(kwargs.get("final_reward_spread", reward_spread_mode))
        # final_spread_gamma   = float(kwargs.get("final_spread_gamma", reward_spread_gamma))

        # === LCF（只用于 prm_reward）参数 ===
        lcf_mode    = str(kwargs.get("lcf_mode", "mean"))   # "focal" | "sigmoid" | "mean" = 保留原始均值
        lcf_gamma   = float(kwargs.get("lcf_gamma", 1.5))
        lcf_q       = float(kwargs.get("lcf_q", 0.30))       # 分位数（建议训练中退火到 0.15）
        lcf_lambda  = float(kwargs.get("lcf_lambda", 0.30))  # sigmoid 温度
        lcf_fallback_zero = bool(kwargs.get("lcf_fallback_zero", True))  # 无低置信时r_hat置0

        sc_topk = int(kwargs.get("sc_topk", 20))   # 你可以调 20/50/100 看权衡


        # 3. Decode prompts
        raw_prompts = self.tokenizer.batch_decode(idx0, skip_special_tokens=True)

        # 4. SamplingParams for intermediate rollout
        base_sp = SamplingParams(
            max_tokens=step_response_len,
            logprobs=sc_topk,
            temperature=temperature,
            n=num_rollout,
            stop=["\n", "<end_of_reasoning>"]
        )
        

        # === 新增：二阶段“补全”用的采样参数（看一眼再决定用）===0831
        # 若未传入，默认与 step_response_len 相同；你也可以改成 step_response_len // 2 以省算力
        step_completion_len = int(kwargs.get("step_completion_length", step_response_len//8))
        comp_sp = SamplingParams(
            max_tokens=step_completion_len,
            logprobs=sc_topk,
            temperature=temperature,
            n=1,
            stop=["<end_of_reasoning>"]
            )

        V = int(getattr(self.inference_engine.llm_engine.tokenizer, "vocab_size",
                getattr(self.tokenizer, "vocab_size", 0)))
        assert V > 0, "vocab_size 未取到"

        import numpy as np
        import itertools
        import torch

        import math, numpy as np

        def _as_float_lp(x):
            if isinstance(x, (int, float, np.floating)): return float(x)
            lp = getattr(x, "logprob", None)
            if lp is not None: return float(lp)
            if isinstance(x, (list, tuple)) and len(x) >= 2:
                try: return float(x[1])
                except: pass
            return None

        # def _sc_from_topk_entry(entry, V: int, eps: float = 1e-8) -> float:
        #     """
        #     entry: 一步的 logprobs 返回（可能是 dict[token_id->obj] 或 list[obj]）
        #     V: 词表大小
        #     近似 SC_t = - (1/K) * sum_i log p_i：
        #     已知 top-k 的 log p_i，尾部 (V-k) 假设均匀：p_tail_each = tail/(V-k)
        #     """
        #     # 取出这一步返回的若干 log p_i
        #     logs = []
        #     if isinstance(entry, dict):
        #         for obj in entry.values():
        #             lp = _as_float_lp(obj)
        #             if lp is not None: logs.append(lp)
        #     elif isinstance(entry, (list, tuple)):
        #         for obj in entry:
        #             lp = _as_float_lp(obj)
        #             if lp is not None: logs.append(lp)

        #     k = len(logs)
        #     if k == 0:
        #         # 没有任何候选：返回均匀的最小值 log K
        #         return math.log(V)

        #     probs = np.clip(np.exp(np.asarray(logs, dtype=np.float64)), eps, 1.0)
        #     s = float(probs.sum())
        #     tail = max(0.0, 1.0 - s)

        #     if V > k:
        #         tail_logp_each = math.log(max(eps, tail / (V - k)))
        #         sum_logp = float(np.log(probs).sum()) + (V - k) * tail_logp_each
        #     else:
        #         # 罕见：top-k 覆盖了全部词（k==V）
        #         sum_logp = float(np.log(probs).sum())

        #     sc_t = - sum_logp / float(V)            # H(U,p) 的一步值
        #     return sc_t
        def _sc_from_topk_entry(entry, V: int, eps: float = 1e-12) -> float:
            logs = []
            if isinstance(entry, dict):
                for obj in entry.values():
                    lp = _as_float_lp(obj); 
                    if lp is not None: logs.append(lp)
            else:
                for obj in entry:
                    lp = _as_float_lp(obj)
                    if lp is not None: logs.append(lp)

            k = len(logs)
            if k == 0:
                return math.log(V)

            logs = np.asarray(logs, dtype=np.float64)
            m = float(logs.max())
            s_top = float(np.exp(logs - m).sum() * math.exp(m))
            s_top = min(1.0 - 1e-12, max(0.0, s_top))

            if V > k:
                tail_logp_each = math.log1p(-s_top) - math.log(V - k)
                sum_logp = float(logs.sum()) + (V - k) * tail_logp_each
            else:
                sum_logp = float(logs.sum())

            return - sum_logp / float(V)

        def _avg_sc_for_output(o, V: int) -> float:
            """
            对单个 vLLM Output：返回“该段生成的平均 SC”（逐 token 求均值）。
            若 o.logprobs 不可用，退化为只用已选 token 的 logprob 做 k=1 的均匀尾近似。
            """
            entries = getattr(o, "logprobs", None)
            if entries and len(entries) > 0:
                scs = [_sc_from_topk_entry(entries[t], V) for t in range(len(entries))]
                return float(np.mean(scs))

            # 兜底：只知道 chosen 的 logprob 列表
            chosen_lps = _extract_chosen_token_logprobs(o)  # 你已有这个函数
            if not chosen_lps: 
                return math.log(V)  # 最小信息时返回下界
            scs = []
            for lp in chosen_lps:
                p = float(np.clip(np.exp(lp), 1e-8, 1.0 - 1e-8))
                # k=1 的特例：sum log p ≈ log p* + (V-1) * log( (1-p*)/(V-1) )
                tail_each = max(1e-8, (1.0 - p) / max(1, V - 1))
                sum_logp = math.log(p) + (V - 1) * math.log(tail_each)
                scs.append(- sum_logp / float(V))
            return float(np.mean(scs))

        def _stable_softmax(x, T=1.0):
            x = np.asarray(x, dtype=np.float32)
            logits = x / max(1e-8, float(T))
            logits -= logits.max()
            p = np.exp(logits)
            s = p.sum()
            if not np.isfinite(s) or s <= 0:
                return np.full_like(p, 1.0 / len(p))
            return p / s

        def _extract_chosen_token_logprobs(one_output) -> list:
            """
            从 vLLM 的单个 output 中提取“已选 token”的逐步 logprob。
            兼容：
            - one_output.token_logprobs: List[float]
            - one_output.logprobs: List[Dict[token_id -> Logprob or float]]
            - one_output.logprobs: List[List[Logprob-like or (id, logprob) 元组]]
            若无法可靠提取，则用 cumulative_logprob / T 兜底。
            """
            import numpy as np

            token_ids = getattr(one_output, "token_ids", None) or []
            L = len(token_ids)
            if L == 0:
                return []

            # 1) 最简单：直接有逐步 logprob 列表
            tlp = getattr(one_output, "token_logprobs", None)
            if tlp is not None and len(tlp) == L:
                # 确保可转为 float
                out = []
                for v in tlp:
                    try:
                        out.append(float(v))
                    except Exception:
                        out.append(float("-10.0"))
                return out

            # 工具：把各种对象取成 float logprob
            def _as_float_lp(x):
                # 直接数值
                if isinstance(x, (int, float, np.floating)):
                    return float(x)
                # vLLM 的 Logprob 类：有 .logprob 字段
                lp = getattr(x, "logprob", None)
                if lp is not None:
                    try:
                        return float(lp)
                    except Exception:
                        pass
                # 可能是 (token_id, logprob) 或类似二元组
                if isinstance(x, (list, tuple)) and len(x) >= 2:
                    try:
                        return float(x[1])
                    except Exception:
                        pass
                return None

            # 2) 通用：逐步候选的结构
            lps = getattr(one_output, "logprobs", None)
            if lps is not None and len(lps) == L:
                chosen = []
                for t in range(L):
                    entry = lps[t]
                    tok_id = token_ids[t]
                    lp_val = None

                    if isinstance(entry, dict):
                        obj = entry.get(tok_id, None)
                        if obj is not None:
                            lp_val = _as_float_lp(obj)
                        # 若没找到精确 token_id，遍历 values，优先匹配 token_id，其次取最大 logprob
                        if lp_val is None:
                            best_lp = None
                            best_lp_val = None
                            for obj2 in entry.values():
                                # 若对象带 token_id 且匹配，直接用
                                cand_id = getattr(obj2, "token_id", None)
                                cand_lp = _as_float_lp(obj2)
                                if cand_id == tok_id and cand_lp is not None:
                                    lp_val = cand_lp
                                    break
                                # 否则保留一个最大 cand_lp 作为兜底
                                if cand_lp is not None and (best_lp is None or cand_lp > best_lp):
                                    best_lp = cand_lp
                                    best_lp_val = cand_lp
                            if lp_val is None and best_lp_val is not None:
                                lp_val = best_lp_val

                    elif isinstance(entry, (list, tuple)):
                        # 候选列表：找 token_id 匹配的，否则取最大 logprob
                        best_lp = None
                        for obj in entry:
                            cand_id = getattr(obj, "token_id", None)
                            cand_lp = _as_float_lp(obj)
                            if cand_id == tok_id and cand_lp is not None:
                                lp_val = cand_lp
                                break
                            if cand_lp is not None and (best_lp is None or cand_lp > best_lp):
                                best_lp = cand_lp
                        if lp_val is None and best_lp is not None:
                            lp_val = best_lp

                    # 最后兜底：用平均 logprob
                    if lp_val is None or not np.isfinite(lp_val):
                        lp_val = float(getattr(one_output, "cumulative_logprob", -10.0)) / (L + 1e-8)

                    chosen.append(float(lp_val))
                return chosen

            # 3) 最差兜底：均匀平均
            avg = float(getattr(one_output, "cumulative_logprob", -10.0)) / (L + 1e-8)
            return [avg] * L


        # === LCF：基于逐token logprob 计算 r_hat（只用于 prm_reward）===
        def _lcf_weighted_mean_from_logprobs(lp_list: list) -> float:
            """
            计算 LCF 加权均值（用于 prm_reward）。新增 lcf_mode:
            - "raw" / "plain" / "none": 不做软掩码，直接返回未加权的 lp 均值
            - "focal":    w = (1 - p)**gamma
            - "sigmoid":  w = sigmoid((tau - lp)/lambda)

            返回:
            float 标量（与旧版一致）
            """
            if not lp_list:
                return 0.0

            lp = np.asarray(lp_list, dtype=np.float32)

            # --- 新增：保留原始 lp 均值 ---
            if lcf_mode in ("raw", "mean", "none"):
                return float(lp.mean())

            # --- 原有两种软掩码 ---
            if lcf_mode == "focal":
                p = np.clip(np.exp(lp), 1e-6, 1.0 - 1e-6)
                w = (1.0 - p) ** float(lcf_gamma)
            elif lcf_mode == "sigmoid":
                tau = float(np.quantile(lp, lcf_q))
                w = 1.0 / (1.0 + np.exp((lp - tau) / max(1e-6, float(lcf_lambda))))
            else:
                # 未知模式时退化为原始均值（更安全）
                return float(lp.mean())

            denom = float(w.sum())
            if denom <= 1e-12:
                # 没有有效权重时：按你的开关决定返回 0 或原始均值
                return 0.0 if lcf_fallback_zero else float(lp.mean())
            return float((w * lp).sum() / denom)

        def _spread_segment_reward(total_reward: float,
                                L: int,
                                w: float = 1.0,
                                mode: str = "exp",        # "exp" | "uniform" | "linear" | "last"
                                gamma: float = 0.95) -> list:
            """
            把整段的奖励 total_reward * w 按照指定模式分配到段内 L 个 token 上。
            返回长度为 L 的 list，且 sum(seg) == total_reward * w。
            """
            if L <= 0:
                return []
            R = float(total_reward) * max(float(w), 0.0)
            if mode == "last":
                # 原始版本：不衰减、不摊分，全部堆到最后一个 token
                return [0.0] * (L - 1) + [float(R)]

            if mode == "uniform":
                v = R / L
                return [float(v)] * L

            if mode == "linear":
                coeffs = np.arange(1, L + 1, dtype=np.float32)  # 1,2,...,L
            else:
                coeffs = (gamma ** np.arange(L - 1, -1, -1, dtype=np.float32))  # L-1,...,0

            s = float(coeffs.sum())
            if s <= 1e-12:
                return [0.0] * L
            coeffs = coeffs / s
            return list((R * coeffs).astype(np.float32))

        # 5. Initialize beam histories
        prev_steps      = [["" for _ in range(beam_size)] for _ in range(bs0)]
        prev_values     = [[0.0 for _ in range(beam_size)] for _ in range(bs0)]  # 平均logprob基线（保持原样）
        weights_history = [[[] for _ in range(beam_size)] for _ in range(bs0)]   # 存每段最后token的“奖励脉冲”
        prev_gains      = [[0.0 for _ in range(beam_size)] for _ in range(bs0)]
        gain_history    = [[[] for _ in range(beam_size)] for _ in range(bs0)]  # 记录每条beam的step-gain轨迹

        # --- 放在 num_foresight 外层超参里（可选开关与打印频率） ---
        debug_prune = bool(kwargs.get("debug_prune", True))
        debug_prune_every = int(kwargs.get("debug_prune_every", 1))  # 每多少个 depth 打一次

        # 6. Multi-step foresight
        for depth in range(num_foresight):
            step_inputs = []
            for b in range(bs0):
                for k in range(beam_size):
                    prefix = (
                        f"User: {raw_prompts[b].strip()}\n"
                        f"Reasoning so far:\n{prev_steps[b][k]}"
                    )
                    ids = self.inference_engine.llm_engine.tokenizer.encode(
                        prefix, add_special_tokens=False
                    )
                    step_inputs.append({"prompt_token_ids": ids})

            outs = self.inference_engine.generate(
                prompts=step_inputs,
                sampling_params=base_sp,
                use_tqdm=False
            )

            all_resp, all_lp, all_rhat = [], [], []  # === 新增 all_rhat：仅用于 prm_reward ===
            for out in outs:
                for o in out.outputs:
                    txt = o.text.strip()
                    # lp  = o.cumulative_logprob / (len(o.token_ids) + 1e-8)  # 平均logprob（用于搜索/剪枝/adv）
                    sc  = _avg_sc_for_output(o, V)          # 平均 self-certainty（越大越“确定”）
                    lp = sc  # 用 avg_sc 替代 avg_logprob 作为搜索/剪枝/adv 的基线 0905
                    # === 仅用于 prm_reward 的 r_hat ===
                    lp_list = _extract_chosen_token_logprobs(o)
                    r_hat   = _lcf_weighted_mean_from_logprobs(lp_list)

                    all_resp.append(txt)
                    all_lp.append(lp)
                    all_rhat.append(r_hat)

            # compute advantage per beam rollout（仍基于 lp 与 prev_values）
            all_adv = []
            group_baseline = {}  # 记录每个(b,k)在当前depth的基线
            for b in range(bs0):
                for k in range(beam_size):
                    start = (b * beam_size + k) * num_rollout
                    end   = start + num_rollout
                    # 旧：prev_v = prev_values[b][k]
                    if depth == 0:
                        mu0 = float(np.mean(all_lp[start:end]))   # 组内均值（SC的均值）
                        baseline = mu0
                    else:
                        baseline = float(prev_values[b][k])
                    group_baseline[(b, k)] = baseline
                    for j in range(num_rollout):
                        all_adv.append(all_lp[start + j] - baseline)
                    # prev_v = prev_values[b][k]
                    # for j in range(num_rollout):
                    #     all_adv.append(all_lp[start + j] - prev_v)  # 动态增益（不改）

            new_steps   = [["" for _ in range(beam_size)] for _ in range(bs0)]
            new_values  = [[0.0 for _ in range(beam_size)] for _ in range(bs0)]
            new_weights = [[[] for _ in range(beam_size)] for _ in range(bs0)]
            new_gains   = [[0.0 for _ in range(beam_size)] for _ in range(bs0)]
            new_gain_history  = [[[] for _ in range(beam_size)] for _ in range(bs0)]

            # === Debug 累计器（本 depth 内聚合） ===
            kept_pre_cnt = 0      # 剪枝前保留数量（按阈）
            kept_post_cnt = 0     # 补足后保留数量
            total_cnt = 0         # 总候选数
            supplemented_cnt = 0  # 通过 softmax pool 补上的个数
            ########################################
            comp_inputs_all, meta = [], []  # meta 记录 (b, origin) 方便回填
            for b in range(bs0):
                start = b * beam_size * num_rollout
                end   = start + beam_size * num_rollout
                lp_slice   = np.array(all_lp[start:end],    dtype=np.float32)  # 仍用 lp 做筛选/打分
                adv_slice  = np.array(all_adv[start:end],   dtype=np.float32)
                rhat_slice = np.array(all_rhat[start:end],  dtype=np.float32)  # 仅用于 prm_reward
                resp_slice = all_resp[start:end]

                # low-sigma width pruning（用 lp_slice）
                mu, sigma = float(lp_slice.mean()), float(lp_slice.std())
                keep = [i for i, v in enumerate(lp_slice) if v > mu - sigma_rate * sigma]
                keep_num = list(keep)  # 拷贝一份再做补足
                # --- 统计（剪枝前） ---
                N = len(lp_slice)
                total_cnt += N
                kept_pre_cnt += len(keep)

                if len(keep) < beam_size: # 按照绝对似然补足
                    pool = np.setdiff1d(np.arange(len(lp_slice)), np.array(keep), assume_unique=False)
                    if len(pool) > 0:
                        p_pool = _stable_softmax(lp_slice[pool], temperature)
                        extra = np.random.choice(pool, beam_size - len(keep), replace=False, p=p_pool).tolist()
                        keep += extra
                        # --- 统计（补足数量） ---
                        supplemented_cnt += len(extra)
                # --- 统计（补足后） ---
                kept_post_cnt += len(keep)


                # === 对 keep 中所有候选先做一次“补全”，得到 post-completion 的 lp2 与 adv2 ===
                comp_inputs   = []
                origins_keep  = []
                resps_keep    = []
                rhat_keep     = []

                for kk in keep:
                    origin = kk // num_rollout
                    resp   = resp_slice[kk]
                    rhat   = float(rhat_slice[kk])  # 只用于 prm_reward 段末脉冲

                    comp_prefix = (
                        f"User: {raw_prompts[b].strip()}\n"
                        f"Reasoning so far:\n{prev_steps[b][origin]}{resp}"
                    )
                    ids = self.inference_engine.llm_engine.tokenizer.encode(
                        comp_prefix, add_special_tokens=False
                    )
                    comp_inputs_all.append({"prompt_token_ids": ids})
                    meta.append((b, origin, resp, float(rhat_slice[kk])))
                    comp_inputs.append({"prompt_token_ids": ids})
                    origins_keep.append(origin)
                    resps_keep.append(resp)
                    rhat_keep.append(rhat)

            # —— 一次性调用 vLLM 生成 —— 
            comp_outs_all = self.inference_engine.generate(
                prompts=comp_inputs_all, sampling_params=comp_sp, use_tqdm=False
            )

            # —— 把结果按 b 分桶，计算 lp2/adv2，并在各自 b 内做选样 —— 
            cursor = 0
            buckets = {b: {"lp2": [], "adv2": [], "origin": [], "resp": [], "rhat": []} for b in range(bs0)}
            for i, out in enumerate(comp_outs_all):
                o = out.outputs[0]
                L = len(o.token_ids)
                sc2  = _avg_sc_for_output(o, V)
                #lp2 = float(o.cumulative_logprob) / (L + 1e-8) if L > 0 else -10.0
                lp2 = sc2  # 用 avg_sc 替代 avg_logprob 作为搜索/剪枝/adv 的基线 0905
                b, origin, resp, rhat = meta[i]
                # adv2 = lp2 - float(prev_values[b][origin])
                base = group_baseline[(b, origin)]  # 首轮=mu0，其后=prev_values[b][origin]
                adv2 = lp2 - base
                bk = buckets[b]
                bk["lp2"].append(lp2); bk["adv2"].append(adv2)
                bk["origin"].append(origin); bk["resp"].append(resp); bk["rhat"].append(rhat)

            # —— 对每个 b：用 lp2/adv2 做 zscore + softmax 选 beam_size 个，并更新 new_* —— 
            for b in range(bs0):
                if not buckets[b]["lp2"]: continue
                abs2_k = np.asarray(buckets[b]["lp2"], dtype=np.float32)
                adv2_k = np.asarray(buckets[b]["adv2"], dtype=np.float32)
                z_abs2 = (abs2_k - abs2_k.mean()) / (abs2_k.std() + 1e-8)
                z_gain2= (adv2_k - adv2_k.mean()) / (adv2_k.std()+ 1e-8)
                logits = z_gain2 
                combined = _stable_softmax(logits, temperature)
                sel = np.random.choice(len(abs2_k), size=beam_size, replace=False, p=combined)

                for k_idx, sel_idx in enumerate(sel):
                    origin = buckets[b]["origin"][sel_idx]
                    resp   = buckets[b]["resp"][sel_idx]
                    rhat   = buckets[b]["rhat"][sel_idx]
                    lp2    = float(abs2_k[sel_idx])
                    adv2   = float(adv2_k[sel_idx])

                    tok_ids = self.inference_engine.llm_engine.tokenizer.encode(resp, add_special_tokens=False)
                    #seg = [0.0]*(len(tok_ids)-1) + ([rhat] if len(tok_ids)>0 else [])
                    if prm_mode == "gain":
                        # 用 adv2 作为本段奖励，并按策略分配到段内 token
                        seg = _spread_segment_reward(
                            total_reward = prm_gain_scale * adv2,
                            L            = len(tok_ids),
                            mode         = reward_spread_mode,
                            gamma        = reward_spread_gamma
                        )
                    else:
                        # 兼容旧逻辑（LCF 均值 logprob 作为段尾脉冲）
                        seg = [0.0]*(len(tok_ids)-1) + ([rhat] if len(tok_ids)>0 else [])

                    new_weights[b][k_idx] = weights_history[b][origin] + seg

                    new_steps[b][k_idx]   = prev_steps[b][origin] + resp + "\n"
                    new_values[b][k_idx]  = lp2
                    new_gains[b][k_idx]   = adv2
                    new_gain_history[b][k_idx] = gain_history[b][origin] + [adv2]



            prev_steps, prev_values, weights_history = new_steps, new_values, new_weights
            prev_gains = new_gains  # 更新增益
            gain_history = new_gain_history

            # === depth 级别的调试输出（可按 debug_prune_every 控制频率） ===
            if debug_prune and (depth % max(1, debug_prune_every) == 0):
                ratio_pre  = (kept_pre_cnt  / total_cnt) if total_cnt > 0 else 0.0
                ratio_post = (kept_post_cnt / total_cnt) if total_cnt > 0 else 0.0
                print(f"[PRUNE-DEBUG] depth={depth} "
                    f"keep_pre={kept_pre_cnt}/{total_cnt} ({ratio_pre:.2%}), "
                    f"keep_post={kept_post_cnt}/{total_cnt} ({ratio_post:.2%}), "
                    f"supplemented={supplemented_cnt}, "
                    f"beam_size={beam_size}, num_rollout={num_rollout}")

        # 7. Final answer generation — 用 prev_values 的 abs+gain 计算 combined 采样 beam，并写入其概率
        final_prompts, history_list, final_ws, final_probs = [], [], [], []
        final_gain_hists, seq_gain_list = [], [] 
        # for b in range(bs0):
        #     L = np.array(prev_values[b], dtype=np.float32)  # 平均 logprob（绝对置信度）
        #     G = np.array(prev_gains[b],  dtype=np.float32)  # 真正的动态增益

        #     def zscore(x, eps=1e-8):
        #         s = x.std()
        #         return (x - x.mean()) / (s + eps)

        #     zL = zscore(L)
        #     zG = zscore(G)  

        #     tau_abs  = float(kwargs.get("tau_abs",  temperature))
        #     tau_gain = float(kwargs.get("tau_gain", temperature))
        #     lam      = float(kwargs.get("mix_lambda", 0.6))

        #     logits = lam * (zL / tau_abs) + (1.0 - lam) * (zG / tau_gain)
        #     combined_beam = _stable_softmax(logits, temperature)

        #     # 采样
        #     choice   = int(np.random.choice(len(combined_beam), p=combined_beam))
        #     p_choice = float(combined_beam[choice])  # 被选中的组合概率

        #     history_list.append(prev_steps[b][choice])
        #     final_ws.append(weights_history[b][choice])
        #     final_probs.append(p_choice)
        #     final_gain_hists.append(gain_history[b][choice])

        #     # ★ 新增：记录该 prompt 进入最终阶段时的 baseline（均值 logprob 基线）
        #     #    就是 prev_values[b][choice]
        #     if 'final_baselines' not in locals():
        #         final_baselines = []
        #     final_baselines.append(float(prev_values[b][choice]))

        #     prompt_txt = (
        #         f"User: {raw_prompts[b].strip()}\n"
        #         f"Reasoning so far:\n{prev_steps[b][choice]}"
        #     )
        #     ids = self.tokenizer.encode(prompt_txt, add_special_tokens=False)
        #     final_prompts.append({"prompt_token_ids": ids})
        final_baselines = []

        # 可选：单独给最终一步一个更短的补全长度，默认复用 comp_sp
        final_step_completion_len = int(kwargs.get("final_step_completion_length",
                                                128//2))
        final_comp_sp = SamplingParams(
            max_tokens=final_step_completion_len,
            logprobs=sc_topk,
            temperature=temperature,
            n=1,
            stop=["<end_of_reasoning>"],
        )

        for b in range(bs0):
            # 对当前 b 的每个 beam 做一次“短补全”来打分
            comp_inputs = []
            for k in range(beam_size):
                prefix = (
                    f"User: {raw_prompts[b].strip()}\n"
                    f"Reasoning so far:\n{prev_steps[b][k]}"
                )
                ids = self.inference_engine.llm_engine.tokenizer.encode(
                    prefix, add_special_tokens=False
                )
                comp_inputs.append({"prompt_token_ids": ids})

            comp_outs = self.inference_engine.generate(
                prompts=comp_inputs,
                sampling_params=final_comp_sp,
                use_tqdm=False
            )

            # 计算“补全后的均值 logprob”和对应的增益 adv2_final
            lp2_k, adv2_k = [], []
            for k, out in enumerate(comp_outs):
                o = out.outputs[0]
                L = len(o.token_ids)
                sc2  = _avg_sc_for_output(o, V)
                lp2 = sc2  # 用 avg_sc 替代 avg_logprob 作为搜索/剪枝/adv 的基线 0905
                #lp2 = float(o.cumulative_logprob) / (L + 1e-8) if L > 0 else -10.0
                lp2_k.append(lp2)
                adv2_k.append(lp2 - float(prev_values[b][k]))

            abs2_k = np.asarray(lp2_k, dtype=np.float32)
            adv2_k = np.asarray(adv2_k, dtype=np.float32)
            z_abs2  = (abs2_k - abs2_k.mean()) / (abs2_k.std()  + 1e-8)
            z_gain2 = (adv2_k - adv2_k.mean()) / (adv2_k.std() + 1e-8)

            # 只用增益 or 混合打分，默认只用增益更稳
            if bool(kwargs.get("final_use_gain_only", True)):
                logits = z_gain2
            else:
                tau_abs  = float(kwargs.get("tau_abs",  temperature))
                tau_gain = float(kwargs.get("tau_gain", temperature))
                lam      = float(kwargs.get("mix_lambda", mix_lambda))
                logits   = lam * (z_abs2 / max(1e-8, tau_abs)) + (1.0 - lam) * (z_gain2 / max(1e-8, tau_gain))

            combined_beam = _stable_softmax(logits, temperature)
            choice   = int(np.random.choice(len(combined_beam), p=combined_beam))
            p_choice = float(combined_beam[choice])

            # 记录被选中的轨迹与基线，用于之后 final_adv 计算与最终生成
            history_list.append(prev_steps[b][choice])
            final_ws.append(weights_history[b][choice])
            final_probs.append(p_choice)
            final_gain_hists.append(gain_history[b][choice])
            final_baselines.append(float(prev_values[b][choice]))

            prompt_txt = (
                f"User: {raw_prompts[b].strip()}\n"
                f"Reasoning so far:\n{prev_steps[b][choice]}"
            )
            ids = self.tokenizer.encode(prompt_txt, add_special_tokens=False)
            final_prompts.append({"prompt_token_ids": ids})
        

        # 8. Generate final sequences
        final_sp = SamplingParams(
            max_tokens=response_len,
            logprobs=sc_topk,                # 必须开，后面要用 logprob
            temperature=temperature,
            n=1,
            stop=["<end_of_reasoning>"]
        )
        final_outs = self.inference_engine.generate(
            prompts=final_prompts,
            sampling_params=final_sp,
            use_tqdm=False
        )

        import numpy as np
        ############# seq gain 计算函数 #############
        ############# 越稳越好 ####################
        def _seq_gain_pos_neg(
            gh,
            w_pos=1.0,         # 正增益权重：鼓励稳定上升
            w_neg=1.0,         # 负增益权重：显式惩罚回撤（>= w_pos 会更保守）
            var_coef=0.10,     # 振荡惩罚（对 gain 方差的系数）
            dd_coef=0.15,      # （可选）回撤面积惩罚：抑制“先跌后涨”
            second_diff_coef=0.0,  # （可选）二阶差分（加速度）的方差惩罚，进一步抑制抖动
        ):
            """
            gh: list[float]，每步 gain = φ_t - φ_{t-1}（φ=mean logprob）
            返回：一个单调反映“越稳越好”的标量，供 compute_grpo_outcome_advantage 做组内 z-score + 映射
            """
            if len(gh) == 0:
                return 0.0

            gh = np.asarray(gh, dtype=np.float32)

            # 1) 正/负增益分解
            pos = float(np.sum(np.clip(gh,  0.0, None)))
            neg = float(np.sum(np.clip(-gh, 0.0, None)))

            # 2) 振荡惩罚（方差）
            var_pen = float(np.var(gh)) if gh.size > 1 else 0.0

            # 3) （可选）回撤面积惩罚：对累计轨迹 φ 的“低于历史峰值”的面积做惩罚
            #    这能抑制“先刻意跌、再猛涨”的刷分行为
            phi = np.cumsum(gh)                 # φ_t - φ_0
            peak = np.maximum.accumulate(phi)   # 历史峰值
            drawdown = np.maximum(0.0, peak - phi)
            dd_area = float(np.sum(drawdown))   # 回撤面积

            # 4) （可选）二阶差分方差：进一步抑制高频抖动（默认关）
            if second_diff_coef > 0.0 and gh.size > 2:
                second_diff_var = float(np.var(np.diff(gh, n=2)))
            else:
                second_diff_var = 0.0

            score = (w_pos * pos
                    - w_neg * neg
                    - var_coef * var_pen
                    - dd_coef * dd_area
                    - second_diff_coef * second_diff_var)
            return score


        # 9. Parse final outputs；在最终段落的最后一个 token 写入 LCF r_hat_final
        full_texts, padded_ws = [], []
        for i, out in enumerate(final_outs):
            o = out.outputs[0]
            gen = o.text.strip()
            full = history_list[i] + gen
            full_texts.append(full)
            tok_ids = self.tokenizer.encode(gen, add_special_tokens=False)

            # # === 仅 prm_reward：最后段落用软掩码 r_hat_final ===
            # lp_list = _extract_chosen_token_logprobs(o)
            # r_hat_final = _lcf_weighted_mean_from_logprobs(lp_list)
            # seg = [0.0] * (len(tok_ids) - 1) + [r_hat_final]

            # padded_ws.append(final_ws[i] + seg)

            L = len(o.token_ids)
            sc_final  = _avg_sc_for_output(o, V)
            mean_lp_final = sc_final    # 用 avg_sc 替代 avg_logprob 作为搜索/剪枝/adv 的基线 0905
            #mean_lp_final = float(o.cumulative_logprob) / (L + 1e-8) if L > 0 else -10.0
            baseline = float(final_baselines[i])
            final_adv = mean_lp_final - baseline   # ★ 最终段的增益

            if prm_mode == "gain":
                seg = _spread_segment_reward(
                    total_reward = prm_gain_scale * final_adv,
                    L            = len(tok_ids),
                    mode         = reward_spread_mode,
                    gamma        = reward_spread_gamma
                )
            else:
                # 兼容旧逻辑
                lp_list = _extract_chosen_token_logprobs(o)
                r_hat_final = _lcf_weighted_mean_from_logprobs(lp_list)
                seg = [0.0] * (len(tok_ids) - 1) + [r_hat_final]

            padded_ws.append(final_ws[i] + seg)

            gh = final_gain_hists[i]
            sg = _seq_gain_pos_neg(
                gh,
                w_pos=1.0,
                w_neg=1.0,       # 想更保守可设 1.2~1.5
                var_coef=0.10,   # 0.05~0.15 常用
                dd_coef=0.15,    # 0.10~0.30 常用；更重视“别回撤”就调大
                second_diff_coef=0.0  # 如仍抖可设 0.02~0.05
            ) if len(gh) > 0 else 0.0
            seq_gain_list.append(float(sg))
            #seq_gain_list.append(float(np.mean(gh)) if len(gh) > 0 else 0.0) # 求gain均值（旧版）

        full_ids = [self.tokenizer.encode(t, add_special_tokens=False) for t in full_texts]
        resp_pad = pad_2d_list_to_length(full_ids, self.pad_token_id, response_len).to(idx0.device)

        # 10. Build prm_reward（仅此采用 LCF 段奖；其他逻辑不变）
        pr_tensors = []
        for r in padded_ws:
            if len(r) >= response_len:
                row = r[:response_len]
            else:
                row = r + [0.0] * (response_len - len(r))
            pr_tensors.append(row)

        prm_reward = torch.tensor(pr_tensors, device=idx0.device, dtype=torch.float32)
        seq_gain = torch.tensor(seq_gain_list, device=idx0.device, dtype=torch.float32)  # [Bn]

        # 11. Rebuild batch tensors
        Bn = resp_pad.size(0)
        repeat = Bn // bs0
        idx   = idx0.repeat_interleave(repeat, dim=0)
        mask  = prompts.batch["attention_mask"].repeat_interleave(repeat, dim=0)
        pos   = prompts.batch["position_ids"].repeat_interleave(repeat, dim=0)
        seq   = torch.cat([idx, resp_pad], dim=1)
        delta = torch.arange(1, response_len+1, device=pos.device).unsqueeze(0).expand(Bn, -1)
        last  = pos[:, -1:].expand(-1, response_len)
        pos   = torch.cat([pos, last + delta], dim=1)
        attn  = get_response_mask(resp_pad, prompts.meta_info["eos_token_id"], mask.dtype)
        mask  = torch.cat([mask, attn], dim=1)

        batch = TensorDict({
            "prompts":        idx,
            "responses":      resp_pad,
            "input_ids":      seq,
            "attention_mask": mask,
            "position_ids":   pos,
            "prm_reward":     prm_reward,   # ← 只这部分采用 LCF
            "seq_gain":       seq_gain,
        }, batch_size=Bn)

        # 12. Expand non_tensor_batch
        new_ntb = {}
        for k, v in non_tensor_batch.items():
            if isinstance(v, list):
                new_ntb[k] = list(itertools.chain.from_iterable([v] * repeat))
            elif isinstance(v, np.ndarray):
                new_ntb[k] = np.repeat(v, repeat, axis=0)
            elif torch.is_tensor(v):
                new_ntb[k] = v.repeat_interleave(repeat, dim=0)
            else:
                new_ntb[k] = [v] * Bn

        print(f"[DEBUG] resp_pad.shape={resp_pad.shape}, seq.shape={seq.shape}, prm_reward.shape={prm_reward.shape}")
        return DataProto(batch=batch, non_tensor_batch=new_ntb)'''
        






#########################
# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = original_compute_logits(hidden_states, sampling_metadata)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        self.tokenizer = tokenizer

        # Engine is deferred to be initialized in init_worker
        self.config = config
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False
        self.address = self._init_zeromq()

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock("/tmp/verl_vllm_zmq.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        self.loop_thread = threading.Thread(target=self._loop_forever)
        self.loop_thread.start()

        return address

    def _get_free_port(self):
        ip = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    def _loop_forever(self):
        while True:
            message = self.socket.recv()
            method, args, kwargs = pickle.loads(message)
            result = self.execute_method(method, *args, **kwargs)
            self.socket.send(pickle.dumps(result))

    def get_zeromq_address(self):
        return self.address

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
