from __future__ import annotations

"""
WIP: 
draft for ater attention backend
"""

import os
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInfo

# dummy wrapper for ater decode
class AterAttnDecodeWrapper:
    def __init__(self):
        pass

# dummy wrapper for ater prefill
class AterAttnPrefillWrapper:
    def __init__(self):
        pass

class AterAttnBackend(AttentionBackend):
    """Ater attention kernels."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__()

        self.max_context_len = model_runner.model_config.context_len
        max_bs = model_runner.req_to_token_pool.size
        self.kv_indptr = torch.zeros((max_bs + 1,), dtype=torch.int32, device=model_runner.device)
        self.kv_last_page_len = torch.ones((max_bs,), dtype=torch.int32, device=model_runner.device)
        self.qo_indptr = torch.zeros((max_bs + 1,), dtype=torch.int32, device=model_runner.device)

        # Create wrappers
        self.prefill_wrapper = AterAttnDecodeWrapper()
        self.decode_wrapper = AterAttnPrefillWrapper()

        # Create indices updater
        self.indices_updater_decode = None
        self.indices_updater_prefill = None
        
        # Other metadata
        self.forward_metadata = {}
        self.prefill_cuda_graph_metadata = {}
        self.decode_cuda_graph_metadata = {}
        

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_decode():
            self.indices_updater_decode.update()
        elif forward_batch.forward_mode.is_draft_extend():
            self.indices_updater_prefill.update()
        elif forward_batch.forward_mode.is_target_verify():
            self.indices_updater_prefill.update()
        else:
            self.indices_updater_prefill.update()

    def init_cuda_graph_state(self, max_bs: int):
        self.cuda_graph_kv_indices = torch.zeros(
            (max_bs * self.max_context_len,),
            dtype=torch.int32,
            device="cuda",
        )

        self.cuda_graph_custom_mask = torch.zeros(
            (max_bs * self.max_context_len),
            dtype=torch.uint8,
            device="cuda",
        )
        self.cuda_graph_qk_indptr = [x.clone() for x in self.kv_indptr]
        self.cuda_graph_qo_indptr = [x.clone() for x in self.kv_indptr]

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
    ):
        if forward_mode.is_decode():
            decode_wrapper = AterAttnDecodeWrapper()
            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_decode.update()
            self.decode_cuda_graph_metadata[bs] = decode_wrapper
            self.forward_metadata = None
        elif forward_mode.is_target_verify():
            prefill_wrapper = AterAttnPrefillWrapper()
            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_prefill.update()
            self.prefill_cuda_graph_metadata[bs] = prefill_wrapper
            self.forward_metadata = None
        else:
            raise ValueError(f"Invalid mode: {forward_mode=}")

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
    ):
        if forward_mode.is_decode():
            self.indices_updater_decode.update()
        elif forward_mode.is_target_verify():
            self.indices_updater_prefill.update()
        else:
            raise ValueError("Invalid forward mode")

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        
        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        o = self.prefill_wrapper.forward(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            causal=not layer.is_cross_attention,
            sm_scale=layer.scaling,
            window_left=layer.sliding_window_size,
            logits_soft_cap=layer.logit_cap,
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):        
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        
        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        o = self.decode_wrapper.forward(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)



# index update
class AterIndicesUpdaterDecode:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // model_runner.tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            model_runner.tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrapper: AterAttnDecodeWrapper,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInfo],
    ):
        decode_wrapper = decode_wrapper or self.decode_wrapper
        # TODO: update function for decoder
        # call create_ater_kv_indices_triton here


class AterIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // model_runner.tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            model_runner.tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token


    def update(
        self,
        req_pool_indices: torch.Tnesor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrapper: AterAttnPrefillWrapper,
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInfo],
    ):
        prefill_wrapper = prefill_wrapper or self.prefill_wrapper
        # TODO: prefill update function here
        # call create_ater_kv_indices_triton



@triton.jit
def create_ater_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)

