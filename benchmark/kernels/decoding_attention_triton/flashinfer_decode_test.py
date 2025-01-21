"""
This script script is used for rpd profiling on flashinfer decoder
"""
import itertools

import torch
import triton.language as tl
from flashinfer import BatchDecodeWithPagedKVCacheWrapper

from sglang.srt.layers.attention.flashinfer_backend import should_use_tensor_core


def decode_attention_flashinfer(dtype, head_num_q, head_num_kv):
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
    use_tensor_cores = should_use_tensor_core(
        kv_cache_dtype=dtype,
        num_attention_heads=head_num_q,
        num_kv_heads=head_num_kv,
    )
    flashinfer_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_tensor_cores=use_tensor_cores
    )

    class FlashinferAttention(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            q,
            kv_data,
            batch_size,
            kv_len,
            head_num_q,
            head_num_kv,
            head_dim,
            dtype,
            warmup=10,
        ):
            total_tokens = batch_size * kv_len
            kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * kv_len
            kv_indices = torch.arange(0, total_tokens).to(0).int()
            kv_last_page_len = torch.full(
                (batch_size,), 1, dtype=torch.int32, device="cuda"
            )

            flashinfer_decode_wrapper.end_forward()
            flashinfer_decode_wrapper.begin_forward(
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                head_num_q,
                head_num_kv,
                head_dim,
                1,
                pos_encoding_mode="NONE",
                data_type=dtype,
            )

            for _ in range(warmup):
                o = flashinfer_decode_wrapper.forward(
                    q.contiguous().view(-1, head_num_q, head_dim), kv_data
                )

            return o

    return FlashinferAttention


if __name__ == "__main__":
    head_dim = 128
    dtype = torch.float16
    batch_size_range = [2**i for i in [8]]
    kv_len_range = [1024]
    configs = list(itertools.product(batch_size_range, kv_len_range))

    for head_num_q, head_num_kv in [[32, 32]]:
        if (head_num_q / head_num_kv in [1,2,4,8]): # this condition sets use_tensor_cores = False and using decode module instead of prefill module
            attn_flashinfer = decode_attention_flashinfer(
                dtype, head_num_q, head_num_kv
            ).apply
            for batch_size, kv_len in configs:
                q = torch.randn(
                    batch_size, head_num_q, head_dim, dtype=dtype, device="cuda"
                )
                kv_data = (
                    torch.randn(
                        batch_size * kv_len,
                        head_num_kv,
                        head_dim,
                        dtype=dtype,
                        device="cuda",
                    ),
                    torch.randn(
                        batch_size * kv_len,
                        head_num_kv,
                        head_dim,
                        dtype=dtype,
                        device="cuda",
                    ),
                )

                _ = attn_flashinfer(
                    q, kv_data, batch_size, kv_len, head_num_q, head_num_kv, head_dim, dtype
                )
                print(
                    head_num_q,
                    "  ",
                    head_num_kv,
                    "  ",
                    batch_size,
                    "  ",
                    kv_len
                )
