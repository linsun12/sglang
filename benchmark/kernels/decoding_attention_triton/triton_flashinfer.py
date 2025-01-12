import itertools
import math

import torch
import torch.utils.benchmark as benchmark
import triton
import triton.language as tl
from flashinfer import BatchDecodeWithPagedKVCacheWrapper

from sglang.srt.layers.attention.triton_ops.decode_attention import decode_attention_fwd
from sglang.srt.layers.attention.flashinfer_backend import should_use_tensor_core


def benchmark_forward(
    fn,
    *inputs,
    repeats=10,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    return t, m


def time_fwd(func, *args, **kwargs):
    time_f = benchmark_forward(func, *args, **kwargs)
    return time_f[1].mean * 1e6


#============triton decoder===========
def decode_attention_sglang(
    q,
    kv_data,
    batch_size,
    kv_len,
    head_num_q,
    head_num_kv,
    head_dim,
    num_kv_splits,
    warmup=10,
):

    k_buffer = kv_data[0].view(-1, head_num_kv, head_dim)
    v_buffer = kv_data[1].view(-1, head_num_kv, head_dim)
    o = torch.empty_like(q)
    total_tokens = batch_size * kv_len
    req_to_token = torch.arange(0, total_tokens).to(0).int().view(batch_size, kv_len)
    b_req_idx = torch.arange(0, batch_size).to(0).int()
    b_seq_len = torch.full((batch_size,), kv_len, dtype=torch.int32, device="cuda")
    max_len_in_batch = kv_len
    sm_scale = 1.0 / (head_dim**0.5)

    attn_logits = torch.empty(
        (batch_size, head_num_q, num_kv_splits, head_dim + 1),
        dtype=torch.float32,
        device="cuda",
    )

    for _ in range(warmup):
        decode_attention_fwd(
            q,
            k_buffer,
            v_buffer,
            o,
            req_to_token,
            b_req_idx,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            sm_scale,
        )

    f = time_fwd(
        decode_attention_fwd,
        q,
        k_buffer,
        v_buffer,
        o,
        req_to_token,
        b_req_idx,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
    )

    return f, o



#==============flashinfer decoder==================
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
            
            # some dummy page indices related data
            # kv_indptr: the index ptr of the paged kv cache, shape [batch_size+1], kv_len is the number of tokens in each sequence
            # so if each sequence has 100 tokens, and batch size = 4, this is [0, 100, 200, 300, 400]
            # kv_indices: page indices of the paged kv cache, total_tokens = total number of tokens in this batch saved in cache
            # example [0,1,2,3,...,399,400]
            # kv_last_page_len: the number of entries in the last page OF EACH REQUEST in the paged kv cache. Each request has its own
            # pages, like 4 requests in each batch, each request has 100 tokens in kv cache, each request's kv cache is split in pages
            # each request's last page has this many entries in it, rest is empty and can be used to insert new kv cache from decoder -> we need to update this
            # this dummy data makes last page of each request in the batch has 1 entries filled
            # in this dummy example page_size = 1, each page already has 1 token's kv, so each token's new kv data needs to open a new page
            
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
                1, # page size
                pos_encoding_mode="NONE",
                data_type=dtype,
            )

            for _ in range(warmup):
                o = flashinfer_decode_wrapper.forward(
                    q.contiguous().view(-1, head_num_q, head_dim), kv_data
                )

            f = time_fwd(
                flashinfer_decode_wrapper.forward,
                q.contiguous().view(-1, head_num_q, head_dim),
                kv_data,
            )

            return f, o

    return FlashinferAttention



def calculate_diff():

    dtype = torch.float16
    batch_size = 64
    kv_len = 4096
    head_num_q = 64
    head_num_kv = 8
    head_dim = 128

    q = torch.randn(batch_size, head_num_q, head_dim, dtype=dtype, device="cuda")
    kv_data = (
        torch.randn(
            batch_size * kv_len, head_num_kv, head_dim, dtype=dtype, device="cuda"
        ),
        torch.randn(
            batch_size * kv_len, head_num_kv, head_dim, dtype=dtype, device="cuda"
        ),
    )

    _, output_sglang = decode_attention_sglang(
        q,
        kv_data,
        batch_size,
        kv_len,
        head_num_q,
        head_num_kv,
        head_dim,
        num_kv_splits=8,
    )

    attn_flashinfer = decode_attention_flashinfer(dtype, head_num_q, head_num_kv).apply
    _, output_flashinfer = attn_flashinfer(
        q, kv_data, batch_size, kv_len, head_num_q, head_num_kv, head_dim, dtype
    )

    print(f"SGLang output={output_sglang}")
    print(f"FlashInfer output={output_flashinfer}")
    if torch.allclose(output_sglang, output_flashinfer, atol=1e-2, rtol=1e-2):
        print("✅ SGLang[Triton] and FlashInfer match")
    else:
        print("❌ SGLang[Triton] and FlashInfer differ")


if __name__ == "__main__":
    calculate_diff()

    head_dim = 128
    dtype = torch.float16
    batch_size_range = [2**i for i in range(0, 8, 2)]
    kv_len_range = [2**i for i in range(6, 13, 1)]
    
    # total number of tokens in each batch
    configs = list(itertools.product(batch_size_range, kv_len_range))

    for head_num_q, head_num_kv in [[32, 32], [64, 8], [40, 8]]:
        attn_flashinfer = decode_attention_flashinfer(
            dtype, head_num_q, head_num_kv
        ).apply
        for batch_size, kv_len in configs:
            # q shape: (batch size, number of heads for q, embedding len)
            # in decoding stage q is one newly generated vector of size head_dim
            q = torch.randn(
                batch_size, head_num_q, head_dim, dtype=dtype, device="cuda"
            )
            
            # k shape: for each head, for each element in batch, shape kv_len(number of tokens in seq) x head_dim(embedding len)
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
            
            # triton's decoder has num_kv_splits
            us_sglang, output_sglang = decode_attention_sglang(
                q,
                kv_data,
                batch_size,
                kv_len,
                head_num_q,
                head_num_kv,
                head_dim,
                num_kv_splits=8,
            )
            
            # flashinfer's decoder has paged attention implemented
            us_flashinfer, _ = attn_flashinfer(
                q, 
                kv_data, 
                batch_size, 
                kv_len, 
                head_num_q, 
                head_num_kv, 
                head_dim, 
                dtype
            )
            print(
                head_num_q,
                "  ",
                head_num_kv,
                "  ",
                batch_size,
                "  ",
                kv_len,
                "  ",
                us_sglang,
                "  ",
                us_flashinfer,
            )
