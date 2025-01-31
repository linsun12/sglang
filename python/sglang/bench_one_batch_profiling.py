"""
this script is to for running end to end profiling with same config as flashinfer unit test
input length is set to 54 to match kv_len==54 in unit tests, in real scenario kv_len != input len
tp = 1 to match unit test
batch = 128 match unit test
counting warm up runs since we don't have warm up in unit tests


"""

import argparse
import dataclasses
import itertools
import json
import logging
import multiprocessing
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server import _set_envs_and_config
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import configure_logger, kill_process_tree, suppress_other_loggers


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (128,)
    input_len: Tuple[int] = (54,)
    output_len: Tuple[int] = (2,) # for shorter profiling file
    result_filename: str = "result.jsonl"

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to case the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


def load_model(server_args, port_args, tp_rank):
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    model_config = ModelConfig(
        server_args.model_path,
        trust_remote_code=server_args.trust_remote_code,
        revision=server_args.revision,
        context_length=server_args.context_length,
        model_override_args=server_args.json_model_override_args,
        is_embedding=server_args.is_embedding,
        dtype=server_args.dtype,
        quantization=server_args.quantization,
    )
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    rank_print(f"max_total_num_tokens={model_runner.max_total_num_tokens}")
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    if server_args.tp_size > 1:
        dist.barrier()
    return model_runner, tokenizer


def prepare_synthetic_inputs_for_latency_test(batch_size, input_len):
    input_ids = np.ones((batch_size, input_len), dtype=np.int32)
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(input_ids)):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(input_ids[i]),
            sampling_params=sampling_params,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        reqs.append(req)

    return reqs


@torch.no_grad
def extend(reqs, model_runner):
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool=model_runner.token_to_kv_pool,
        tree_cache=None,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits, batch


@torch.no_grad
def decode(input_token_ids, batch, model_runner):
    batch.output_ids = input_token_ids
    batch.prepare_for_decode()
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits


def synchronize(device):
    torch.get_device_module(device).synchronize()


def latency_test_run_once(
    run_name, model_runner, rank_print, reqs, batch_size, input_len, output_len, device
):
    max_batch_size = model_runner.max_total_num_tokens // (input_len + output_len)
    if batch_size > max_batch_size:
        rank_print(
            f"skipping ({batch_size}, {input_len}, {output_len}) due to max batch size limit"
        )
        return

    # Clear the pools.
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool.clear()

    measurement_results = {
        "run_name": run_name,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
    }

    tot_latency = 0

    # Prefill
    synchronize(device)
    tic = time.time()
    next_token_ids, _, batch = extend(reqs, model_runner)
    synchronize(device)
    prefill_latency = time.time() - tic
    tot_latency += prefill_latency
    throughput = input_len * batch_size / prefill_latency
    rank_print(
        f"Prefill. latency: {prefill_latency:6.5f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["prefill_latency"] = prefill_latency
    measurement_results["prefill_throughput"] = throughput

    # Decode
    decode_latencies = []
    for i in range(output_len - 1):
        synchronize(device)
        tic = time.time()
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        synchronize(device)
        latency = time.time() - tic
        tot_latency += latency
        throughput = batch_size / latency
        decode_latencies.append(latency)
        if i < 5:
            rank_print(
                f"Decode.  latency: {latency:6.5f} s, throughput: {throughput:9.2f} token/s"
            )

    # Record decode timing from 2nd output
    if output_len > 1:
        med_decode_latency = np.median(decode_latencies)
        med_decode_throughput = batch_size / med_decode_latency
        rank_print(
            f"Decode.  median latency: {med_decode_latency:6.5f} s, median throughput: {med_decode_throughput:9.2f} token/s"
        )
        measurement_results["median_decode_latency"] = med_decode_latency
        measurement_results["median_decode_throughput"] = med_decode_throughput

    throughput = (input_len + output_len) * batch_size / tot_latency
    rank_print(
        f"Total. latency: {tot_latency:6.3f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["total_latency"] = tot_latency
    measurement_results["overall_throughput"] = throughput
    return measurement_results


def latency_test(
    server_args,
    port_args,
    bench_args,
    tp_rank,
):
    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, tp_rank)

    # Prepare inputs for warm up
    reqs = prepare_synthetic_inputs_for_latency_test(
        bench_args.batch_size[0], bench_args.input_len[0]
    )

    # Warm up
    # rank_print("Warmup ...")
    # latency_test_run_once(
    #     bench_args.run_name,
    #     model_runner,
    #     rank_print,
    #     reqs,
    #     bench_args.batch_size[0],
    #     bench_args.input_len[0],
    #     1,  # shorter decoding to speed up the warmup
    #     server_args.device,
    # )

    rank_print("Benchmark ...")

    # Run the sweep
    result_list = []
    for bs, il, ol in itertools.product(
        bench_args.batch_size, bench_args.input_len, bench_args.output_len
    ):
        reqs = prepare_synthetic_inputs_for_latency_test(bs, il)
        ret = latency_test_run_once(
            bench_args.run_name,
            model_runner,
            rank_print,
            reqs,
            bs,
            il,
            ol,
            server_args.device,
        )
        if ret is not None:
            result_list.append(ret)

    # Write results in jsonlines format on rank 0.
    if tp_rank == 0 and bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            for result in result_list:
                fout.write(json.dumps(result) + "\n")


def main(server_args, bench_args):
    _set_envs_and_config(server_args)

    if server_args.model_path:

        work_func = latency_test
    else:
        raise ValueError(
            "Provide --model-path for running the tests or "
            "provide --result-filename for plotting the results"
        )

    port_args = PortArgs.init_new(server_args)

    if server_args.tp_size == 1:
        work_func(server_args, port_args, bench_args, 0)
    else:
        workers = []
        for tp_rank in range(server_args.tp_size):
            proc = multiprocessing.Process(
                target=work_func,
                args=(
                    server_args,
                    port_args,
                    bench_args,
                    tp_rank,
                ),
            )
            proc.start()
            workers.append(proc)

        for proc in workers:
            proc.join()

        proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)
