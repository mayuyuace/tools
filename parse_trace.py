import json
from typing import List
import os
import argparse
from enum import Enum
import numpy as np
from dataclasses import dataclass

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
HF_CONFIG_PATH = os.path.join(CUR_PATH, "hf_configs/")


DTYPE_TO_BYTES = {
    "fp8": 1,
    "fp16": 2,
    "fp32": 4,
    "int8": 1,
    # qweight + fp16 scale
    "int4": 0.5234375,
}

# for BMG-24G

HW_CONFIG = {
    "xpu-bmg": {
        "tflops": 98,
        "mem_bw": 451,
    },
    "cuda-4090d": {
        "tflops": 98,
        "mem_bw": 451
    }
}

# TFLOPS_PEAK = 98
# MEM_BANDWIDTH_PEAK = 451

class EfficiencyMetrics(Enum):
    TFLOPS = 0
    MEM_BANDWIDTH = 1


@dataclass
class TRACE_STATS():
    total_kernels: int = 0
    total_gemm_kernels: int = 0
    total_qkv_gemm_kernels: int = 0
    total_out_gemm_kernels: int = 0
    total_router_gemm_kernels: int = 0
    total_gateup_gemm_kernels: int = 0
    total_down_gemm_kernels: int = 0
    total_w13_grouped_gemm_kernels: int = 0
    total_w2_grouped_gemm_kernels: int = 0
    total_fmha_kernels: int = 0
    total_flash_fwd_splitkv_kernels: int = 0
    total_flash_fwd_splitkv_combine_kernels: int = 0
    total_norm_kernels: int = 0
    total_act_kernels: int = 0
    total_rope_kernels: int = 0
    total_reshape_and_cache_kernels: int = 0
    total_copy_kernels: int = 0
    total_allreduce_kernels: int = 0
    total_dynamic_per_token_scaled_fp8_quant_kernels: int = 0
    total_other_kernels: int = 0

    total_kernel_time: float = 0.0
    total_qkv_gemm_time: float = 0.0
    total_out_gemm_time: float = 0.0
    total_router_gemm_time: float = 0.0
    total_gateup_gemm_time: float = 0.0
    total_down_gemm_time: float = 0.0
    total_w13_grouped_gemm_time: float = 0.0
    total_w2_grouped_gemm_time: float = 0.0
    total_fmha_time: float = 0.0
    total_flash_fwd_splitkv_time: float = 0.0
    total_flash_fwd_splitkv_combine_time: float = 0.0
    total_norm_time: float = 0.0
    total_act_time: float = 0.0
    total_rope_time: float = 0.0
    total_reshape_and_cache_time: float = 0.0
    total_copy_time: float = 0.0
    total_allreduce_time: float = 0.0
    total_dynamic_per_token_scaled_fp8_quant_time: float = 0.0
    total_other_time: float = 0.0
    total_avg_time: float = 0.0

    qkv_gemm_avg_time: float = 0.0
    qkv_gemm_time_std: float = 0.0
    out_gemm_avg_time: float = 0.0
    out_gemm_time_std: float = 0.0
    router_gemm_avg_time: float = 0.0
    router_gemm_time_std: float = 0.0
    gateup_gemm_avg_time: float = 0.0
    gateup_gemm_time_std: float = 0.0
    down_gemm_avg_time: float = 0.0
    down_gemm_time_std: float = 0.0
    w13_grouped_gemm_avg_time: float = 0.0
    w13_grouped_gemm_time_std: float = 0.0
    w2_grouped_gemm_avg_time: float = 0.0
    w2_grouped_gemm_time_std: float = 0.0
    fmha_avg_time: float = 0.0
    fmha_time_std: float = 0.0
    flash_fwd_splitkv_avg_time: float = 0.0
    flash_fwd_splitkv_time_std: float = 0.0
    flash_fwd_splitkv_combine_avg_time: float = 0.0
    flash_fwd_splitkv_combine_time_std: float = 0.0
    norm_avg_time: float = 0.0
    norm_time_std: float = 0.0
    act_avg_time: float = 0.0
    act_time_std: float = 0.0
    rope_avg_time: float = 0.0
    rope_time_std: float = 0.0
    reshape_and_cache_avg_time: float = 0.0
    reshape_and_cache_time_std: float = 0.0
    copy_avg_time: float = 0.0
    copy_time_std: float = 0.0
    allreduce_avg_time: float = 0.0
    allreduce_time_std: float = 0.0
    dynamic_per_token_scaled_fp8_quant_avg_time: float = 0.0
    dynamic_per_token_scaled_fp8_quant_time_std: float = 0.0
    other_avg_time: float = 0.0
    other_time_std: float = 0.0

    qkv_gemm_tflops_or_mem_bandwidth: float = 0.0
    qkv_gemm_tflops_or_mem_bandwidth_utilization: float = 0.0
    out_gemm_tflops_or_mem_bandwidth: float = 0.0
    out_gemm_tflops_or_mem_bandwidth_utilization: float = 0.0
    router_gemm_tflops_or_mem_bandwidth: float = 0.0
    router_gemm_tflops_or_mem_bandwidth_utilization: float = 0.0
    gateup_gemm_tflops_or_mem_bandwidth: float = 0.0
    gateup_gemm_tflops_or_mem_bandwidth_utilization: float = 0.0
    down_gemm_tflops_or_mem_bandwidth: float = 0.0
    down_gemm_tflops_or_mem_bandwidth_utilization: float = 0.0
    w13_grouped_gemm_tflops_or_mem_bandwidth: float = 0.0
    w13_grouped_gemm_tflops_or_mem_bandwidth_utilization: float = 0.0
    w2_grouped_gemm_tflops_or_mem_bandwidth: float = 0.0
    w2_grouped_gemm_tflops_or_mem_bandwidth_utilization: float = 0.0
    fmha_tflops_or_mem_bandwidth: float = 0.0
    fmha_tflops_or_mem_bandwidth_utilization: float = 0.0

    tflops_or_mem_bandwidth_unavailble: str = "N/A"

def load_model_config(args):
    if args.model == "llama3-8b":
        config_file = os.path.join(HF_CONFIG_PATH, "llama3-8b/config.json")
        print(config_file)
        with open(config_file, "r") as f:
            config = json.load(f)
    elif args.model == "qwen2.5-32b":
        config_file = os.path.join(HF_CONFIG_PATH, "qwen2.5-32b/config.json")
        with open(config_file, "r") as f:
            config = json.load(f)
    elif args.model == "llama3-70b":
        config_file = os.path.join(HF_CONFIG_PATH, "llama3-70b/config.json")
        with open(config_file, "r") as f:
            config = json.load(f)
    elif args.model == "qwen2.5-14b":
        config_file = os.path.join(HF_CONFIG_PATH, "qwen2.5-14b/config.json")
        with open(config_file, "r") as f:
            config = json.load(f)
    elif args.model == "llama4-scout":
        config_file = os.path.join(HF_CONFIG_PATH, "llama4-scout/config.json")
        with open(config_file, "r") as f:
            config = json.load(f)["text_config"]
    else:
        raise ValueError(f"Model {args.model} not in supported model list: llama3-8b, qwen2.5-32b, qwen2.5-14b, llama3-70b. Please provide a valid model name.")
    print(f"Loading model config from {config_file}...")
    # print(config)
    return config


def load_trace_json(args):
    print(f"Loading trace json file from {args.trace_json_file}...")
    with open(args.trace_json_file, "r") as f:
        trace_dict = json.load(f)
    return trace_dict["traceEvents"]


def get_gemm_shape(config, m, tp, is_moe=False, topk=1, num_experts=1):
    hidden_size = config["hidden_size"]
    num_attention_heads = config["num_attention_heads"]
    num_key_value_heads = config["num_key_value_heads"]
    head_dim = config.get("head_dim", None) or hidden_size // num_attention_heads
    intermediate_size = config["intermediate_size"]
    if is_moe:
        topk = config["num_experts_per_tok"]

    gemm_shapes = []
    # qkv gemm shape
    qkv_gemm_shape = (m, hidden_size, (num_attention_heads + num_key_value_heads * 2) * head_dim // tp)
    gemm_shapes.append(qkv_gemm_shape)
    # out gemm shape
    out_gemm_shape = (m, num_attention_heads * head_dim // tp, hidden_size)
    gemm_shapes.append(out_gemm_shape)
    if is_moe:
        # router gemm shape
        router_gemm_shape = (m, hidden_size, num_experts)
        gemm_shapes.append(router_gemm_shape)
    # gateup gemm shape
    gateup_gemm_shape = (m, hidden_size, intermediate_size * 2 // tp)
    gemm_shapes.append(gateup_gemm_shape)
    # down gemm shape
    down_gemm_shape = (m, intermediate_size // tp, hidden_size)
    gemm_shapes.append(down_gemm_shape)
    if is_moe:
        # w13 grouped gemm shape
        w13_grouped_gemm_shape = (m * topk, hidden_size, intermediate_size * 2 // tp)
        gemm_shapes.append(w13_grouped_gemm_shape)
        # w2 grouped gemm shape
        w2_grouped_gemm_shape = (m * topk, intermediate_size // tp, hidden_size)
        gemm_shapes.append(w2_grouped_gemm_shape)

    print(f"{'qkv_gemm shape (m, k, n):':>30} {qkv_gemm_shape}")
    print(f"{'out_gemm shape (m, k, n):':>30} {out_gemm_shape}")
    if is_moe:
        print(f"{'router_gemm shape (m, k, n):':>30} {router_gemm_shape}")
    print(f"{'gateup_gemm shape (m, k, n):':>30} {gateup_gemm_shape}")
    print(f"{'down_gemm shape (m, k, n):':>30} {down_gemm_shape}")
    if is_moe:
        print(f"{'w13_grouped_gemm shape (m * topk, k, n):':>30} {w13_grouped_gemm_shape}")
        print(f"{'w2_grouped_gemm shape (m * topk, k, n):':>30} {w2_grouped_gemm_shape}")

    return gemm_shapes

def compute_gemm_tflops_or_mem_bandwidth(trace_stats: TRACE_STATS, gemm_shape_list: List, weight_dtype: str, metric: EfficiencyMetrics, is_moe=False, real_num_experts=1):
    print("[INFO] Computing TFlops or memory bandwidth...")
    if is_moe:
        gemm_time_list = [
                trace_stats.qkv_gemm_avg_time,
                trace_stats.out_gemm_avg_time,
                trace_stats.router_gemm_avg_time,
                trace_stats.gateup_gemm_avg_time,
                trace_stats.down_gemm_avg_time,
                trace_stats.w13_grouped_gemm_avg_time,
                trace_stats.w2_grouped_gemm_avg_time,
            ]
    else:
        gemm_time_list = [
                trace_stats.qkv_gemm_avg_time,
                trace_stats.out_gemm_avg_time,
                trace_stats.gateup_gemm_avg_time,
                trace_stats.down_gemm_avg_time,
            ]
    if metric == EfficiencyMetrics.TFLOPS:
        # print("[INFO] Computing TFlops...")
        gemm_gflops = []
        for gemm_time, gemm_shape in zip(gemm_time_list, gemm_shape_list):
            m, k, n = gemm_shape
            flops = 2 * m * n * k  # FLOPs for GEMM
            gflops = flops / (gemm_time * 1e6)
            gemm_gflops.append(gflops)
        return gemm_gflops
    elif metric == EfficiencyMetrics.MEM_BANDWIDTH:
        # print("[INFO] Computing memory bandwidth...")
        gemm_bandwidth = []
        if is_moe:
            gemm_time_list[-1] /= real_num_experts
            gemm_time_list[-2] /= real_num_experts
        for gemm_time, gemm_shape in zip(gemm_time_list, gemm_shape_list):
            m, k, n = gemm_shape
            bytes_transferred = k * n * DTYPE_TO_BYTES[weight_dtype]
            bandwidth = bytes_transferred / (gemm_time * 1e3)
            gemm_bandwidth.append(bandwidth)
        return gemm_bandwidth


def compute_fmha_tflops_or_membandwidth(trace_stats: TRACE_STATS, context_len: List[int], seq_len: List[int], model_config, tp: int, kv_cache_dtype: str, metric: EfficiencyMetrics):
    assert len(context_len) == len(seq_len), "context_len and seq_len must have the same length."

    hidden_size = model_config["hidden_size"]
    num_attention_heads = model_config["num_attention_heads"]
    num_attention_heads_per_rank = num_attention_heads // tp
    num_key_value_heads = model_config["num_key_value_heads"]
    num_key_value_heads_per_rank = num_key_value_heads // tp
    head_dim = model_config.get("head_dim", None) or hidden_size // num_attention_heads

    if metric == EfficiencyMetrics.TFLOPS:
        total_flops = 0
        for s_len in seq_len:
            # # only consider gemm in prefill which is the dominant term
            # if s_len > 1:
            total_flops += 4 * s_len * s_len * num_attention_heads_per_rank * head_dim
        return total_flops / (trace_stats.fmha_avg_time * 1e6)  # in TFLOPs

    elif metric == EfficiencyMetrics.MEM_BANDWIDTH:
        total_context_len = 0
        for c_len, s_len in zip(context_len, seq_len):
            # if s_len == 1:
            total_context_len += c_len
        total_bytes_transferred = total_context_len * num_key_value_heads_per_rank * head_dim * 2 * DTYPE_TO_BYTES[kv_cache_dtype]
        return total_bytes_transferred / (trace_stats.fmha_avg_time * 1e3)  # in GB/s


def safe_mean(time_list):
    return np.mean(time_list) if len(time_list) > 0 else 0.0

def safe_std(time_list):
    return np.std(time_list) if len(time_list) > 0 else 0.0

def parse_kernel_info(trace_events: List, is_moe=False) -> TRACE_STATS:
    print("[INFO] Parsing kernel information from trace events...")
    qkv_gemm_time_list = []
    out_gemm_time_list = []
    router_gemm_time_list = []
    gateup_gemm_time_list = []
    down_gemm_time_list = []
    w13_grouped_gemm_time_list = []
    w2_grouped_gemm_time_list = []

    fmha_time_list = []
    flash_fwd_splitkv_time_list = []
    flash_fwd_splitkv_combine_time_list = []
    norm_time_list = []
    act_time_list = []
    rope_time_list = []
    reshape_and_cache_time_list = []
    copy_time_list = []
    allreduce_time_list = []
    dynamic_per_token_scaled_fp8_quant_time_list = []
    other_time_list = []

    stats = TRACE_STATS()

    for event in trace_events:
        if isinstance(event, dict):
            if 'cat' in event.keys() and event['cat'] == 'kernel':
                kernel_name = event['name'].lower()
                duration = event["dur"]
                if ('fmha' in kernel_name) or ('paged_attention' in kernel_name):
                    stats.total_fmha_time += duration
                    fmha_time_list.append(duration)
                    stats.total_fmha_kernels += 1
                elif 'gemm' in kernel_name:
                    if is_moe:
                        if stats.total_gemm_kernels % 7 == 0:
                            stats.total_qkv_gemm_time += duration
                            qkv_gemm_time_list.append(duration)
                            stats.total_qkv_gemm_kernels += 1
                        elif stats.total_gemm_kernels % 7 == 1:
                            stats.total_out_gemm_time += duration
                            out_gemm_time_list.append(duration)
                            stats.total_out_gemm_kernels += 1
                        elif stats.total_gemm_kernels % 7 == 2:
                            stats.total_router_gemm_time += duration
                            router_gemm_time_list.append(duration)
                            stats.total_router_gemm_kernels += 1
                        elif stats.total_gemm_kernels % 7 == 3:
                            stats.total_gateup_gemm_time += duration
                            gateup_gemm_time_list.append(duration)
                            stats.total_gateup_gemm_kernels += 1
                        elif stats.total_gemm_kernels % 7 == 4:
                            stats.total_down_gemm_time += duration
                            down_gemm_time_list.append(duration)
                            stats.total_down_gemm_kernels += 1
                        elif stats.total_gemm_kernels % 7 == 5:
                            stats.total_w13_grouped_gemm_time += duration
                            w13_grouped_gemm_time_list.append(duration)
                            stats.total_w13_grouped_gemm_kernels += 1
                        elif stats.total_gemm_kernels % 7 == 6:
                            stats.total_w2_grouped_gemm_time += duration
                            w2_grouped_gemm_time_list.append(duration)
                            stats.total_w2_grouped_gemm_kernels += 1
                    else:
                        if stats.total_gemm_kernels % 4 == 0:
                            stats.total_qkv_gemm_time += duration
                            qkv_gemm_time_list.append(duration)
                            stats.total_qkv_gemm_kernels += 1
                        elif stats.total_gemm_kernels % 4 == 1:
                            stats.total_out_gemm_time += duration
                            out_gemm_time_list.append(duration)
                            stats.total_out_gemm_kernels += 1
                        elif stats.total_gemm_kernels % 4 == 2:
                            stats.total_gateup_gemm_time += duration
                            gateup_gemm_time_list.append(duration)
                            stats.total_gateup_gemm_kernels += 1
                        elif stats.total_gemm_kernels % 4 == 3:
                            stats.total_down_gemm_time += duration
                            down_gemm_time_list.append(duration)
                            stats.total_down_gemm_kernels += 1
                    stats.total_gemm_kernels += 1
                elif 'flash_fwd_splitkv_combine' in kernel_name:
                    stats.total_flash_fwd_splitkv_combine_time += duration
                    flash_fwd_splitkv_combine_time_list.append(duration)
                    stats.total_flash_fwd_splitkv_combine_kernels += 1
                elif 'flash_fwd_splitkv' in kernel_name:
                    stats.total_flash_fwd_splitkv_time += duration
                    flash_fwd_splitkv_time_list.append(duration)
                    stats.total_flash_fwd_splitkv_kernels += 1
                elif 'norm' in kernel_name:
                    stats.total_norm_time += duration
                    norm_time_list.append(duration)
                    stats.total_norm_kernels += 1
                elif 'allreduce' in kernel_name:
                    stats.total_allreduce_time += duration
                    allreduce_time_list.append(duration)
                    stats.total_allreduce_kernels += 1
                elif 'and_mul' in kernel_name:
                    stats.total_act_time += duration
                    act_time_list.append(duration)
                    stats.total_act_kernels += 1
                elif 'rotary' in kernel_name:
                    stats.total_rope_time += duration
                    rope_time_list.append(duration)
                    stats.total_rope_kernels += 1
                elif 'reshapeandcache' in kernel_name or 'reshape_and_cache' in kernel_name:
                    stats.total_reshape_and_cache_time += duration
                    reshape_and_cache_time_list.append(duration)
                    stats.total_reshape_and_cache_kernels += 1
                elif 'copy' in kernel_name and 'globalrange' not in kernel_name:
                    stats.total_copy_time += duration
                    copy_time_list.append(duration)
                    stats.total_copy_kernels += 1
                elif 'dynamic_per_token_scaled_fp8_quant_kernel' in kernel_name:
                    stats.total_dynamic_per_token_scaled_fp8_quant_time += duration
                    dynamic_per_token_scaled_fp8_quant_time_list.append(duration)
                    stats.total_dynamic_per_token_scaled_fp8_quant_kernels += 1
                else:
                    stats.total_other_time += duration
                    other_time_list.append(duration)
                    stats.total_other_kernels += 1
                stats.total_kernels += 1
                stats.total_kernel_time += duration

    stats.qkv_gemm_avg_time = safe_mean(qkv_gemm_time_list)
    stats.out_gemm_avg_time = safe_mean(out_gemm_time_list)
    stats.router_gemm_avg_time = safe_mean(router_gemm_time_list)
    stats.gateup_gemm_avg_time = safe_mean(gateup_gemm_time_list)
    stats.down_gemm_avg_time = safe_mean(down_gemm_time_list)
    stats.w13_grouped_gemm_avg_time = safe_mean(w13_grouped_gemm_time_list)
    stats.w2_grouped_gemm_avg_time = safe_mean(w2_grouped_gemm_time_list)
    stats.fmha_avg_time = safe_mean(fmha_time_list)
    stats.flash_fwd_splitkv_avg_time = safe_mean(flash_fwd_splitkv_time_list)
    stats.flash_fwd_splitkv_combine_avg_time = safe_mean(flash_fwd_splitkv_combine_time_list)
    stats.norm_avg_time = safe_mean(norm_time_list)
    stats.act_avg_time = safe_mean(act_time_list)
    stats.rope_avg_time = safe_mean(rope_time_list)
    stats.reshape_and_cache_avg_time = safe_mean(reshape_and_cache_time_list)
    stats.copy_avg_time = safe_mean(copy_time_list)
    stats.allreduce_avg_time = safe_mean(allreduce_time_list)
    stats.dynamic_per_token_scaled_fp8_quant_avg_time = safe_mean(dynamic_per_token_scaled_fp8_quant_time_list)
    stats.other_avg_time = safe_mean(other_time_list)
    stats.total_avg_time = stats.qkv_gemm_avg_time + stats.out_gemm_avg_time + stats.gateup_gemm_avg_time + stats.down_gemm_avg_time + \
        stats.fmha_avg_time + stats.norm_avg_time + stats.act_avg_time + stats.rope_avg_time + \
        stats.reshape_and_cache_avg_time + stats.copy_avg_time + stats.allreduce_avg_time + stats.other_avg_time

    stats.qkv_gemm_time_std = safe_std(qkv_gemm_time_list)
    stats.out_gemm_time_std = safe_std(out_gemm_time_list)
    stats.router_gemm_time_std = safe_std(router_gemm_time_list)
    stats.gateup_gemm_time_std = safe_std(gateup_gemm_time_list)
    stats.down_gemm_time_std = safe_std(down_gemm_time_list)
    stats.w13_grouped_gemm_time_std = safe_std(w13_grouped_gemm_time_list)
    stats.w2_grouped_gemm_time_std = safe_std(w2_grouped_gemm_time_list)
    stats.fmha_time_std = safe_std(fmha_time_list)
    stats.flash_fwd_splitkv_time_std = safe_std(flash_fwd_splitkv_time_list)
    stats.flash_fwd_splitkv_combine_time_std = safe_std(flash_fwd_splitkv_combine_time_list)
    stats.norm_time_std = safe_std(norm_time_list)
    stats.act_time_std = safe_std(act_time_list)
    stats.rope_time_std = safe_std(rope_time_list)
    stats.reshape_and_cache_time_std = safe_std(reshape_and_cache_time_list)
    stats.copy_time_std = safe_std(copy_time_list)
    stats.allreduce_time_std = safe_std(allreduce_time_list)
    stats.dynamic_per_token_scaled_fp8_quant_time_std = safe_std(dynamic_per_token_scaled_fp8_quant_time_list)
    stats.other_time_std = safe_std(other_time_list)

    return stats

def print_onlyif_appeared(kernel_name, num_call, total_time, total_time_percentage, avg_time, std_time, tflops_or_membw, tflops_or_membw_utilization):
    if num_call <= 0:
        return
    content = f"{kernel_name:<25} {num_call:<10} {total_time:<15.2f} {total_time_percentage:<15.2f} {avg_time:<15.2f} {std_time:<15.2f} "
    if tflops_or_membw_utilization == "N/A":
        content += f"{tflops_or_membw:<10} {tflops_or_membw_utilization:<10}"
    else:
        content += f"{tflops_or_membw:<10.2f} {tflops_or_membw_utilization:<10.2f}"
    print(content)

def print_trace_stats(stats: TRACE_STATS, metric: EfficiencyMetrics):
    header = f"{'Kernel':<25} {'calls':<10} {'Total time(us)':<15} {'Total Time(%)':<15} {'Avg Time(us)':<15} {'Std Dev(us)':<15}"
    header += f" {'TFlops':<10}" if metric == EfficiencyMetrics.TFLOPS else f" {'Mem BW':<10}"
    header += f" {'Utilization(%)':<10}"
    print(header)
    print("=" * len(header))
    print_onlyif_appeared('qkv_gemm', stats.total_qkv_gemm_kernels, stats.total_qkv_gemm_time, stats.total_qkv_gemm_time/stats.total_kernel_time*100, stats.qkv_gemm_avg_time, stats.qkv_gemm_time_std, stats.qkv_gemm_tflops_or_mem_bandwidth, stats.qkv_gemm_tflops_or_mem_bandwidth_utilization)
    print_onlyif_appeared('out_gemm', stats.total_out_gemm_kernels, stats.total_out_gemm_time, stats.total_out_gemm_time/stats.total_kernel_time*100, stats.out_gemm_avg_time, stats.out_gemm_time_std, stats.out_gemm_tflops_or_mem_bandwidth, stats.out_gemm_tflops_or_mem_bandwidth_utilization)
    print_onlyif_appeared('router_gemm', stats.total_router_gemm_kernels, stats.total_router_gemm_time, stats.total_router_gemm_time/stats.total_kernel_time*100, stats.router_gemm_avg_time, stats.router_gemm_time_std, stats.router_gemm_tflops_or_mem_bandwidth, stats.router_gemm_tflops_or_mem_bandwidth_utilization)
    print_onlyif_appeared('gate_up_gemm', stats.total_gateup_gemm_kernels, stats.total_gateup_gemm_time, stats.total_gateup_gemm_time/stats.total_kernel_time*100, stats.gateup_gemm_avg_time, stats.gateup_gemm_time_std, stats.gateup_gemm_tflops_or_mem_bandwidth, stats.gateup_gemm_tflops_or_mem_bandwidth_utilization)
    print_onlyif_appeared('down_gemm', stats.total_down_gemm_kernels, stats.total_down_gemm_time, stats.total_down_gemm_time/stats.total_kernel_time*100, stats.down_gemm_avg_time, stats.down_gemm_time_std, stats.down_gemm_tflops_or_mem_bandwidth, stats.down_gemm_tflops_or_mem_bandwidth_utilization)
    print_onlyif_appeared('w13_grouped_gemm', stats.total_w13_grouped_gemm_kernels, stats.total_w13_grouped_gemm_time, stats.total_w13_grouped_gemm_time/stats.total_kernel_time*100, stats.w13_grouped_gemm_avg_time, stats.w13_grouped_gemm_time_std, stats.w13_grouped_gemm_tflops_or_mem_bandwidth, stats.w13_grouped_gemm_tflops_or_mem_bandwidth_utilization)
    print_onlyif_appeared('w2_grouped_gemm', stats.total_w2_grouped_gemm_kernels, stats.total_w2_grouped_gemm_time, stats.total_w2_grouped_gemm_time/stats.total_kernel_time*100, stats.w2_grouped_gemm_avg_time, stats.w2_grouped_gemm_time_std, stats.w2_grouped_gemm_tflops_or_mem_bandwidth, stats.w2_grouped_gemm_tflops_or_mem_bandwidth_utilization)
    print_onlyif_appeared('fmha', stats.total_fmha_kernels, stats.total_fmha_time, stats.total_fmha_time/stats.total_kernel_time*100, stats.fmha_avg_time, stats.fmha_time_std, stats.fmha_tflops_or_mem_bandwidth, stats.fmha_tflops_or_mem_bandwidth_utilization)
    print_onlyif_appeared('flash_fwd_splitkv', stats.total_flash_fwd_splitkv_kernels, stats.total_flash_fwd_splitkv_time, stats.total_flash_fwd_splitkv_time/stats.total_kernel_time*100, stats.flash_fwd_splitkv_avg_time, stats.flash_fwd_splitkv_time_std, stats.tflops_or_mem_bandwidth_unavailble, stats.tflops_or_mem_bandwidth_unavailble)
    print_onlyif_appeared('flash_fwd_splitkv_combine', stats.total_flash_fwd_splitkv_combine_kernels, stats.total_flash_fwd_splitkv_combine_time, stats.total_flash_fwd_splitkv_combine_time/stats.total_kernel_time*100, stats.flash_fwd_splitkv_combine_avg_time, stats.flash_fwd_splitkv_combine_time_std, stats.tflops_or_mem_bandwidth_unavailble, stats.tflops_or_mem_bandwidth_unavailble)
    print_onlyif_appeared('norm', stats.total_norm_kernels, stats.total_norm_time, stats.total_norm_time/stats.total_kernel_time*100, stats.norm_avg_time, stats.norm_time_std, stats.tflops_or_mem_bandwidth_unavailble, stats.tflops_or_mem_bandwidth_unavailble)
    print_onlyif_appeared('silu_and_mul', stats.total_act_kernels, stats.total_act_time, stats.total_act_time/stats.total_kernel_time*100, stats.act_avg_time, stats.act_time_std, stats.tflops_or_mem_bandwidth_unavailble, stats.tflops_or_mem_bandwidth_unavailble)
    print_onlyif_appeared('rope', stats.total_rope_kernels, stats.total_rope_time, stats.total_rope_time/stats.total_kernel_time*100, stats.rope_avg_time, stats.rope_time_std, stats.tflops_or_mem_bandwidth_unavailble, stats.tflops_or_mem_bandwidth_unavailble)
    print_onlyif_appeared('reshape_and_cache', stats.total_reshape_and_cache_kernels, stats.total_reshape_and_cache_time, stats.total_reshape_and_cache_time/stats.total_kernel_time*100, stats.reshape_and_cache_avg_time, stats.reshape_and_cache_time_std, stats.tflops_or_mem_bandwidth_unavailble, stats.tflops_or_mem_bandwidth_unavailble)
    print_onlyif_appeared('copy', stats.total_copy_kernels, stats.total_copy_time, stats.total_copy_time/stats.total_kernel_time*100, stats.copy_avg_time, stats.copy_time_std, stats.tflops_or_mem_bandwidth_unavailble, stats.tflops_or_mem_bandwidth_unavailble)
    print_onlyif_appeared('all_reduce', stats.total_allreduce_kernels, stats.total_allreduce_time, stats.total_allreduce_time/stats.total_kernel_time*100, stats.allreduce_avg_time, stats.allreduce_time_std, stats.tflops_or_mem_bandwidth_unavailble, stats.tflops_or_mem_bandwidth_unavailble)
    print_onlyif_appeared('dynamic_fp8_quant', stats.total_dynamic_per_token_scaled_fp8_quant_kernels, stats.total_dynamic_per_token_scaled_fp8_quant_time, stats.total_dynamic_per_token_scaled_fp8_quant_time/stats.total_kernel_time*100, stats.dynamic_per_token_scaled_fp8_quant_avg_time, stats.dynamic_per_token_scaled_fp8_quant_time_std, stats.tflops_or_mem_bandwidth_unavailble, stats.tflops_or_mem_bandwidth_unavailble)
    print_onlyif_appeared('other', stats.total_other_kernels, stats.total_other_time, stats.total_other_time/stats.total_kernel_time*100, stats.other_avg_time, stats.other_time_std, stats.tflops_or_mem_bandwidth_unavailble, stats.tflops_or_mem_bandwidth_unavailble)

    print("=" * len(header))
    print(f"{'Total kernels:':<25} {stats.total_kernels:<10}")
    print(f"{'Total kernel time(us):':<25} {stats.total_kernel_time:<10.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse vLLM trace json file.")
    parser.add_argument(
        "--trace_json_file",
        type=str,
        help="Path to the vLLM trace json file.",
    )
    parser.add_argument(
        "--scheinfo_json_file",
        type=str,
        help="Path to the vLLM scheinfo json file with profiler on.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="xpu-bmg",
        help="Device name, e.g., xpu-bmg, cuda-4090d",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3-8b",
        help="Model name, e.g., llama3-8b, qwen2.5-32b, qwen2.5-14b, llama3-70b, llama4-scout, etc.",
    )
    parser.add_argument(
        "--weight_dtype",
        type=str,
        default="fp8",
        help="Weight data type, e.g., fp8, fp16, fp32, int8, int4.",
    )
    parser.add_argument(
        "--kv_dtype",
        type=str,
        default="fp16",
        help="KV cache data type, e.g., fp8, fp16, int8.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=4,
        help="Number of ranks for tensor parallelism (TP).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["tflops", "mem_bandwidth"],
        default="tflops",
        help="Efficiency metric to compute: tflops or mem_bandwidth.",
    )
    args = parser.parse_args()

    is_moe = args.model in ["llama4-scout"]

    m = int(args.trace_json_file.split("/")[-1].split(".")[0].split("_")[-1])
    step = int(args.trace_json_file.split("/")[-1].split(".")[0].split("_")[-3])

    config = load_model_config(args)
    trace_events = load_trace_json(args)
    print("Trace events loaded successfully.")
    with open(args.scheinfo_json_file, "r") as f:
        scheinfo = json.load(f)

    num_experts = 1
    if is_moe:
        real_num_experts = scheinfo["steps"][step - 1]["real_num_experts"]
        if args.model == "llama4-scout":
            num_experts = config["num_local_experts"]
        else:
            raise ValueError(f"Please provide num_experts for moe model {args.model}!")
        print(f"num_experts: {num_experts}, real_num_experts: {real_num_experts}, ")

    gemm_shapes = get_gemm_shape(config, m, tp=args.tp, is_moe=is_moe, num_experts=num_experts)
    trace_stats = parse_kernel_info(trace_events, is_moe=is_moe)
    gemm_bandwidth_or_tflops = compute_gemm_tflops_or_mem_bandwidth(
        trace_stats, gemm_shapes, args.weight_dtype, EfficiencyMetrics[args.metric.upper()], is_moe=is_moe, real_num_experts=real_num_experts
    )
    context_lens = scheinfo["steps"][step - 1]["context_lens"]
    seq_lens = scheinfo["steps"][step - 1]["tokens"]
    fmha_bandwidth_or_tflops = compute_fmha_tflops_or_membandwidth(
        trace_stats, context_lens, seq_lens, config, args.tp, args.kv_dtype, EfficiencyMetrics[args.metric.upper()]
    )
    trace_stats.fmha_tflops_or_mem_bandwidth = fmha_bandwidth_or_tflops
    trace_stats.qkv_gemm_tflops_or_mem_bandwidth = gemm_bandwidth_or_tflops[0]
    trace_stats.out_gemm_tflops_or_mem_bandwidth = gemm_bandwidth_or_tflops[1]
    trace_stats.router_gemm_tflops_or_mem_bandwidth = gemm_bandwidth_or_tflops[2]
    trace_stats.gateup_gemm_tflops_or_mem_bandwidth = gemm_bandwidth_or_tflops[3]
    trace_stats.down_gemm_tflops_or_mem_bandwidth = gemm_bandwidth_or_tflops[4]
    trace_stats.w13_grouped_gemm_tflops_or_mem_bandwidth = gemm_bandwidth_or_tflops[5]
    trace_stats.w2_grouped_gemm_tflops_or_mem_bandwidth = gemm_bandwidth_or_tflops[6]

    device = args.device
    assert device in ["xpu-bmg", "cuda-4090d"], f"Unsupported device {device}!"

    TFLOPS_PEAK = HW_CONFIG[device]["tflops"]
    MEM_BANDWIDTH_PEAK = HW_CONFIG[device]["mem_bw"]
    print(f"[WARNING] Pls confirm the HW spec for {device}: tflops = {TFLOPS_PEAK}, mem_bw = {MEM_BANDWIDTH_PEAK}")

    trace_stats.qkv_gemm_tflops_or_mem_bandwidth_utilization = (
        trace_stats.qkv_gemm_tflops_or_mem_bandwidth / TFLOPS_PEAK * 100
        if args.metric == "tflops" else
        trace_stats.qkv_gemm_tflops_or_mem_bandwidth / MEM_BANDWIDTH_PEAK * 100
    )
    trace_stats.out_gemm_tflops_or_mem_bandwidth_utilization = (
        trace_stats.out_gemm_tflops_or_mem_bandwidth / TFLOPS_PEAK * 100
        if args.metric == "tflops" else
        trace_stats.out_gemm_tflops_or_mem_bandwidth / MEM_BANDWIDTH_PEAK * 100
    )
    trace_stats.router_gemm_tflops_or_mem_bandwidth_utilization = (
        trace_stats.router_gemm_tflops_or_mem_bandwidth / TFLOPS_PEAK * 100
        if args.metric == "tflops" else
        trace_stats.router_gemm_tflops_or_mem_bandwidth / MEM_BANDWIDTH_PEAK * 100
    )
    trace_stats.gateup_gemm_tflops_or_mem_bandwidth_utilization = (
        trace_stats.gateup_gemm_tflops_or_mem_bandwidth / TFLOPS_PEAK * 100
        if args.metric == "tflops" else
        trace_stats.gateup_gemm_tflops_or_mem_bandwidth / MEM_BANDWIDTH_PEAK * 100
    )
    trace_stats.down_gemm_tflops_or_mem_bandwidth_utilization = (
        trace_stats.down_gemm_tflops_or_mem_bandwidth / TFLOPS_PEAK * 100
        if args.metric == "tflops" else
        trace_stats.down_gemm_tflops_or_mem_bandwidth / MEM_BANDWIDTH_PEAK * 100
    )
    trace_stats.fmha_tflops_or_mem_bandwidth_utilization = (
        trace_stats.fmha_tflops_or_mem_bandwidth / TFLOPS_PEAK * 100
        if args.metric == "tflops" else
        trace_stats.fmha_tflops_or_mem_bandwidth / MEM_BANDWIDTH_PEAK * 100
    )
    trace_stats.w13_grouped_gemm_tflops_or_mem_bandwidth_utilization = (
        trace_stats.w13_grouped_gemm_tflops_or_mem_bandwidth / TFLOPS_PEAK * 100
        if args.metric == "tflops" else
        trace_stats.w13_grouped_gemm_tflops_or_mem_bandwidth / MEM_BANDWIDTH_PEAK * 100
    )
    trace_stats.w2_grouped_gemm_tflops_or_mem_bandwidth_utilization = (
        trace_stats.w2_grouped_gemm_tflops_or_mem_bandwidth / TFLOPS_PEAK * 100
        if args.metric == "tflops" else
        trace_stats.w2_grouped_gemm_tflops_or_mem_bandwidth / MEM_BANDWIDTH_PEAK * 100
    )

    print_trace_stats(trace_stats, EfficiencyMetrics[args.metric.upper()])
