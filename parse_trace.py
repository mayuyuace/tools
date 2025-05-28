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
    "int4": 0.5,
}

TFLOPS_PEAK = 98
MEM_BANDWIDTH_PEAK = 451

class EfficiencyMetrics(Enum):
    TFLOPS = 0
    MEM_BANDWIDTH = 1


@dataclass
class TRACE_STATS():
    total_kernels: int = 0
    total_gemm_kernels: int = 0
    total_qkv_gemm_kernels: int = 0
    total_out_gemm_kernels: int = 0
    total_gateup_gemm_kernels: int = 0
    total_down_gemm_kernels: int = 0
    total_fmha_kernels: int = 0
    total_norm_kernels: int = 0
    total_act_kernels: int = 0
    total_rope_kernels: int = 0
    total_reshape_and_cache_kernels: int = 0
    total_copy_kernels: int = 0
    total_allreduce_kernels: int = 0
    total_other_kernels: int = 0

    total_kernel_time: float = 0.0
    total_qkv_gemm_time: float = 0.0
    total_out_gemm_time: float = 0.0
    total_gateup_gemm_time: float = 0.0
    total_down_gemm_time: float = 0.0
    total_fmha_time: float = 0.0
    total_norm_time: float = 0.0
    total_act_time: float = 0.0
    total_rope_time: float = 0.0
    total_reshape_and_cache_time: float = 0.0
    total_copy_time: float = 0.0
    total_allreduce_time: float = 0.0
    total_other_time: float = 0.0
    total_avg_time: float = 0.0

    qkv_gemm_avg_time: float = 0.0
    qkv_gemm_time_std: float = 0.0
    out_gemm_avg_time: float = 0.0
    out_gemm_time_std: float = 0.0
    gateup_gemm_avg_time: float = 0.0
    gateup_gemm_time_std: float = 0.0
    down_gemm_avg_time: float = 0.0
    down_gemm_time_std: float = 0.0
    fmha_avg_time: float = 0.0
    fmha_time_std: float = 0.0
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
    other_avg_time: float = 0.0
    other_time_std: float = 0.0

    qkv_gemm_tflops_or_mem_bandwidth: float = 0.0
    qkv_gemm_tflops_or_mem_bandwidth_utilization: float = 0.0
    out_gemm_tflops_or_mem_bandwidth: float = 0.0
    out_gemm_tflops_or_mem_bandwidth_utilization: float = 0.0
    gateup_gemm_tflops_or_mem_bandwidth: float = 0.0
    gateup_gemm_tflops_or_mem_bandwidth_utilization: float = 0.0
    down_gemm_tflops_or_mem_bandwidth: float = 0.0
    down_gemm_tflops_or_mem_bandwidth_utilization: float = 0.0

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
    else:
        raise ValueError(f"Unsupported model: {args.model}. Only supports llama3-8b and qwen2.5-32b for now! Please provide a valid model name.")
    print(f"Loading model config from {config_file}...")
    # print(config)
    return config


def load_trace_json(args):
    print(f"Loading trace json file from {args.trace_json_file}...")
    with open(args.trace_json_file, "r") as f:
        trace_dict = json.load(f)
    return trace_dict["traceEvents"]


def get_gemm_shape(config, m, tp):
    hidden_size = config["hidden_size"]
    num_attention_heads = config["num_attention_heads"]
    num_key_value_heads = config["num_key_value_heads"]
    head_dim = config.get("head_dim", None) or hidden_size // num_attention_heads
    intermediate_size = config["intermediate_size"]

    gemm_shapes = []
    # qkv gemm shape
    qkv_gemm_shape = (m, hidden_size, (num_attention_heads + num_key_value_heads * 2) * head_dim // tp)
    gemm_shapes.append(qkv_gemm_shape)
    # out gemm shape
    out_gemm_shape = (m, num_attention_heads * head_dim // tp, hidden_size)
    gemm_shapes.append(out_gemm_shape)
    # gateup gemm shape
    gateup_gemm_shape = (m, hidden_size, intermediate_size * 2 // tp)
    gemm_shapes.append(gateup_gemm_shape)
    # down gemm shape
    down_gemm_shape = (m, intermediate_size // tp, hidden_size)
    gemm_shapes.append(down_gemm_shape)

    return gemm_shapes

def compute_tflops_or_mem_bandwidth(trace_stats: TRACE_STATS, gemm_shape_list: List, weight_dtype: str, metric: EfficiencyMetrics):
    print("[INFO] Computing TFlops or memory bandwidth...")
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
        for gemm_time, gemm_shape in zip(gemm_time_list, gemm_shape_list):
            m, k, n = gemm_shape
            bytes_transferred = k * n * DTYPE_TO_BYTES[weight_dtype]
            bandwidth = bytes_transferred / (gemm_time * 1e3)
            gemm_bandwidth.append(bandwidth)
        return gemm_bandwidth


def parse_kernel_info(trace_events: List):
    print("[INFO] Parsing kernel information from trace events...")
    qkv_gemm_time_list = []
    out_gemm_time_list = []
    gateup_gemm_time_list = []
    down_gemm_time_list = []

    fmha_time_list = []
    norm_time_list = []
    act_time_list = []
    rope_time_list = []
    reshape_and_cache_time_list = []
    copy_time_list = []
    allreduce_time_list = []
    other_time_list = []

    stats = TRACE_STATS()

    for event in trace_events:
        if isinstance(event, dict):
            if 'cat' in event.keys() and event['cat'] == 'kernel':
                kernel_name = event['name'].lower()
                duration = event["dur"]
                if 'gemm' in kernel_name:
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
                elif 'fmha' in kernel_name:
                    stats.total_fmha_time += duration
                    fmha_time_list.append(duration)
                    stats.total_fmha_kernels += 1
                elif 'norm' in kernel_name:
                    stats.total_norm_time += duration
                    norm_time_list.append(duration)
                    stats.total_norm_kernels += 1
                elif 'allreduce' in kernel_name:
                    stats.total_allreduce_time += duration
                    allreduce_time_list.append(duration)
                    stats.total_allreduce_kernels += 1
                elif 'op_and_mul' in kernel_name:
                    stats.total_act_time += duration
                    act_time_list.append(duration)
                    stats.total_act_kernels += 1
                elif 'rotaryembedding' in kernel_name:
                    stats.total_rope_time += duration
                    rope_time_list.append(duration)
                    stats.total_rope_kernels += 1
                elif 'reshapeandcache' in kernel_name:
                    stats.total_reshape_and_cache_time += duration
                    reshape_and_cache_time_list.append(duration)
                    stats.total_reshape_and_cache_kernels += 1
                elif 'copy' in kernel_name and 'globalrange' not in kernel_name:
                    stats.total_copy_time += duration
                    copy_time_list.append(duration)
                    stats.total_copy_kernels += 1
                else:
                    stats.total_other_time += duration
                    other_time_list.append(duration)
                    stats.total_other_kernels += 1
                stats.total_kernels += 1
                stats.total_kernel_time += duration

    stats.qkv_gemm_avg_time = np.mean(qkv_gemm_time_list)
    stats.out_gemm_avg_time = np.mean(out_gemm_time_list)
    stats.gateup_gemm_avg_time = np.mean(gateup_gemm_time_list)
    stats.down_gemm_avg_time = np.mean(down_gemm_time_list)
    stats.fmha_avg_time = np.mean(fmha_time_list)
    stats.norm_avg_time = np.mean(norm_time_list)
    stats.act_avg_time = np.mean(act_time_list)
    stats.rope_avg_time = np.mean(rope_time_list)
    stats.reshape_and_cache_avg_time = np.mean(reshape_and_cache_time_list)
    stats.copy_avg_time = np.mean(copy_time_list)
    stats.allreduce_avg_time = np.mean(allreduce_time_list)
    stats.other_avg_time = np.mean(other_time_list)
    stats.total_avg_time = stats.qkv_gemm_avg_time + stats.out_gemm_avg_time + stats.gateup_gemm_avg_time + stats.down_gemm_avg_time + \
        stats.fmha_avg_time + stats.norm_avg_time + stats.act_avg_time + stats.rope_avg_time + \
        stats.reshape_and_cache_avg_time + stats.copy_avg_time + stats.allreduce_avg_time + stats.other_avg_time

    stats.qkv_gemm_time_std = np.std(qkv_gemm_time_list)
    stats.out_gemm_time_std = np.std(out_gemm_time_list)
    stats.gateup_gemm_time_std = np.std(gateup_gemm_time_list)
    stats.down_gemm_time_std = np.std(down_gemm_time_list)
    stats.fmha_time_std = np.std(fmha_time_list)
    stats.norm_time_std = np.std(norm_time_list)
    stats.act_time_std = np.std(act_time_list)
    stats.rope_time_std = np.std(rope_time_list)
    stats.reshape_and_cache_time_std = np.std(reshape_and_cache_time_list)
    stats.copy_time_std = np.std(copy_time_list)
    stats.allreduce_time_std = np.std(allreduce_time_list)
    stats.other_time_std = np.std(other_time_list)

    return stats


def print_trace_stats(stats: TRACE_STATS, metric: EfficiencyMetrics):
    header = f"{'Kernel':<20} {'calls':<10} {'Total time(us)':<15} {'Total Time(%)':<15} {'Avg Time(us)':<15} {'Std Dev(us)':<15}"
    header += f" {'TFlops':<10}" if metric == EfficiencyMetrics.TFLOPS else f" {'Mem BW':<10}"
    header += f" {'Utilization(%)':<10}"
    print(header)
    print("=" * len(header))
    print(f"{'qkv_gemm':<20} {stats.total_qkv_gemm_kernels:<10} {stats.total_qkv_gemm_time:<15.2f} {stats.total_qkv_gemm_time/stats.total_kernel_time*100:<15.2f} {stats.qkv_gemm_avg_time:<15.2f} {stats.qkv_gemm_time_std:<15.2f} {stats.qkv_gemm_tflops_or_mem_bandwidth:<10.2f} {stats.qkv_gemm_tflops_or_mem_bandwidth_utilization:<10.2f}")
    print(f"{'out_gemm':<20} {stats.total_out_gemm_kernels:<10} {stats.total_out_gemm_time:<15.2f} {stats.total_out_gemm_time/stats.total_kernel_time*100:<15.2f} {stats.out_gemm_avg_time:<15.2f} {stats.out_gemm_time_std:<15.2f} {stats.out_gemm_tflops_or_mem_bandwidth:<10.2f} {stats.out_gemm_tflops_or_mem_bandwidth_utilization:<10.2f}")
    print(f"{'gate_up_gemm':<20} {stats.total_gateup_gemm_kernels:<10} {stats.total_gateup_gemm_time:<15.2f} {stats.total_gateup_gemm_time/stats.total_kernel_time*100:<15.2f} {stats.gateup_gemm_avg_time:<15.2f} {stats.gateup_gemm_time_std:<15.2f} {stats.gateup_gemm_tflops_or_mem_bandwidth:<10.2f} {stats.gateup_gemm_tflops_or_mem_bandwidth_utilization:<10.2f}")
    print(f"{'down_gemm':<20} {stats.total_down_gemm_kernels:<10} {stats.total_down_gemm_time:<15.2f} {stats.total_down_gemm_time/stats.total_kernel_time*100:<15.2f} {stats.down_gemm_avg_time:<15.2f} {stats.down_gemm_time_std:<15.2f} {stats.down_gemm_tflops_or_mem_bandwidth:<10.2f} {stats.down_gemm_tflops_or_mem_bandwidth_utilization:<10.2f}")
    print(f"{'fmha':<20} {stats.total_fmha_kernels:<10} {stats.total_fmha_time:<15.2f} {stats.total_fmha_time/stats.total_kernel_time*100:<15.2f} {stats.fmha_avg_time:<15.2f} {stats.fmha_time_std:<15.2f} {stats.tflops_or_mem_bandwidth_unavailble:<10} {stats.tflops_or_mem_bandwidth_unavailble:<10}")
    print(f"{'norm':<20} {stats.total_norm_kernels:<10} {stats.total_norm_time:<15.2f} {stats.total_norm_time/stats.total_kernel_time*100:<15.2f} {stats.norm_avg_time:<15.2f} {stats.norm_time_std:<15.2f} {stats.tflops_or_mem_bandwidth_unavailble:<10} {stats.tflops_or_mem_bandwidth_unavailble:<10}")
    print(f"{'silu_and_mul':<20} {stats.total_act_kernels:<10} {stats.total_act_time:<15.2f} {stats.total_act_time/stats.total_kernel_time*100:<15.2f} {stats.act_avg_time:<15.2f} {stats.act_time_std:<15.2f} {stats.tflops_or_mem_bandwidth_unavailble:<10} {stats.tflops_or_mem_bandwidth_unavailble:<10}")
    print(f"{'rope':<20} {stats.total_rope_kernels:<10} {stats.total_rope_time:<15.2f} {stats.total_rope_time/stats.total_kernel_time*100:<15.2f} {stats.rope_avg_time:<15.2f} {stats.rope_time_std:<15.2f} {stats.tflops_or_mem_bandwidth_unavailble:<10} {stats.tflops_or_mem_bandwidth_unavailble:<10}")
    print(f"{'reshape_and_cache':<20} {stats.total_reshape_and_cache_kernels:<10} {stats.total_reshape_and_cache_time:<15.2f} {stats.total_reshape_and_cache_time/stats.total_kernel_time*100:<15.2f} {stats.reshape_and_cache_avg_time:<15.2f} {stats.reshape_and_cache_time_std:<15.2f} {stats.tflops_or_mem_bandwidth_unavailble:<10} {stats.tflops_or_mem_bandwidth_unavailble:<10}")
    print(f"{'copy':<20} {stats.total_copy_kernels:<10} {stats.total_copy_time:<15.2f} {stats.total_copy_time/stats.total_kernel_time*100:<15.2f} {stats.copy_avg_time:<15.2f} {stats.copy_time_std:<15.2f} {stats.tflops_or_mem_bandwidth_unavailble:<10} {stats.tflops_or_mem_bandwidth_unavailble:<10}")
    print(f"{'all_reduce':<20} {stats.total_allreduce_kernels:<10} {stats.total_allreduce_time:<15.2f} {stats.total_allreduce_time/stats.total_kernel_time*100:<15.2f} {stats.allreduce_avg_time:<15.2f} {stats.allreduce_time_std:<15.2f} {stats.tflops_or_mem_bandwidth_unavailble:<10} {stats.tflops_or_mem_bandwidth_unavailble:<10}")
    print(f"{'other':<20} {stats.total_other_kernels:<10} {stats.total_other_time:<15.2f} {stats.total_other_time/stats.total_kernel_time*100:<15.2f} {stats.other_avg_time:<15.2f} {stats.other_time_std:<15.2f} {stats.tflops_or_mem_bandwidth_unavailble:<10} {stats.tflops_or_mem_bandwidth_unavailble:<10}")
    print("=" * len(header))
    print(f"{'Total kernels:':<20} {stats.total_kernels:<10}")
    print(f"{'Total kernel time(us):':<20} {stats.total_kernel_time:<10.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse vLLM trace json file.")
    parser.add_argument(
        "--trace_json_file",
        type=str,
        help="Path to the vLLM trace json file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3-8b",
        help="Model name, e.g., llama3-8b, qwen2.5-32b, etc.",
    )
    parser.add_argument(
        "--weight_dtype",
        type=str,
        default="fp8",
        help="Weight data type, e.g., fp8, fp16, fp32, int8, int4.",
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

    m = int(args.trace_json_file.split("/")[-1].split(".")[0].split("_")[-1])

    config = load_model_config(args)
    trace_events = load_trace_json(args)
    print("Trace events loaded successfully.")

    gemm_shapes = get_gemm_shape(config, m, tp=args.tp)
    trace_stats = parse_kernel_info(trace_events)
    gemm_bandwidth_or_tflops = compute_tflops_or_mem_bandwidth(
        trace_stats, gemm_shapes, args.weight_dtype, EfficiencyMetrics[args.metric.upper()]
    )
    
    trace_stats.qkv_gemm_tflops_or_mem_bandwidth = gemm_bandwidth_or_tflops[0]
    trace_stats.out_gemm_tflops_or_mem_bandwidth = gemm_bandwidth_or_tflops[1]
    trace_stats.gateup_gemm_tflops_or_mem_bandwidth = gemm_bandwidth_or_tflops[2]
    trace_stats.down_gemm_tflops_or_mem_bandwidth = gemm_bandwidth_or_tflops[3]

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

    print_trace_stats(trace_stats, EfficiencyMetrics[args.metric.upper()])
