# tools
## Usage
### Convert the server log to json file
```shell
python parse_log.py --log_file scheinfo.log --output_file scheinfo.json
```

### Parse the trace file to kernel perf breakdown
```shell
python parse_trace.py --trace_json_file trace.json --scheinfo_json_file scheinfo.json --model qwen2.5-32b --weight_dtype fp8 --kv_dtype fp16 --tp 4 --metric tflops (or mem_bandwidth) --device xpu-bmg
```