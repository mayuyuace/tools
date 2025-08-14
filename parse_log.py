import re
import json
import argparse

def read_log_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return lines

def save_scheduling_info(sche_info, output_file):
    with open(output_file, "w") as f:
        json.dump(sche_info, f, indent=4)
    print(f"Schedule info parsed and saved to {output_file}.")

"""
sche_info = {
    "steps": [
    { 'total_requests_num': int,
    'total_num_scheduled_tokens': int,
    'context_lens': [int, int, ...],
    'tokens': [int, int, ...],
    'prefill_request_num': int,
    'decode_request_num': int,
    'prefill_tokens': int,
    'decode_tokens': int,
    'model_forward_time': float,
    'duration': float,
    'step_id': int,
    }]
}
"""
def extract_scheduling_info(lines):
    sche_info = {
        "displayTimeUnit": 'ms',
        "steps": [],
    }

    for line_start_id in range(len(lines)):
        line = lines[line_start_id]
        if 'total_requests_num' in line:
            step_info = {
                'total_requests_num': 0,
                'total_num_scheduled_tokens': 0,
                'context_lens': [],
                'tokens': [],
                'prefill_request_num': 0,
                'decode_request_num': 0,
                'prefill_tokens': 0,
                'decode_tokens': 0,
                'model_forward_time': 0.0,
                'duration': 0.0,
                'step_id': -1
            }
            total_requests_num = int(line.split(':')[-1].strip())
            step_info['total_requests_num'] = total_requests_num
            for line_id in range(line_start_id + 1, len(lines)):
                cur_line = lines[line_id]
                if 'total_requests_num' in cur_line:
                    line_start_id = line_id
                    break
                if 'total_num_scheduled_tokens' in cur_line:
                    total_num_scheduled_tokens = int(cur_line.split(':')[-1].strip())
                    step_info['total_num_scheduled_tokens'] = total_num_scheduled_tokens
                elif 'context_len' in cur_line:
                    context_len = int(cur_line.split(':')[-1].strip())
                    step_info['context_lens'].append(context_len)
                elif 'num_scheduled_token' in cur_line:
                    num_scheduled_token = int(cur_line.split(':')[-1].strip())
                    step_info['tokens'].append(num_scheduled_token)
                    if num_scheduled_token > 1:
                        step_info['prefill_request_num'] += 1
                        step_info['prefill_tokens'] += num_scheduled_token
                    else:
                        step_info['decode_request_num'] += 1
                        step_info['decode_tokens'] += num_scheduled_token
                elif 'step' in cur_line:
                    temp = cur_line.split("=")[-2]
                    step = int(re.split(':| ', temp)[3].strip())
                    model_forward_time = float(cur_line.split("=")[-1].strip()) * 1000
                    step_info['step_id'] = step
                    step_info['model_forward_time'] = model_forward_time
                elif 'execution time' in cur_line:
                    execution_time = float(cur_line.split(":")[-1].split(" ")[1].strip())
                    step_info['duration'] = execution_time
            assert len(step_info['context_lens']) == step_info['total_requests_num'], f"total_requests_num {step_info['total_requests_num']} does not match context_lens length {len(step_info['context_lens'])}"
            assert len(step_info['tokens']) == step_info['total_requests_num'], f"total_requests_num {step_info['total_requests_num']} does not match tokens length {len(step_info['tokens'])}"
            assert sum(step_info['tokens']) == step_info['total_num_scheduled_tokens'], f"total_num_scheduled_tokens {step_info['total_num_scheduled_tokens']} does not match sum of tokens {sum(step_info['tokens'])}"
            assert step_info['prefill_request_num'] + step_info['decode_request_num'] == step_info['total_requests_num'], f"prefill_request_num {step_info['prefill_request_num']} + decode_request_num {step_info['decode_request_num']} does not equal total_requests_num {step_info['total_requests_num']}"
            assert step_info['prefill_tokens'] + step_info['decode_tokens'] == step_info['total_num_scheduled_tokens'], f"prefill_tokens {step_info['prefill_tokens']} + decode_tokens {step_info['decode_tokens']} does not equal total_num_scheduled_tokens {step_info['total_num_scheduled_tokens']}"
            assert step_info['decode_request_num'] == step_info['decode_tokens'], f"decode_request_num {step_info['decode_request_num']} does not equal decode_tokens {step_info['decode_tokens']}"

            if step_info['step_id'] != -1:
                sche_info['steps'].append(step_info)
    return sche_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse vLLM scheduling info from log file.")
    parser.add_argument("--log_file", type=str, help="Path to the vLLM server log file.")
    parser.add_argument("--output_file", type=str, default="sche_info.json", help="Path to save the parsed scheduling info.")
    args = parser.parse_args()

    lines = read_log_file(args.log_file)
    sche_info = extract_scheduling_info(lines)
    save_scheduling_info(sche_info, args.output_file)
