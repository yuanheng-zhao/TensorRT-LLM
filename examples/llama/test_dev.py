import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaTokenizer

import tensorrt_llm
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import ModelConfig, SamplingConfig

from build import get_engine_name  # isort:skip

EOS_TOKEN = 2
PAD_TOKEN = 2


# examples/llama/run.py
def read_config(config_path: Path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    remove_input_padding = config['plugin_config']['remove_input_padding']
    dtype = config['builder_config']['precision']
    tp_size = config['builder_config']['tensor_parallel']
    pp_size = config['builder_config']['pipeline_parallel']
    world_size = tp_size * pp_size
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    num_heads = config['builder_config']['num_heads'] // tp_size
    hidden_size = config['builder_config']['hidden_size'] // tp_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    num_kv_heads = config['builder_config'].get('num_kv_heads', num_heads)
    paged_kv_cache = config['plugin_config']['paged_kv_cache']
    tokens_per_block = config['plugin_config']['tokens_per_block']
    sliding_window_length = config['plugin_config']['sliding_window_length']
    quant_mode = QuantMode(config['builder_config']['quant_mode'])
    if config['builder_config'].get('multi_query_mode', False):
        tensorrt_llm.logger.warning(
            "`multi_query_mode` config is deprecated. Please rebuild the engine."
        )
        num_kv_heads = 1
    num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size
    use_custom_all_reduce = config['plugin_config'].get('use_custom_all_reduce',
                                                        False)

    model_config = ModelConfig(num_heads=num_heads,
                               num_kv_heads=num_kv_heads,
                               hidden_size=hidden_size,
                               vocab_size=vocab_size,
                               num_layers=num_layers,
                               gpt_attention_plugin=use_gpt_attention_plugin,
                               paged_kv_cache=paged_kv_cache,
                               tokens_per_block=tokens_per_block,
                               remove_input_padding=remove_input_padding,
                               dtype=dtype,
                               quant_mode=quant_mode,
                               use_custom_all_reduce=use_custom_all_reduce,
                               sliding_window_length=sliding_window_length)

    return model_config, tp_size, pp_size, dtype


# Simplified versionof parse_input in examples/llama/run.py
def parse_input(input_text: str, tokenizer, end_id: int, remove_input_padding: bool):
    input_tokens = []
    input_tokens.append(
        tokenizer.encode(input_text, add_special_tokens=False))

    input_ids = None
    input_lengths = torch.tensor([len(x) for x in input_tokens],
                                 dtype=torch.int32,
                                 device='cuda')
    if remove_input_padding:
        input_ids = np.concatenate(input_tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.int32,
                                 device='cuda').unsqueeze(0)
    else:
        input_ids = torch.nested.to_padded_tensor(
            torch.nested.nested_tensor(input_tokens, dtype=torch.int32),
            end_id).cuda()

    return input_ids, input_lengths


# simplified version of print_output in exampels/llama/run.py
def print_output(output_ids, input_lengths, max_output_len, tokenizer):
    num_beams = output_ids.size(1)
    for b in range(input_lengths.size(0)):
        inputs = output_ids[b][0][:input_lengths[b]].tolist()
        input_text = tokenizer.decode(inputs)
        print(f'Input: \"{input_text}\"')
        for beam in range(num_beams):
            output_begin = input_lengths[b]
            output_end = input_lengths[b] + max_output_len
            outputs = output_ids[b][beam][output_begin:output_end].tolist()
            output_text = tokenizer.decode(outputs)
            print(f'Output: \"{output_text}\"')


# simplified version of generate in exampels/llama/run.py
# added sliding_window_length
# ignore sliding window when sliding_window_length <= 0
def generate(
    max_output_len: int,
    log_level: str = 'error',
    engine_dir: str = 'llama_outputs',
    input_text: str = 'Born in north-east France, Soyer trained as a',
    tokenizer_dir: str = None,
    num_beams: int = 1,
    sliding_window_length: int = 0
):
    tensorrt_llm.logger.set_level(log_level)
    tensorrt_llm.logger.info(f"[test_dev.py:generate] max_output_len: {max_output_len}; engine_dir: {engine_dir}; tokenizer_dir: {tokenizer_dir}")

    engine_dir = Path(engine_dir)
    config_path = engine_dir / 'config.json'
    model_config, tp_size, pp_size, dtype = read_config(config_path)

    tensorrt_llm.logger.info(f"[test_dev.py:generate] model_config: {model_config}")

    assert pp_size == 1, "Pipeline parallelism is not supported in this script."

    world_size = tp_size * pp_size

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=tp_size,
                                           pp_size=pp_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_dir, legacy=False)

    sampling_config = SamplingConfig(end_id=EOS_TOKEN,
                                     pad_id=PAD_TOKEN,
                                     num_beams=num_beams)

    engine_name = get_engine_name('llama', dtype, tp_size, pp_size,
                                  runtime_rank)
    serialize_path = engine_dir / engine_name
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                     engine_buffer,
                                                     runtime_mapping,
                                                     debug_mode=False,
                                                     debug_tensors_to_save=None)
    if runtime_rank == 0:
        print(f"Running the {dtype} engine ...")

    input_ids, input_lengths = parse_input(input_text, tokenizer, EOS_TOKEN,
                                           model_config.remove_input_padding)

    max_input_length = torch.max(input_lengths).item()
    # setup sliding window length (e.g. cache size) for the session
    decoder.setup(input_lengths.size(0), max_input_length, max_output_len,
                  num_beams, sliding_window_length)

    output_gen_ids = decoder.decode(input_ids,
                                    input_lengths,
                                    sampling_config)
    torch.cuda.synchronize()
    output_ids = output_gen_ids
    if runtime_rank == 0:
        print_output(output_ids, input_lengths, max_output_len, tokenizer)


# Usage
# 1. build the engine
# python build.py --dtype float16 --remove_input_padding --use_gpt_attention_plugin float16 --enable_context_fmha --use_gemm_plugin float16 --model_dir /home/lczyh/share/models/llama-7b-hf --output_dir llama7b_outputs
# 2. load and run the engine to generate text output
# python test_dev.py --max_output_len=50 --engine_dir llama7b_outputs --tokenizer_dir /home/lczyh/share/models/llama-7b-hf
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, required=True)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--engine_dir', type=str, default='llama7b_outputs')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        default="/home/lczyh/share/models/llama-7b-hf",
                        help="Directory containing the tokenizer.model.")
    parser.add_argument('--input_text',
                        type=str,
                        default='Born in north-east France, Soyer trained as a')
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    args = parser.parse_args()

    generate(**vars(args))
