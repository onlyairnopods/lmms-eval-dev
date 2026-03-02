#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
# export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=true

conda activate llava

# model='llava_hf'
# model_args='/workspace2/caizhenyang/lzz/models/llava-1.5-7b-hf'

# model='qwen3_vl'
# model_args='/workspace2/caizhenyang/lzz/models/Qwen3-VL-4B-Instruct'

# model='gemma3'
# model_args='/workspace2/caizhenyang/lzz/models/gemma-3-4b-it'

# model_name="${model_args%/}"
# model_name="${model_name##*/}"


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

models=(
    # "llava_hf"
    
    # "qwen3_vl"
    # "qwen3_vl"
        
    # "llava_hf"

    # "gemma3"
    "gemma3"
)

model_args_list=(
    # "/workspace2/caizhenyang/lzz/models/llava-1.5-7b-hf" # 完成除 xmmmu 外

    # "/workspace2/caizhenyang/lzz/models/Qwen3-VL-4B-Instruct" # 完成
    # "/workspace2/caizhenyang/lzz/models/Qwen3-VL-8B-Instruct" # 完成

    # "/workspace2/caizhenyang/lzz/models/llava-v1.6-mistral-7b-hf" # 完成除 xmmmu 外

    # "/workspace2/caizhenyang/lzz/models/gemma-3-4b-it" # 完成
    "/workspace2/caizhenyang/lzz/models/gemma-3-12b-it"
)

tasks=(
    'cvqa'
    'maxm'
    'xmmmu' # llava报错，可能有问题
    # 'xchat' # 没配置文件
    # 'flores' # 没配置文件，没数据
) # list of tasks for evaluation

# output_path='eval_outputs'
# mkdir -p ${output_path}

for i in "${!models[@]}"; do
    model="${models[$i]}"
    model_args="${model_args_list[$i]}"

    case "$model" in
        qwen3_vl)
            extra_args=",interleave_visuals=False"
            ;;
        llava_hf)
            extra_args=",device_map=auto"
            ;;
        gemma3)
            extra_args=",device_map=balanced_low_0"
            ;;
        *)
            # 默认没有额外参数
            extra_args=""
            ;;
    esac

    # 提取模型名（自动去掉尾部 /）
    model_name="${model_args%/}"
    model_name="${model_name##*/}"

    output_path="eval_${model_name}_outputs"
    mkdir -p "${output_path}"

    echo "=============================="
    echo "Testing model: ${model_name}"
    echo "=============================="

    for task in "${tasks[@]}"; do
        echo "Running task: ${task}"

        # python3 -m lmms_eval \
        python3 -m accelerate.commands.launch \
            --num_processes=4 --mixed_precision=bf16 \
            -m lmms_eval \
            --model ${model} \
            --model_args pretrained=${model_args}${extra_args} \
            --tasks ${task} \
            --batch_size 1 \
            --log_samples \
            --log_samples_suffix ${task} \
            --output_path ${output_path} \
            --show_config \
            --write_out \
            2>&1 | tee -a ${output_path}/${model_name}_${task}_run.log
    done
done