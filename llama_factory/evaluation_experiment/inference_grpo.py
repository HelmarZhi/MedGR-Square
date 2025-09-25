import os
import json
import multiprocessing
from tqdm import tqdm
from llamafactory.chat import ChatModel


def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def run_inference_single_model(args):
    input_jsonl_path, output_jsonl_path, model_path = args

    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    # 判断是否是合并后的全模型（无 adapter）
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        raise ValueError(f"'{model_path}' contains adapter_config.json. Please merge LoRA before inference.")

    # 初始化完整模型
    infer_args = {
        "model_name_or_path": model_path,
        "infer_backend": "huggingface",
        "top_k": 1,
    }
    model = ChatModel(args=infer_args)

    data = load_jsonl(input_jsonl_path)

    with open(output_jsonl_path, 'w', encoding='utf-8') as out_file:
        for item in tqdm(data, desc=f"Inference: {os.path.basename(model_path)}"):
            image_path = item["image_path"]
            question = item["question"]
            question_id = item["question_id"]
            gt_answer = item["gt_answer"]

            messages = [{
                "role": "user",
                "content": f"<|vision_start|>{image_path}<|vision_end|>\n{question}\n"
                           f"Please answer with only one of the given options. Do not explain cause I just want the final answer."
            }]

            try:
                responses = model.chat(
                    messages=messages,
                    image=image_path,
                    generate_kwargs={
                        "temperature": 0,
                        "top_k": 1
                    }
                )
                model_answer = responses[0].response_text if responses else "No response"
            except Exception as e:
                model_answer = f"[ERROR] {str(e)}"

            result = {
                "question_id": question_id,
                "model_answer": model_answer,
                "gt_answer": gt_answer,
                "question": question,
                "image_path": image_path
            }

            out_file.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    multiprocessing.set_start_method("spawn", force=True)

    # === 配置项 ===
    input_jsonl = '/home/one/Codes/LLaMA-Factory-main/formatted_test.jsonl'
    output_dir = '/home/one/Codes/LLaMA-Factory-main/results/grpo_infer'
    merged_checkpoints = [
        # '/home/one/Codes/Qwen2-VL-Finetune-master/output/test_grpo1_10000and100/checkpoint-1600',
        '/home/one/Codes/Qwen2-VL-Finetune-master/output/test_grpo1_10000and100/checkpoint-1650',
        '/home/one/Codes/Qwen2-VL-Finetune-master/output/test_grpo1_10000and100/checkpoint-1700',
        '/home/one/Codes/Qwen2-VL-Finetune-master/output/test_grpo1_10000and100/checkpoint-1750',
        # '/home/one/Codes/Qwen2-VL-Finetune-master/output/test_grpo1_10000and100/checkpoint-1800',
    ]

    task_args = [
        (
            input_jsonl,
            os.path.join(output_dir, f"{os.path.basename(ckpt)}.jsonl"),
            ckpt
        )
        for ckpt in merged_checkpoints
    ]

    with multiprocessing.Pool(processes=min(4, len(task_args))) as pool:
        pool.map(run_inference_single_model, task_args)
