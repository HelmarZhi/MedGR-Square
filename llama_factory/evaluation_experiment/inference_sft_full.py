import os
import json
import torch
from tqdm import tqdm
from llamafactory.chat import ChatModel


def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def run_inference(input_jsonl_path, output_jsonl_path, base_model_path):
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    infer_args = {
        "model_name_or_path": base_model_path,
        "infer_backend": "huggingface",
        "top_k": 1,
    }

    model = ChatModel(args=infer_args)
    if hasattr(model.engine.model, 'is_parallelizable'):
        model.engine.model.is_parallelizable = False
        model.engine.model.model_parallel = False
    model.engine.model = model.engine.model.to("cuda:0")

    data = load_jsonl(input_jsonl_path)

    with open(output_jsonl_path, 'w', encoding='utf-8') as out_file:
        for idx, item in enumerate(tqdm(data, desc="Inference")):
            image_path = item["image_path"]
            question = item["question"]
            question_id = item["question_id"]
            gt_answer = item["gt_answer"]

            messages = [{
                "role": "user",
                "content": f"<|vision_start|>{image_path}<|vision_end|>\n{question}\nPlease answer with only one of the given options. Do not explain cause i just want the final answer."
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
    # 使用 3 个 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["USE_TORCH_DISTRIBUTED"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    input_jsonl = '/home/one/Codes/LLaMA-Factory-main/formatted_test.jsonl'
    base_model_path = '/home/one/Codes/LLaMA-Factory-main/saves/qwen2_5vl-7b/full/sft_ixc_wo_training_top1000/checkpoint-100'
    output_path = '/home/one/Codes/LLaMA-Factory-main/results/full_sft/ixc_top1000_infer_output.jsonl'

    run_inference(input_jsonl, output_path, base_model_path)
