import os
import json
import multiprocessing
from tqdm import tqdm
from llamafactory.chat import ChatModel


def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def run_batch_inference_task(args):
    input_jsonl_path, output_jsonl_path, base_model_path, lora_path = args

    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    infer_args = {
        "model_name_or_path": base_model_path,
        "adapter_name_or_path": lora_path,
        "infer_backend": "huggingface",
        "top_k": 1,
    }
    model = ChatModel(args=infer_args)
    # model.model = model.model.to("cuda:0")

    data = load_jsonl(input_jsonl_path)

    with open(output_jsonl_path, 'w', encoding='utf-8') as out_file:
        for item in tqdm(data, desc=f"Inference: {os.path.basename(lora_path)}"):
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    multiprocessing.set_start_method("spawn", force=True)

    input_jsonl = '/home/one/Codes/LLaMA-Factory-main/formatted_test.jsonl'
    base_model_path = '/home/one/Codes/LLaMA-Factory-main/saves/qwen2_5vl-7b/full/sft_ixc_wo_training_top1000/checkpoint-100'
    output_rootpath = '/home/one/Codes/LLaMA-Factory-main/results/sft_sample_rawgt_11000_7b'

    checkpoints_list = [
        # '/home/one/Codes/LLaMA-Factory-main/saves/qwen2_5vl-7b/lora/sft_11000_rawGT/checkpoint-50',
        # '/home/one/Codes/Qwen2-VL-Finetune-master/output/test_grpo1/checkpoint-90',
        # '/openbayes/home/LLaMA-Factory-main/saves/qwen2_5vl-3b/lora/sft_mixed_final_11000/checkpoint-396',
    ]

    task_args = [
        (
            input_jsonl,
            os.path.join(output_rootpath, f"{os.path.basename(ckpt)}.jsonl"),
            base_model_path,
            ckpt
        )
        for ckpt in checkpoints_list
    ]

    with multiprocessing.Pool(processes=4) as pool:
        pool.map(run_batch_inference_task, task_args)
