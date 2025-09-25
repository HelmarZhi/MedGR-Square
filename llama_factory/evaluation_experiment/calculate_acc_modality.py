from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import json
import os
from collections import defaultdict
import pandas as pd
import string

def safe_load_jsonl(filepath):
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"⚠️ Skipped invalid line in {filepath}")
                continue
    return records

def normalize_answer(ans):
    if not isinstance(ans, str):
        return ""
    ans = ans.lower()
    ans = ans.translate(str.maketrans('', '', string.punctuation))
    return ans.strip()

def compute_accuracy_by_modality(checkpoint_path, testdata_path, max_skip_ratio=0.5):
    checkpoint_data = safe_load_jsonl(checkpoint_path)
    filtered_data = safe_load_jsonl(testdata_path)

    qid_to_info = {item["question_id"]: item for item in filtered_data}

    modality_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    all_preds, all_gts = [], []

    total_correct, total_count, skipped_invalid = 0, 0, 0
    total_samples = len(checkpoint_data)
    max_skips_allowed = int(total_samples * max_skip_ratio)
    force_include = False

    for entry in checkpoint_data:
        qid = entry.get("question_id")
        pred = entry.get("model_answer")
        ref = qid_to_info.get(qid)

        if not ref or not pred or not isinstance(pred, str):
            if not force_include:
                skipped_invalid += 1
                if skipped_invalid >= max_skips_allowed:
                    force_include = True
                continue

        gt = ref.get("gt_answer", "")
        options = ref.get("options", [])
        modality = ref.get("modality_type", "Unknown")

        norm_pred = normalize_answer(pred)
        norm_gt = normalize_answer(gt)
        norm_options = [normalize_answer(opt) for opt in options]

        if not force_include and (not gt or not options or norm_pred not in norm_options):
            skipped_invalid += 1
            if skipped_invalid >= max_skips_allowed:
                force_include = True
            continue

        modality_stats[modality]["total"] += 1
        total_count += 1
        all_preds.append(norm_pred)
        all_gts.append(norm_gt)

        if norm_pred == norm_gt:
            modality_stats[modality]["correct"] += 1
            total_correct += 1

    # 计算全局 F1 和 AUC
    all_labels = list(set(all_gts + all_preds))
    overall_f1 = f1_score(all_gts, all_preds, average='macro') if total_count > 0 else 0.0
    try:
        gts_bin = label_binarize(all_gts, classes=all_labels)
        preds_bin = label_binarize(all_preds, classes=all_labels)
        overall_auc = roc_auc_score(gts_bin, preds_bin, average='macro', multi_class='ovo') if len(all_labels) > 1 else 0.0
    except Exception:
        overall_auc = 0.0

    # 每个 modality 的 accuracy
    modality_accuracy = []
    for modality, stats in modality_stats.items():
        accuracy = round(stats["correct"] / stats["total"], 4) if stats["total"] > 0 else 0.0
        modality_accuracy.append({
            "modality_type": modality,
            "accuracy": accuracy,
            "correct": stats["correct"],
            "total": stats["total"]
        })

    # 汇总行
    overall_accuracy = round(total_correct / total_count, 4) if total_count > 0 else 0.0
    overall_row = pd.DataFrame([{
        "modality_type": "Overall",
        "accuracy": overall_accuracy,
        "f1": round(overall_f1, 4),
        "auc": round(overall_auc, 4),
        "correct": total_correct,
        "total": total_count
    }])

    df = pd.DataFrame(modality_accuracy).sort_values("accuracy", ascending=False)
    df = pd.concat([df, overall_row], ignore_index=True)

    print(f"✅ 总有效样本数: {total_count}")
    print(f"⏭️ 跳过不合法答案数: {skipped_invalid}（最多跳过 {max_skips_allowed}）")
    return df

if __name__ == "__main__":
    df = compute_accuracy_by_modality(
        checkpoint_path="/home/one/Codes/LLaMA-Factory-main/evaluation_experiment/checkpoint_30.jsonl",
        testdata_path="/home/one/Codes/LLaMA-Factory-main/filtered_test_images.jsonl",
        max_skip_ratio=0.00
    )
    print(df.to_string(index=False))
