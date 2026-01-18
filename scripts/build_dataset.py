import argparse
import json
import math
import os
import random
import re
import sys
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TaskItem:
    suite: str
    prompt: str
    gold: str
    meta: Dict[str, Any]


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_gsm8k(limit: int, seed: int) -> List[TaskItem]:
    ds = load_dataset("gsm8k", "main", split="train")
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    items = []
    for idx in indices[:limit]:
        row = ds[idx]
        prompt = row["question"].strip()
        gold = extract_gsm8k_answer(row["answer"])
        items.append(TaskItem("GSM8K", prompt, gold, {}))
    return items


def extract_gsm8k_answer(answer: str) -> str:
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer.strip()


def load_math(limit: int, seed: int) -> List[TaskItem]:
    ds = load_dataset("HuggingFaceH4/MATH", "default", split="train")
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    items = []
    for idx in indices[:limit]:
        row = ds[idx]
        prompt = row["problem"].strip()
        gold = extract_math_answer(row["solution"])
        items.append(TaskItem("MATH", prompt, gold, {"level": row.get("level"), "type": row.get("type")}))
    return items


def extract_math_answer(solution: str) -> str:
    boxed = re.findall(r"\\boxed\{([^}]+)\}", solution)
    if boxed:
        return boxed[-1].strip()
    lines = [line.strip() for line in solution.splitlines() if line.strip()]
    return lines[-1] if lines else solution.strip()


def load_mbpp(limit: int, seed: int) -> List[TaskItem]:
    ds = load_dataset("mbpp", "full", split="train")
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    items = []
    for idx in indices[:limit]:
        row = ds[idx]
        prompt = row["text"].strip()
        gold = row["code"].strip()
        meta = {"test_list": row.get("test_list", [])}
        items.append(TaskItem("MBPP+", prompt, gold, meta))
    return items


def load_gaia(limit: int, seed: int) -> List[TaskItem]:
    # GAIA access can be restricted; gracefully skip if unavailable.
    def _load_with_token(token: Optional[str]) -> Optional[Any]:
        configs = ["2023_all", "2023_level1", "2023_level2", "2023_level3", "default", None]
        splits = ["train", "validation", "test"]
        for cfg in configs:
            for split in splits:
                try:
                    if cfg is None:
                        return load_dataset("gaia-benchmark/GAIA", split=split, token=token)
                    return load_dataset("gaia-benchmark/GAIA", cfg, split=split, token=token)
                except Exception:
                    continue
        return None

    try:
        ds = load_dataset("gaia-benchmark/GAIA", "default", split="train")
    except Exception:
        ds = None
        env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if env_token:
            ds = _load_with_token(env_token)
        if ds is None:
            ds = _load_with_token(True)
        if ds is None:
            return []
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    items = []
    for idx in indices[:limit]:
        row = ds[idx]
        prompt = row.get("question") or row.get("prompt") or row.get("task") or ""
        prompt = str(prompt).strip()
        gold = str(row.get("answer") or row.get("gold") or "").strip()
        items.append(TaskItem("GAIA", prompt, gold, {}))
    return items


def compute_suite_quotas(suites: List[str], total: int) -> Dict[str, int]:
    base = total // len(suites)
    remainder = total % len(suites)
    quotas = {s: base for s in suites}
    for s in suites[:remainder]:
        quotas[s] += 1
    return quotas


def build_tasks(total: int, seed: int, include_gaia: bool) -> List[TaskItem]:
    if include_gaia:
        suites = ["MBPP+", "GAIA", "MATH", "GSM8K"]
        suite_quotas = compute_suite_quotas(suites, total)

        gaia_items = load_gaia(suite_quotas["GAIA"], seed + 3)
        if not gaia_items:
            print("GAIA unavailable; redistributing quotas to MBPP+/MATH/GSM8K.", file=sys.stderr)
            suites = ["MBPP+", "MATH", "GSM8K"]
            suite_quotas = compute_suite_quotas(suites, total)
            items: List[TaskItem] = []
            items.extend(load_mbpp(suite_quotas["MBPP+"], seed))
            items.extend(load_math(suite_quotas["MATH"], seed + 1))
            items.extend(load_gsm8k(suite_quotas["GSM8K"], seed + 2))
        else:
            gaia_shortfall = suite_quotas["GAIA"] - len(gaia_items)
            if gaia_shortfall > 0:
                extras = {"MBPP+": 0, "MATH": 0, "GSM8K": 0}
                for i in range(gaia_shortfall):
                    extras[["MBPP+", "MATH", "GSM8K"][i % 3]] += 1
                suite_quotas["MBPP+"] += extras["MBPP+"]
                suite_quotas["MATH"] += extras["MATH"]
                suite_quotas["GSM8K"] += extras["GSM8K"]
            items = []
            items.extend(load_mbpp(suite_quotas["MBPP+"], seed))
            items.extend(load_math(suite_quotas["MATH"], seed + 1))
            items.extend(load_gsm8k(suite_quotas["GSM8K"], seed + 2))
            items.extend(gaia_items)
    else:
        suites = ["MBPP+", "MATH", "GSM8K"]
        suite_quotas = compute_suite_quotas(suites, total)
        items = []
        items.extend(load_mbpp(suite_quotas["MBPP+"], seed))
        items.extend(load_math(suite_quotas["MATH"], seed + 1))
        items.extend(load_gsm8k(suite_quotas["GSM8K"], seed + 2))

    random.Random(seed).shuffle(items)
    return items


def framework_roles(framework: str) -> List[str]:
    if framework == "smolagents":
        return ["planner", "executor", "tool", "assistant"]
    if framework == "autogen":
        return ["assistant", "critic", "assistant", "assistant"]
    if framework == "crewai":
        return ["researcher", "analyst", "writer", "reviewer"]
    return ["assistant", "assistant", "assistant", "assistant"]


def make_steps(task: TaskItem, framework: str) -> List[Dict[str, str]]:
    roles = framework_roles(framework)
    prompt = task.prompt
    gold = task.gold
    suite = task.suite

    steps: List[Dict[str, str]] = []
    steps.append({"r": roles[0], "c": f"Plan: identify requirements from task. Task: {prompt}"})

    if suite == "MBPP+":
        steps.append({"r": roles[1], "c": "Draft a Python function that satisfies the prompt and tests."})
        steps.append({"r": roles[2], "c": "Tool result: basic checks pass on the draft."})
        steps.append({"r": roles[3], "c": f"Final code:\n{gold}"})
    elif suite in ("GSM8K", "MATH"):
        steps.append({"r": roles[1], "c": "Work through the math carefully and keep intermediate values."})
        steps.append({"r": roles[2], "c": "Tool result: arithmetic verified for key steps."})
        steps.append({"r": roles[3], "c": f"Final answer: {gold}"})
    else:  # GAIA or fallback
        steps.append({"r": roles[1], "c": "Collect key facts and constraints needed to answer."})
        steps.append({"r": roles[2], "c": "Tool result: cross-check facts with a quick lookup."})
        steps.append({"r": roles[3], "c": f"Final answer: {gold}"})

    return steps


def perturb_step(text: str, mode: str) -> str:
    if mode == "value_flip":
        nums = list(re.finditer(r"-?\d+", text))
        if nums:
            match = nums[0]
            val = int(match.group())
            flipped = val + 1 if val != 0 else 1
            return text[: match.start()] + str(flipped) + text[match.end() :]
        return text + " (incorrect)"
    if mode == "tool_corrupt":
        if "Tool result" in text or "Result" in text:
            return re.sub(r"result:.*", "result: corrupted output", text, flags=re.IGNORECASE)
        return text + "\nTool result: corrupted output"
    if mode == "deletion":
        return "[omitted]"
    if mode == "truncation":
        cutoff = max(1, len(text) // 2)
        return text[:cutoff]
    return text


def make_wrong_final(task: TaskItem) -> str:
    if task.suite == "MBPP+":
        return "raise AssertionError('synthetic failure')"
    nums = re.findall(r"-?\d+", task.gold)
    if nums:
        val = int(nums[-1])
        return str(val + 1)
    return task.gold + "_wrong"


def apply_synthetic_failure(sample: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    steps = sample["steps"]
    if len(steps) < 2:
        return sample

    k = rng.randint(0, len(steps) - 2)
    mode = rng.choice(["value_flip", "tool_corrupt", "deletion", "truncation"])
    steps[k]["c"] = perturb_step(steps[k]["c"], mode)

    suite = sample["meta"]["suite"]
    task = TaskItem(suite, sample["task"]["prompt"], sample["task"]["gold"], {})
    wrong_final = make_wrong_final(task)
    if task.suite == "MBPP+":
        steps[-1]["c"] = f"Final code:\n{wrong_final}"
    else:
        steps[-1]["c"] = f"Final answer: {wrong_final}"

    sample["label"]["when"] = k
    sample["label"]["who"] = steps[k]["r"]
    return sample


def judge_sample(sample: Dict[str, Any]) -> Tuple[bool, str]:
    suite = sample["meta"]["suite"]
    if suite == "MBPP+":
        code = sample["steps"][-1]["c"].split("Final code:")[-1].strip()
        tests = sample.get("meta", {}).get("test_list", [])
        ok = run_mbpp_tests(code, tests)
        return ok, "tests" if ok else "tests failed"
    if suite in ("GSM8K", "MATH"):
        gold = sample["task"]["gold"].strip()
        pred = extract_final_answer(sample["steps"][-1]["c"]) or ""
        return pred.strip() == gold, "exact match"
    if suite == "GAIA":
        gold = sample["task"]["gold"].strip()
        pred = extract_final_answer(sample["steps"][-1]["c"]) or ""
        if not gold:
            return True, "no gold"
        return pred.strip() == gold, "exact match"
    return False, "unknown suite"


def run_mbpp_tests(code: str, tests: List[str]) -> bool:
    script = code + "\n\n" + "\n".join(tests) + "\n"
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py") as f:
        f.write(script)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def extract_final_answer(text: str) -> str:
    match = re.search(r"Final answer:\s*(.*)", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"Final code:\s*(.*)", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def serialize_for_graph(sample: Dict[str, Any]) -> Tuple[List[int], List[List[int]]]:
    prompt = sample["task"]["prompt"]
    steps = sample["steps"]
    parts = []
    step_token_indices: List[List[int]] = []
    parts.append((f"Task:\n{prompt}\n", None))
    for idx, step in enumerate(steps):
        header = f"\n[{idx}:{step['r']}]\n"
        body = step["c"] + "\n"
        parts.append((header + body, idx))

    input_ids: List[int] = []
    for text, step_idx in parts:
        tokens = TOKENIZER.encode(text, add_special_tokens=False)
        if step_idx is not None:
            token_positions = list(range(len(input_ids), len(input_ids) + len(tokens)))
            step_token_indices.append(token_positions)
        input_ids.extend(tokens)
    return input_ids, step_token_indices


def build_graph(sample: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    input_ids, step_token_indices = serialize_for_graph(sample)
    if len(input_ids) > cfg["max_tokens"]:
        original_len = len(input_ids)
        offset = original_len - cfg["max_tokens"]
        input_ids = input_ids[offset:]
        # When truncating, remap step indices to avoid mismatch.
        new_step_indices: List[List[int]] = []
        for indices in step_token_indices:
            remapped = [i - offset for i in indices if i >= offset]
            if remapped:
                new_step_indices.append(remapped)
            else:
                new_step_indices.append([])
        step_token_indices = new_step_indices

    if not step_token_indices:
        edges = []
    else:
        with torch.no_grad():
            input_tensor = torch.tensor([input_ids], device=DEVICE)
            outputs = MODEL(input_tensor, output_attentions=True)
        layers = outputs.attentions
        last_n = min(cfg["last_n_layers"], len(layers))
        attn = torch.stack(layers[-last_n:])
        attn_mean = attn.mean(dim=(0, 2)).squeeze(0)  # layers+heads, drop batch
        attn_mean = attn_mean.cpu()

        edges = []
        for dst_idx, dst_tokens in enumerate(step_token_indices):
            if not dst_tokens:
                continue
            weights = []
            for src_idx, src_tokens in enumerate(step_token_indices):
                if src_idx == dst_idx or not src_tokens:
                    weights.append(0.0)
                    continue
                sub = attn_mean[dst_tokens][:, src_tokens]
                weights.append(float(sub.mean().item()))
            top_k = cfg["top_k"]
            pairs = sorted(enumerate(weights), key=lambda x: x[1], reverse=True)
            kept = [(i, w) for i, w in pairs if w > 0.0][:top_k]
            if cfg["threshold"] is not None:
                kept = [(i, w) for i, w in kept if w >= cfg["threshold"]]
            if cfg["renorm"] and kept:
                total = sum(w for _, w in kept)
                kept = [(i, w / total) for i, w in kept]
            for src_idx, w in kept:
                edges.append([src_idx, dst_idx, round(w, 6)])

    return {
        "kind": "attn_step",
        "tgt": "final",
        "edges": edges,
        "cfg": {
            "agg": cfg["agg"],
            "top_k": cfg["top_k"],
            "threshold": cfg["threshold"],
            "renorm": cfg["renorm"],
            "last_n_layers": cfg["last_n_layers"],
            "normalize_by_src_len": cfg["normalize_by_src_len"],
        },
    }


def compute_path(sample: Dict[str, Any]) -> None:
    when = sample["label"].get("when")
    if when is None:
        return

    steps = sample["steps"]
    target = len(steps) - 1
    edges = sample["graph"]["edges"]

    graph: Dict[int, List[Tuple[int, float]]] = {}
    for src, dst, weight in edges:
        graph.setdefault(src, []).append((dst, weight))

    # Dijkstra on -log(weight)
    dist = {when: 0.0}
    prev: Dict[int, int] = {}
    visited = set()
    while True:
        candidates = [(node, d) for node, d in dist.items() if node not in visited]
        if not candidates:
            break
        node, d = min(candidates, key=lambda x: x[1])
        visited.add(node)
        if node == target:
            break
        for dst, weight in graph.get(node, []):
            if weight <= 0:
                continue
            cost = d - math.log(weight)
            if cost < dist.get(dst, float("inf")):
                dist[dst] = cost
                prev[dst] = node

    if target not in prev and target != when:
        sample["label"]["path"] = [when, target]
        return

    path = [target]
    cur = target
    while cur != when:
        cur = prev[cur]
        path.append(cur)
    sample["label"]["path"] = list(reversed(path))


def build_samples(total: int, seed: int, include_gaia: bool) -> List[Dict[str, Any]]:
    tasks = build_tasks(total, seed, include_gaia)
    frameworks = ["smolagents", "autogen", "crewai"]
    per_framework = total // len(frameworks)
    remainder = total % len(frameworks)
    framework_counts = {f: per_framework for f in frameworks}
    for f in frameworks[:remainder]:
        framework_counts[f] += 1

    samples = []
    task_idx = 0
    for framework in frameworks:
        for _ in range(framework_counts[framework]):
            task = tasks[task_idx]
            task_idx += 1
            steps = make_steps(task, framework)
            sample = {
                "id": "",
                "task": {
                    "prompt": task.prompt,
                    "gold": task.gold,
                },
                "steps": steps,
                "graph": {},
                "label": {
                    "when": None,
                    "who": None,
                    "path": None,
                },
                "views": {
                    "w_gold": {"use_gold": True},
                    "wo_gold": {"use_gold": False},
                },
                "meta": {
                    **task.meta,
                    "suite": task.suite,
                    "framework": framework,
                },
            }
            samples.append(sample)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--synthetic", type=int, default=330)
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--last_n_layers", type=int, default=4)
    parser.add_argument(
        "--skip-gaia",
        action="store_true",
        help="Skip GAIA entirely and redistribute its quota to MBPP+/MATH/GSM8K.",
    )
    args = parser.parse_args()

    seed_all(args.seed)

    global MODEL, TOKENIZER, DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TOKENIZER = AutoTokenizer.from_pretrained(args.model)
    MODEL = AutoModelForCausalLM.from_pretrained(args.model)
    MODEL.to(DEVICE)
    MODEL.eval()

    samples = build_samples(args.total, args.seed, not args.skip_gaia)
    rng = random.Random(args.seed + 99)
    synth_indices = set(rng.sample(range(len(samples)), min(args.synthetic, len(samples))))

    cfg = {
        "agg": "mean",
        "top_k": args.top_k,
        "threshold": None,
        "renorm": True,
        "last_n_layers": args.last_n_layers,
        "normalize_by_src_len": False,
        "max_tokens": args.max_tokens,
    }

    for i, sample in enumerate(samples):
        sample["id"] = f"gtl_{i:06d}"
        if i in synth_indices:
            sample = apply_synthetic_failure(sample, rng)
        sample["graph"] = build_graph(sample, cfg)
        compute_path(sample)
        success, reason = judge_sample(sample)
        sample.setdefault("meta", {})
        sample["meta"]["judge"] = {"success": success, "reason": reason}
        samples[i] = sample

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
