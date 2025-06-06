import json
from googleapiclient import discovery
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import (
    get_dataset_config_names,
    load_dataset,
    concatenate_datasets
)
from tqdm import tqdm
import evaluate
import torch
import os
from src.utils import build_metrics_eval_messages, preprocess_to_perspective_message


device = 'cuda' if torch.cuda.is_available() else \
    'mps' if torch.mps.is_available() else 'cpu'


def metrics():
    def eval_metrics(cfg):
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.evals.metrics.model_path
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.evals.metrics.model_path
        ).to(device)

        if 'gemma' in model.config._name_or_path:
            model.generation_config.eos_token_id = [1, 107]

        model.eval()
        configs = get_dataset_config_names("HuggingFaceH4/hhh_alignment")
        parts = []
        for cfg in configs:
            ds = load_dataset("HuggingFaceH4/hhh_alignment", cfg, split="test")
            ds = ds.add_column("subset", [cfg] * len(ds))
            parts.append(ds)
            print(f"Loaded {cfg:8} subset with {len(ds)} rows.")
        hhh_all = concatenate_datasets(parts)
        assert len(hhh_all) == 221

        print(f"Total rows: {len(hhh_all)}")
        print("-" * 60)

        metric_global = evaluate.load("accuracy")
        metrics_by_subset = defaultdict(lambda: evaluate.load("accuracy"))

        tok_id_A = tokenizer.convert_tokens_to_ids("A")
        tok_id_B = tokenizer.convert_tokens_to_ids("B")

        with torch.no_grad():
            for i, ex in tqdm(enumerate(hhh_all)):
                q = ex["input"]
                choices = ex["targets"]["choices"]
                gold_idx = ex["targets"]["labels"].index(1)

                msgs = build_metrics_eval_messages(q, choices[0], choices[1])
                input_ids = tokenizer.apply_chat_template(
                    msgs,
                    return_tensors="pt",
                    add_generation_prompt=True,
                ).to(device)

                logits = model(input_ids).logits
                last_logits = logits[0, -1]

                logp_A = last_logits[tok_id_A].item()
                logp_B = last_logits[tok_id_B].item()
                pred = 1 if logp_B > logp_A else 0

                metric_global.add(prediction=pred, reference=gold_idx)
                metrics_by_subset[ex["subset"]].add(
                    prediction=pred, reference=gold_idx
                )

        results = {"overall": metric_global.compute()["accuracy"]}
        for subset, m in metrics_by_subset.items():
            results[subset] = m.compute()["accuracy"]

        print("\nFinal accuracy summary")
        for k, v in results.items():
            print(f"{k:8}: {v:.3%}")
        return results

    return eval_metrics


def auto_interp():
    def eval_auto_interp(cfg):
        # tokenizer = AutoTokenizer.from_pretrained(
        #     cfg.checkpoint_dir,
        #     use_fast=True
        # )
        # model = AutoModelForCausalLM.from_pretrained(
        #     cfg.checkpoint_dir,
        #     torch_dtype="auto"
        # ).to(device)
        # model.eval()

        # topk_store = defaultdict(
        #     lambda: defaultdict(lambda: {"pos": [], "neg": []})
        # )
        # example_counter = 0        # global running index over all examples

        print('evaluating auto interp')
    return eval_auto_interp


def toxicity():
    def eval_toxicity(cfg):
        client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=os.getenv('GOOGLE_API_KEY'),
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.evals.toxicity.model_path
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.evals.toxicity.model_path
        ).to(device)
        dataset = load_dataset(
            cfg.evals.toxicity.dataset_name, split="train"
        )

        if 'gemma' in model.config._name_or_path:
            model.generation_config.eos_token_id = [1, 107]
            model.generation_config.max_length = 512

        challenging_dataset = dataset.filter(
            lambda example: example["challenging"]
        )

        prompt_only = challenging_dataset.map(
            preprocess_to_perspective_message,
            batched=False,
            remove_columns=dataset.column_names
        )

        model.eval()

        full_outputs = []
        completions_only = []

        pad_id = tokenizer.pad_token_id
        batch_size = cfg.evals.toxicity.batch_size

        for start in tqdm(
            range(0, len(prompt_only), batch_size),
            desc="Generating"
        ):
            # ── Slice a (shallow‑copied) mini‑dataset and pull the column as a python list
            batch_messages = prompt_only[start: start + batch_size]["messages"]

            # ── Convert each conversation into a single prompt string
            batch_prompts = [
                tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False)
                for msgs in batch_messages
            ]

            # ── Tokenise & push to GPU
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to("cuda")

            # ── Autoregressive generation
            gen_ids = model.generate(**enc)

            # ── Decode full strings
            full_outputs.extend(
                tokenizer.batch_decode(
                    gen_ids, skip_special_tokens=True
                )
            )

            # ── Extract only the newly generated continuation
            for prompt_ids, output_ids in zip(enc["input_ids"], gen_ids):
                # Locate the first END‑OF‑TURN token in the *generated* sequence
                eot_positions = (
                    output_ids == cfg.evals.toxicity.end_of_turn_id
                ).nonzero(as_tuple=True)[0]
                if len(eot_positions) > 0:
                    first_eot = eot_positions[0].item()
                    # after the EOT
                    completion_ids = output_ids[first_eot + 1:]
                else:
                    # Fallback: trim the prompt length, same as before
                    prompt_ids_no_pad = prompt_ids[prompt_ids != pad_id]
                    completion_ids = output_ids[len(prompt_ids_no_pad):]

                completions_only.append(
                    # TODO: decode or batch decode?
                    tokenizer.decode(
                        completion_ids, skip_special_tokens=True
                    ).strip()
                )

            # optional: free VRAM early
            del enc, gen_ids

        model.detach().cpu()

        if device == 'cuda':
            torch.cuda.empty_cache()

        if cfg.evals.toxicity.dump_generated:
            with open(cfg.evals.toxicity.dump_path, 'w+') as f:
                json.dump(
                    [full_outputs, completions_only], f
                )

    return eval_toxicity
