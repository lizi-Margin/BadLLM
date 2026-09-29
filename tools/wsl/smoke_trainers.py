"""Run tiny trainer jobs to validate installation; never use these as model quality evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
MODEL = Path("/home/hulc/badllm/smoke/qwen3-114m-init")


def main(stage):
    work = Path(tempfile.mkdtemp(prefix=f"{stage}-trainer-", dir=MODEL.parent))
    env = {**os.environ, "WANDB_MODE": "disabled", "OMP_NUM_THREADS": "4"}
    if stage == "pt":
        (work / "texts.json").write_text(json.dumps([
            {"text": "This is an installation smoke test. A small model learns by predicting the next token. " * 8}
            for _ in range(16)
        ]))
        (work / "dataset_info.json").write_text(json.dumps({
            "env_smoke": {"file_name": "texts.json", "columns": {"prompt": "text"}}
        }))
        config = {
            "model_name_or_path": str(MODEL), "stage": "pt", "do_train": True,
            "finetuning_type": "full", "dataset": "env_smoke", "dataset_dir": str(work),
            "cutoff_len": 64, "packing": True, "output_dir": str(work / "output"),
            "per_device_train_batch_size": 1, "gradient_accumulation_steps": 1,
            "learning_rate": 0.0003, "max_steps": 2, "bf16": True,
            "flash_attn": "sdpa", "logging_steps": 1, "save_strategy": "no",
            "report_to": "none", "preprocessing_num_workers": 1,
            "dataloader_num_workers": 0, "gradient_checkpointing": False,
        }
        config_path = work / "pt.json"
        config_path.write_text(json.dumps(config))
        command = [str(Path(sys.executable).parent / "llamafactory-cli"), "train", str(config_path)]
    else:
        import pyarrow as pa
        import pyarrow.parquet as pq
        rows = [{"question": f"What is {n} + 1?", "answer": str(n + 1)} for n in range(4)]
        pq.write_table(pa.Table.from_pylist(rows), work / "train.parquet")
        pq.write_table(pa.Table.from_pylist([
            {"question": "What is 5 + 1?", "answer": "6"},
            {"question": "What is 6 + 1?", "answer": "7"},
        ]), work / "val.parquet")
        command = [
            str(Path(sys.executable).parent / "torchrun"), "--standalone", "--nproc_per_node=1",
            "-m", "verl.trainer.fsdp_sft_trainer",
            f"model.partial_pretrain={MODEL}", "model.strategy=fsdp2",
            "model.fsdp_config.model_dtype=bf16", "model.enable_gradient_checkpointing=False",
            f"data.train_files={work / 'train.parquet'}", f"data.val_files={work / 'val.parquet'}",
            "data.train_batch_size=1", "data.micro_batch_size_per_gpu=1", "data.max_length=64",
            "+data.apply_chat_template_kwargs.enable_thinking=False", "use_remove_padding=False",
            "trainer.n_gpus_per_node=1", "trainer.total_epochs=1", "trainer.total_training_steps=1",
            "trainer.logger=[console]", "trainer.resume_mode=disable", "trainer.save_freq=-1",
            "trainer.test_freq=-1", f"trainer.default_local_dir={work / 'output'}",
            "trainer.project_name=badllm-env", "trainer.experiment_name=smoke",
        ]
    print(f"Smoke work directory: {work}", flush=True)
    subprocess.run(command, env=env, check=True)
    result = {"stage": stage, "status": "passed", "work_dir": str(work),
              "steps": 2 if stage == "pt" else 1, "command": command,
              "note": "Synthetic installation test only; no useful model has been trained."}
    (REPO / ".artifacts/env" / f"trainer-{stage}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["pt", "sft"])
    main(parser.parse_args().stage)
