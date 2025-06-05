import wandb
import torch
import hydra
import random
import numpy as np
from trl import setup_chat_format
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from src.train import lukas_sft, run_sft, run_dpo
from omegaconf import DictConfig, OmegaConf
import logging


@hydra.main(
    version_base=None,
    config_path="config/eval_config",
    config_name="default"
)
def main():
    pass


if __name__ == '__main__':
    main()
