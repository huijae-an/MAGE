import os

import config
from google.oauth2 import service_account
from llama_index.core.llms.llm import LLM
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.vertex import Vertex
from pydantic import BaseModel

from llama_index.llms.openai import OpenAI

from .log_utils import get_logger

logger = get_logger(__name__)


class Config:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.file_config = {}
        if self.file_path and os.path.isfile(self.file_path):
            self.file_config = config.Config(self.file_path)
        self.fallback_config = {}
        self.fallback_config["OPENAI_API_BASE_URL"] = ""

    def __getitem__(self, index):
        # Values in key.cfg has priority over env variables
        if index in self.file_config:
            return self.file_config[index]
        if index in os.environ:
            return os.environ[index]
        if index in self.fallback_config:
            return self.fallback_config[index]
        raise KeyError(
            f"Cannot find {index} in either cfg file '{self.file_path}' or env variables"
        )


def get_llm(**kwargs) -> LLM:
    print("right here")
    cfg = Config(kwargs["cfg_path"])
    provider: str = kwargs["provider"]
    provider = provider.lower()
    if provider == "vllm":
        model = kwargs["model"]
        temperature = kwargs["temperature"] 
        top_p = kwargs["top_p"]
        max_tokens = kwargs["max_token"]
        api_key = "EMPTY"
        # edit the port if necessary
        base_url = "http://localhost:8000/v1"
        llm: LLM = OpenAI(model = model,
                              temperature = temperature,
                              top_p = top_p,
                              max_tokens = max_tokens,
                              api_key = api_key,
                              api_base = base_url)
    else:
        raise ValueError(f"gen_config: Invalid provider: {provider}")

    return llm


class ExperimentSetting(BaseModel):
    """
    Global setting for experiment
    """

    temperature: float = 0.85  # Chat temperature
    top_p: float = 0.95  # Chat top_p


global_exp_setting = ExperimentSetting()


def get_exp_setting() -> ExperimentSetting:
    return global_exp_setting


def set_exp_setting(temperature: float | None = None, top_p: float | None = None):
    if temperature is not None:
        global_exp_setting.temperature = temperature
    if top_p is not None:
        global_exp_setting.top_p = top_p
    return global_exp_setting
