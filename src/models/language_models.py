"""Module related to language models used with llamaindex"""

from pathlib import Path

from llama_index.llms import HuggingFaceLLM, LlamaCPP
from llama_index.llms.custom import CustomLLM
from llama_index.llms.llama_utils import completion_to_prompt, messages_to_prompt
from llama_index.prompts import PromptTemplate

LANGUAGE_MODEL_PATH = Path.cwd().joinpath("models", "llama-2-13b-chat.gguf")


def get_llama2(
    max_new_tokens: int = 256, model_temperature: int = 0.1, context_window: int = 3800
) -> CustomLLM:
    """Init llama-cpp-python https://github.com/abetlen/llama-cpp-python via llama_index.llms"""

    # llama2 has a context window of 4096 tokens
    return LlamaCPP(
        model_path=str(LANGUAGE_MODEL_PATH),
        context_window=context_window,
        temperature=model_temperature,
        max_new_tokens=max_new_tokens,
        model_kwargs={"n_gpu_layers": 50, "n_batch": 8, "use_mlock": False},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=custom_completion_to_prompt,
        verbose=True,
    )


def custom_completion_to_prompt(completion: str) -> str:
    return completion_to_prompt(
        completion,
        system_prompt=(
            "You are a Q&A assistant. Your goal is to answer questions as "
            "accurately as possible is the instructions and context provided."
        ),
    )


def get_huggingface_llm(
    model_name: str,
    max_new_tokens: int = 256,
    model_temperature: int = 0.1,
    context_window: int = 2048,
) -> HuggingFaceLLM:
    """Return a hugginface LLM"""

    query_wrapper_prompt = PromptTemplate(
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{query_str}\n\n### Response:"
    )

    return HuggingFaceLLM(
        model_name=model_name,
        tokenizer_name=model_name,
        context_window=context_window,
        max_new_tokens=max_new_tokens,
        generate_kwargs={"temperature": model_temperature, "do_sample": True},
        device_map="auto",
        tokenizer_kwargs={"max_length": 2048},
        query_wrapper_prompt=query_wrapper_prompt,
        model_kwargs={"max_memory": {0: "18GB"}, "offload_folder": "/tmp/offload"},
    )
