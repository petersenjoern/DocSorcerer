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


def get_zephyr(
    max_new_tokens: int = 256, model_temperature: int = 0.1, context_window: int = 3800
):
    """Return zephyr"""

    def zepyhr_messages_to_prompt(messages):
        prompt = ""
        for message in messages:
            if message.role == "system":
                prompt += f"<|system|>\n{message.content}</s>\n"
            elif message.role == "user":
                prompt += f"<|user|>\n{message.content}</s>\n"
            elif message.role == "assistant":
                prompt += f"<|assistant|>\n{message.content}</s>\n"

        # ensure we start with a system prompt, insert blank if needed
        if not prompt.startswith("<|system|>\n"):
            prompt = "<|system|>\n</s>\n" + prompt

        # add final assistant prompt
        prompt = prompt + "<|assistant|>\n"

        return prompt

    llm = HuggingFaceLLM(
        model_name="TheBloke/zephyr-7B-beta-GPTQ",
        tokenizer_name="TheBloke/zephyr-7B-beta-GPTQ",
        query_wrapper_prompt=PromptTemplate(
            "<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"
        ),
        context_window=context_window,
        max_new_tokens=max_new_tokens,
        model_kwargs={
            "max_memory": {0: "20GB"},
            "offload_folder": "/tmp/offload",
        },
        # tokenizer_kwargs={},
        generate_kwargs={
            "temperature": model_temperature,
            "top_k": 40,
            "top_p": 0.95,
            "repetition_penalty": 1.1,
            "do_sample": True,
        },
        messages_to_prompt=zepyhr_messages_to_prompt,
        device_map="auto",
    )
    return llm
