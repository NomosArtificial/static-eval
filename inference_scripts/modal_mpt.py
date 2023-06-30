from modal import Image, Stub, gpu, method, web_endpoint


MODEL_NAME = "mosaicml/mpt-30b-instruct"

# Spec for an image where model is cached locally
def download_model():
    from huggingface_hub import snapshot_download

    model_name = MODEL_NAME
    snapshot_download(model_name)


image = (
    Image.micromamba()
    .micromamba_install(
        "cudatoolkit=11.7",
        "cudnn=8.1.0",
        "cuda-nvcc",
        "scipy",
        channels=["conda-forge", "nvidia"],
    )
    .apt_install("git")
    .pip_install(
        "bitsandbytes==0.39.0",
        "bitsandbytes-cuda117==0.26.0.post2",
        "peft @ git+https://github.com/huggingface/peft.git",
        "transformers @ git+https://github.com/huggingface/transformers.git",
        "accelerate @ git+https://github.com/huggingface/accelerate.git",
        "torch==2.0.0",
        "torchvision==0.15.1",
        "sentencepiece==0.1.97",
        "huggingface_hub==0.14.1",
        "einops==0.6.1",
    )
    .run_function(download_model)
)


from transformers import StoppingCriteria, StoppingCriteriaList
import torch


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids: list):
        self.keywords = keywords_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False


stub = Stub(image=image, name="mpt")


@stub.cls(
    gpu=gpu.A100(count=4),  # Use A100s
    # memory=43008,
    timeout=60 * 10,  # 10 minute timeout on inputs
    container_idle_timeout=60 * 5,  # Keep runner alive for 5 minutes
)
class MPT30B:
    def __enter__(self):
        import torch
        from transformers import (
            AutoTokenizer,
            AutoConfig,
            AutoModelForCausalLM,
        )

        model_name = MODEL_NAME

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.max_seq_len = 16384  # (input + output) tokens can now be up to 16384

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,  # Model is downloaded to cache dir
            device_map="auto",
            load_in_8bit=True,
            config=config,
        )
        model.tie_weights()
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
            device_map="auto",
        )
        tokenizer.bos_token_id = 1

        self.model = torch.compile(model)
        self.tokenizer = tokenizer

    @method()
    def generate(self, prompt: str):
        import torch
        from threading import Thread
        from transformers import TextIteratorStreamer
        from transformers import GenerationConfig

        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.1,
            max_new_tokens=512,
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokenized = self.tokenizer(prompt, return_tensors="pt")
            input_ids = tokenized.input_ids
            # print("input_ids")
            # print(input_ids)
            # print()
            input_ids = input_ids.to(self.model.device)
            # print("input_ids")
            # print(input_ids)
            # print()

            stop_words = ["\n"]
            stop_ids = [self.tokenizer.encode(w)[0] for w in stop_words]
            stop_criteria = KeywordsStoppingCriteria(stop_ids)

            streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
            generate_kwargs = dict(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,  # False,  # True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                attention_mask=tokenized.attention_mask,
                output_scores=True,
                stopping_criteria=StoppingCriteriaList([stop_criteria]),
                streamer=streamer,
            )

            thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
            thread.start()
            for new_text in streamer:
                yield new_text

            thread.join()


prompt_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n###Instruction\n{}\n\n### Response\n"


@stub.local_entrypoint()
def cli():
    question = "Can you describe the main differences between Python and JavaScript programming languages."
    model = MPT30B()
    # output = model.generate.call(prompt_template.format(question))
    # print(output)
    for text in model.generate.call(prompt_template.format(question)):
        print(text, end="", flush=True)


@stub.function(timeout=60 * 10)
@web_endpoint(method="POST")
def get(body: dict):
    from fastapi.responses import JSONResponse
    from itertools import chain

    model = MPT30B()
    buffer = ""

    for text in model.generate.call(body["question"]):
        buffer += text

    return JSONResponse({"output": buffer})
