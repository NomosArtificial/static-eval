# adapted from https://modal.com/docs/guide/ex/falcon_gptq,
# https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/falcon_gptq.py


# see https://modal.com/docs/guide/ex/falcon_gptq for intructions on how to serve with modal

from modal import Image, Stub, gpu, method, web_endpoint


IMAGE_MODEL_DIR = "/model"
MODEL_NAME = "TheBloke/falcon-7b-instruct-GPTQ"


def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_NAME)


image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "huggingface_hub==0.14.1",
        "transformers @ git+https://github.com/huggingface/transformers.git@f49a3453caa6fe606bb31c571423f72264152fce",
        "auto-gptq @ git+https://github.com/PanQiWei/AutoGPTQ.git@b5db750c00e5f3f195382068433a3408ec3e8f3c",
        "einops==0.6.1",
    )
    .run_function(download_model)
)


stub = Stub(name="example-falcon-gptq", image=image)


@stub.cls(gpu=gpu.A100(), timeout=60 * 10, container_idle_timeout=60 * 5)
class Falcon40BGPTQ:
    def __enter__(self):
        from transformers import AutoTokenizer
        from auto_gptq import AutoGPTQForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        print("Loaded tokenizer.")

        self.model = AutoGPTQForCausalLM.from_quantized(
            MODEL_NAME,
            trust_remote_code=True,
            use_safetensors=True,
            device_map="auto",
            use_triton=False,
            strict=False,
        )
        print("Loaded model.")

    @method()
    def generate(self, prompt: str):
        from threading import Thread
        from transformers import TextIteratorStreamer

        inputs = self.tokenizer(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        generation_kwargs = dict(
            inputs=inputs.input_ids.cuda(),
            attention_mask=inputs.attention_mask,
            temperature=0.1,
            max_new_tokens=512,
            streamer=streamer,
        )

        # Run generation on separate thread to enable response streaming.
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for new_text in streamer:
            yield new_text

        thread.join()


prompt_template = (
    "A chat between a curious human user and an artificial intelligence assistant. The assistant give a helpful, detailed, and accurate answer to the user's question."
    "\n\nUser:\n{}\n\nAssistant:\n"
)


@stub.local_entrypoint()
def cli():
    question = "Tell me a joke about horses."
    model = Falcon40BGPTQ()
    for text in model.generate.call(prompt_template.format(question)):
        print(text, end="", flush=True)


@stub.function(timeout=60 * 10)
@web_endpoint(method="POST")
def get(body: dict):
    from fastapi.responses import JSONResponse

    model = Falcon40BGPTQ()
    buffer = ""

    for text in model.generate.call(body["question"]):
        buffer += text

    return JSONResponse({"output": buffer})
