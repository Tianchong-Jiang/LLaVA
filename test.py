import os
os.environ['TRANSFORMERS_CACHE'] = '/mount/'

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

base_prompt = "The image shows a synthetic scene of a robot working in a kitchen. The only objects in the image are: a microwave oven, a kettle, four stoves, four stove knobs, a light knob, three cabinets, a robot, a sink, and a water tap. Whatever is not on this list is not in the image and should not be mentioned. "
image_file = "/mount/data/moving_kettle.png"

# load model
model_name = get_model_name_from_path(model_path)

# input from keyboard
while True:
    prompt = input("Enter prompt: ")
    if prompt == "exit":
        break

    prompt = base_prompt + prompt

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": model_name,
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    eval_model(args, tokenizer, model, image_processor)