import sys
sys.path.insert(0, "/home/mazaveri/hpc-prog/sankarshan/repos/concept-graphs")
from conceptgraph.llava.llava_model_hf import LLaVaChat

chat = LLaVaChat("/home/mazaveri/hpc-prog/sankarshan/checkpoints/llava-1.5-7b-hf")
image = chat.load_image("/home/mazaveri/hpc-prog/sankarshan/outputs/pointclouds/room0_inputs.png")
pixel_values = chat.image_processor(image, return_tensors="pt")["pixel_values"][0]
features = chat.encode_image(pixel_values[None, ...].half().cuda())
chat.reset()
print(chat(query="Describe the central object in the image.", image_features=features))
