import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
MID = "apple/FastVLM-1.5B"
IMAGE_TOKEN_INDEX = -200  # what the model code looks for
# Load
dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32

tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MID,
    dtype=dtype,             
    device_map=None,         
    trust_remote_code=True,
)
model = model.to("mps" if torch.backends.mps.is_available() else "cpu")

# Build chat -> render to string (not tokens) so we can place <image> exactly
messages = [
    {"role": "user", "content": "<image>\nDescribe this image in detail."}
]
rendered = tok.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)
pre, post = rendered.split("<image>", 1)
# Tokenize the text *around* the image token (no extra specials!)
pre_ids  = tok(pre,  return_tensors="pt", add_special_tokens=False).input_ids
post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids
# Splice in the IMAGE token id (-200) at the placeholder position
img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(model.device)
attention_mask = torch.ones_like(input_ids, device=model.device)
img = Image.open("image copy 3.png").convert("RGB")
px = model.get_vision_tower().image_processor(images=img, return_tensors="pt")["pixel_values"]
px = px.to(model.device, dtype=model.dtype)
# Generate
with torch.no_grad():
    out = model.generate(
        inputs=input_ids,
        attention_mask=attention_mask,
        images=px,
        max_new_tokens=256*3,
    )
print(tok.decode(out[0], skip_special_tokens=True))
