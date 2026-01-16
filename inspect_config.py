from transformers import AutoConfig
import traceback

print("Starting inspection...")
try:
    config = AutoConfig.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True)
    print("Config loaded.")
    print("Config class:", config.__class__.__name__)
    print("Has num_attention_heads:", hasattr(config, "num_attention_heads"))
    
    # Check for other common attention head attributes
    for attr in ["n_head", "num_heads", "attention_heads", "encoder_attention_heads", "decoder_attention_heads"]:
        if hasattr(config, attr):
            print(f"{attr}:", getattr(config, attr))
            
    # Check inside vision_config or text_config if they exist
    if hasattr(config, "vision_config"):
        print("Has vision_config")
    if hasattr(config, "text_config"):
        print("Has text_config")
        
except Exception:
    traceback.print_exc()
