# utils/generation_wrapper.py

from transformers import AutoTokenizer, AutoModelForCausalLM

class GenerationWrapper:
    def __init__(self, model_name, device):
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    def generate_text(self, prompt, max_length=50, temperature=0.8, do_sample=True):
        """
        Generate text using the model based on a given prompt.
        """
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample
            )
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

