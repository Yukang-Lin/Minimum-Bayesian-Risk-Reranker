import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.flop_counter import FlopCounterMode
import time
import debugpy
try:
    debugpy.listen(('localhost', 9503))
    print('waiting for debugger attach')
    debugpy.wait_for_client()
except:
    raise Exception("Debugpy is not installed. Please install it by running 'pip install debugpy'.")

time1 = time.time()
# 加载模型和分词器
model_name = "/new_disk/models_for_all/llama-2-7b-chat-hf"  # 请替换为实际的模型名称
# model_name = '/new_disk/models_for_all/gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
time2 = time.time()
print(f"加载模型耗时: {time2 - time1:.2f}s")

inputs = tokenizer('hello, how are you?', return_tensors='pt').to(model.device)
## FLOPS
# flop_counter = FlopCounterMode(model)
# with flop_counter:
#     outputs = model(**inputs)
#     # outputs = model.generate(**inputs, max_length=50)
# # print(tokenizer.decode(outputs[0]))
# print(f"总FLOPs: {flop_counter.get_total_flops() / 1e9:.2f} GFLOPs")
# print(f"每token FLOPs: {flop_counter.get_total_flops() / 50 / 1e9:.2f} GFLOPs/token")

# from thop import profile

# inputs = {"input_ids": torch.tensor([[1, 2, 3]])}  # 示例输入
# flops, params = profile(model, inputs=(inputs,))
# print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")

## 吞吐量
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()
start_event.record()
outputs = model.generate(**inputs, max_new_tokens=50)
end_event.record()
torch.cuda.synchronize()

time_ms = start_event.elapsed_time(end_event)
num_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
print(f"吞吐量: {num_tokens / (time_ms / 1000):.3f} tokens/second")