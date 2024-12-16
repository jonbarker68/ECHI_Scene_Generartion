import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torchaudio
import json


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

audio, fs = torchaudio.load("/fastdata/acp19rws/CHiME9_Pilots/EarlyTests/test0_close_talk_short.wav")
audio = audio.squeeze(0).detach().cpu().numpy()
result = pipe(audio, return_timestamps=True)
print(result)

with open("whisper_eg.json", "w") as file:
    json.dump(result, file, indent=4)