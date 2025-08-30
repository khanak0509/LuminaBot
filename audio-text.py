import whisper

with open('data.txt','a') as f:
    f.write('Now the file has more content')
    f.write('khanak')


model = whisper.load_model("turbo")
result = model.transcribe("RAG vs Fine-Tuning vs Prompt Engineering_ Optimizing AI Models.mp3")
print(result["text"])
