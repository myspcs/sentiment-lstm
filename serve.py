import torch
import gradio as gr
import pickle
from model import LSTMModel

device = "cpu"

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

def encode(text, max_len=30):
    ids = [vocab.get(w, 0) for w in text.lower().split()]
    ids = ids[:max_len]
    return ids + [0] * (max_len - len(ids))

model = LSTMModel(len(vocab))
model.load_state_dict(torch.load("sentiment_lstm.pth", map_location=device))
model.eval()

def predict(text):
    x = torch.tensor([encode(text)])
    with torch.no_grad():
        p = model(x).item()
    return "Pozitif" if p > 0.5 else "Negatif"

gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=2, placeholder="Film yorumu gir"),
    outputs="text",
    title="Türkçe Duygu Analizi (LSTM)",
    description="LSTM tabanlı Türkçe film yorumu duygu analizi").launch(share=True)
