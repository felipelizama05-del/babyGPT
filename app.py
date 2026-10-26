import json
import os
import math
import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_functional
import streamlit as st

st.set_page_config(
    page_title="MiniGPT Generador",
    page_icon="🤖",
    layout="centered"
)

FICHERO_PESOS = "modelo_pesos.pth"
FICHERO_VOCAB = "vocabulario.json"

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class AtencionCausal(torch_nn.Module):
    def __init__(self, d_model, n_head, max_seq_len, dropout):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv_proj = torch_nn.Linear(d_model, 3 * d_model)
        self.out_proj = torch_nn.Linear(d_model, d_model)
        self.attn_dropout = torch_nn.Dropout(dropout)
        self.res_dropout = torch_nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Calculamos la máscara al vuelo exactamente igual que en el entrenamiento
        mask = torch.tril(torch.ones(T, T, device=x.device)) == 0
        att = att.masked_fill(mask, float('-inf'))
        
        att = torch_functional.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.res_dropout(self.out_proj(y))

class BloqueTransformer(torch_nn.Module):
    def __init__(self, d_model, n_head, max_seq_len, dropout):
        super().__init__()
        self.ln1 = torch_nn.LayerNorm(d_model)
        self.attn = AtencionCausal(d_model, n_head, max_seq_len, dropout)
        self.ln2 = torch_nn.LayerNorm(d_model)
        self.mlp = torch_nn.Sequential(
            torch_nn.Linear(d_model, 4 * d_model),
            torch_nn.GELU(),
            torch_nn.Linear(4 * d_model, d_model),
            torch_nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MiniGPT(torch_nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, max_seq_len, dropout):
        super().__init__()
        self.token_emb = torch_nn.Embedding(vocab_size, d_model)
        self.pos_emb = torch_nn.Embedding(max_seq_len, d_model)
        self.drop = torch_nn.Dropout(dropout)
        self.blocks = torch_nn.Sequential(*[BloqueTransformer(d_model, n_head, max_seq_len, dropout) for _ in range(n_layer)])
        self.ln_f = torch_nn.LayerNorm(d_model)
        self.head = torch_nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

# --- HIPERPARÁMETROS ACTUALIZADOS AL ENTRENAMIENTO ---
d_model = 192
n_head = 6
n_layer = 4
max_seq_len = 128
dropout = 0.1

@st.cache_resource
def cargar_recursos():
    if not os.path.exists(FICHERO_PESOS) or not os.path.exists(FICHERO_VOCAB):
        return None, None, None, None
    with open(FICHERO_VOCAB, "r", encoding="utf-8") as f:
        contenido = json.load(f)
        caracteres = contenido["caracteres"] if isinstance(contenido, dict) and "caracteres" in contenido else contenido
        
    vocab_size = len(caracteres)
    char_a_int = {ch: i for i, ch in enumerate(caracteres)}
    int_a_char = {i: ch for i, ch in enumerate(caracteres)}
    modelo = MiniGPT(vocab_size, d_model, n_head, n_layer, max_seq_len, dropout).to(device)
    modelo.load_state_dict(torch.load(FICHERO_PESOS, map_location=device))
    modelo.eval()
    return modelo, char_a_int, int_a_char, vocab_size

modelo, char_a_int, int_a_char, vocab_size = cargar_recursos()

st.title("🤖 Generador de Texto con FlariscaAI")
st.write("Interfaz web interactiva para generar texto.")
st.write("Parámetros: 1.84M")

if modelo is None:
    st.error("Error: No se encuentran 'modelo_pesos.pth' o 'vocabulario.json'. Asegúrate de haber entrenado el modelo primero.")
else:
    st.sidebar.header("Configuración")
    caracteres_a_generar = st.sidebar.slider("Caracteres a generar", 50, 1000, 300, 50)
    temperatura = st.sidebar.slider("Temperatura", 0.1, 2.0, 1.0, 0.1)

    prompt = st.text_input("Ingresa una frase inicial:", "En un lugar de la mancha")

    if st.button("Generar Texto", type="primary"):
        encode = lambda s: [char_a_int[c] for c in s if c in char_a_int]
        decode = lambda l: "".join([int_a_char[i] for i in l])
        tokens_iniciales = encode(prompt)
        
        if not tokens_iniciales:
            st.warning("⚠️ Los caracteres ingresados no están en el vocabulario.")
        else:
            contexto = torch.tensor([tokens_iniciales], dtype=torch.long, device=device)
            with st.spinner("Generando..."):
                with torch.no_grad():
                    for _ in range(caracteres_a_generar):
                        cond = contexto[:, -max_seq_len:]
                        logits = modelo(cond)
                        logits = logits[:, -1, :] / temperatura
                        probs = torch_functional.softmax(logits, dim=-1)
                        siguiente_char = torch.multinomial(probs, num_samples=1)
                        contexto = torch.cat((contexto, siguiente_char), dim=1)
            
            st.subheader("Resultado:")
            st.text_area("Texto generado:", value=decode(contexto[0].tolist()), height=200)
