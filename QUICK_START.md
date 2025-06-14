# 🚀 Guia de Início Rápido - GPU

## ⚡ 3 Passos para Usar sua RTX

### 1️⃣ Verificar Sistema
```bash
python check_system.py
```
- Verifica se sua RTX está funcionando
- Confirma se Python está OK
- Lista arquivos de áudio encontrados

### 2️⃣ Instalar (só na primeira vez)
```bash
python install_gpu_deps.py
```
- Instala PyTorch com CUDA
- Instala bibliotecas de áudio otimizadas
- Configura tudo automaticamente

### 3️⃣ Processar Áudio
```bash
python manage-silence.py
```
- Processa todos os arquivos na pasta `audio/`
- Usa sua GPU para máxima velocidade
- Remove silêncio automaticamente

## 📁 Preparação

1. **Coloque arquivos** na pasta `audio/`
2. **Faça backup** (recomendado)
3. **Execute** o processador

## 🎯 Formatos Suportados

✅ WAV, MP3, FLAC, OGG, AAC, M4A, WMA

## ⚡ Performance Esperada

| GPU | Speedup | Tempo (10 arquivos) |
|-----|---------|-------------------|
| RTX 4090 | 50x | ~30s |
| RTX 4070 | 30x | ~45s |
| RTX 3080 | 25x | ~60s |
| RTX 3060 | 15x | ~90s |

## 🔧 Problemas Comuns

### GPU não detectada?
```bash
nvidia-smi
```
Se não funcionar, atualize drivers NVIDIA.

### Erro de instalação?
Execute novamente:
```bash
python install_gpu_deps.py
```

### Quer testar performance?
```bash
python benchmark.py
```

---

🎉 **Pronto! Sua RTX está otimizada para processamento de áudio!**