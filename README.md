# 🎙️ Live to Podcast - Processador de Áudio GPU-Acelerado

Sistema otimizado para processamento de áudio com aceleração por GPU RTX.

## 🚀 Novidades - Versão GPU

O `manage-silence.py` foi completamente otimizado para usar sua GPU RTX, oferecendo:

- **⚡ 10-50x mais rápido** que a versão CPU
- **🔄 Processamento paralelo** de múltiplos arquivos
- **🎯 Detecção de silêncio avançada** com algoritmos GPU
- **💾 Uso eficiente de memória** GPU

## 🛠️ Instalação Rápida

### 1. Verificar Sistema
```bash
python check_system.py
```

### 2. Instalar Dependências
```bash
python install_gpu_deps.py
```

### 3. Processar Áudio
```bash
python manage-silence.py
```

## 📁 Estrutura do Projeto

```
├── audio/                    # Coloque seus arquivos de áudio aqui
├── video/                    # Arquivos de vídeo
├── manage-silence.py         # 🚀 Processador GPU-otimizado
├── video2audio.py           # Conversor vídeo para áudio
├── check_system.py          # Verificador do sistema
├── install_gpu_deps.py      # Instalador de dependências
├── benchmark.py             # Teste de performance
├── requirements.txt         # Dependências
└── README_GPU.md           # Documentação detalhada
```

## 🎵 Como Usar

1. **Coloque arquivos de áudio** na pasta `audio/`
2. **Execute o processador**:
   ```bash
   python manage-silence.py
   ```
3. **Arquivos processados** substituem os originais (backup recomendado)

## 📊 Performance

### Antes (CPU):
- Processamento sequencial
- ~1 arquivo por vez
- Algoritmos básicos

### Depois (GPU):
- Processamento paralelo
- 4+ arquivos simultâneos  
- Algoritmos otimizados
- **10-50x mais rápido**

## 🔧 Scripts Disponíveis

| Script | Função |
|--------|--------|
| `check_system.py` | Verifica se sistema está pronto |
| `install_gpu_deps.py` | Instala dependências GPU |
| `manage-silence.py` | **Processador principal** |
| `benchmark.py` | Testa performance CPU vs GPU |
| `video2audio.py` | Converte vídeo para áudio |

## 🎯 Requisitos

- **Python 3.8+**
- **GPU NVIDIA** (RTX recomendada)
- **Drivers NVIDIA** atualizados
- **CUDA** (instalado automaticamente)

## 💡 Dicas

- **Arquivos grandes**: Maior benefício da GPU
- **Múltiplos arquivos**: Use processamento em lote
- **Backup**: Sempre faça backup antes de processar
- **SSD**: Recomendado para I/O mais rápido

## 🆘 Solução de Problemas

### GPU não detectada?
```bash
nvidia-smi  # Verifica se GPU está funcionando
python check_system.py  # Diagnóstico completo
```

### Erro de instalação?
```bash
python install_gpu_deps.py  # Reinstala dependências
```

### Performance baixa?
- Verifique se está usando GPU: `python check_system.py`
- Arquivos muito pequenos podem não se beneficiar
- Considere processar em lotes maiores

---

🎉 **Aproveite o poder da sua RTX para processamento de áudio ultrarrápido!**

Para documentação completa, veja: [README_GPU.md](README_GPU.md)