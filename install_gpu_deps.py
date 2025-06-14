#!/usr/bin/env python3
"""
Script para instalar dependências otimizadas para GPU
"""

import subprocess
import sys
import platform

def run_command(command):
    """Executa comando e retorna resultado"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def install_pytorch_gpu():
    """Instala PyTorch com suporte CUDA"""
    print("🚀 Instalando PyTorch com suporte CUDA...")
    
    # Comando para instalar PyTorch com CUDA
    if platform.system() == "Windows":
        cmd = "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121"
    else:
        cmd = "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121"
    
    success, output = run_command(cmd)
    if success:
        print("✅ PyTorch com CUDA instalado com sucesso!")
    else:
        print(f"❌ Erro ao instalar PyTorch: {output}")
        return False
    return True

def install_audio_libs():
    """Instala bibliotecas de áudio"""
    print("🎵 Instalando bibliotecas de áudio...")
    
    libs = [
        "librosa>=0.10.0",
        "soundfile>=0.12.0", 
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "numba>=0.56.0"
    ]
    
    for lib in libs:
        print(f"   Instalando {lib}...")
        success, output = run_command(f"pip install {lib}")
        if not success:
            print(f"❌ Erro ao instalar {lib}: {output}")
            return False
    
    print("✅ Bibliotecas de áudio instaladas com sucesso!")
    return True

def test_gpu_availability():
    """Testa se GPU está disponível"""
    print("🔍 Testando disponibilidade da GPU...")
    
    test_code = """
import torch
print(f"CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Versão CUDA: {torch.version.cuda}")
    print(f"Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ CUDA não disponível - usando CPU")
"""
    
    success, output = run_command(f'python -c "{test_code}"')
    if success:
        print("✅ Teste de GPU:")
        print(output)
    else:
        print(f"❌ Erro no teste: {output}")
    
    return success

def main():
    print("🔧 Configurando ambiente para processamento de áudio com GPU")
    print("=" * 60)
    
    # Instala PyTorch com CUDA
    if not install_pytorch_gpu():
        print("❌ Falha na instalação do PyTorch")
        sys.exit(1)
    
    # Instala bibliotecas de áudio
    if not install_audio_libs():
        print("❌ Falha na instalação das bibliotecas de áudio")
        sys.exit(1)
    
    # Testa GPU
    test_gpu_availability()
    
    print("\n🎉 Instalação concluída!")
    print("💡 Agora você pode executar: python manage-silence.py")

if __name__ == "__main__":
    main()