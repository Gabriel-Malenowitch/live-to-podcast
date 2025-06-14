#!/usr/bin/env python3
"""
Script para verificar se o sistema está pronto para processamento GPU
"""

import sys
import subprocess
import importlib

def check_python_version():
    """Verifica versão do Python"""
    print("🐍 Verificando Python...")
    version = sys.version_info
    print(f"   Versão: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("   ✅ Python OK")
        return True
    else:
        print("   ❌ Python 3.8+ necessário")
        return False

def check_nvidia_gpu():
    """Verifica se há GPU NVIDIA disponível"""
    print("\n🎮 Verificando GPU NVIDIA...")
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GeForce' in line or 'RTX' in line or 'GTX' in line:
                    gpu_info = line.strip()
                    print(f"   GPU encontrada: {gpu_info}")
                    print("   ✅ GPU NVIDIA OK")
                    return True
            print("   ⚠️  nvidia-smi funcionou mas GPU não identificada")
            return False
    except FileNotFoundError:
        print("   ❌ nvidia-smi não encontrado")
        print("   💡 Instale os drivers NVIDIA")
        return False
    except Exception as e:
        print(f"   ❌ Erro ao verificar GPU: {e}")
        return False

def check_cuda():
    """Verifica se CUDA está disponível"""
    print("\n🔥 Verificando CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   Versão CUDA: {torch.version.cuda}")
            print(f"   Dispositivos CUDA: {torch.cuda.device_count()}")
            print(f"   GPU atual: {torch.cuda.get_device_name()}")
            print("   ✅ CUDA OK")
            return True
        else:
            print("   ❌ CUDA não disponível no PyTorch")
            return False
    except ImportError:
        print("   ❌ PyTorch não instalado")
        return False

def check_required_packages():
    """Verifica se os pacotes necessários estão instalados"""
    print("\n📦 Verificando pacotes necessários...")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchaudio': 'TorchAudio', 
        'librosa': 'Librosa',
        'soundfile': 'SoundFile',
        'numpy': 'NumPy',
        'scipy': 'SciPy'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} não instalado")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_audio_files():
    """Verifica se há arquivos de áudio para processar"""
    print("\n🎵 Verificando arquivos de áudio...")
    
    import os
    audio_dir = "audio"
    
    if not os.path.exists(audio_dir):
        print(f"   ❌ Pasta '{audio_dir}' não encontrada")
        return False
    
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']
    audio_files = []
    
    for file in os.listdir(audio_dir):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            audio_files.append(file)
    
    if audio_files:
        print(f"   ✅ {len(audio_files)} arquivo(s) encontrado(s):")
        for file in audio_files[:5]:  # Mostra até 5 arquivos
            print(f"      • {file}")
        if len(audio_files) > 5:
            print(f"      ... e mais {len(audio_files) - 5} arquivo(s)")
        return True
    else:
        print("   ⚠️  Nenhum arquivo de áudio encontrado")
        print("   💡 Adicione arquivos na pasta 'audio/'")
        return False

def estimate_performance():
    """Estima performance esperada"""
    print("\n📊 Estimativa de Performance...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"   GPU: {gpu_name}")
            print(f"   Memória: {memory_gb:.1f} GB")
            
            # Estimativas baseadas na GPU
            if "RTX 40" in gpu_name or "RTX 30" in gpu_name:
                speedup = "20-50x"
                rating = "🚀 Excelente"
            elif "RTX 20" in gpu_name or "GTX 16" in gpu_name:
                speedup = "10-30x"
                rating = "⚡ Muito Bom"
            elif "GTX" in gpu_name:
                speedup = "5-15x"
                rating = "👍 Bom"
            else:
                speedup = "2-10x"
                rating = "✅ Adequado"
            
            print(f"   Speedup esperado: {speedup}")
            print(f"   Classificação: {rating}")
            
        else:
            print("   🖥️  Modo CPU apenas")
            print("   Speedup: 1x (sem aceleração)")
            
    except ImportError:
        print("   ❌ Não foi possível estimar performance")

def main():
    """Função principal"""
    print("🔍 VERIFICAÇÃO DO SISTEMA")
    print("=" * 50)
    
    checks = []
    
    # Verifica Python
    checks.append(check_python_version())
    
    # Verifica GPU
    checks.append(check_nvidia_gpu())
    
    # Verifica CUDA
    checks.append(check_cuda())
    
    # Verifica pacotes
    packages_ok, missing = check_required_packages()
    checks.append(packages_ok)
    
    # Verifica arquivos
    checks.append(check_audio_files())
    
    # Estima performance
    estimate_performance()
    
    # Resumo final
    print("\n" + "=" * 50)
    print("📋 RESUMO")
    print("=" * 50)
    
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print("🎉 Sistema totalmente pronto!")
        print("   Execute: python manage-silence.py")
    elif passed >= 3:
        print("⚠️  Sistema parcialmente pronto")
        if not packages_ok:
            print("   Execute: python install_gpu_deps.py")
        print("   Algumas funcionalidades podem não funcionar")
    else:
        print("❌ Sistema não está pronto")
        print("   Resolva os problemas acima antes de continuar")
    
    print(f"\n✅ Verificações passaram: {passed}/{total}")
    
    if not packages_ok:
        print(f"\n📦 Pacotes faltando: {', '.join(missing)}")
        print("💡 Execute: python install_gpu_deps.py")

if __name__ == "__main__":
    main()