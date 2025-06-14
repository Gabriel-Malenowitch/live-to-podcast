#!/usr/bin/env python3
"""
Script de benchmark para comparar performance CPU vs GPU
"""

import time
import os
import torch
import numpy as np
import sys
sys.path.append('.')
from manage_silence import GPUAudioProcessor
import librosa
import soundfile as sf

def create_test_audio(duration_seconds=60, sample_rate=44100, output_path="test_audio.wav"):
    """Cria um arquivo de áudio de teste com silêncio no início e fim"""
    
    # Gera áudio de teste
    t = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate))
    
    # Sinal principal (música simulada)
    main_signal = (
        0.3 * np.sin(2 * np.pi * 440 * t) +  # Nota A4
        0.2 * np.sin(2 * np.pi * 880 * t) +  # Nota A5
        0.1 * np.sin(2 * np.pi * 220 * t)    # Nota A3
    )
    
    # Adiciona silêncio no início (5 segundos)
    silence_start = np.zeros(int(5 * sample_rate))
    
    # Adiciona silêncio no fim (3 segundos)
    silence_end = np.zeros(int(3 * sample_rate))
    
    # Combina tudo
    full_audio = np.concatenate([silence_start, main_signal, silence_end])
    
    # Salva arquivo
    sf.write(output_path, full_audio, sample_rate)
    print(f"✅ Arquivo de teste criado: {output_path} ({len(full_audio)/sample_rate:.1f}s)")
    
    return output_path

def benchmark_cpu_processing(file_path):
    """Benchmark do processamento CPU (simulado)"""
    print("\n🖥️  BENCHMARK CPU")
    print("-" * 40)
    
    start_time = time.time()
    
    # Simula processamento CPU tradicional
    audio_data, sr = librosa.load(file_path, sr=None)
    
    # Detecção de silêncio simples (CPU)
    threshold = 10 ** (-40 / 20)  # -40 dB
    
    # Encontra regiões não-silenciosas
    non_silent = np.abs(audio_data) > threshold
    
    if non_silent.any():
        start_idx = np.where(non_silent)[0][0]
        end_idx = np.where(non_silent)[0][-1]
        trimmed_audio = audio_data[start_idx:end_idx]
    else:
        trimmed_audio = audio_data
    
    cpu_time = time.time() - start_time
    
    print(f"⏱️  Tempo CPU: {cpu_time:.3f}s")
    print(f"📊 Duração original: {len(audio_data)/sr:.1f}s")
    print(f"📊 Duração processada: {len(trimmed_audio)/sr:.1f}s")
    
    return cpu_time, len(trimmed_audio)

def benchmark_gpu_processing(file_path):
    """Benchmark do processamento GPU"""
    print("\n🚀 BENCHMARK GPU")
    print("-" * 40)
    
    start_time = time.time()
    
    # Usa o processador GPU otimizado
    processor = GPUAudioProcessor(silence_threshold_db=-40)
    
    # Carrega e processa
    audio_data, sr = librosa.load(file_path, sr=None)
    audio_tensor = torch.from_numpy(audio_data).float().to(processor.device)
    
    start_idx, end_idx = processor.detect_silence_gpu(audio_tensor, sr)
    trimmed_tensor = audio_tensor[start_idx:end_idx]
    
    gpu_time = time.time() - start_time
    
    print(f"⏱️  Tempo GPU: {gpu_time:.3f}s")
    print(f"📊 Duração original: {len(audio_data)/sr:.1f}s")
    print(f"📊 Duração processada: {len(trimmed_tensor)/sr:.1f}s")
    print(f"🎯 Dispositivo: {processor.device}")
    
    return gpu_time, len(trimmed_tensor)

def run_comprehensive_benchmark():
    """Executa benchmark completo"""
    print("🏁 BENCHMARK COMPLETO - CPU vs GPU")
    print("=" * 60)
    
    # Verifica se GPU está disponível
    if not torch.cuda.is_available():
        print("⚠️  GPU não disponível - executando apenas teste CPU")
        gpu_available = False
    else:
        gpu_available = True
        print(f"🚀 GPU detectada: {torch.cuda.get_device_name()}")
    
    # Cria arquivos de teste de diferentes tamanhos
    test_files = []
    durations = [30, 60, 120, 300]  # 30s, 1min, 2min, 5min
    
    print(f"\n📁 Criando arquivos de teste...")
    for duration in durations:
        filename = f"test_{duration}s.wav"
        create_test_audio(duration, output_path=filename)
        test_files.append((filename, duration))
    
    # Executa benchmarks
    results = []
    
    for filename, duration in test_files:
        print(f"\n" + "="*60)
        print(f"🎵 TESTANDO ARQUIVO: {filename} ({duration}s)")
        print("="*60)
        
        # Benchmark CPU
        cpu_time, cpu_samples = benchmark_cpu_processing(filename)
        
        # Benchmark GPU (se disponível)
        if gpu_available:
            gpu_time, gpu_samples = benchmark_gpu_processing(filename)
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        else:
            gpu_time = 0
            gpu_samples = 0
            speedup = 0
        
        results.append({
            'duration': duration,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup,
            'filename': filename
        })
        
        # Limpa arquivo de teste
        os.remove(filename)
    
    # Mostra resultados finais
    print(f"\n" + "="*60)
    print("📊 RESULTADOS FINAIS")
    print("="*60)
    
    print(f"{'Duração':<10} {'CPU (s)':<10} {'GPU (s)':<10} {'Speedup':<10}")
    print("-" * 45)
    
    total_cpu_time = 0
    total_gpu_time = 0
    
    for result in results:
        total_cpu_time += result['cpu_time']
        total_gpu_time += result['gpu_time']
        
        if gpu_available:
            print(f"{result['duration']}s{'':<6} {result['cpu_time']:<10.3f} "
                  f"{result['gpu_time']:<10.3f} {result['speedup']:<10.1f}x")
        else:
            print(f"{result['duration']}s{'':<6} {result['cpu_time']:<10.3f} {'N/A':<10} {'N/A':<10}")
    
    print("-" * 45)
    
    if gpu_available and total_gpu_time > 0:
        overall_speedup = total_cpu_time / total_gpu_time
        print(f"{'TOTAL':<10} {total_cpu_time:<10.3f} {total_gpu_time:<10.3f} {overall_speedup:<10.1f}x")
        
        print(f"\n🎉 RESUMO:")
        print(f"   🖥️  Tempo total CPU: {total_cpu_time:.3f}s")
        print(f"   🚀 Tempo total GPU: {total_gpu_time:.3f}s")
        print(f"   ⚡ Speedup médio: {overall_speedup:.1f}x")
        print(f"   💾 Economia de tempo: {((total_cpu_time - total_gpu_time) / total_cpu_time * 100):.1f}%")
    else:
        print(f"{'TOTAL':<10} {total_cpu_time:<10.3f} {'N/A':<10} {'N/A':<10}")
        print(f"\n📝 Para ver o speedup da GPU, instale CUDA e PyTorch com suporte GPU")

def main():
    """Função principal"""
    print("🔥 BENCHMARK: Processamento de Áudio CPU vs GPU")
    print("Este benchmark vai testar a performance do seu sistema")
    print()
    
    response = input("Deseja executar o benchmark completo? (s/n): ").lower().strip()
    
    if response in ['s', 'sim', 'y', 'yes']:
        run_comprehensive_benchmark()
    else:
        print("Benchmark cancelado.")

if __name__ == "__main__":
    main()