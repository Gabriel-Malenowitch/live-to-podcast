import os
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import List, Tuple, Optional

# Configuração para usar GPU se disponível
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Usando dispositivo: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name()}")
    print(f"   Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

class GPUAudioProcessor:
    def __init__(self, silence_threshold_db: float = -40, min_silence_duration: float = 1.0):
        """
        Processador de áudio otimizado para GPU
        
        Args:
            silence_threshold_db: Limiar de silêncio em dB
            min_silence_duration: Duração mínima de silêncio em segundos
        """
        self.silence_threshold_db = silence_threshold_db
        self.min_silence_duration = min_silence_duration
        self.device = device
        
    def db_to_amplitude(self, db: float) -> float:
        """Converte dB para amplitude"""
        return 10 ** (db / 20)
    
    def detect_silence_gpu(self, audio_tensor: torch.Tensor, sr: int) -> Tuple[int, int]:
        """
        Detecta início e fim do áudio não-silencioso usando GPU
        
        Args:
            audio_tensor: Tensor do áudio no dispositivo GPU
            sr: Taxa de amostragem
            
        Returns:
            Tuple com índices de início e fim do áudio não-silencioso
        """
        # Converte threshold de dB para amplitude
        threshold = self.db_to_amplitude(self.silence_threshold_db)
        
        # Calcula RMS em janelas para detecção mais robusta
        window_size = int(0.1 * sr)  # Janelas de 100ms
        hop_length = window_size // 4
        
        # Padding para garantir que temos janelas completas
        audio_padded = F.pad(audio_tensor, (0, window_size))
        
        # Calcula RMS usando convolução (mais eficiente na GPU)
        windows = audio_padded.unfold(0, window_size, hop_length)
        rms = torch.sqrt(torch.mean(windows ** 2, dim=1))
        
        # Encontra regiões não-silenciosas
        non_silent_mask = rms > threshold
        
        if not non_silent_mask.any():
            return 0, len(audio_tensor)
        
        # Encontra primeiro e último índice não-silencioso
        non_silent_indices = torch.where(non_silent_mask)[0]
        start_window = non_silent_indices[0].item()
        end_window = non_silent_indices[-1].item()
        
        # Converte índices de janela para índices de amostra
        start_sample = start_window * hop_length
        end_sample = min((end_window + 1) * hop_length + window_size, len(audio_tensor))
        
        return start_sample, end_sample
    
    def process_audio_file(self, file_path: str) -> bool:
        """
        Processa um arquivo de áudio individual
        
        Args:
            file_path: Caminho para o arquivo de áudio
            
        Returns:
            True se processado com sucesso, False caso contrário
        """
        try:
            filename = os.path.basename(file_path)
            print(f"  📁 Carregando: {filename}")
            
            # Carrega áudio usando librosa (mais rápido que pydub)
            audio_data, sr = librosa.load(file_path, sr=None, mono=True)
            original_duration = len(audio_data) / sr
            
            # Move para GPU
            audio_tensor = torch.from_numpy(audio_data).float().to(self.device)
            
            print(f"  🔍 Analisando silêncio...")
            start_idx, end_idx = self.detect_silence_gpu(audio_tensor, sr)
            
            # Extrai áudio não-silencioso
            trimmed_tensor = audio_tensor[start_idx:end_idx]
            trimmed_duration = len(trimmed_tensor) / sr
            
            if trimmed_duration < 0.1:  # Se muito curto, mantém original
                print(f"  ⚠️  Áudio muito curto após remoção, mantendo original")
                return True
            
            # Move de volta para CPU para salvar
            trimmed_audio = trimmed_tensor.cpu().numpy()
            
            print(f"  ⏱️  Duração: {original_duration:.1f}s → {trimmed_duration:.1f}s "
                  f"({((original_duration - trimmed_duration) / original_duration * 100):.1f}% removido)")
            
            # Salva o arquivo processado
            print(f"  💾 Salvando: {filename}")
            sf.write(file_path, trimmed_audio, sr)
            
            print(f"  ✅ Concluído: {filename}")
            return True
            
        except Exception as e:
            print(f"  ❌ Erro ao processar {filename}: {str(e)}")
            return False
    
    def process_batch_parallel(self, file_paths: List[str], max_workers: int = 4) -> None:
        """
        Processa múltiplos arquivos em paralelo
        
        Args:
            file_paths: Lista de caminhos dos arquivos
            max_workers: Número máximo de workers paralelos
        """
        print(f"🔄 Processando {len(file_paths)} arquivos com {max_workers} workers paralelos...")
        
        start_time = time.time()
        successful = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submete todas as tarefas
            future_to_file = {
                executor.submit(self.process_audio_file, file_path): file_path 
                for file_path in file_paths
            }
            
            # Processa resultados conforme completam
            for i, future in enumerate(as_completed(future_to_file), 1):
                file_path = future_to_file[future]
                filename = os.path.basename(file_path)
                
                print(f"\n[{i}/{len(file_paths)}] Processando: {filename}")
                
                try:
                    success = future.result()
                    if success:
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"  ❌ Erro inesperado: {str(e)}")
                    failed += 1
        
        elapsed_time = time.time() - start_time
        print(f"\n🎉 Processamento concluído!")
        print(f"   ⏱️  Tempo total: {elapsed_time:.1f}s")
        print(f"   ✅ Sucessos: {successful}")
        print(f"   ❌ Falhas: {failed}")
        print(f"   🚀 Velocidade média: {len(file_paths)/elapsed_time:.1f} arquivos/s")

def trim_silence_from_audio():
    """Função principal otimizada para GPU"""
    audio_dir = "audio"
    
    if not os.path.exists(audio_dir):
        print(f"❌ Pasta '{audio_dir}' não encontrada!")
        return

    # Extensões suportadas (librosa suporta mais formatos)
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma']

    # Encontra arquivos de áudio
    audio_files = []
    for file in os.listdir(audio_dir):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            audio_files.append(os.path.join(audio_dir, file))
    
    if not audio_files:
        print(f"❌ Nenhum arquivo de áudio encontrado na pasta '{audio_dir}'!")
        return
    
    print(f"🎵 Encontrados {len(audio_files)} arquivo(s) de áudio:")
    for file_path in audio_files:
        filename = os.path.basename(file_path)
        print(f"   • {filename}")

    # Configura processador GPU
    processor = GPUAudioProcessor(
        silence_threshold_db=-40,  # Ajuste conforme necessário
        min_silence_duration=1.0   # 1 segundo
    )
    
    # Determina número de workers baseado na GPU
    if torch.cuda.is_available():
        # Para GPU, usa menos workers para evitar sobrecarga de memória
        max_workers = min(4, len(audio_files))
    else:
        # Para CPU, pode usar mais workers
        max_workers = min(8, len(audio_files))
    
    # Processa arquivos em paralelo
    processor.process_batch_parallel(audio_files, max_workers=max_workers)

if __name__ == "__main__":
    trim_silence_from_audio()