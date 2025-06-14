import os
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

def trim_silence_from_audio():
    audio_dir = "audio"
    silence_threshold = -40  # dBFS (ajuste conforme necessário)
    min_silence_len = 1000   # ms (1 segundo)

    if not os.path.exists(audio_dir):
        print(f"Pasta '{audio_dir}' não encontrada!")
        return

    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']

    audio_files = []
    for file in os.listdir(audio_dir):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            audio_files.append(file)
    
    if not audio_files:
        print(f"Nenhum arquivo de áudio encontrado na pasta '{audio_dir}'!")
        return
    
    print(f"Encontrados {len(audio_files)} arquivo(s) de áudio para processar:")
    for file in audio_files:
        print(f"  - {file}")

    for i, audio_file in enumerate(audio_files, 1):
        try:
            print(f"\n[{i}/{len(audio_files)}] Processando: {audio_file}")
            

            audio_path = os.path.join(audio_dir, audio_file)
            

            print(f"  Carregando áudio: {audio_file}")
            audio = AudioSegment.from_file(audio_path)
            
            print(f"  Removendo silêncio...")
            nonsilent_ranges = detect_nonsilent(
                audio, 
                min_silence_len=min_silence_len, 
                silence_thresh=silence_threshold
            )

            if nonsilent_ranges:
                start_trim = nonsilent_ranges[0][0]
                end_trim = nonsilent_ranges[-1][1]
                trimmed_audio = audio[start_trim:end_trim]
                
                duration_before = len(audio) / 1000
                duration_after = len(trimmed_audio) / 1000
                print(f"  Duração: {duration_before:.1f}s → {duration_after:.1f}s")
                
                # Salva o áudio processado (substitui o original)
                print(f"  Salvando áudio processado: {audio_file}")
                trimmed_audio.export(audio_path, format="wav")
                print(f"  ✓ Concluído: {audio_file}")
                
            else:
                print(f"  ⚠️ Áudio muito silencioso, mantendo original")
                print(f"  ✓ Mantido sem alterações: {audio_file}")
            
        except Exception as e:
            print(f"  ✗ Erro ao processar {audio_file}: {str(e)}")
            continue
    
    print(f"\n🎉 Processamento de remoção de silêncio concluído!")

if __name__ == "__main__":
    trim_silence_from_audio()