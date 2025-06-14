import os
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

def convert_videos_to_audio():
    video_dir = r"D:\shorts-igreja\to-do podcasts"
    audio_dir = r"D:\shorts-igreja\podcasts-done"
    silence_threshold = -40  # dBFS (ajuste conforme necessário)
    min_silence_len = 1000   # ms (1 segundo)
    
    if not os.path.exists(video_dir):
        print(f"Pasta '{video_dir}' não encontrada!")
        return
    
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
        print(f"Pasta '{audio_dir}' criada.")
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    
    video_files = []
    for file in os.listdir(video_dir):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(file)
    
    if not video_files:
        print(f"Nenhum arquivo de vídeo encontrado na pasta '{video_dir}'!")
        return
    
    print(f"Encontrados {len(video_files)} arquivo(s) de vídeo para processar:")
    for file in video_files:
        print(f"  - {file}")
    
    for i, video_file in enumerate(video_files, 1):
        try:
            print(f"\n[{i}/{len(video_files)}] Processando: {video_file}")
            
            video_path = os.path.join(video_dir, video_file)
            
            audio_filename = os.path.splitext(video_file)[0] + ".wav"
            audio_path = os.path.join(audio_dir, audio_filename)
            
            print(f"  Extraindo áudio de: {video_file}")
            audio = AudioSegment.from_file(video_path)

            print(f"  Salvando áudio como: {audio_filename}")
            audio.export(audio_path, format="wav")
            
            print(f"  ✓ Concluído: {audio_filename}")
            
        except Exception as e:
            print(f"  ✗ Erro ao processar {video_file}: {str(e)}")
            continue
    
    print(f"\n🎉 Processamento concluído! Arquivos de áudio salvos na pasta '{audio_dir}'.")

if __name__ == "__main__":
    convert_videos_to_audio()