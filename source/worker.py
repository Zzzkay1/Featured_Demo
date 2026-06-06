import sys
import json
import traceback
import os

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python worker.py <task> <audio_path> [lang]")
        sys.exit(1)

    task = sys.argv[1]
    audio_path = sys.argv[2]
    
    try:
        from MusicTools import MusicTools
        if task == "asr":
            input_language = sys.argv[3] if len(sys.argv) > 3 else "zh"
            result = MusicTools.run_ASR(audio_path, input_language)
            print("\n---RESULT_START---")
            print(json.dumps(result, ensure_ascii=False))
            print("---RESULT_END---\n")
            
        elif task == "separation":
            result = MusicTools.run_separation(audio_path)
            print("\n---RESULT_START---")
            print(json.dumps(result, ensure_ascii=False))
            print("---RESULT_END---\n")
            
        else:
            raise ValueError(f"Unknown task: {task}")
            
    except Exception as e:
        error_msg = f"Worker Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        sys.exit(1)
