import json
import os
import threading
import queue
from timeit import default_timer as timer
from typing_extensions import final
import whisper

import torch
import torchaudio

from pyannote.audio import Pipeline
from pyannote_whisper.utils import diarize_text
from pyannote.audio.pipelines.utils.hook import ProgressHook



torch.cuda.init()

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token="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")
pipeline = pipeline.to(torch.device('cuda'))


model = whisper.load_model("large-v1", device="cuda")
print(model)

audio_dir = "../../data/audios/audios/"
json_dir = "../../data/transcripts/raw-large/"


files = os.listdir(audio_dir)

def transcribe_par(model, audio_dir, audio_file, result_queue):
    print(f"Transcribing {audio_file}...")
    start = timer()
    asr_result = model.transcribe(audio_dir + audio_file)
    print("Transcription time: %r" % (timer() - start))
    result_queue.put(("asr", asr_result))

def diarize_par(pipeline, audio_dir, audio_file, result_queue):
    print(f"Diarizing {audio_file}...")
    

    start = timer()
    with ProgressHook() as hook:
        waveform, sample_rate = torchaudio.load(audio_dir + audio_file)
        diarization_result = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)
    print("Diarization time: %r" % (timer() - start))
    result_queue.put(("dia", diarization_result))


print("STARTING TO TRANSCRIBE %r FILES" % len(files))

for audio_file in files: 
    
    if audio_file.replace(".wav", ".json") in os.listdir(json_dir):
        print(f"Continue {audio_file}...")
        continue
    try: 
        result_queue = queue.Queue()
        transcription_thread = threading.Thread(target=transcribe_par, args=(model, audio_dir, audio_file, result_queue))
        diarization_thread = threading.Thread(target=diarize_par, args=(pipeline, audio_dir, audio_file, result_queue))

        transcription_thread.start()
        diarization_thread.start()

        transcription_thread.join()
        diarization_thread.join()

        while not result_queue.empty():
            idx, result = result_queue.get()
            if idx=="asr":
                asr_result = result
            else:
                diarization_result = result

        
        final_result = diarize_text(asr_result, diarization_result)
        
        final_result = [{"seg_start":seg.start, "seg_end": seg.end, "speaker":spk, "sentence": sent } for seg, spk, sent in final_result]

        with open(json_dir + audio_file.replace(".wav", ".json"), "w") as f:
            json.dump(final_result,f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(e)
        print(audio_file)