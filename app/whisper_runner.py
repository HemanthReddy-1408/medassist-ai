import os
import tempfile
import sounddevice as sd
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
from app.agent.graph import build_medassist_graph

# 🎙️ Config
RECORD_DURATION = 5  # seconds
SAMPLE_RATE = 16000  # Hz
THREAD_ID = "voice-user-001"  # Needed for memory checkpointer

# 🧠 Load LangGraph agent
graph = build_medassist_graph()

# 🗣️ Load Whisper (CPU-only for now)
model_size = "base"  # Try "medium" later if you want higher quality
whisper_model = WhisperModel(model_size, compute_type="int8", device="cpu")

# 🎧 Record audio
def record_audio(duration=RECORD_DURATION, fs=SAMPLE_RATE):
    print(f"🎙️ Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return audio

# 💾 Save as temp WAV
def save_temp_wav(audio_data, fs=SAMPLE_RATE):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp_file.name, fs, audio_data)
    return temp_file.name

# 📝 Transcribe using FasterWhisper
def transcribe_audio(file_path):
    segments, _ = whisper_model.transcribe(file_path)
    full_text = " ".join([seg.text for seg in segments])
    return full_text.strip()

# 🚀 Main loop
def main():
    while True:
        print("\n🩺 MedAssist – Speak your medical query (or type 'exit')")
        cmd = input("▶️  Press ENTER to record (or type 'exit'): ").strip().lower()
        if cmd == "exit":
            break

        # Step 1: Record
        audio_data = record_audio()

        # Step 2: Save temp file
        wav_path = save_temp_wav(audio_data)

        # Step 3: Transcribe
        transcript = transcribe_audio(wav_path)
        print(f"\n📝 Transcribed: {transcript}")

        # Step 4: Invoke LangGraph Agent
        response = graph.invoke({
            "messages": [{"role": "user", "content": transcript}],
            "thread_id": THREAD_ID
        })

        # Step 5: Output final response
        print(f"\n🤖 MedAssist:\n{response['final_response']}")

        # Step 6: Cleanup
        os.remove(wav_path)

if __name__ == "__main__":
    main()
