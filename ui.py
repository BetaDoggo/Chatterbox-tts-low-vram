from chatterbox.tts import ChatterboxTTS
import gradio as gr
import torch
import re 
import numpy as np  

# Load model once (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxTTS.from_pretrained(device=device)

def split_text(text, max_chunk_size):
    """Split text into chunks respecting sentence boundaries"""
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Check if adding this sentence would exceed max size
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start new chunk with current sentence
            current_chunk = sentence + " "
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def trim_silence(audio, threshold=0.01, margin=0.01, sr=24000):
    """
    Remove leading and trailing silence from audio.
    
    Args:
        audio (np.array): Audio signal as 1D numpy array
        threshold (float): Amplitude threshold for silence detection
        margin (float): Additional silence to leave at boundaries (seconds)
        sr (int): Sample rate
    
    Returns:
        np.array: Trimmed audio signal
    """
    if len(audio) == 0:
        return audio
    
    # Calculate amplitude envelope
    amplitude = np.abs(audio)
    
    # Set threshold based on peak amplitude
    peak = np.max(amplitude)
    silence_threshold = peak * threshold
    
    # Find where audio exceeds the threshold
    above_threshold = np.where(amplitude > silence_threshold)[0]
    
    if len(above_threshold) == 0:
        return audio
    
    # Calculate start and end points
    start = max(0, above_threshold[0] - int(margin * sr))
    end = min(len(audio), above_threshold[-1] + int(margin * sr) + 1)
    
    return audio[start:end]

def generate_speech(text, audio_prompt, exaggeration, cfg_weight, temperature, chunk_size):
    # Validate inputs
    if audio_prompt is None:
        raise gr.Error("Please provide an audio prompt")
    if not text.strip():
        raise gr.Error("Please enter text to convert to speech")
    
    # Split text into chunks if needed
    if len(text) > chunk_size:
        chunks = split_text(text, chunk_size)
        gr.Info(f"Text split into {len(chunks)} chunks for processing")
        
        # Process each chunk sequentially
        all_audio = []
        for i, chunk in enumerate(chunks):
            gr.Info(f"Processing chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
            wav_tensor = model.generate(
                chunk,
                audio_prompt_path=audio_prompt,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature
            )
            
            # Convert to numpy array and trim
            wav_numpy = wav_tensor.cpu().numpy().squeeze()
            trimmed_wav = trim_silence(wav_numpy, sr=model.sr)
            all_audio.append(trimmed_wav)
        
        # Concatenate all chunks
        audio_numpy = np.concatenate(all_audio)
    else:
        # Process as single chunk
        wav_tensor = model.generate(
            text,
            audio_prompt_path=audio_prompt,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature
        )
        # Convert to numpy array and trim
        wav_numpy = wav_tensor.cpu().numpy().squeeze()
        audio_numpy = trim_silence(wav_numpy, sr=model.sr)
    
    return model.sr, audio_numpy

# Create Gradio interface
with gr.Blocks(title="Chatterbox TTS") as demo:
    gr.Markdown("# Chatterbox Text-to-Speech")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Text Prompt",
                placeholder="Enter text to speak...",
                value="Hello, how are you today?",
                lines=3
            )
            audio_input = gr.Audio(
                label="Voice Prompt (Record or Upload)",
                type="filepath",
                sources=["upload", "microphone"]
            )
            with gr.Row():
                exaggeration = gr.Slider(0, 1, value=0.5, label="Exaggeration")
                cfg_weight = gr.Slider(0, 1, value=0.5, label="CFG Weight")
            temperature = gr.Slider(0, 1, value=0.8, label="Temperature")
            # Add chunk size slider
            chunk_size = gr.Slider(
                50, 1000, 
                value=200, 
                step=10,
                label="Chunk Size (characters)",
                info="Split long texts into chunks of this size"
            )
            submit_btn = gr.Button("Generate Speech", variant="primary")
        
        with gr.Column():
            audio_output = gr.Audio(label="Generated Speech", interactive=False)
    
    # Update examples to include chunk_size
    gr.Examples(
        examples=[
            ["Good morning! Have a wonderful day.", "sample.mp3", 0.6, 0.5, 0.8, 200],
            ["What the fuck did you just fucking say about me, you little bitch? I'll have you know I graduated top of my class in the Navy Seals, and I've been involved in numerous secret raids on Al-Quaeda, and I have over 300 confirmed kills. I am trained in gorilla warfare and I'm the top sniper in the entire US armed forces. You are nothing to me but just another target. I will wipe you the fuck out with precision the likes of which has never been seen before on this Earth, mark my fucking words. You think you can get away with saying that shit to me over the Internet? Think again, fucker. As we speak I am contacting my secret network of spies across the USA and your IP is being traced right now so you better prepare for the storm, maggot. The storm that wipes out the pathetic little thing you call your life. You're fucking dead, kid. I can be anywhere, anytime, and I can kill you in over seven hundred ways, and that's just with my bare hands. Not only am I extensively trained in unarmed combat, but I have access to the entire arsenal of the United States Marine Corps and I will use it to its full extent to wipe your miserable ass off the face of the continent, you little shit. If only you could have known what unholy retribution your little \"clever\" comment was about to bring down upon you, maybe you would have held your fucking tongue. But you couldn't, you didn't, and now you're paying the price, you goddamn idiot. I will shit fury all over you and you will drown in it. You're fucking dead, kiddo.", "sample.mp3", 0.7, 0.5, 0.8, 200]
        ],
        inputs=[text_input, audio_input, exaggeration, cfg_weight, temperature, chunk_size],
        outputs=audio_output,
        fn=generate_speech,
        cache_examples=True
    )
    
    submit_btn.click(
        fn=generate_speech,
        inputs=[text_input, audio_input, exaggeration, cfg_weight, temperature, chunk_size],
        outputs=audio_output
    )

if __name__ == "__main__":
    demo.launch()