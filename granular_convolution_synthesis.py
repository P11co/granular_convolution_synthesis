import gradio as gr
import numpy as np
import librosa
import random


def extract_grain(grain_orig, grain_start, grain_dur):
    return grain_orig[grain_start:grain_start + grain_dur + 1]


def extract_grains(grain_orig, grain_start, grain_dur, hop_size):
    grains = []
    for start in range(grain_start, len(grain_orig) - grain_dur + 1, hop_size):
        new_grain = extract_grain(grain_orig, start, grain_dur)
        grains.append(new_grain)
    return grains


def apply_window_to_grains(grains):
    window = np.hanning(len(grains[0]))
    for i in range(len(grains)):
        # print(len(grains[i]))
        grains[i] = grains[i] * window
    return grains


def overlap_add(windowed_grains, hop_size, min_overlap=0.5):
    # Calculate minimum grain copies needed to fill gaps
    grain_length = len(windowed_grains[0])
    copies_needed = max(1, int(np.ceil(hop_size / (grain_length * min_overlap))))
    
    output_length = hop_size * (len(windowed_grains) - 1) + grain_length
    output = np.zeros(output_length)
    
    # Add each grain multiple times if needed
    for i, grain in enumerate(windowed_grains):
        base_pos = i * hop_size
        for copy in range(copies_needed):
            start_pos = base_pos + (copy * hop_size // copies_needed)
            end_pos = start_pos + len(grain)
            if end_pos <= output_length:
                output[start_pos:end_pos] += grain
    
    return output


def extract_impulses(impulse, sr, idur, ihop, irand, grains, gdur, ghop, min_overlap=0.25):
    # istart = int(irand * sr * random())
    istart = 0
    # total_grains_reconstructed = max(1, int(np.ceil(ghop / (gdur * min_overlap))))
    impulse_responses = []
    for start in range(istart, len(impulse) - idur + 1, ihop):
        new_impulse = extract_grain(impulse, start, idur)
        impulse_responses.append(new_impulse)
    return impulse_responses


# a function that creates a bouncing index
# ie if len(impulse)=3, then i=3 implies impulse[2] and i=4 implies impulse[1]
def mirror(index, length):
    if length == 0:
        raise ValueError("Length of impulses cannot be zero.")
    
    cycle = 2 * (length - 1) 
    mod_index = index % cycle
    
    if mod_index < length:
        return mod_index
    else:
        return cycle - mod_index


def impulse_grain_convolve(impulses, grains, dry_wet_ratio=0.7, ir_scale=0.3):
    convolved_grains = []
    for i, grain in enumerate(grains):
        # grain is already the current grain, don't use grain[i]
        impulse_index = mirror(i, len(impulses))
        cgrain = np.convolve(grain, impulses[impulse_index] * ir_scale)
        
        # Normalize lengths
        max_len = max(len(grain), len(cgrain))
        grain_padded = np.pad(grain, (0, max_len - len(grain)))
        conv_padded = np.pad(cgrain, (0, max_len - len(cgrain)))
        
        # Mix dry and wet signals
        mixed = (grain_padded * dry_wet_ratio) + (conv_padded * (1 - dry_wet_ratio))
        convolved_grains.append(mixed)
    return convolved_grains


def process_audio(grain_audio_path, impulse_audio_path, gstart, gdur, ghop, idur, ihop, irand, ohop, ir_scale, dry_wet_ratio):
    # load audio files
    grain_audio, sr = librosa.load(grain_audio_path, sr=None)
    impulse_audio, _ = librosa.load(impulse_audio_path, sr=sr)  # Use same sr as grain_audio
    
    # Check 1: ensure audio files are loaded
    if len(grain_audio) == 0 or len(impulse_audio) == 0:
        raise ValueError("Audio files not loaded properly")
    
    # Check 2: ensure gstart is not beyond audio length
    if gstart >= len(grain_audio):
        gstart = 0
        
    grains = extract_grains(grain_audio, gstart, gdur, ghop)
    impulses = extract_impulses(impulse_audio, sr, idur, ihop, irand, grains, gdur, ghop)
    cgrains = impulse_grain_convolve(impulses, grains, ir_scale, dry_wet_ratio)
    windowed_grains = apply_window_to_grains(cgrains)
    output_audio = overlap_add(windowed_grains, ohop, min_overlap=0.25)
    
    # Debug print
    print(f"Number of grains extracted: {len(grains)}")
    print(f"Audio length: {len(grain_audio)}, gstart: {gstart}, gdur: {gdur}, ghop: {ghop}")
    
    
    # Normalize the output to prevent clipping
    normalized_audio = output_audio / np.max(np.abs(output_audio))
    
    return (sr, normalized_audio)  # Return tuple of (sample_rate, audio_data)


demo = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Audio(label="Grain Audio", type="filepath"),
        gr.Audio(label="Impulse Audio", type="filepath"),
        gr.Slider(minimum=0, maximum=44100*5, value=88200, step=100, label="Grain Start (gstart)", info="Starting point for grain extraction"),
        gr.Slider(minimum=100, maximum=44100*2, value=22050, step=100, label="Grain Duration (gdur)", info="Duration of each grain"),
        gr.Slider(minimum=100, maximum=44100, value=8820, step=100, label="Grain Hop Size (ghop)", info="Distance between consecutive grains"),
        gr.Slider(minimum=100, maximum=44100*2, value=22050, step=100, label="Impulse Duration (idur)", info="Duration of impulse response"),
        gr.Slider(minimum=100, maximum=44100, value=8820, step=100, label="Impulse Hop Size (ihop)", info="Distance between impulse responses"),
        gr.Slider(minimum=0, maximum=44100, value=0, step=100, label="Impulse Random Offset (irand)", info="Random offset for impulse selection"),
        gr.Slider(minimum=100, maximum=44100, value=8820, step=100, label="Output Hop Size (ohop)", info="Hop size for output reconstruction"),
        gr.Slider(minimum=0, maximum=1, value=0.3, step=0.01, label="Impulse Response Scale (ir_scale)", info="Scaling factor for impulse response"),
        gr.Slider(minimum=0, maximum=1, value=0.7, step=0.01, label="Dry/Wet Ratio", info="Balance between original and processed signal")
    ],
    outputs=gr.Audio(label="Processed Audio", type="numpy"),
    title="Granular Convolution Processor",
    description="Process audio using granular convolution with customizable parameters"
)

if __name__ == "__main__":
    demo.launch()
