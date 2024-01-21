import streamlit as st
import whisper
import datetime
import subprocess
import torch
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import wave
import contextlib
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from pyannote.audio import Audio
from pyannote.core import Segment
import tempfile
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import os
import tempfile
from pyannote.audio import Audio
from pyannote.core import Segment
import io


# Function to convert MP4 to audio


def mp4_to_audio(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    video_clip = VideoFileClip(tfile.name)
    video_duration = video_clip.duration  # Video duration in seconds

    chunk_size = 600  # 10 minutes in seconds
    temp_audio_filenames = []

    for start in range(0, int(video_duration), chunk_size):
        end = min(start + chunk_size, video_duration)
        print(f"Processing video from {start} to {end} seconds")

        # Subclip video
        sub_video = video_clip.subclip(start, end)

        # Create a temporary file for each subclip
        temp_audio_filename = tempfile.mktemp('.wav')
        sub_video.audio.write_audiofile(temp_audio_filename)

        temp_audio_filenames.append(temp_audio_filename)

    return temp_audio_filenames  # return filenames for further processing


def transcribe_and_identify_speakers(audio_path, num_speakers=1, language='any', model_size='tiny'):
    if audio_path[-3:] != 'wav':
        subprocess.call(['ffmpeg', '-i', audio_path, 'audio.wav', '-y'])
        audio_path = 'audio.wav'

    model_name = model_size
    if language == 'English' and model_size != 'large':
        model_name += '.en'

    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    segments = result["segments"]

    with contextlib.closing(wave.open(audio_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    audio = Audio()
    embedding_model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb",
        device=torch.device("cpu"))

    def segment_embedding(segment):
        start = segment["start"]
        end = min(duration, segment["end"])
        clip = Segment(start, end)
        waveform, sample_rate = audio.crop(audio_path, clip)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return embedding_model(waveform[None])

    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment)

    embeddings = np.nan_to_num(embeddings)

    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_
    for i in range(len(segments)):
        segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

    transcript = ""
    for (i, segment) in enumerate(segments):
        if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
            transcript += f"\n{segment['speaker']} {str(datetime.timedelta(seconds=round(segment['start'])))}\n"
        transcript += f"{segment['text'][1:]} "

    return transcript


# Streamlit UI
st.title('Whisper ASR Transcription')

# File uploader
uploaded_file = st.file_uploader(
    "Choose an audio file...", type=["wav", "mp3", "mp4"])

# Picker for number of speakers
num_speakers = st.selectbox('Number of Speakers:', (1, 2, 3, 4, 5))

# Picker for language
language_options = {
    'Any Language': 'any',
    'English': 'English'
}
language = st.selectbox('Language:', list(language_options.keys()))
language = language_options[language]  # Convert to code used in function

# Picker for model size
model_size = st.selectbox(
    'Model Size:', ('tiny', 'base', 'small', 'medium', 'large'))

# Transcription button
if st.button('Transcribe'):
    if uploaded_file is not None:
        file_content = uploaded_file.read()  # Read the file content once
        file_extension = uploaded_file.name.split(
            '.')[-1]  # Get the file extension

        with st.spinner('Transcribing...'):
            # Create a temporary directory to store the audio file
            with tempfile.TemporaryDirectory() as temp_dir:
                audio_path = os.path.join(temp_dir, uploaded_file.name)
                with open(audio_path, "wb") as f:
                    f.write(file_content)  # Use the stored file content

                if file_extension.lower() == 'mp4':
                    # Use the stored file content for mp4_to_audio
                    temp_audio_filenames = mp4_to_audio(
                        io.BytesIO(file_content))
                else:
                    # If it's an audio file, no need to extract audio
                    temp_audio_filenames = [audio_path]

                master_transcript = ""
                for temp_audio_filename in temp_audio_filenames:
                    transcript = transcribe_and_identify_speakers(
                        temp_audio_filename, num_speakers, language, model_size)
                    master_transcript += transcript

        if master_transcript:
            st.write('Transcription:')
            transcript_area = st.text_area(
                "", value=master_transcript, height=200, max_chars=None)
            st.write('Copy the text above and paste it wherever you need.')

            # Create a download button
            st.download_button(
                label="Download Transcript",
                data=master_transcript.encode(),
                file_name='transcript.txt',
                mime='text/plain'
            )
        else:
            st.error('No transcription was generated.')
