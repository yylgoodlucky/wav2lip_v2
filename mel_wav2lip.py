
import audio

auio_path = "/nfslocal/data/kaixinwang/LRS2_datasets/resolution_2x/6393267985458244248/00017/audio.wav"

wav = audio.load_wav(auio_path, 16000)
orig_mel = audio.melspectrogram(wav).T

print(wav.shape)
print(orig_mel.shape)