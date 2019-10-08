import os
import librosa
import numpy as np
from configuration import get_config
from utils import keyword_spot

config = get_config()   # get arguments from parser

# downloaded dataset path
clean_path = './Dataset/clean_trainset_wav'  # clean dataset
noisy_path = './Dataset/noisy_trainset_wav'  # noisy dataset

def extract_noise():
    """ Extract noise and save the spectrogram (as numpy array in config.noise_path)
        Need: paired clean and noisy data set
    """
    print("start noise extraction!")
    os.makedirs(config.noise_path, exist_ok=True)              # make folder to save noise file
    batch_frames = config.N * config.M * config.tdsv_frame     # TD-SV frame number of each batch
    stacked_noise = []
    stacked_len = 0
    k = 0
    for i, path in enumerate(os.listdir(clean_path)):
        if path[-3:] != 'wav':
            continue;

        dur1 = librosa.get_duration(filename=os.path.join(clean_path, path));
        dur2 = librosa.get_duration(filename=os.path.join(noisy_path, path));
        dur = min(dur1, dur2)   # process the starting overlapping audio

        clean, sr = librosa.core.load(os.path.join(clean_path, path), sr=8000, duration=dur)  # load clean audio
        noisy, _ = librosa.core.load(os.path.join(noisy_path, path), sr=sr, duration=dur)     # load noisy audio

        noise = clean - noisy       # get noise audio by subtract clean voice from the noisy audio
        S = librosa.core.stft(y=noise, n_fft=config.nfft,
                              win_length=int(config.window * sr), hop_length=int(config.hop * sr))   # perform STFT
        stacked_noise.append(S)
        stacked_len += S.shape[1]

        if stacked_len < batch_frames:   # if noise frames is short than batch frames, then continue to stack the noise
            continue

        stacked_noise = np.concatenate(stacked_noise, axis=1)[:,:batch_frames]          # concat noise and slice
        np.save(os.path.join(config.noise_path, "noise_%d.npy" % k), stacked_noise)   # save spectrogram as numpy file
        print(" %dth file saved" % k, stacked_noise.shape)
        stacked_noise = []     # reset list
        stacked_len = 0
        k += 1

    print("noise extraction is end! %d noise files" % k)


def save_spectrogram_tdsv(path, data_type):
    """ Select text specific utterance and perform STFT with the audio file.
        Audio spectrogram files are divided as train set and test set and saved as numpy file. 
        Need : utterance data set (VTCK)
    """
    print('Preprocess ' + data_type)
    utterances_spec = []
    for folder in os.listdir(path):
        if not os.path.isdir(os.path.join(path, folder)):
            continue
        audios = os.listdir(os.path.join(path, folder))
        audios.sort()
        utter_path= os.path.join(path, folder, audios[0])
        if config.train and os.path.splitext(os.path.basename(utter_path))[0][-3:] != '001':  # if the text utterance doesn't exist pass
            print(os.path.basename(utter_path)[:4], "001 file doesn't exist")
            continue

        utter, sr = librosa.core.load(utter_path, config.sr)               # load the utterance audio
        utter_trim, index = librosa.effects.trim(utter, top_db=14)         # trim the beginning and end blank
        if utter_trim.shape[0]/sr <= config.hop*(config.tdsv_frame+2):     # if trimmed file is too short, then pass
            print(os.path.basename(utter_path), "voice trim fail")
            continue

        S = librosa.core.stft(y=utter_trim, n_fft=config.nfft,
                              win_length=int(config.window * sr), hop_length=int(config.hop * sr))  # perform STFT
        S = keyword_spot(S)          # keyword spot (for now, just slice last 80 frames which contains "Call Stella")
        utterances_spec.append(S)    # make spectrograms list

    utterances_spec = np.array(utterances_spec)  # list to numpy array
    np.random.shuffle(utterances_spec)           # shuffle spectrogram (by person)
    total_num = utterances_spec.shape[0]
    print("Speaker number : %d"%total_num, ", shape : ", utterances_spec.shape)

    np.save(os.path.join(path, data_type + ".npy"), utterances_spec) # save spectrogram as numpy file


def preprocess_train_tdsv():
    save_spectrogram_tdsv(config.train_path, 'train')


def preprocess_test_tdsv():
    for folder in os.listdir(config.enroll_path):
        if not os.path.isdir(os.path.join(config.enroll_path, folder)):
            continue
        save_spectrogram_tdsv(os.path.join(config.enroll_path, folder), 'enroll')

    for folder in os.listdir(config.verification_path):
        if not os.path.isdir(os.path.join(config.verification_path, folder)):
            continue
        save_spectrogram_tdsv(os.path.join(config.verification_path, folder), 'verification')


if __name__ == "__main__":
    extract_noise()