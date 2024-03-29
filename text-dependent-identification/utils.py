import tensorflow as tf
import numpy as np
import os
import librosa
from configuration import get_config

config = get_config()


def keyword_spot(spec):
    """ Keyword detection for data preprocess
        For VTCK data I truncate last 80 frames of trimmed audio - "Call Stella"
    :return: 80 frames spectrogram
    """
    return spec[:, -config.tdsv_frame:]


def random_batch(path=config.train_path, speaker_num=config.N, utter_num=config.M, shuffle=True, utter_start=0):
    """ Generate 1 batch.
        For TD-SV, noise is added to each utterance.
        For TI-SV, random frame length is applied to each batch of utterances (140-180 frames)
        speaker_num : number of speaker of each batch
        utter_num : number of utterance per speaker of each batch
        shuffle : random sampling or not
        utter_start : start point of slicing (TI-SV)
    :return: 1 random numpy batch (frames x batch(NM) x n_mels)
    """

    # TD-SV
    if config.tdsv:
        idx = 0
        np_file = ''
        while not np_file.endswith('.npy'):
            np_file = os.listdir(path)[idx]
            idx += 1
        path = os.path.join(path, np_file)  # path of numpy file
        utters = np.load(path)              # load text specific utterance spectrogram
        if shuffle:
            np.random.shuffle(utters)       # shuffle for random sampling
        utters = utters[:speaker_num]       # select N speaker

        # concat utterances (M utters per each speaker)
        # ex) M=2, N=2 => utter_batch = [speaker1, speaker1, speaker2, speaker2]
        utter_batch = np.concatenate([np.concatenate([utters[i]]*utter_num, axis=1) for i in range(speaker_num)], axis=1)

        if config.train:
            noise_filenum = np.random.randint(0, config.noise_filenum)                    # random selection of noise
            noise = np.load(os.path.join(config.noise_path, "noise_%d.npy"%noise_filenum))  # load noise
            utter_batch += noise[:,:utter_batch.shape[1]]   # add noise to utterance

        utter_batch = np.abs(utter_batch) ** 2
        mel_basis = librosa.filters.mel(sr=config.sr, n_fft=config.nfft, n_mels=40)
        utter_batch = np.log10(np.dot(mel_basis, utter_batch) + 1e-6)          # log mel spectrogram of utterances

        utter_batch = np.array([utter_batch[:,config.tdsv_frame*i:config.tdsv_frame*(i+1)]
                                for i in range(speaker_num*utter_num)])        # reshape [batch, n_mels, frames]


    utter_batch = np.transpose(utter_batch, axes=(2,0,1))     # transpose [frames, batch, n_mels]
    return utter_batch


def normalize(x):
    """ normalize the last dimension vector of the input matrix
    :return: normalized input
    """
    return x/tf.sqrt(tf.reduce_sum(x**2, axis=-1, keep_dims=True)+1e-6)


def cossim(x,y, normalized=True):
    """ calculate similarity between tensors
    :return: cos similarity tf op node
    """
    if normalized:
        return tf.reduce_sum(x*y)
    else:
        x_norm = tf.sqrt(tf.reduce_sum(x**2)+1e-6)
        y_norm = tf.sqrt(tf.reduce_sum(y**2)+1e-6)
        return tf.reduce_sum(x*y)/x_norm/y_norm


def similarity(embedded, w, b, N=config.N, M=config.M, P=config.proj, center=None):
    """ Calculate similarity matrix from embedded utterance batch (NM x embed_dim) eq. (9)
        Input center to test enrollment. (embedded for verification)
    :return: tf similarity matrix (NM x N)
    """
    embedded_split = tf.reshape(embedded, shape=[N, M, P])

    if center is None:
        center = normalize(tf.reduce_mean(embedded_split, axis=1))              # [N,P] normalized center vectors eq.(1)
        center_except = normalize(tf.reshape(tf.reduce_sum(embedded_split, axis=1, keep_dims=True)
                                             - embedded_split, shape=[N*M,P]))  # [NM,P] center vectors eq.(8)
        # make similarity matrix eq.(9)
        S = tf.concat(
            [tf.concat([tf.reduce_sum(center_except[i*M:(i+1)*M,:]*embedded_split[j,:,:], axis=1, keep_dims=True) if i==j
                        else tf.reduce_sum(center[i:(i+1),:]*embedded_split[j,:,:], axis=1, keep_dims=True) for i in range(N)],
                       axis=1) for j in range(N)], axis=0)
    else :
        # If center(enrollment) exist, use it.
        S = tf.concat(
            [tf.concat([tf.reduce_sum(center[i:(i + 1), :] * embedded_split[j, :, :], axis=1, keep_dims=True) for i
                        in range(N)],
                       axis=1) for j in range(N)], axis=0)

    S = tf.abs(w)*S+b   # rescaling

    return S


def loss_cal(S, type="softmax", N=config.N, M=config.M):
    """ calculate loss with similarity matrix(S) eq.(6) (7) 
    :type: "softmax" or "contrast"
    :return: loss
    """
    S_correct = tf.concat([S[i*M:(i+1)*M, i:(i+1)] for i in range(N)], axis=0)  # colored entries in Fig.1

    if type == "softmax":
        total = -tf.reduce_sum(S_correct-tf.log(tf.reduce_sum(tf.exp(S), axis=1, keep_dims=True) + 1e-6))
    elif type == "contrast":
        S_sig = tf.sigmoid(S)
        S_sig = tf.concat([tf.concat([0*S_sig[i*M:(i+1)*M, j:(j+1)] if i==j
                              else S_sig[i*M:(i+1)*M, j:(j+1)] for j in range(N)], axis=1)
                             for i in range(N)], axis=0)
        total = tf.reduce_sum(1-tf.sigmoid(S_correct)+tf.reduce_max(S_sig, axis=1, keep_dims=True))
    else:
        raise AssertionError("loss type should be softmax or contrast !")

    return total


def optim(lr):
    """ return optimizer determined by configuration
    :return: tf optimizer
    """
    if config.optim == "sgd":
        return tf.train.GradientDescentOptimizer(lr)
    elif config.optim == "rmsprop":
        return tf.train.RMSPropOptimizer(lr)
    elif config.optim == "adam":
        return tf.train.AdamOptimizer(lr, beta1=config.beta1, beta2=config.beta2)
    else:
        raise AssertionError("Wrong optimizer type!")


# for check
if __name__ == "__main__":
    random_batch()
    w= tf.constant([1], dtype= tf.float32)
    b= tf.constant([0], dtype= tf.float32)
    embedded = tf.constant([[0,1,0],[0,0,1], [0,1,0], [0,1,0], [1,0,0], [1,0,0]], dtype= tf.float32)
    sim_matrix = similarity(embedded,w,b,3,2,3)
    loss1 = loss_cal(sim_matrix, type="softmax",N=3,M=2)
    loss2 = loss_cal(sim_matrix, type="contrast",N=3,M=2)
    with tf.Session() as sess:
        print(sess.run(sim_matrix))
        print(sess.run(loss1))
        print(sess.run(loss2))