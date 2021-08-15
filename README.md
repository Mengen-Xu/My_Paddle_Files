# 『行远见大』告白模拟器

## 项目简介

眼瞅着又到一年一度的七夕节，有人欢喜有狗愁。在这个狂撒狗粮的节日，怎么能少得了用飞桨搞事呢？这不我来整活了，悄咪咪地把飞桨的颜值担当——依依小姐姐的照片给拿来了，再通过使用语音合成和唇部合成技术，让依依小姐姐对你说出告白~

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/cf60acc21ba84d9197aa42ee3828bbd45ed6b5e85bcb421dbad83cd5452a2c6c" width = "224"></center>
<br>

于是乎一款拯救单身狗的神器诞生了，今年七夕单身狗不用再发愁了。飞桨语音合成套件Parakeet提供的音色克隆的能力以及飞桨对抗网络套件PaddleGAN中唇形合成的技术，轻松实现让暗恋的对象对你说出告白！

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/aa27a2a685f5472c84a8876ea2c2477cc6ca29f89b854009bc07974e5e7148dd" width = "224"></center>

## 课程链接

飞桨领航团AI达人创造营：[https://aistudio.baidu.com/aistudio/education/group/info/24607](https://aistudio.baidu.com/aistudio/education/group/info/24607)

## 致敬开源

大家好，我是[行远见大](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/347967)。欢迎你与我一同建设飞桨开源社区，知识分享是一种美德，让我们向开源致敬！

# 环境配置

## 查看环境


```python
import paddle
print("本项目基于 PaddlePaddle 的版本号为："+ paddle.__version__)
```

    本项目基于 PaddlePaddle 的版本号为：2.1.0


## 数据准备

本项目使用七夕节专属语音合成数据集：[依依小姐姐语音包](https://aistudio.baidu.com/aistudio/datasetdetail/99999)。


```python
# 加载第99999号寓意天长地久数据集
!unzip -oq /home/aistudio/data/data99999/nltk_data.zip
!unzip -oq /home/aistudio/data/data99999/work.zip
```

## 下载并安装 Parakeet 包

温馨提示：此节若报错，请重启项目后再运行。


```python
# 下载 Parakeet 包
!git clone https://gitee.com/paddlepaddle/Parakeet.git -b release/v0.3 work/Parakeet
```


```python
# 安装 Parakeet 包
!pip install -e work/Parakeet/
```


```python
# 把必要的路径添加到 sys.path，避免找不到已安装的包
import sys
sys.path.append("/home/aistudio/work/Parakeet")
sys.path.append("/home/aistudio/work/Parakeet/examples/tacotron2_aishell3")

import numpy as np
import paddle
from matplotlib import pyplot as plt
from IPython import display as ipd
import soundfile as sf
import librosa.display
from parakeet.utils import display
paddle.set_device("gpu:0")
```


```python
%matplotlib inline
```

# 语音合成

## 语音合成原理

在训练语音克隆模型时，目标音色作为 **Speaker Encoder** 的输入，模型会提取这段语音的说话人特征（音色）作为 **Speaker Embedding**。

接着，在训练模型重新合成此类音色的语音时，除了输入的目标文本外，说话人的特征也将成为额外条件加入模型的训练。

在预测时，选取一段新的目标音色作为 **Speaker Encoder** 的输入，并提取其说话人特征，最终实现**输入为一段文本和一段目标音色，模型生成目标音色说出此段文本的语音片段，完美的实现音色克隆~**

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/982ab955b87244d3bae3b003aff8e28d9ec159ff0d6246a79757339076dfe7d4" width = "800"></center>

<br>

**Parakeet 语音合成相关项目：**

Parakeet 语音合成原理详情可见：[Parakeet音色克隆：柯南的变声器成真啦](https://aistudio.baidu.com/aistudio/projectdetail/2020850?channelType=0&channel=0)

手把手教你定制一个朗诵机器人：[『行远见大』《行远见大论》合纵连横篇](https://aistudio.baidu.com/aistudio/projectdetail/2188888)

## 加载语音克隆模型


```python
from examples.ge2e.audio_processor import SpeakerVerificationPreprocessor
from parakeet.models.lstm_speaker_encoder import LSTMSpeakerEncoder

# speaker encoder
p = SpeakerVerificationPreprocessor(
    sampling_rate=16000, 
    audio_norm_target_dBFS=-30, 
    vad_window_length=30, 
    vad_moving_average_width=8, 
    vad_max_silence_length=6, 
    mel_window_length=25, 
    mel_window_step=10, 
    n_mels=40, 
    partial_n_frames=160, 
    min_pad_coverage=0.75, 
    partial_overlap_ratio=0.5)
speaker_encoder = LSTMSpeakerEncoder(n_mels=40, num_layers=3, hidden_size=256, output_size=256)
speaker_encoder_params_path = "/home/aistudio/work/pretrained/ge2e_ckpt_0.3/step-3000000.pdparams"
speaker_encoder.set_state_dict(paddle.load(speaker_encoder_params_path))
speaker_encoder.eval()

# synthesizer
from parakeet.models.tacotron2 import Tacotron2
from examples.tacotron2_aishell3.chinese_g2p import convert_sentence
from examples.tacotron2_aishell3.aishell3 import voc_phones, voc_tones

from yacs.config import CfgNode
synthesizer = Tacotron2(
    vocab_size=68,
    n_tones=10,
    d_mels= 80,
    d_encoder= 512,
    encoder_conv_layers = 3,
    encoder_kernel_size= 5,
    d_prenet= 256,
    d_attention_rnn= 1024,
    d_decoder_rnn = 1024,
    attention_filters = 32,
    attention_kernel_size = 31,
    d_attention= 128,
    d_postnet = 512,
    postnet_kernel_size = 5,
    postnet_conv_layers = 5,
    reduction_factor = 1,
    p_encoder_dropout = 0.5,
    p_prenet_dropout= 0.5,
    p_attention_dropout= 0.1,
    p_decoder_dropout= 0.1,
    p_postnet_dropout= 0.5,
    d_global_condition=256,
    use_stop_token=False
)
params_path = "/home/aistudio/work/pretrained/tacotron2_aishell3_ckpt_0.3/step-450000.pdparams"
synthesizer.set_state_dict(paddle.load(params_path))
synthesizer.eval()

# vocoder
from parakeet.models import ConditionalWaveFlow
vocoder = ConditionalWaveFlow(upsample_factors=[16, 16], n_flows=8, n_layers=8, n_group=16, channels=128, n_mels=80, kernel_size=[3, 3])
params_path = "/home/aistudio/work/pretrained/waveflow_ljspeech_ckpt_0.3/step-2000000.pdparams"
vocoder.set_state_dict(paddle.load(params_path))
vocoder.eval()
```

    vocab_phones:
     Vocab(size: 68,
    stoi:
    OrderedDict([('<pad>', 0), ('<unk>', 1), ('<s>', 2), ('</s>', 3), ('$', 4), ('%', 5), ('&r', 6), ('a', 7), ('ai', 8), ('an', 9), ('ang', 10), ('ao', 11), ('b', 12), ('c', 13), ('ch', 14), ('d', 15), ('e', 16), ('ea', 17), ('ei', 18), ('en', 19), ('eng', 20), ('er', 21), ('f', 22), ('g', 23), ('h', 24), ('i', 25), ('ia', 26), ('iai', 27), ('ian', 28), ('iang', 29), ('iao', 30), ('ie', 31), ('ien', 32), ('ieng', 33), ('ii', 34), ('iii', 35), ('io', 36), ('iou', 37), ('j', 38), ('k', 39), ('l', 40), ('m', 41), ('n', 42), ('o', 43), ('ou', 44), ('p', 45), ('q', 46), ('r', 47), ('s', 48), ('sh', 49), ('t', 50), ('u', 51), ('ua', 52), ('uai', 53), ('uan', 54), ('uang', 55), ('uei', 56), ('uen', 57), ('ueng', 58), ('uo', 59), ('v', 60), ('van', 61), ('ve', 62), ('ven', 63), ('veng', 64), ('x', 65), ('z', 66), ('zh', 67)]))
    vocab_tones:
     Vocab(size: 10,
    stoi:
    OrderedDict([('<pad>', 0), ('<unk>', 1), ('<s>', 2), ('</s>', 3), ('0', 4), ('1', 5), ('2', 6), ('3', 7), ('4', 8), ('5', 9)]))


## 提取目标音色的声音特征

我们把参考音频放在 'work/ref_audio' 目录下，如果需要试用自己的声音，也可以把录音文件上传至这个目录。

注意：支持音频格式为 wav 和 flac ，如有其他格式音频，建议使用软件进行转换。

温馨提示：请使用自己的声音，使用他人的声音请先获得对方的同意后再使用。installed.wav 是 Parakeet 内置的女声语音包，yiyi.wav 是依依小姐姐的语言包（已获授权）~

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/5fb99568d87044b098d5aa3bc0dbfc4139aa750df0ae441791a52d0449cab722" width = "224"></center>


```python
ref_name = "yiyi.wav"
ref_audio_path = f"/home/aistudio/work/ref_audio/{ref_name}"
ipd.Audio(filename=ref_audio_path)
```


```python
mel_sequences = p.extract_mel_partials(p.preprocess_wav(ref_audio_path))
print("mel_sequences: ", mel_sequences.shape)
with paddle.no_grad():
    embed = speaker_encoder.embed_utterance(paddle.to_tensor(mel_sequences))
print("embed shape: ", embed.shape)
```

    mel_sequences:  (6, 160, 40)
    embed shape:  [256]


## 合成频谱

提取到了参考语音的特征向量之后，给定需要合成的文本，通过 Tacotron2 模型生成频谱。

目前只支持汉字以及两个表示停顿的特殊符号，'%' 表示句中较短的停顿，'$' 表示较长的停顿。

温馨提示：合成频谱呈现斜率接近 $k = 1$ 时，合成语音的效果较好。

![](https://ai-studio-static-online.cdn.bcebos.com/97220cb91975410e8af40151bae873bb764d7cfbf4f94470b834dae0796a4bb8)


```python
# 替换 sentence 里的内容，书写依依小姐姐对你的七夕专属告白
sentence = "月老%有意%揽红线$银河%万里%话情缘$今晚%月色真美$依依在这里$祝%各位飞桨%开发者们$七夕%情人节%快乐$"

phones, tones = convert_sentence(sentence)
print(phones)
print(tones)

phones = np.array([voc_phones.lookup(item) for item in phones], dtype=np.int64)
tones = np.array([voc_tones.lookup(item) for item in tones], dtype=np.int64)

phones = paddle.to_tensor(phones).unsqueeze(0)
tones = paddle.to_tensor(tones).unsqueeze(0)
utterance_embeds = paddle.unsqueeze(embed, 0)
```

    ['ve', 'l', 'ao', '%', 'iou', 'i', '%', 'l', 'an', 'h', 'ueng', 'x', 'ian', '$', 'ien', 'h', 'e', '%', 'uan', 'l', 'i', '%', 'h', 'ua', 'q', 'ieng', 'van', '$', 'j', 'ien', 'uan', '%', 've', 's', 'e', 'zh', 'en', 'm', 'ei', '$', 'i', 'i', 'z', 'ai', 'zh', 'e', 'l', 'i', '$', 'zh', 'u', '%', 'g', 'e', 'uei', 'f', 'ei', 'j', 'iang', '%', 'k', 'ai', 'f', 'a', 'zh', 'e', 'm', 'en', '$', 'q', 'i', 'x', 'i', '%', 'q', 'ieng', 'r', 'en', 'j', 'ie', '%', 'k', 'uai', 'l', 'e', '$']
    ['4', '0', '3', '0', '3', '4', '0', '0', '3', '0', '2', '0', '4', '0', '2', '0', '2', '0', '4', '0', '3', '0', '0', '4', '0', '2', '2', '0', '0', '1', '3', '0', '4', '0', '4', '0', '1', '0', '3', '0', '1', '1', '0', '4', '0', '4', '0', '3', '0', '0', '4', '0', '0', '4', '4', '0', '1', '0', '3', '0', '0', '1', '0', '1', '0', '3', '0', '5', '0', '0', '1', '0', '1', '0', '0', '2', '0', '2', '0', '2', '0', '0', '4', '0', '4', '0']



```python
with paddle.no_grad():
    outputs = synthesizer.infer(phones, tones=tones, global_condition=utterance_embeds)
mel_input = paddle.transpose(outputs["mel_outputs_postnet"], [0, 2, 1])
fig = display.plot_alignment(outputs["alignments"][0].numpy().T)
```

## 合成最终语音

使用 waveflow 声码器，将生成的频谱转换为音频。合成的音频保存在 'data/syn_audio' 文件夹。

![](https://ai-studio-static-online.cdn.bcebos.com/25ea53f301f14b50ac0f1f78b16d379d05ee3bc32c434bf18d66b03491ee44e2)


```python
!mkdir -p data/syn_audio

with paddle.no_grad():
    wav = vocoder.infer(mel_input)
wav = wav.numpy()[0]
sf.write(f"/home/aistudio/data/syn_audio/{ref_name}", wav, samplerate=22050)
librosa.display.waveplot(wav)
```


```python
# 下载到本地，对音频进行后期处理
ipd.Audio(wav, rate=22050)
```

# 唇形合成

## 下载 PaddleGAN


```python
%cd /home/aistudio/work

# 从github上克隆PaddleGAN代码（如下载速度过慢，可用gitee源）
# !git clone https://github.com/PaddlePaddle/PaddleGAN
!git clone https://gitee.com/PaddlePaddle/PaddleGAN

# 安装依赖
%cd /home/aistudio/work/PaddleGAN
!pip install -r requirements.txt
%cd applications/
```

## 输出视频

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/91c333011e594f41b94e01a2ec8d12592844c23aa1514148a2491ce11ab72f8a" width = "500"></center>

<center>yiyi.jpg</center>

<br>

**在 face 处加载依依小姐姐的照片：** --face /home/aistudio/ 'yiyi.jpg'

**在 audio 处导入合成依依小姐姐的语音：** --audio /home/aistudio/ 'voice.mp3'

**输出视频：** --outfile /home/aistudio/ movie.mp4

温馨提示：使用对象的照片前请告知对方呦~


```python
# wav2lip 让照片中的人物对口型，实现唇部合成
!export PYTHONPATH=$PYTHONPATH:/home/aistudio/work/PaddleGAN && python tools/wav2lip.py \
                                --face /home/aistudio/'yiyi.jpg' \
                                --audio /home/aistudio/'voice.mp3' \
                                --outfile /home/aistudio/movie.mp4
```

# 总结

## 效果展示

Bilibili视频：[依依小姐姐唱情歌给你听~](https://www.bilibili.com/video/BV1Gq4y197tW)

## 项目总结

本项目首先提取其说话人特征，生成目标音色并根据文本说出语音片段，再使用PaddleGAN中所提供的唇形合成算法--Wav2lip让视频或图片中的人物根据目标音频拟合唇形，让视频或图片里的人物将一段话自然地念出来。Wav2lip采用预训练好的唇形同步损失函数，重建损失函数以及脸部逼真度判别器，使得生成器能够产生准确而逼真的唇部运动，实现唇形与语音精准同步，从而改善了视觉质量。Wav2Lip适用于任何人脸、任何语言，对任意视频都能达到很高都准确率，可以无缝地与原始视频融合。

## 七夕总结

前年七夕，我一个人过。

去年七夕，我一个人过。

今年就不一样了。

今年七夕，我在 AI Studio 过。

希望明年能有个漂亮的小姐姐陪我过。

嘤嘤嘤~

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/72a60ac6fbc04001b5269706e2aaa6df577c105e866942a09585efe5f87eab91" width = "224"></center>

