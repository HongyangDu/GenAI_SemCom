# [Generative AI-aided Joint Training-free Secure Semantic Communications via Multi-modal Prompts](https://github.com/HongyangDu/GenAI_SemCom)

Semantic communication (SemCom) holds promise for reducing network resource consumption while achieving the communications goal. However, the computational overheads in jointly training semantic encoders and decoders—and the subsequent deployment in network devices—are overlooked. Recent advances in Generative artificial intelligence (GAI) offer a potential solution. The robust learning abilities of GAI models indicate that semantic decoders can reconstruct source messages using a limited amount of semantic information, e.g., prompts, without joint training with the semantic encoder. A notable challenge, however, is the instability introduced by GAI’s diverse generation ability. This instability, evident in outputs like text-generated images, limits the direct application of GAI in scenarios demanding accurate message recovery, such as face image transmission. To solve the above problems, this paper proposes a GAI-aided SemCom system with multi-model prompts for accurate content decoding.

This repository contains the code accompanying the paper 

> **"GENERATIVE AI-AIDED JOINT TRAINING-FREE SECURE SEMANTIC COMMUNICATIONS VIA MULTI-MODAL PROMPTS"**

Authored by *Hongyang Du, Guangyuan Liu, Dusit Niyato, Jiayi Zhang, Jiawen Kang, Zehui Xiong, Bo Ai, Dong In Kim*, accepted by *ICASSP 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*.

The paper can be found at [ArXiv](https://arxiv.org/pdf/2309.02616.pdf) or [IEEE](https://ieeexplore.ieee.org/abstract/document/10447237/)

## ⚡ Coding Environment

To create a new environment, please use:

```bash
requirements.txt, or
install_requirements_for_colab.sh, or
requirement_for_colab.txt
```


## 🔍 Pre-trained Model
Please download the pre-trained diffusion model in [Google Drive](https://drive.google.com/file/d/1UZhnOlsiKyBTYR5cYOcPXTvURk9bNX4z/view?usp=sharing), and put it in
: checkpoints/ffhq256_autoenc

## 🏃‍♀️ Run the Program
Run `encoder.py` to encode the image and save the semantic information.
Run `encoder.py` to load the saved semantic information and decode it.


## Citation
If our method can be used in your paper, please help cite:
```bibtex
@inproceedings{du2024generative,
  title={Generative {AI}-aided Joint Training-free Secure Semantic Communications via Multi-modal Prompts},
  author={Du, Hongyang and Liu, Guangyuan and Niyato, Dusit and Zhang, Jiayi and Kang, Jiawen and Xiong, Zehui and Ai, Bo and Kim, Dong In},
  booktitle={Proc. IEEE Int. Conf. Acoust. Speech Signal Process.},
  pages={12896--12900},
  year={2024}
}
```
## 🔍 Some Discussion

For the optimization part in the paper, please refer to our tutorial paper: [Beyond Deep Reinforcement Learning: A Tutorial on Generative Diffusion Models in Network Optimization](https://arxiv.org/abs/2308.05384) and [the codes](https://github.com/HongyangDu/GDMOPT).

For the semantic encoding part, the method to further reduce the size of semantic information will be given in our journal version. 

Please star this repository to get the latest updates!
