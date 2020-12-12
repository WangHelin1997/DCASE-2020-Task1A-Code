# DCASE-2020-Task1A-Code
A Pytorch implementation of the paper : ["Acoustic Scene Classification with Spectrogram Processing Strategies"](http://dcase.community/documents/workshop2020/proceedings/DCASE2020Workshop_Wang_58.pdf)

## Paper results

Note that the test results are obtained only by the device a for the DCASE2020 Task1A dev set.

Model | Accuracy |  Log loss | Model size
-|-|-|-
DCASE2020 Task 1 Baseline|70.6%|1.356|19.1 MB
Log-Mel CNN|72.1%|0.879|18.9 MB
Gamma CNN|76.1% |0.762 |18.9 MB
MFCC CNN| 63.6%| 1.029| 18.9 MB
SPSMR| 79.4%| 0.696| 75.5 MB
Log-Mel CNN + SPSMF| 75.5%| 1.135| 94.4 MB
CQT CNN + SPSMF| 74.5%| 1.185| 94.4 MB
Gamma CNN + SPSMF| 78.8%| 1.169| 94.4 MB
MFCC CNN + SPSMF| 60.9%| 1.801| 94.4 MB
SPSMR + SPSMF| 80.9%| 0.737| 377.6 MB
Log-Mel CNN + SPSMT| 74.5%| 0.987| 18.9 MB
CQT CNN + SPSMT| 73.3%| 1.032| 18.9 MB
Gamma CNN + SPSMT| 78.2%| 0.866| 18.9 MB
MFCC CNN + SPSMT| 67.6%| 1.081| 18.9 MB
SPSMR + SPSMT| 79.7%| 0.701| 75.5 MB
SPSMR + SPSMF + SPSMT| 81.8%| 0.694| 453.1 MB


## Citation
If this code is helpful, please feel free to cite the following papers:
```
@inproceedings{Wang2020,
    author = "Wang, Helin and Zou, Yuexian and Chong, DaDing",
    title = "Acoustic Scene Classification with Spectrogram Processing Strategies",
    booktitle = "Proceedings of the Detection and Classification of Acoustic Scenes and Events 2020 Workshop (DCASE2020)",
    address = "Tokyo, Japan",
    month = "November",
    year = "2020",
    pages = "210--214",
    abstract = "Recently, convolutional neural networks (CNN) have achieved the state-of-the-art performance in acoustic scene classification (ASC) task. The audio data is often transformed into two-dimensional spectrogram representations, which are then fed to the neural networks. In this paper, we study the problem of efficiently taking advantage of different spectrogram representations through discriminative processing strategies. There are two main contributions. The first contribution is exploring the impact of the combination of multiple spectrogram representations at different stages, which provides a meaningful reference for the effective spectrogram fusion. The second contribution is that the processing strategies in multiple frequency bands and multiple temporal frames are proposed to make fully use of a single spectrogram representation. The proposed spectrogram processing strategies can be easily transferred to any network structures. The experiments are carried out on the DCASE 2020 Task1 datasets, and the results show that our method could achieve the accuracy of 81.8\% (official baseline: 54.1\%) and 92.1\% (official baseline: 87.3\%) on the officially provided fold 1 evaluation dataset of Task1A and Task1B, respectively."
}
```

## Acknowledgment
Thanks for the base code provided by https://github.com/qiuqiangkong/dcase2019_task1/.

```
@article{kong2019cross,
  title={Cross-task learning for audio tagging, sound event detection and spatial localization: DCASE 2019 baseline systems},
  author={Kong, Qiuqiang and Cao, Yin and Iqbal, Turab and Xu, Yong and Wang, Wenwu and Plumbley, Mark D},
  journal={arXiv preprint arXiv:1904.03476},
  year={2019}
}
```

## Contact
If you have any problem about our code, feel free to contact
- wanghl15@pku.edu.cn

or describe your problem in Issues.
