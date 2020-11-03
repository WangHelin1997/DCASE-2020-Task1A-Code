# DCASE-2020-Task1A-Code
A Pytorch implementation of the paper : Acoustic Scene Classification with Spectrogram Processing Strategies.

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
@techreport{Wang2020_t1,
    Author = "Wang, Helin and Chong, Dading and Zou, Yuexian",
    title = "Acoustic Scene Classification with Multiple Decision Schemes",
    institution = "DCASE2020 Challenge",
    year = "2020",
    month = "June",
    abstract = "This technical report describes the ADSPLAB team’s submission for Task1 of DCASE2020 challenge. Our acoustic scene classifi- cation (ASC) system is based on the convolutional neural networks (CNN). Multiple decision schemes are proposed in our system, in- cluding the decision schemes in multiple representations, multiple frequency bands, and multiple temporal frames. The final system is the fusion of models with multiple decision schemes and mod- els pre-trained on AudioSet. The experimental results show that our system could achieve the accuracy of 84.5 \%(official baseline: 54.1\%) and 92.1\% (official baseline: 87.3\%) on the officially provided fold 1 evaluation dataset of Task1A and Task1B, respectively."
}
@misc{wang2020acoustic,
    title={Acoustic Scene Classification with Spectrogram Processing Strategies},
    author={Helin Wang and Yuexian Zou and Dading Chong},
    year={2020},
    eprint={2007.03781},
    archivePrefix={arXiv},
    primaryClass={cs.SD}
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
