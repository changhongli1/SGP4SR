# Overview
This is the source code for SGP4SR.

Code of our paper `SGP4SR: Separated-Modality Guided User Preference Learning for Multimodal Sequential Recommendation`, accepted by AAAI 2026.


## How to Run
The interaction data of the Baby dataset is already provided in our "./dataset/baby" directory. Due to the size limitation of the supplementary materials, please visit the public work MMRec(https://github.com/enoche/MMRec/tree/master/data) and download the preprocessed text and image modalities (corresponding to the "text_feat.npy" and "image_feat.npy" files) from the Baby dataset there. Then, rename them to "baby.text" and "baby.image" respectively and place them in the "./dataset/baby" directory.

After preparing the virtual environment, run the following code directly:

```bash
python run.py
```

## Citation

```
@inproceedings{SGP4SR,
  title={SGP4SR: Separated-Modality Guided User Preference Learning for Multimodal Sequential Recommendation},
  author={Li, Changhong and Guo, Zhiqiang and Li, Guohui and Yang, Zhong and Hong, Chuhang},
  booktitle={Proceedings of the 40th Annual AAAI Conference on Artificial Intelligence},
  volume={40},
  number={18},
  pages={15054--15062},
  address={Singapore},
  year={2026},
}
```

## Acknowledgement

This repository is based on [MISSRec](https://github.com/gimpong/MM23-MISSRec) and [Recbole](https://github.com/RUCAIBox/RecBole).

We would like to extend our sincere appreciation to these repositories for their invaluable code contributions and dedicated efforts, which have lent substantial support to the progress of our work.


