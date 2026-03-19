# Seperated-Modality Guided User Perference Learning for Multimodal Sequential Reconmmendation

## Preparation of data
The interaction data of the Baby dataset is already provided in our "./dataset/baby" directory. Due to the size limitation of the supplementary materials, please visit the public work MMRec(https://github.com/enoche/MMRec/tree/master/data) and download the preprocessed text and image modalities (corresponding to the "text_feat.npy" and "image_feat.npy" files) from the Baby dataset there. Then, rename them to "baby.text" and "baby.image" respectively and place them in the "./dataset/baby" directory.

## Preparation of the running environment
Our source code is mainly based on Recbole. Please prepare a virtual environment according to the corresponding versions of the following dependent libraries:
| Package            | Version                | Package           | Version                |
|-------------------------|------------------------|-------------------------|------------------------|
| absl-py                 | 2.1.0                  | cachetools              | 5.5.2                  |
| certifi                 | 2022.12.7              | charset-normalizer      | 3.4.2                  |
| colorama                | 0.4.6                  | colorlog                | 6.9.0                  |
| cupy-cuda11x            | 11.6.0                 | fastrlock               | 0.8.3                  |
| google-auth             | 2.40.3                 | google-auth-oauthlib    | 0.4.6                  |
| grpcio                  | 1.62.3                 | idna                    | 3.10                   |
| importlib-metadata      | 6.7.0                  | joblib                  | 1.3.2                  |
| Markdown                | 3.4.4                  | MarkupSafe              | 2.1.5                  |
| numpy                   | 1.21.6                 | oauthlib                | 3.2.2                  |
| pandas                  | 1.3.5                  | Pillow                  | 9.5.0                  |
| pip                     | 22.3.1                 | protobuf                | 3.20.3                 |
| pyasn1                  | 0.5.1                  | pyasn1-modules          | 0.3.0                  |
| python-dateutil         | 2.9.0.post0            | pytz                    | 2025.2                 |
| PyYAML                  | 6.0.1                  | requests                | 2.31.0                 |
| requests-oauthlib       | 2.0.0                  | rsa                     | 4.9.1                  |
| scikit-learn            | 1.0.2                  | scipy                   | 1.7.3                  |
| setuptools              | 65.6.3                 | six                     | 1.17.0                 |
| tensorboard             | 2.11.2                 | tensorboard-data-server | 0.6.1                  |
| tensorboard-plugin-wit  | 1.8.1                  | threadpoolctl           | 3.1.0                  |
| torch                   | 1.12.0+cu113           | torchaudio              | 0.12.0+cu113           |
| torchvision             | 0.13.0+cu113           | tqdm                    | 4.67.1                 |
| typing_extensions       | 4.7.1                  | urllib3                 | 2.0.7                  |
| Werkzeug                | 2.2.3                  | wheel                   | 0.38.4                 |
| zipp                    | 3.15.0                 |                        |                        |


Then, run the following code directly:

```bash
python run.py
```

We have also provided our completed running results in the file "./log/SGP/baby.log" to facilitate your verification of our method. If you have any questions regarding the code, please feel free to ask us during the Rebuttal stage. Additionally, we will supplement all datasets mentioned in the paper as well as the code for other experiments once the paper is accepted.

