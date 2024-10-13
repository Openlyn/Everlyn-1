## VAE Training

English | [简体中文](./README_zh-CN.md)

After completing data preprocessing, we can obtain the following dataset:

```
📦 project/
├── 📂 datasets/
│   ├── 📂 internal_datasets/
│       ├── 📂 videos/
│       │   ├── 📄 00000001.mp4
│       │   ├── 📄 00000001.jpg
│       │   └── 📄 .....
│       └── 📄 json_of_internal_datasets.json
```

The json_of_internal_datasets.json is a standard JSON file. The file_path in the json can to be set as relative path, as shown in below:
```json
[
    {
      "file_path": "videos/00000001.mp4",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "video"
    },
    {
      "file_path": "train/00000001.jpg",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "image"
    },
    .....
]
```

You can also set the path as absolute path as follow:
```json
[
    {
      "file_path": "/mnt/data/videos/00000001.mp4",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "video"
    },
    {
      "file_path": "/mnt/data/train/00000001.jpg",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "image"
    },
    .....
]
```

## Train Video VAE
We need to set config in ```easyanimate/vae/configs/autoencoder``` at first. The default config is ```autoencoder_kl_32x32x4_slice.yaml```. We need to set the some params in yaml file. 

- ```data_json_path``` corresponds to the JSON file of the dataset. 
- ```data_root``` corresponds to the root path of the dataset. If you want to use absolute path in json file, please delete this line.
- ```ckpt_path``` corresponds to the pretrained weights of the vae. 
- ```gpus``` and num_nodes need to be set as the actual situation of your machine. 

The we run shell file as follow: 
```
sh scripts/train_vae.sh
```