## VAE 训练

[English](./README.md) | 简体中文

在完成数据预处理后，你可以获得这样的数据格式:

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

json_of_internal_datasets.json是一个标准的json文件。json中的file_path可以被设置为相对路径，如下所示：
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

你也可以将路径设置为绝对路径：
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

## 训练 Video VAE
我们首先需要修改 ```easyanimate/vae/configs/autoencoder``` 中的配置文件。默认的配置文件是 ```autoencoder_kl_32x32x4_slice.yaml```。你需要修改以下参数： 

- ```data_json_path``` json file 所在的目录。 
- ```data_root``` 数据的根目录。如果你在json file中使用了绝对路径，请设置为空。
- ```ckpt_path``` 预训练的vae模型路径。 
- ```gpus``` 以及 ```num_nodes``` 需要设置为你机器的实际gpu数目。 

运行以下的脚本来训练vae: 
```
sh scripts/train_vae.sh
```