# Neural Search (Work-in-Progress)

Neural search on MS MARCO dataset.


```
apt update
apt install -y build-essential
pip install -U pip
pip install -U tensorboard tensorflow torch tqdm kaggle transformers --extra-index-url https://download.pytorch.org/whl/cu121
```

### Prepare tfdata

```
python prepare_tf_data.py
```


### Or download tfdata

```
kaggle datasets download --unzip -p data -d thimac/msmarco-tfdata
```


### Train model

```
python train.py
```

Train on multiple GPUs (torchrun):

```
torchrun --nproc_per_node=8 train.py
```

### Compute query and passage embed

```
python compute_passage_embed.py --batch-size 2560  --ckpt ckpts/step_00030000.pt
python compute_passage_embed.py --ckpt ckpts/step_00030000.pt
```

```
conda create --name faiss  -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
source activate faiss
pip install tensorflow tqdm
python compute_recall.py
```

