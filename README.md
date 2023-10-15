# Neural Search (Work-in-Progress)

Neural search on MS MARCO dataset.


```
apt update
apt install -y build-essential
pip install -U pip
pip install tensorboard tensorflow torch tqdm kaggle transformers
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
torchrun --nproc_per_node=8 train.py --embed-dim 256
```

### Compute query and passage embed

```
python compute_passage_embed.py --batch-size 2560  --ckpt ckpts/step_00019000.pt --embed-dim 256
python compute_passage_embed.py
```

