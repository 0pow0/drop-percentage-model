Task: Predict drop percentage of SINR, which equals (max sinr - min sinr) / max sinr

Input: Features (Distance, TxP, Bandwidth, Subbandwidth, Sub bandoffset) of variable number of interference base stations (3-10).

Output: Drop percentage = (max - min) / max

Data: Included and located under data/folder. `.npy` data file could be generated with `data_gen.py` script.

Overwrite the `--train_folder` and `--val_folder` to the absolute path of data/train and data/test respectively; Overwrite the `--checkpoint_dir` to the absolute path of the folder to store model parameters.

Start training with:

```python
cat args.txt | xargs python main.py
```

Model parameters would be stored in that folder

Test with:
```python
python util/plot.py
```