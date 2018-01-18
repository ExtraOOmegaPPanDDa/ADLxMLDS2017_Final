# NTU ADLxMLDS2017 Final
## HTC Egocentric Hand Detection

---
### 0. download 我們的 best model

```
$ bash download_best_model.sh
```


---
### 1. 如何跑 training

```
$ python3 hand_train.py [data_path] [selection_sets] [restore_exist_path] [restore_bbx_path] [save_exist_path] [save_bbx_path]
```


以下為參數說明

data_path: 資料位置，其架構如下

<img src="https://github.com/ExtraOOmegaPPanDDa/ADLxMLDS2017_Final/blob/master/asset/data_tree.png" width="200">

selection_sets: 有 s001~s009、air、book，輸入時請以*號相連

restore_exist_path: 要 restore 的 exist model path，不 restore 請輸入 None

restore_bbx_path: 要 restore 的 bbx model path，不 restore 請輸入 None

save_exist_path: 要 save 的 exist model path

save_bbx_path: 要 save 的 bbx model path

#### Example

```
$ python3 hand_train.py ../data/ air*book None None ./train_exist_model.h5 ./train_bbx_model.h5
```


---
### 2. 如何跑 testing

```
$ python3 hand_test.py [exist_model_path] [bbx_model_path]
```

#### Example

```
$ python3 hand_test.py ./hand_exist_model.h5 ./hand_bbx_model.h5
```

---
### 3. 實驗環境描述
---