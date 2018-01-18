# NTU ADLxMLDS2017 Final
## HTC Egocentric Hand Detection

---
### 0. model download and pkg install

### download 我們的 best model

```
$ bash download_best_model.sh
```


### 安裝相關套件

```
$ pip3 install -r requirements3.txt
```


---
### 1. 如何跑 training

```
$ python3 hand_train.py [data_path] [selection_sets] [restore_exist_path] [restore_bbx_path] [save_exist_path] [save_bbx_path]
```


以下為參數說明：

data_path: 資料位置，其架構如下

<img src="https://github.com/ExtraOOmegaPPanDDa/ADLxMLDS2017_Final/blob/master/asset/data_tree.png" width="200">

selection_sets: 有 s001~s009、air、book，輸入時請以*號相連

restore_exist_path: 要 restore 的 exist model path，不 restore 請輸入 None

restore_bbx_path: 要 restore 的 bbx model path，不 restore 請輸入 None

save_exist_path: 要 save 的 exist model path

save_bbx_path: 要 save 的 bbx model path

#### Example

```
$ python3 hand_train.py ../data/ s0001*s0002*s0003*s0004 None None ./synth_exist_model.h5 ./synth_bbx_model.h5
```


```
$ python3 hand_train.py ../data/ air*book None None ./synth_exist_model.h5 ./synth_bbx_model.h5 ./vive_exist_model.h5 ./vive_bbx_model.h5
```

---
### 2. 如何跑 testing

```
$ python3 hand_test.py [exist_model_path] [bbx_model_path]
```

以下為參數說明：

exist_model_path: exist model 的位置

bbx_model_path: bbx model 的位置


另外

Testing 的部分，HTC 有提供 judger_hand API，安裝後即可 access testing 資料 


#### Example

```
$ python3 hand_test.py ./hand_exist_model.h5 ./hand_bbx_model.h5
```

---
### 3. 實驗環境描述
---

資料：如第 1. 所示，需含有 DeepQ-Synth-Hand-01、DeepQ-Synth-Hand-02、DeepQ-Vivepaper 在 data_path

系統：Ubuntu 16.04.3 LTS

GPU：GeForce GTX 1080

套件版本：

pkg | version
---- | ---
keras | 2.1.2
tensorflow-gpu | 1.4.1
imageio | 2.2.0
scikit-image | 0.13.1
opencv-python | 3.4.0.12





