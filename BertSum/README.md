# Replicating the BertSum Model.
本專案旨在重現 [BertSum](https://github.com/nlpyang/BertSum) 模型，並解決在本地端環境下建立與執行的挑戰

## 📌 背景說明

由於 Google Colab 免費版本具有以下限制：

- GPU 使用資源有限
- 長時間閒置會自動斷線

因此，選擇改為 **在本機使用 WSL（Windows Subsystem for Linux）** 架設環境，以穩定執行實驗。然而在 Windows 原生系統中安裝 `pyrouge` 時經常遇到相容性問題，因此最終決定使用 **Ubuntu on WSL** 來建立虛擬環境。 

---

## 📁 目錄

- **安裝 CUDA Toolkit**
- **建立 Python 環境**
- **複製並處理 BertSum 原始碼與資料集**
- **開始訓練模型**
- **模型驗證與測試結果**

---

## ⚙️ 安裝 CUDA Toolkit

1. **檢查 GPU 支援的 CUDA 版本：**

   在 Terminal 輸入以下指令：
   
   ```bash
   nvidia-smi
   ```
   你將會看到如下畫面(範例):
   
   ![image](https://github.com/user-attachments/assets/cb1e6265-2630-4f9a-bdf5-1e3556f5e4c9)

   > 提示：請根據 `CUDA Version` 資訊選擇相容的 CUDA Toolkit（建議不超過顯示的版本)

   ![image](https://github.com/user-attachments/assets/afd8fadf-f7f0-49cf-8b1a-e4fa37685141)
   
   > 提示：也可以注意一下選擇的 CUDA Toolkit 與電腦的 `Driver Version` 是否相容

2. 前往 [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)，選擇適合版本 (本例為 12.6.0) 生成下載指令:

   ![image](https://github.com/user-attachments/assets/a6988419-1fad-4c99-955d-56eb14813a0e)
   ``` bash
   wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run
   sudo sh cuda_12.6.0_560.28.03_linux.run
   ```
   > 提示：於 Terminal 依序輸入上方指令，跟著指引安裝就可以安裝完畢


## ⚙️ 設定環境變數

3. 使用 Vim 編輯 `.bashrc` 檔案：

   ```bash
   sudo vim ~/.bashrc
   ```
   在檔案底部新增以下內容:

   ```bash
   export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
   export LS_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
   export CUDA_HOME=/usr/local/cuda-12.6
   ```
   > 提示：將光標滑到文件最末端，按下 `i`，即可開始編輯檔案 <br>
   > 提示：編輯完畢後按下 `esc`，回到一般模式，再輸入 `:wq`，保存文件後離開

4. 更新環境變數

   ```bash
   source ~/.bashrc
   sudo ldconfig
   ```

## ✅ 確認安裝成功

5. 在 Terminal 輸入以下指令檢查 CUDA 是否安裝成功:

   ```bash
   nvcc -V
   ```
   若顯示如下資訊，及代表安裝成功:
   ![image](https://github.com/user-attachments/assets/f86f0a43-955d-4af4-b598-d600633d873d)
   

## 建立 Python 環境

以下步驟將協助你在 WSL 上建立適用於 BertSum 的 Python 開發環境。

---

### 1️⃣ 建立 Conda 虛擬環境

```bash
conda create --name env_bert python=3.10
conda activate
```
>建立名為 env_bert 的環境，並指定 Python 版本為 3.10。

### 2️⃣ 安裝 PyTorch（CUDA 12.6 相容版本）

請前往官方網站確認對應 CUDA 的下載版本指令: 👉https://pytorch.org/get-started/previous-versions/

針對 CUDA 12.6，安裝指令如下:
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

### 3️⃣ 安裝 BertSum 執行所需套件

```bash
pip install pytorch_pretrained_bert
pip install tensorboardX
pip install multiprocess
```

### 4️⃣ 安裝 Pyrouge（評估套件）

由於 `pyrouge` 安裝較為複雜，建議依序執行以下指令:

```bash
pip install pyrouge
pip install pyrouge --upgrade
pip install https://github.com/bheinzerling/pyrouge/archive/master.zip
pip show pyrouge
```

接著 clone 官方 repo 並設定路徑:
```bash
git clone https://github.com/andersjo/pyrouge.git
pyrouge_set_rouge_path '/home/szuting/pyrouge/tools/ROUGE-1.5.5'
```

安裝過程參考截圖:

![image](https://github.com/user-attachments/assets/508ed93b-5704-4954-a0b6-40241c28f73c)

![image](https://github.com/user-attachments/assets/8c2db840-e973-4cd3-af93-d536309f90bf)

![image](https://github.com/user-attachments/assets/1234d295-b886-4038-a7f3-61d8f5171ec3)

![image](https://github.com/user-attachments/assets/b74ff568-3e2a-468b-92be-9747100d3fd0)

![image](https://github.com/user-attachments/assets/3d5aab22-b392-4225-a9ca-108bb6ecf500)

![image](https://github.com/user-attachments/assets/3bf43cbb-99c5-4311-bba4-6e21d59e5110)

![image](https://github.com/user-attachments/assets/e39910d8-67ca-4a2b-b073-a5acdc1ad290)

### 5️⃣ 安裝 Perl 套件（為 ROUGE 工具提供 XML 支援）

```bash
sudo apt-get install libxml-parser-perl
```
執行畫面: 

![image](https://github.com/user-attachments/assets/e2be645b-590b-49e3-b5e7-0657e1ff7e11)

### 6️⃣ 修復 WordNet-2.0.exc.db 錯誤（ROUGE 相關問題）

1. 進入資料夾:

   ```bash
   cd pyrouge/tools/ROUGE-1.5.5/data
   ```
2. 刪除已存在的 `WordNet-2.0.exc.db` (若有):

   ```bash
   rm WordNet-2.0.exc.db
   ```
3. 進入子資料夾:

   ```bash
   cd WordNet-2.0-Exceptions
   rm WordNet-2.0.exc.db  # 若存在則刪除
   ```
4. 重新建立 `.exc.db`:

   ```bash
   ./buildExeptionDB.pl . exc WordNet-2.0.exc.db
   ```
5. 回到 `data` 資料夾，建立符號連結:

   ```bash
   cd ../
   ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db
   ```

## 複製並處理 BertSum 原始碼與資料集

以下步驟將說明如何取得 BertSum 的程式碼，處理資料集，以及修正執行時可能遇到的錯誤。

---

### 1️⃣ 複製 BertSum 原始碼至本地端

使用 Git 將原始碼下載到當前目錄：

```bash
git clone https://github.com/nlpyang/BertSum.git
```
![image](https://github.com/user-attachments/assets/50cf4fe2-cb32-41e2-9682-53266aa12c5a)

可使用以下指令確認內容是否下載成功:

```bash
ls -lt BertSum
```
![image](https://github.com/user-attachments/assets/49038f97-6d46-4c49-873b-3916325c320f)

### 2️⃣ 將資料集下載並解壓縮至 `bert_data/`

1. 前往以下連結下載資料集 (zip 壓縮檔)
   
   👉 [BertSum dataset (Google Drive)](https://drive.usercontent.google.com/download?id=1x0d61LP9UAN389YN00z0Pv-7jQgirVg6&export=download&authuser=0)

2. 將壓縮檔放入 `BertSum/bert_data/` 資料夾

3. 使用以下指令完成解壓縮與確認:

   ```bash
   cd BertSum/bert_data       # 進入資料夾
   du -h bertsum_data.zip     # 查看壓縮檔大小
   unzip bertsum_data.zip     # 解壓縮資料
   ```

   解壓縮成功畫面:

   ![image](https://github.com/user-attachments/assets/dd156348-5fbf-4945-aeb9-ecb815ab3fde)

   ![image](https://github.com/user-attachments/assets/74fd6ca7-f62a-45a8-bd5b-23baab53bd76)

### ⚠️ 修正程式碼中的 mask 錯誤（避免類型錯誤）

在某些環境中執行 BertSum 可能會遇到 tensor 類型不相容的錯誤，請修改以下程式碼：

📄 檔案路徑：`BertSum/src/models/dataloader.py`

修改第 31 行和 34 行如下:

```python
# 第 31 行：
mask = ~(src == 0)

# 第 34 行：
mask_cls = ~(clss == 0)
```

## 🚀 開始訓練模型

使用以下指令透過 `train.py` 腳本開始訓練 BertSum 模型。

```bash
python train.py \
    -mode train \
    -encoder classifier \
    -dropout 0.1 \
    -bert_data_path ../bert_data/cnndm \
    -model_path ../models/bert_classifier \
    -lr 2e-3 \
    -visible_gpus 0 \
    -gpu_ranks 0 \
    -world_size 1 \
    -report_every 50 \
    -save_checkpoint_steps 500 \
    -batch_size 6000 \
    -decay_method noam \
    -train_steps 2000 \
    -accum_count 2 \
    -log_file ../logs/bert_classifier \
    -use_interval true \
    -warmup_steps 1000
```
📌 參數說明：

| 參數                    | 說明                                       |
|-------------------------|--------------------------------------------|
| `-mode train`           | 執行訓練模式                                |
| `-encoder classifier`   | 使用分類器作為 encoder                      |
| `-dropout 0.1`          | dropout 機率為 0.1，防止過擬合              |
| `-bert_data_path`       | 指定訓練資料的路徑                          |
| `-model_path`           | 訓練後模型儲存的位置                        |
| `-lr 2e-3`              | 設定學習率為 0.002                         |
| `-visible_gpus 0`       | 指定使用 GPU 0                              |
| `-gpu_ranks 0`          | 指定訓練時的 GPU 排序                       |
| `-world_size 1`         | GPU 數量設定（單機單 GPU 時為 1）           |
| `-report_every 50`      | 每 50 步輸出一次訓練狀態                    |
| `-save_checkpoint_steps 500` | 每 500 步儲存一次模型                    |
| `-batch_size 6000`      | 每批次使用約 6000 個 tokens                 |
| `-decay_method noam`    | 使用 Noam decay 調整學習率                 |
| `-train_steps 2000`     | 訓練總步數設為 2000                         |
| `-accum_count 2`        | 每 2 批次累積一次梯度，降低記憶體壓力       |
| `-log_file`             | 指定 log 檔輸出位置                         |
| `-use_interval true`    | 啟用句子之間的段落間隔                      |
| `-warmup_steps 1000`    | 訓練前預熱的步數，用於穩定學習率曲線        |

訓練過程輸出結果如下:

>![image](https://github.com/user-attachments/assets/c410eea9-88c4-4332-afd5-dc2bd058ffb6)
>![image](https://github.com/user-attachments/assets/4a69763a-1576-44c5-bb6d-52f2ae81752a)
>![image](https://github.com/user-attachments/assets/92262c44-f89e-4727-8197-71e12b0157e7)
>
>![image](https://github.com/user-attachments/assets/e5f69df2-12de-4181-ac0f-1df33f2a5312)
>![image](https://github.com/user-attachments/assets/86e18bf2-734e-43e6-8b1a-53e4b6044993)
>![image](https://github.com/user-attachments/assets/34560661-ff85-4bb4-b162-625348a8491e)

## 🚀 模型驗證與測試結果

使用以下指令對訓練完成的模型進行驗證：

```bash
python train.py \
      -mode validate \
      -bert_data_path ../bert_data/cnndm \
      -model_path ../models/bert_classifier  \
      -visible_gpus 0 \
      -gpu_ranks 0 \
      -batch_size 10000  \
      -log_file ../models/bert_classifier/log  \
      -result_path ../models/bert_classifier/results \
      -test_all \
      -block_trigram true
```

📌 參數說明：

| 參數 | 說明 |
|------|------|
| `-mode validate` | 設定執行模式為驗證（可選值還有 `train`, `test` 等）。 |
| `-bert_data_path ../bert_data/cnndm` | 指定預處理後的 BERT 格式資料路徑（此處為 CNN/DM 資料集）。 |
| `-model_path ../models/bert_classifier` | 模型參數儲存與載入的路徑。 |
| `-visible_gpus 0` | 指定可用的 GPU 編號（此處使用第 0 張 GPU）。 |
| `-gpu_ranks 0` | 指定訓練時的 GPU 排名，常與多 GPU 設定有關。 |
| `-batch_size 10000` | 單批次處理的樣本總 token 數（而非單純樣本數），通常用於記憶體最佳化。 |
| `-log_file ../models/bert_classifier/log` | 訓練或驗證過程中的日誌儲存位置。 |
| `-result_path ../models/bert_classifier/results` | 驗證過程中預測結果的輸出位置。 |
| `-test_all` | 使用所有檢查點（checkpoints）進行測試 / 驗證（會自動載入多個模型並逐一驗證）。 |
| `-block_trigram true` | 是否啟用三元組重複過濾（防止產出重複句子），常用於摘要任務中提升多樣性。 |

### ⚠️ 修正 weights_only 載入錯誤 (PyTorch 2.6+)

若使用 PyTorch 2.6 或以上版本，可能會遇到如下錯誤：

![image](https://github.com/user-attachments/assets/0e496721-ace6-400a-804a-85a4ecf4c3c6)

🔧 解決方式如下：

📄 修改檔案：`BertSum/src/train.py`

```python
# 在第 173 行和 200 行的 torch.load 加上 weights_only=False

# 原本版本
checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)

# 修改後
checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage, weights_only=False)
```
💡 說明： 加上 weights_only=False，表示允許還原模型檔案中的全部 Python 物件（如 Namespace），避免新版 PyTorch 的安全限制導致載入錯誤。

驗證結果輸出如下:

>![image](https://github.com/user-attachments/assets/0621ecbb-5be1-497e-90f1-6cc29476a50e)
>![image](https://github.com/user-attachments/assets/36853850-8874-48ee-b492-58ea8822e839)
>![image](https://github.com/user-attachments/assets/19c20242-b0c4-45d1-ac1b-cdad0f921061)
>![image](https://github.com/user-attachments/assets/45ed2411-8d85-4c89-ac09-0b63d5529a29)

測試結果輸出如下:

model_step_2000.pt:

>![image](https://github.com/user-attachments/assets/1ea9c561-c1b7-4311-9b71-e6c0c702b38d)
>![image](https://github.com/user-attachments/assets/ff7ee8ed-86c3-447d-bfe8-08d6c1b81ba7)
>![image](https://github.com/user-attachments/assets/b1b4903a-9ef1-4df7-8de9-d135e6fc8c2c)
>![image](https://github.com/user-attachments/assets/304103d0-d15d-4581-93ae-b92a6f4703a0)

model_step_1000.pt:

>![image](https://github.com/user-attachments/assets/47f8df35-dbf7-4504-b53b-0b1117337393)
>![image](https://github.com/user-attachments/assets/ff7b4ab0-85f4-4e1d-b610-9bb71293d352)
>![image](https://github.com/user-attachments/assets/53f9edcb-e0f0-4dd7-8af1-29bbccb07365)

model_step_1500.pt:

>![image](https://github.com/user-attachments/assets/5b4e4070-81b7-4504-9a4d-36a91002ca05)
>![image](https://github.com/user-attachments/assets/1ef981cc-f9fb-4ebe-b2b6-3e04527a00f9)
>![image](https://github.com/user-attachments/assets/ca7e0a8a-75e2-4bd0-97d2-75c6edca189e)







