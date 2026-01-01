# NKUST 機器學習期末報告 - TEAM_9122

## 專案簡介
本專案使用 YOLO 模型進行主動脈瓣 (Aortic Valve) 的檢測任務。包含完整的資料前處理、模型訓練流程以及結果評估程式碼。

## 檔案說明
- **train.ipynb**: 包含資料下載、資料集分割 (Train/Val/Test)、YOLO 模型訓練與預測的完整流程。
- **evaluate_yolo.py**: 用於讀取預測結果，計算 IoU、Precision、Recall 並繪製可視化結果。
- **aortic_valve_colab.yaml**: YOLO 訓練配置文件。

## 環境安裝
本專案建議在 Google Colab 環境下執行，所需套件如下：
```bash
pip install ultralytics
pip install opencv-python-headless
pip install numpy
pip install pyyaml
