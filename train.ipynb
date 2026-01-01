{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "7e5e09ae",
   "metadata": {},
   "source": [
    "安裝依賴項目"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bfdd951c",
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n",
    "!pip install ultralytics"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dff47ee9",
   "metadata": {},
   "source": [
    "驗證環境"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7dac3d96",
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys, pkgutil\n",
    "print(\"Python:\", sys.executable)\n",
    "print(\"ipykernel ok:\", pkgutil.find_loader(\"ipykernel\") is not None)\n",
    "try:\n",
    "    import torch\n",
    "    print(\"Torch:\", torch.__version__)\n",
    "    print(\"CUDA available:\", torch.cuda.is_available())\n",
    "    if torch.cuda.is_available():\n",
    "        print(torch.cuda.get_device_name(0))\n",
    "except Exception as e:\n",
    "    print(\"Torch import error:\", e)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "38f51e23",
   "metadata": {},
   "source": [
    "驗證YOLO ultralytics 套件"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a0ff5bf4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import ultralytics\n",
    "ultralytics.checks()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "03922524",
   "metadata": {},
   "source": [
    "解壓並移動檔案，切分訓練、驗證、與測試數量\n",
    "\n",
    "這邊因省時間\"train\"與\"val\"資料夾中只保留存在標註檔的圖片，只有\"test\"保留完整的"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "688a6683",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os, zipfile, shutil\n",
    "\n",
    "def unzip_if_needed(zip_path, dest_dir):\n",
    "    if os.path.isdir(dest_dir):\n",
    "        return\n",
    "    if os.path.exists(zip_path):\n",
    "        os.makedirs(dest_dir, exist_ok=True)\n",
    "        with zipfile.ZipFile(zip_path, 'r') as zf:\n",
    "            zf.extractall(dest_dir)\n",
    "\n",
    "def find_patient_root(root):\n",
    "    for dirpath, dirnames, _ in os.walk(root):\n",
    "        if any(d.startswith(\"patient\") for d in dirnames):\n",
    "            return dirpath\n",
    "    return root\n",
    "\n",
    "# 解壓縮（不使用外部 unzip 指令）\n",
    "unzip_if_needed(\"training_image.zip\", \"./training_image\")\n",
    "unzip_if_needed(\"training_label.zip\", \"./training_label\")\n",
    "\n",
    "IMG_ROOT = find_patient_root(\"./training_image\")\n",
    "LBL_ROOT = find_patient_root(\"./training_label\")\n",
    "\n",
    "# 建立輸出資料夾\n",
    "for p in [\"./datasets/train/images\",\"./datasets/train/labels\",\n",
    "          \"./datasets/val/images\",\"./datasets/val/labels\",\n",
    "          \"./datasets/test/images\",\"./datasets/test/labels\"]:\n",
    "    os.makedirs(p, exist_ok=True)\n",
    "\n",
    "def move_patients(start, end, split):\n",
    "    # train/val 僅搬移有標註的影像，test 保留全部影像\n",
    "    moved = 0\n",
    "    for i in range(start, end + 1):\n",
    "        patient = f\"patient{i:04d}\"\n",
    "        img_dir = os.path.join(IMG_ROOT, patient)\n",
    "        lbl_dir = os.path.join(LBL_ROOT, patient)\n",
    "        if not os.path.isdir(img_dir):\n",
    "            continue\n",
    "\n",
    "        if split == \"test\":\n",
    "            label_lookup = {}\n",
    "            if os.path.isdir(lbl_dir):\n",
    "                label_lookup = {\n",
    "                    os.path.splitext(fname)[0]: os.path.join(lbl_dir, fname)\n",
    "                    for fname in os.listdir(lbl_dir)\n",
    "                    if fname.endswith(\".txt\")\n",
    "                }\n",
    "            for fname in os.listdir(img_dir):\n",
    "                if not fname.lower().endswith(\".png\"):\n",
    "                    continue\n",
    "                base = os.path.splitext(fname)[0]\n",
    "                img_path = os.path.join(img_dir, fname)\n",
    "                dst_img = f\"./datasets/{split}/images/{base}.png\"\n",
    "                dst_lbl = f\"./datasets/{split}/labels/{base}.txt\"\n",
    "                if os.path.exists(dst_img):\n",
    "                    os.remove(dst_img)\n",
    "                shutil.move(img_path, dst_img)\n",
    "                lbl_path = label_lookup.get(base)\n",
    "                if lbl_path and os.path.exists(lbl_path):\n",
    "                    if os.path.exists(dst_lbl):\n",
    "                        os.remove(dst_lbl)\n",
    "                    shutil.move(lbl_path, dst_lbl)\n",
    "                moved += 1\n",
    "        else:\n",
    "            if not os.path.isdir(lbl_dir):\n",
    "                continue\n",
    "            for lbl_name in os.listdir(lbl_dir):\n",
    "                if not lbl_name.endswith(\".txt\"):\n",
    "                    continue\n",
    "                base = os.path.splitext(lbl_name)[0]\n",
    "                img_path = os.path.join(img_dir, base + \".png\")\n",
    "                lbl_path = os.path.join(lbl_dir, lbl_name)\n",
    "                if not os.path.exists(img_path):\n",
    "                    continue\n",
    "                dst_img = f\"./datasets/{split}/images/{base}.png\"\n",
    "                dst_lbl = f\"./datasets/{split}/labels/{base}.txt\"\n",
    "                if os.path.exists(dst_img):\n",
    "                    os.remove(dst_img)\n",
    "                shutil.move(img_path, dst_img)\n",
    "                if os.path.exists(dst_lbl):\n",
    "                    os.remove(dst_lbl)\n",
    "                shutil.move(lbl_path, dst_lbl)\n",
    "                moved += 1\n",
    "    return moved\n",
    "\n",
    "n_train = move_patients(1, 40, \"train\")\n",
    "n_val   = move_patients(41, 45, \"val\")\n",
    "n_test  = move_patients(46, 50, \"test\")\n",
    "\n",
    "print(f\"完成移動：train {n_train} 筆，val {n_val} 筆, test {n_test} 筆\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9c4a8e34",
   "metadata": {},
   "source": [
    "驗證圖片數量"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "df5f9618",
   "metadata": {},
   "outputs": [],
   "source": [
    "print('訓練集圖片數量 : ',len(os.listdir(\"./datasets/train/images\")))\n",
    "print('訓練集標記數量 : ',len(os.listdir(\"./datasets/train/labels\")))\n",
    "print('驗證集圖片數量 : ',len(os.listdir(\"./datasets/val/images\")))\n",
    "print('驗證集標記數量 : ',len(os.listdir(\"./datasets/val/labels\")))\n",
    "print('測試集圖片數量 : ',len(os.listdir(\"./datasets/test/images\")))\n",
    "print('測試集標記數量 : ',len(os.listdir(\"./datasets/test/labels\")))\n",
    "print('總圖片數量: ',len(os.listdir(\"./datasets/train/images\")) + len(os.listdir(\"./datasets/val/images\")) + len(os.listdir(\"./datasets/test/images\")))\n",
    "print('總標記數量: ',len(os.listdir(\"./datasets/train/labels\")) + len(os.listdir(\"./datasets/val/labels\")) + len(os.listdir(\"./datasets/test/labels\")))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d053c01b",
   "metadata": {},
   "source": [
    "產生我們需要的.yaml檔案"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "46342e67",
   "metadata": {},
   "outputs": [],
   "source": [
    "from textwrap import dedent\n",
    "from pathlib import Path\n",
    "\n",
    "data_yaml = dedent(\n",
    "    \"\"\"\n",
    "train: \"./datasets/train/images\"\n",
    "val: \"./datasets/val/images\"\n",
    "test: \"./datasets/test/images\"\n",
    "\n",
    "names:\n",
    "  0: aortic_valve\n",
    "\"\"\"\n",
    ").strip()\n",
    "\n",
    "config_path = Path(\"aortic_valve_colab.yaml\")\n",
    "config_path.write_text(data_yaml + \"\\n\", encoding=\"utf-8\")\n",
    "\n",
    "print(f\"YAML saved to: {config_path.resolve()}\")\n",
    "print(data_yaml)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "160b5649",
   "metadata": {},
   "source": [
    "開始訓練\n",
    "\n",
    "若擔心GPU 記憶體不足可以嘗試降低batch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "56a31d36",
   "metadata": {},
   "outputs": [],
   "source": [
    "from ultralytics import YOLO\n",
    "\n",
    "model = YOLO('yolo12n.pt') #初次訓練使用YOLO官方的預訓練模型，如要使用自己的模型訓練可以將'yolo12n.pt'替換掉\n",
    "results = model.train(data=\"./aortic_valve_colab.yaml\",\n",
    "            epochs=10, #跑幾個epoch，這邊設定10做範例測試，可依需求調整\n",
    "            batch=16, #batch_size\n",
    "            imgsz=640, #圖片大小640*640\n",
    "            device=0 #使用GPU進行訓練\n",
    "            )"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ca59a302",
   "metadata": {},
   "source": [
    "解壓縮testing_imgae"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e26bc586",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os, zipfile, shutil\n",
    "\n",
    "def unzip_if_needed(zip_path, dest_dir):\n",
    "    if os.path.isdir(dest_dir):\n",
    "        return\n",
    "    if os.path.exists(zip_path):\n",
    "        os.makedirs(dest_dir, exist_ok=True)\n",
    "        with zipfile.ZipFile(zip_path, 'r') as zf:\n",
    "            zf.extractall(dest_dir)\n",
    "\n",
    "def find_patient_root(root):\n",
    "    for dirpath, dirnames, _ in os.walk(root):\n",
    "        if any(d.lower().startswith(\"patient\") for d in dirnames):\n",
    "            return dirpath\n",
    "    return root\n",
    "\n",
    "# 解壓縮（不使用外部 unzip 指令）\n",
    "unzip_if_needed(\"testing_image.zip\", \"./testing_image\")\n",
    "\n",
    "TEST_ROOT = find_patient_root(\"./testing_image\")\n",
    "\n",
    "# 建立輸出資料夾\n",
    "DST_TEST = \"./datasets/predict/images\"\n",
    "os.makedirs(DST_TEST, exist_ok=True)\n",
    "\n",
    "# 收集所有圖片路徑（只看直屬的 patient 資料夾）\n",
    "all_files = []\n",
    "for patient_folder in os.listdir(TEST_ROOT):\n",
    "    patient_path = os.path.join(TEST_ROOT, patient_folder)\n",
    "    if os.path.isdir(patient_path) and patient_folder.lower().startswith(\"patient\"):\n",
    "        for fname in os.listdir(patient_path):\n",
    "            if fname.lower().endswith(\".png\"):\n",
    "                all_files.append(os.path.join(patient_path, fname))\n",
    "\n",
    "# 按名稱排序並複製\n",
    "all_files.sort()\n",
    "copied = 0\n",
    "for f in all_files:\n",
    "    dst = os.path.join(DST_TEST, os.path.basename(f))\n",
    "    if os.path.exists(dst): os.remove(dst)  # 不想覆蓋就刪掉這行\n",
    "    shutil.copy2(f, dst)\n",
    "    copied += 1\n",
    "\n",
    "print(f\"來源根目錄：{TEST_ROOT}\")\n",
    "print(f\"完成複製！總共 {copied} 張，全部到 {DST_TEST}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4fe1d15a",
   "metadata": {},
   "source": [
    "預測競賽分數約10分鐘"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "023e98eb",
   "metadata": {},
   "outputs": [],
   "source": [
    "from ultralytics import YOLO\n",
    "\n",
    "model = YOLO(r'.\\runs\\detect\\train\\weights\\best.pt') #請自行更改最新的best.pt檔路徑\n",
    "results = model.predict(source=\"./datasets/predict/images/\",\n",
    "              save=True,\n",
    "              imgsz=640,\n",
    "              device=0\n",
    "              )"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f718fcd2",
   "metadata": {},
   "source": [
    "預測數量"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2e235e58",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(len(results))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e90bef58",
   "metadata": {},
   "source": [
    "將偵測框數值寫進.txt檔"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "db0e2e04",
   "metadata": {},
   "outputs": [],
   "source": [
    "os.makedirs('./predict_txt', exist_ok=True)\n",
    "\n",
    "\n",
    "output_file = open('./predict_txt/images.txt', 'w', encoding='utf-8')\n",
    "\n",
    "for i in range(len(results)):\n",
    "    filename = str(results[i].path).replace('\\\\', '/').split('/')[-1].split('.png')[0]\n",
    "    boxes = results[i].boxes\n",
    "    box_num = len(boxes.cls.tolist())\n",
    "    if box_num > 0:\n",
    "        for j in range(box_num):\n",
    "            label = int(boxes.cls[j].item())\n",
    "            conf = boxes.conf[j].item()\n",
    "            x1, y1, x2, y2 = boxes.xyxy[j].tolist()\n",
    "            line = f\"{filename} {label} {conf:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\\n\"\n",
    "            output_file.write(line)\n",
    "output_file.close()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e1ad6e1d",
   "metadata": {},
   "source": [
    "釋放記憶體"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3292b2e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch ,gc\n",
    "\n",
    "# 刪除大型變數\n",
    "del boxes,all_files,results\n",
    "gc.collect()\n",
    "torch.cuda.empty_cache()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5397d0ee",
   "metadata": {},
   "source": [
    "驗證模型(本周作業報告) 利用 YOLO 輸出預測結果"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "df16a98b",
   "metadata": {},
   "outputs": [],
   "source": [
    "!yolo detect val model=runs\\detect\\train\\weights\\best.pt data=aortic_valve_colab.yaml split=test save_txt save_conf"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "ML1141",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
