# ### 這裡開始


import pandas as pd
from datasets import load_dataset, Audio ,DatasetDict

#模型變數設置
#medium 
useModel = "openai/whisper-small"
language1 = "English" #"Chinese"
language = "en" #"zh"

#讀取原始CSV
df = pd.read_csv("processed_songs/metadata_all.csv")

df['file'] = "processed_songs/" + df['file'].astype(str)
#改欄位名稱符合Hugging Face標準
df = df.rename(columns={"file": "audio", "text": "sentence"})
df = df[["audio","sentence"]]

#存成新的CSV
df.to_csv("processed_songs/metadata_hf.csv", index=False)

#用Hugging Face載入新的CSV
dataset = load_dataset(
    "csv",
    data_files={"train": "processed_songs/metadata_hf.csv"}
)

#直接在cast_column中指定目標採樣率
#這會告訴datasets在讀取音檔時，自動解碼並重採樣到16000Hz
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
common_voice = dataset

#特徵提取器(預處理音檔)_選擇模型
#模型輸出後處理為文字格式的分詞器
#language要自己改
from transformers import WhisperFeatureExtractor, WhisperTokenizer

feature_extractor = WhisperFeatureExtractor.from_pretrained(useModel)
tokenizer = WhisperTokenizer.from_pretrained(useModel, language=language1, task="transcribe")

input_str = common_voice["train"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")


from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig
"""輸出格式範例(採樣率48K)
{'audio': {'path': '/home/sanchit_huggingface_co/.cache/huggingface/datasets/downloads/extracted/607848c7e74a89a3b5225c0fa5ffb9470e39b7f11112db614962076a847f3abf/cv-corpus-11.0-2022-09-21/hi/clips/common_voice_hi_25998259.mp3', 
           'array': array([0.0000000e+00, 0.0000000e+00, 0.0000000e+00, ..., 9.6724887e-07,
       1.5334779e-06, 1.0415988e-06], dtype=float32), 
           'sampling_rate': 48000},
 'sentence': 'खीर की मिठास पर गरमाई बिहार की सियासत, कुशवाहा ने दी सफाई'}
"""
processor = WhisperProcessor.from_pretrained(useModel, language=language, task="transcribe")

model = WhisperForConditionalGeneration.from_pretrained(useModel)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.decoder_start_token_id = 50258 
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id

generation_config = GenerationConfig.from_model_config(model.config)
generation_config.update(
    language=language, 
    task="transcribe",
    forced_decoder_ids=None,
    max_length=225, # 配合你後面的設定
    suppress_tokens=[],
    begin_suppress_tokens=[220, 50257] # 抑制非文字符號
)
model.generation_config = generation_config

#print(common_voice["train"][0])


#from datasets import Audio
"""輸出格式範例(下採樣至16K)
{'audio': {'path': '/home/sanchit_huggingface_co/.cache/huggingface/datasets/downloads/extracted/607848c7e74a89a3b5225c0fa5ffb9470e39b7f11112db614962076a847f3abf/cv-corpus-11.0-2022-09-21/hi/clips/common_voice_hi_25998259.mp3', 
           'array': array([ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00, ...,
       -3.4206650e-07,  3.2979898e-07,  1.0042874e-06], dtype=float32),
           'sampling_rate': 16000},
 'sentence': 'खीर की मिठास पर गरमाई बिहार की सियासत, कुशवाहा ने दी सफाई'}
"""
#common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

print(common_voice["train"][0])


#初始化processor(包含feature_extractor + tokenizer)
import torch
def prepare_dataset(batch, processor, torch):
    #processor = WhisperProcessor.from_pretrained(useModel, language="zh", task="transcribe")
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    batch["labels"] = torch.tensor(batch["labels"], dtype=torch.long)
    return batch

common_voice = common_voice.map(
    prepare_dataset,
    fn_kwargs={"processor":processor,"torch":torch},
    remove_columns=common_voice["train"].column_names,
    num_proc=14  #取決於你的CPU核心，自己填數字
)



#定義數據收集器
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        #split inputs and labels since they have to be of different lengths and need different padding methods
        #first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        #get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        #pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        #replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        #if bos token is appended in previous tokenization step,
        #cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


#初始化數據整理器
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id, # 現在這是正確的 50258
)

import evaluate
import numpy as np

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    
    if pred_ids.ndim == 3:
        pred_ids = pred_ids.argmax(-1)

    # 統一使用 processor.tokenizer
    label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # 根據全域變數 language 自動切換
    if language == "zh":
        # 中文：計算字元錯誤率 (CER)
        pred_str = [" ".join(list(s.replace(" ", ""))) for s in pred_str]
        label_str = [" ".join(list(s.replace(" ", ""))) for s in label_str]
    else:
        # 英文：轉小寫計算 WER
        pred_str = [s.lower() for s in pred_str]
        label_str = [s.lower() for s in label_str]

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

#載入預訓練checkpoint
from transformers import WhisperForConditionalGeneration

#模型自己改


from transformers import Seq2SeqTrainingArguments,EarlyStoppingCallback

#如果不想將模型checkpoint上傳到Hub，設定push_to_hub=False
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-en",
    per_device_train_batch_size=4,  #如果出現OOM(顯存超載)錯誤將此減半(初始值8)
    gradient_accumulation_steps=4,  #並將此乘2(初始值2)
    learning_rate=1e-5,
    warmup_steps=200,   # 縮短warmup，因為模型很快就收斂了
    max_steps=10000,
    gradient_checkpointing=True, #開啟這個可以節省顯存，雖然會慢一點
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    
    # --- 關鍵改動處 ---
    save_steps=100,            #每100步就檢查一次
    eval_steps=100,            #每100步就評估一次
    save_total_limit=2,        #只保留最好的兩個模型，省硬碟空間
    weight_decay=0.05,         #增加權重衰減，防止過擬合
    label_smoothing_factor=0.1, #增加標籤平滑，提高泛化能力
    # ----------------
    
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=5 # 連續 5 次評估沒有改善則停止
)


#從訓練資料集中抓出測試資料
common_voice = common_voice["train"].train_test_split(test_size=0.1)


from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[early_stopping_callback],
)


#訓練模型
trainer.train()


#設定上傳模型數值
kwargs = {
    "dataset_tags": "Zzzkay1/song", #標籤，記得改
    "dataset": "English songs * 25",  #a 'pretty' name for the training dataset (翻譯)給資料集設定一個你喜歡的名子
    "dataset_args": "config: en, split: test",
    "language": "en", #語言自己改
    "model_name": "Whisper smail en - Song train",  #a 'pretty' name for your model (翻譯)給模型一個好名子
    "finetuned_from": useModel,
    "tasks": "automatic-speech-recognition",
}
# ### 上傳


#儲存模型pipeline使用的資料
processor.save_pretrained("./" + "whisper-small-en")
#上傳模型
#trainer.push_to_hub(**kwargs)