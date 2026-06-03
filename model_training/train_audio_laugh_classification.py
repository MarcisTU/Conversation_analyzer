import torch
import torch.nn as nn
import torchaudio
from datasets import load_dataset
import evaluate
from transformers import (
    AutoModel,
    TrainingArguments,
    Trainer,
)


class EATAudioClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("worstchan/EAT-base_epoch30_finetune_AS2M", trust_remote_code=True)

        hidden_dim = 768  # EAT-base hidden size
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_mels, labels=None):
        # input_mels shape: [Batch, 1, T, F]
        feats = self.backbone.extract_features(input_mels)

        # Global mean pooling over the time dimension (dim 1)
        pooled_feat = torch.mean(feats, dim=1)
        logits = self.classifier(pooled_feat)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


class EATAudioDataCollator:
    def __init__(self, target_length=1024):
        self.target_length = target_length
        self.norm_mean = -4.268
        self.norm_std = 4.569

    def __call__(self, batch):
        mels_list = []
        labels_list = []

        for item in batch:
            # 'audio' key contains {'array': np.array, 'sampling_rate': int} from AudioFolder
            audio_data = item['audio']['array']
            sr = item['audio']['sampling_rate']
            label = item['label']

            waveform = torch.tensor(audio_data).float()

            # Resample to 16kHz if necessary
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)

            # Normalize and convert to mel-spectrogram
            waveform = waveform - waveform.mean()
            mel = torchaudio.compliance.kaldi.fbank(
                waveform.unsqueeze(0),
                htk_compat=True,
                sample_frequency=16000,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10
            )

            # Pad or truncate time frames
            n_frames = mel.shape[0]
            if n_frames < self.target_length:
                pad_amount = self.target_length - n_frames
                mel = torch.nn.functional.pad(mel, (0, 0, 0, pad_amount), mode='constant', value=0)
            else:
                mel = mel[:self.target_length, :]

            # Apply EAT specific normalization
            mel = (mel - self.norm_mean) / (self.norm_std * 2)
            mel = mel.unsqueeze(0)  # [1, T, F]

            mels_list.append(mel)
            labels_list.append(label)

        # Stack into [Batch, 1, T, F] and [Batch] tensors
        return {
            "input_mels": torch.stack(mels_list),
            "labels": torch.tensor(labels_list, dtype=torch.long)
        }


if __name__ == "__main__":
    dataset = load_dataset("audiofolder", data_dir="./data/audio_laugh")
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    model = EATAudioClassifier(num_labels=2)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(pred):
        logits, labels = pred
        preds = logits.argmax(axis=1)

        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels, average="binary")["f1"],
        }

    data_collator = EATAudioDataCollator(target_length=1024)

    # Set up Hugging Face Training Arguments
    training_args = TrainingArguments(
        output_dir="./results/run_1_laugh_detector",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        eval_on_start=True,
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=10,
        save_total_limit=6,
        learning_rate=8e-5,
        num_train_epochs=100,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to=["tensorboard"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
