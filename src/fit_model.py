import torch
import torch.nn as nn
import re
import numpy as np
from tqdm.auto import tqdm
import mlflow
from time import time
import os
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
)

import matplotlib.pyplot as plt


def binary_metrics(outputs, labels, device):
    acc = BinaryAccuracy().to(device)
    prec = BinaryPrecision().to(device)
    rec = BinaryRecall().to(device)
    f1 = BinaryF1Score().to(device)

    preds = outputs.squeeze().float()
    labels = labels.squeeze().float()
    return (
        acc(preds, labels).item(),
        prec(preds, labels).item(),
        rec(preds, labels).item(),
        f1(preds, labels).item(),
    )


def fit_model(
    epochs: int,
    model: nn.Module,
    model_name: str,
    optimizer: torch.optim.Optimizer,
    criterion,
    train_loader,
    valid_loader,
    device,
    use_mlflow=False,
):

    log = dict()
    log["train_loss"] = []
    log["valid_loss"] = []
    log["train_accuracy"] = []
    log["valid_accuracy"] = []
    log["train_precision"] = []
    log["valid_precision"] = []
    log["train_recall"] = []
    log["valid_recall"] = []
    log["train_f1"] = []
    log["valid_f1"] = []

    time_start = time()

    start_epoch = len(log["train_loss"])

    ### Создаем папку для записи весов
    # -----------------------------------------------------------------
    # Создаём корневую папку weights, если её нет
    folder_path = f"weights/"
    model_folder_path = os.path.join(folder_path, f"{model_name}")

    os.makedirs(model_folder_path, exist_ok=True)

    # Список номеров run_*
    run_nums = []

    # Ищем все подпапки с именем run_число
    for item_name in os.listdir(model_folder_path):
        full_path = os.path.join(model_folder_path, item_name)
        if os.path.isdir(full_path):
            match = re.search(r"run_(\d+)", item_name)
            if match:
                run_nums.append(int(match.group(1)))

    # Определяем следующий номер
    run = max(run_nums) + 1 if run_nums else 1

    # Создаём новую папку
    new_folder = os.path.join(model_folder_path, f"run_{run}")
    os.makedirs(new_folder, exist_ok=True)
    # -----------------------------------------------------------------

    ### Цикл обучения
    # -----------------------------------------------------------------
    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):

        curr_run_path = os.path.join(folder_path, model_name, f"run_{run}")

        epoch_time_start = time()

        print(f'{"-"*13} Epoch {epoch} {"-"*13}')

        ### Обучение

        batch_acc = []
        batch_prec = []
        batch_recall = []
        batch_loss = []
        batch_f1 = []

        model.train()

        # Прогресс бар

        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=True
        )

        for inputs, labels in train_pbar:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Функции потерь

            outputs, _ = model(inputs)
            # outputs = model(inputs).squeeze()

            loss = criterion(outputs, labels.float())
            batch_loss.append(loss.item())

            # Метрики
            acc, prec, rec, f1 = binary_metrics(outputs, labels, device=device)

            batch_acc.append(acc)
            batch_prec.append(prec)
            batch_recall.append(rec)
            batch_f1.append(f1)

            loss.backward()
            optimizer.step()

        train_pbar.set_postfix(
            {
                "Loss": loss,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-score": f1,
            }
        )

        log["train_loss"].append(np.mean(batch_loss))
        log["train_accuracy"].append(np.mean(batch_acc))
        log["train_precision"].append(np.mean(batch_prec))
        log["train_recall"].append(np.mean(batch_recall))
        log["train_f1"].append(np.mean(batch_f1))

        ### Валидация

        batch_acc = []
        batch_prec = []
        batch_recall = []
        batch_loss = []
        batch_f1 = []

        model.eval()

        valid_pbar = tqdm(
            valid_loader, desc=f"Epoch {epoch}/{epochs} [Test]", leave=True
        )
        for inputs, labels in valid_pbar:

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs, _ = model(inputs)
                # outputs = model(inputs).squeeze()

            loss = criterion(outputs, labels.float())
            batch_loss.append(loss.item())

            # Метрики
            acc, prec, rec, f1 = binary_metrics(outputs, labels, device=device)

            batch_acc.append(acc)
            batch_prec.append(prec)
            batch_recall.append(rec)
            batch_f1.append(f1)

        valid_pbar.set_postfix(
            {
                "Loss": loss,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-score": f1,
            }
        )
        ### Метрики и логирование

        log["valid_loss"].append(np.mean(batch_loss))
        log["valid_accuracy"].append(np.mean(batch_acc))
        log["valid_precision"].append(np.mean(batch_prec))
        log["valid_recall"].append(np.mean(batch_recall))
        log["valid_f1"].append(np.mean(batch_f1))

        # [MLflow] Логируем метрики
        if use_mlflow:
            # epoch – номер шага (можно указывать step=epoch)
            for c in log.keys():
                mlflow.log_metric(c, log[c][-1], step=epoch)

        epoch_time = time() - epoch_time_start

        ### Выводим результаты эпохи
        # Train stage
        print(
            f"Train stage: "
            f"loss: {log['train_loss'][-1]:>6.3f}  "
            f"Accuracy: {log['train_accuracy'][-1]:>6.3f}  "
            f"Precision: {log['train_precision'][-1]:>6.3f}  "
            f"Recall: {log['train_recall'][-1]:>6.3f}  "
            f"F1-score: {log['train_f1'][-1]:>6.3f}  "
        )

        # Valid stage
        print(
            f"Valid stage: "
            f"loss: {log['valid_loss'][-1]:>6.3f}  "
            f"Accuracy: {log['valid_accuracy'][-1]:>6.3f}  "
            f"Precision: {log['valid_precision'][-1]:>6.3f}  "
            f"Recall: {log['valid_recall'][-1]:>6.3f}  "
            f"F1-score: {log['valid_f1'][-1]:>6.3f}  "
        )
        print(f"Time: {epoch_time}")

        print(f'{"-"*35}\n')
        torch.save(
            model.state_dict(), os.path.join(curr_run_path, f"weight_epoch_{epoch}.pth")
        )

    total_training_time = time() - time_start
    print(f"Total time = {total_training_time:>5.1f} сек")
    # -----------------------------------------------------------------

    return log, total_training_time, run


def fit_with_mlflow(
    model,
    model_name,
    epochs,
    optimizer,
    criterion,
    train_loader,
    valid_loader,
    device,
    batch_size,
    lr,
):
    mlflow.set_experiment(
        f"{model_name} experiment"
    )  # установить (или создать) эксперимент
    with mlflow.start_run(run_name=f"{model_name}_BS = {batch_size}_lr_{lr}"):
        # Логируем гиперпараметры из config
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("device", device)
        mlflow.log_param("optimizer", optimizer)
        mlflow.log_param("criterion", criterion)

        # mlflow.pytorch.autolog(
        #     checkpoint=True,
        #     checkpoint_save_best_only=False,
        #     checkpoint_save_weights_only=False,
        #     checkpoint_save_freq="epoch",
        # )
        # mlflow.log_param("augmentation", ("Yes" if augmentation else "No"))
        print("начало обучения...")
        # Запускаем обучение
        logs, tot_time, run = fit_model(
            model=model,
            model_name=model_name,
            epochs=epochs,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=device,
            use_mlflow=True,
        )
        mlflow.log_param("Total time", tot_time)

        # Сохраняем модель в MLflow (опционально)
        # mlflow.pytorch.log_model(base_cnn, "model")

    # После выхода из `with` Run автоматически завершается
    return logs, tot_time, run


def plot_history(history, grid=True, suptitle="model 1"):
    fig, ax = plt.subplots(3, 2, figsize=(16, 20))
    fig.suptitle(suptitle, fontsize=24, fontweight="bold", y=0.85)
    ax[0][0].plot(history["train_loss"], label="train loss")
    ax[0][0].plot(history["valid_loss"], label="valid loss")
    ax[0][0].set_title(f'Loss on epoch {len(history["train_loss"])}', fontsize=16)
    ax[0][0].grid(grid)
    ax[0][0].set_ylim((0, max(history["train_loss"] + history["valid_loss"]) + 0.1))
    ax[0][0].legend(fontsize=14)
    ax[0][0].set_xlabel("Epoch", fontsize=14)
    ax[0][0].set_ylabel("Loss", fontsize=14)

    ax[0][1].plot(history["train_accuracy"], label="train accuracy")
    ax[0][1].plot(history["valid_accuracy"], label="valid accuracy")
    ax[0][1].set_title(
        f'Accuracy on epoch {len(history["train_loss"])}',
        fontsize=16,
        fontweight="bold",
    )
    ax[0][1].grid(grid)
    # ax[0][1].set_ylim((min(history["train_accuracy"]) - 0.05, 1))
    ax[0][1].set_ylim(0.5, 1)
    ax[0][1].legend(fontsize=14)
    ax[0][1].set_xlabel("Epoch", fontsize=14)
    ax[0][1].set_ylabel("Accuracy", fontsize=14)

    ax[1][0].plot(history["train_precision"], label="train precision")
    ax[1][0].plot(history["valid_precision"], label="valid precision")
    ax[1][0].set_title(
        f'Precision on epoch {len(history["train_loss"])}',
        fontsize=16,
        fontweight="bold",
    )
    ax[1][0].grid(grid)
    ax[1][0].set_ylim(0.5, 1)
    # ax[1][0].set_ylim(min(history["train_precision"]) - 0.05, 1)
    ax[1][0].legend(fontsize=14)
    ax[1][0].set_xlabel("Epoch", fontsize=14)
    ax[1][0].set_ylabel("Precision", fontsize=14)

    ax[1][1].plot(history["train_recall"], label="train recall")
    ax[1][1].plot(history["valid_recall"], label="valid recall")
    ax[1][1].set_title(
        f'Recal on epoch {len(history["train_loss"])}', fontsize=16, fontweight="bold"
    )
    ax[1][1].grid(grid)
    ax[1][1].set_ylim(0.5, 1)
    # ax[1][1].set_ylim((min(history["train_recall"]) - 0.05, 1))
    ax[1][1].legend(fontsize=14)
    ax[1][1].set_xlabel("Epoch", fontsize=14)
    ax[1][1].set_ylabel("Recal", fontsize=14)

    ax[2][0].plot(history["train_f1"], label="train f1")
    ax[2][0].plot(history["valid_f1"], label="valid f1")
    ax[2][0].set_title(
        f'F1-score on epoch {len(history["train_loss"])}',
        fontsize=16,
        fontweight="bold",
    )
    ax[2][0].grid(grid)
    ax[2][0].set_ylim(0.5, 1)
    # ax[2][0].set_ylim((min(history["train_f1"]) - 0.05, 1))
    ax[2][0].legend(fontsize=14)
    ax[2][0].set_xlabel("Epoch", fontsize=14)
    ax[2][0].set_ylabel("F1", fontsize=14)

    ax[2][1].remove()
    plt.subplots_adjust(top=0.8)
    # plt.tight_layout(rect=[0, 0, 1, 0.8])
    plt.show()
    return fig
