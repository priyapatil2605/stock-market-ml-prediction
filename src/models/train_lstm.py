from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==================================================
# PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "lstm_ready"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# SETTINGS
# ==================================================

SEQUENCE_LENGTH = 20
INPUT_SIZE = 20
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_CLASSES = 3

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20


CLASS_NAMES = [
    "DOWN",
    "FLAT",
    "UP"
]


# ==================================================
# DEVICE
# ==================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==================================================
# LSTM MODEL
# ==================================================

class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.dropout = nn.Dropout(
            0.2
        )

        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )


    def forward(
        self,
        x
    ):

        lstm_out, (
            hidden,
            cell
        ) = self.lstm(x)

        last_output = lstm_out[
            :,
            -1,
            :
        ]

        output = self.dropout(
            last_output
        )

        output = self.fc(
            output
        )

        return output


# ==================================================
# EVALUATION FUNCTION
# ==================================================

def evaluate_model(
    model,
    data_loader,
    criterion
):

    model.eval()

    total_loss = 0

    all_predictions = []
    all_targets = []


    with torch.no_grad():

        for (
            X_batch,
            y_batch
        ) in data_loader:

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )


            outputs = model(
                X_batch
            )


            loss = criterion(
                outputs,
                y_batch
            )


            total_loss += (
                loss.item()
                * X_batch.size(0)
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            all_predictions.extend(
                predictions.cpu().numpy()
            )


            all_targets.extend(
                y_batch.cpu().numpy()
            )


    average_loss = (
        total_loss
        / len(data_loader.dataset)
    )


    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )


    f1 = f1_score(
        all_targets,
        all_predictions,
        average="weighted"
    )


    return (
        average_loss,
        accuracy,
        f1,
        all_predictions,
        all_targets
    )


# ==================================================
# MAIN
# ==================================================

def main():

    print(
        "Loading LSTM-ready data..."
    )


    # ----------------------------------------------
    # LOAD DATA
    # ----------------------------------------------

    X_train = np.load(
        DATA_DIR
        / "X_train_lstm.npy"
    )


    X_test = np.load(
        DATA_DIR
        / "X_test_lstm.npy"
    )


    y_train = np.load(
        DATA_DIR
        / "y_train_lstm.npy"
    )


    y_test = np.load(
        DATA_DIR
        / "y_test_lstm.npy"
    )


    print(
        "\nDataset shapes:"
    )

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_test:  {X_test.shape}"
    )

    print(
        f"y_train: {y_train.shape}"
    )

    print(
        f"y_test:  {y_test.shape}"
    )


    # ----------------------------------------------
    # CONVERT TO PYTORCH TENSORS
    # ----------------------------------------------

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )


    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )


    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long
    )


    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.long
    )


    # ----------------------------------------------
    # CREATE DATASETS
    # ----------------------------------------------

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )


    test_dataset = TensorDataset(
        X_test_tensor,
        y_test_tensor
    )


    # ----------------------------------------------
    # CREATE DATA LOADERS
    # ----------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    # ----------------------------------------------
    # CREATE MODEL
    # ----------------------------------------------

    model = LSTMClassifier(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES
    )


    model = model.to(
        DEVICE
    )


    # ----------------------------------------------
    # LOSS FUNCTION
    # ----------------------------------------------

    criterion = nn.CrossEntropyLoss()


    # ----------------------------------------------
    # OPTIMIZER
    # ----------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # ----------------------------------------------
    # DEVICE INFO
    # ----------------------------------------------

    print(
        "\n" + "=" * 55
    )

    print(
        "LSTM TRAINING"
    )

    print(
        "=" * 55
    )


    print(
        f"\nDevice: {DEVICE}"
    )


    print(
        f"Epochs: {EPOCHS}"
    )


    print(
        f"Batch size: {BATCH_SIZE}"
    )


    print(
        f"Hidden size: {HIDDEN_SIZE}"
    )


    # ----------------------------------------------
    # TRAINING
    # ----------------------------------------------

    best_accuracy = 0

    training_history = []

    start_time = time.time()


    for epoch in range(
        EPOCHS
    ):

        model.train()

        running_loss = 0


        for (
            X_batch,
            y_batch
        ) in train_loader:


            X_batch = X_batch.to(
                DEVICE
            )


            y_batch = y_batch.to(
                DEVICE
            )


            # Forward pass

            outputs = model(
                X_batch
            )


            loss = criterion(
                outputs,
                y_batch
            )


            # Backward pass

            optimizer.zero_grad()


            loss.backward()


            optimizer.step()


            running_loss += (
                loss.item()
                * X_batch.size(0)
            )


        # ------------------------------------------
        # TRAIN LOSS
        # ------------------------------------------

        train_loss = (
            running_loss
            / len(train_loader.dataset)
        )


        # ------------------------------------------
        # TEST EVALUATION
        # ------------------------------------------

        (
            test_loss,
            test_accuracy,
            test_f1,
            _,
            _
        ) = evaluate_model(
            model,
            test_loader,
            criterion
        )


        training_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(
                    train_loss
                ),
                "test_loss": float(
                    test_loss
                ),
                "test_accuracy": float(
                    test_accuracy
                ),
                "test_f1": float(
                    test_f1
                )
            }
        )


        print(
            f"\nEpoch "
            f"[{epoch + 1}/{EPOCHS}]"
        )


        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )


        print(
            f"Test Loss: "
            f"{test_loss:.4f}"
        )


        print(
            f"Test Accuracy: "
            f"{test_accuracy:.4f}"
        )


        print(
            f"Test F1: "
            f"{test_f1:.4f}"
        )


        # ------------------------------------------
        # SAVE BEST MODEL
        # ------------------------------------------

        if (
            test_accuracy
            > best_accuracy
        ):

            best_accuracy = (
                test_accuracy
            )


            torch.save(
                model.state_dict(),
                MODEL_DIR
                / "lstm_model.pt"
            )


            print(
                "Best model saved."
            )


    # ==================================================
    # FINAL EVALUATION
    # ==================================================

    print(
        "\n" + "=" * 55
    )

    print(
        "FINAL LSTM EVALUATION"
    )

    print(
        "=" * 55
    )


    # Load best model

    model.load_state_dict(
        torch.load(
            MODEL_DIR
            / "lstm_model.pt",
            map_location=DEVICE
        )
    )


    (
        final_loss,
        final_accuracy,
        final_f1,
        predictions,
        targets
    ) = evaluate_model(
        model,
        test_loader,
        criterion
    )


    print(
        f"\nFinal Accuracy: "
        f"{final_accuracy:.4f}"
    )


    print(
        f"Final Weighted F1-score: "
        f"{final_f1:.4f}"
    )


    # ----------------------------------------------
    # CLASSIFICATION REPORT
    # ----------------------------------------------

    print(
        "\nClassification Report:"
    )


    print(
        classification_report(
            targets,
            predictions,
            target_names=CLASS_NAMES,
            digits=4
        )
    )


    # ----------------------------------------------
    # CONFUSION MATRIX
    # ----------------------------------------------

    matrix = confusion_matrix(
        targets,
        predictions
    )


    print(
        "Confusion Matrix:"
    )


    print(
        matrix
    )


    # ----------------------------------------------
    # SAVE HISTORY
    # ----------------------------------------------

    with open(
        MODEL_DIR
        / "lstm_training_history.json",
        "w"
    ) as file:

        json.dump(
            training_history,
            file,
            indent=4
        )


    # ----------------------------------------------
    # TRAINING TIME
    # ----------------------------------------------

    training_time = (
        time.time()
        - start_time
    )


    print(
        "\nTraining Time:"
    )


    print(
        f"{training_time:.2f} seconds"
    )


    # ----------------------------------------------
    # COMPLETE
    # ----------------------------------------------

    print(
        "\n" + "=" * 55
    )

    print(
        "LSTM TRAINING COMPLETE"
    )

    print(
        "=" * 55
    )


    print(
        "\nBest model saved to:"
    )

    print(
        MODEL_DIR
        / "lstm_model.pt"
    )


    print(
        "\nTraining history saved to:"
    )

    print(
        MODEL_DIR
        / "lstm_training_history.json"
    )


if __name__ == "__main__":
    main()