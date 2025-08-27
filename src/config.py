FEATURES = ["Voltage_measured", "Current_measured", "Temperature_measured"]
TIME_COL = "Time"
SEQ_LEN = 10

# Training hyperparameters
BATCH_SIZE = 1
LR = 4e-4
EPOCHS = 1
PATIENCE = 10
EMBED_DIM = 256
NUM_LAYERS = 2

# Paths
ARTIFACTS_DIR = "artifacts"
SCALER_PATH = f"{ARTIFACTS_DIR}/scaler.pkl"
MODEL_PATH = f"{ARTIFACTS_DIR}/best_model.pth"
