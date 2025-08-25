FEATURES = ["Voltage_measured", "Current_measured", "Temperature_measured"]
TIME_COL = "Time"
SEQ_LEN = 300

# Training hyperparameters
BATCH_SIZE = 64
LR = 4e-4
EPOCHS = 100
PATIENCE = 10
EMBED_DIM = 256
NUM_LAYERS = 2

# Paths
ARTIFACTS_DIR = "artifacts"
SCALER_PATH = f"{ARTIFACTS_DIR}/scaler.pkl"
MODEL_PATH = f"{ARTIFACTS_DIR}/best_model.pth"
