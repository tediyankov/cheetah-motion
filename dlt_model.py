# import the get data loader function

from data.data_loader import get_data_loaders

# optional overrides for data laders 
# BATCH_SIZE = 32
# VAL_FRAC = 0.15
# TEST_FRAC = 0.2
# NUM_WORKERS = 4 #?
train_loader, val_loader, test_loader = get_data_loaders()
