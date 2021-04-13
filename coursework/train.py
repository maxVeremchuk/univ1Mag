from utils.data_processing import *
from model.predict import *
from tqdm import tqdm

args = {}
args['batch_size'] = 1

train, dev, test, max_fertility = prepare_data(args)

pbar = tqdm(enumerate(dev), total=len(dev), desc="evaluating", ncols=0)

model = create_model(
    dictionary, domain_dicrionary, slot_dictionary, max_fertility, args)

for _, data in pbar:
    predict(data, None, None, None, None, [])
    break




