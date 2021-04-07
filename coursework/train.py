from utils.data_processing import *
from model.predict import *
from tqdm import tqdm

args = {}
args['batch_size'] = 1

train, dev, test = prepare_data(args)

pbar = tqdm(enumerate(dev), total=len(dev), desc="evaluating", ncols=0)

for _, data in pbar:
    predict(data, None, None, None, None, [])
    break



# model = create_model(
#     src_dict = src_dict,
#     domain_dict = domain_dict, slot_dict = slot_dict,
#     max_fertiltiy=max_fertiltiy)
