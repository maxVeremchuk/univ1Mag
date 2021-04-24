from config import *
from utils.data_processing import *
from model.predict import *
from model.model import *
from tqdm import tqdm

train, dev, test, max_fertility, dictionary, domain_dicrionary, slot_dictionary = prepare_data(args)

pbar = tqdm(enumerate(dev), total=len(dev), desc="evaluating", ncols=0)

model = create_model(
    dictionary, domain_dicrionary, slot_dictionary, max_fertility, args)

fertility_criterion = nn.CrossEntropyLoss()
gate_criterion = nn.CrossEntropyLoss()

# for _, data in pbar:
#     prediction = predict(data, model, dictionary, domain_dicrionary, slot_dictionary, {})
#     print(prediction)
#     break




