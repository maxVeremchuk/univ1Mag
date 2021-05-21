from config import *
from utils.data_processing import *
from model.predict import *
from model.model import *
from model.loss import *
from model.predict import *
from model.evaluator import *
from tqdm import tqdm
import os
import pickle as pkl


def run_train_epoch(epoch, max_epoch, data, model):
    avg_fertility_loss = 0
    avg_gate_loss = 0
    avg_state_loss = 0

    avg_slot_nb_tokens = 0
    avg_state_nb_tokens = 0
    avg_gate_nb_tokens = 0

    pbar = tqdm(enumerate(data), total=len(data),
                desc="epoch {}/{}".format(epoch+1, max_epoch), ncols=0)

    for i, data in pbar:
        # print(data)
        out = model.forward(data)
        losses, nb_tokens = loss_compute(
            data, out['generated_y'], out['generated_fertility'], out['generated_gates'])

        avg_fertility_loss += losses['fertility_loss']
        avg_gate_loss += losses['gate_loss']
        avg_state_loss += losses['state_loss']

        avg_gate_nb_tokens += nb_tokens['gate']
        avg_slot_nb_tokens += nb_tokens['slot']
        avg_state_nb_tokens += nb_tokens['state']

        if (i+1) % args['reporting'] == 0:
            avg_fertility_loss /= avg_slot_nb_tokens
            avg_state_loss /= avg_state_nb_tokens
            avg_gate_loss /= avg_gate_nb_tokens
            print("Step {} gate loss {:.4f} fertility loss {:.4f} state loss {:.4f}".
                  format(i+1, avg_gate_loss, avg_fertility_loss, avg_state_loss))
            with open(args['path'] + '/train_log.csv', 'a') as f:
                f.write('{},{},{},{},{}\n'.format(epoch+1, i+1,
                                                  avg_gate_loss, avg_fertility_loss, avg_state_loss))
            avg_fertility_loss = 0
            avg_slot_nb_tokens = 0
            avg_state_loss = 0
            avg_state_nb_tokens = 0
            avg_gate_loss = 0
            avg_gate_nb_tokens = 0

    # epoch_fertility_loss /= epoch_slot_nb_tokens
    # epoch_state_loss /= epoch_state_nb_tokens
    # epoch_gate_loss /= epoch_gate_nb_tokens
    # joint_gate_acc, joint_fertility_acc, joint_acc_score, F1_score, turn_acc_score = 0, 0, 0, 0, 0


def run_val_epoch(epoch, data, model):
    epoch_fertility_loss = 0
    epoch_gate_loss = 0
    epoch_state_loss = 0

    epoch_slot_nb_tokens = 0
    epoch_state_nb_tokens = 0
    epoch_gate_nb_tokens = 0

    epoch_joint_fertility_matches = 0
    epoch_joint_gate_matches = 0
    total_samples = 0

    pbar = tqdm(enumerate(data), total=len(data),
                desc="epoch validation {}".format(epoch+1), ncols=0)
    predictions = {}
    for i, data in pbar:
        out = model.forward(data)
        losses, nb_tokens = loss_compute(
            data, out['generated_y'], out['generated_fertility'], out['generated_gates'], True)

        epoch_slot_nb_tokens += nb_tokens['slot']
        epoch_state_nb_tokens += nb_tokens['state']
        epoch_gate_nb_tokens += nb_tokens['gate']

        epoch_fertility_loss += losses['fertility_loss']
        epoch_state_loss += losses['state_loss']
        epoch_gate_loss += losses['gate_loss']
        print("epoch_gate_loss:   " + str(epoch_gate_loss))
        print("epoch_gate_nb_tokens:   " + str(epoch_gate_nb_tokens))
        # print(data)
        matches, predictions = get_matches(
            out, data, model, dictionary, domain_dicrionary, slot_dictionary, predictions)
        epoch_joint_fertility_matches += matches['joint_fertility']
        epoch_joint_gate_matches += matches['joint_gate']
        total_samples += len(data['turn_id'])

    epoch_fertility_loss /= epoch_slot_nb_tokens
    epoch_state_loss /= epoch_state_nb_tokens
    epoch_gate_loss /= epoch_gate_nb_tokens
    joint_gate_acc, joint_fertility_acc, joint_acc_score, F1_score, turn_acc_score = 0, 0, 0, 0, 0

    joint_fertility_acc = 1.0 * epoch_joint_fertility_matches / \
        (total_samples * len(all_slots))
    joint_gate_acc = 1.0 * epoch_joint_gate_matches / \
        (total_samples * len(all_slots))
    joint_acc_score, F1_score, turn_acc_score = -1, -1, -1
    joint_acc_score, F1_score, turn_acc_score = evaluator.evaluate_metrics(
        predictions)

    print("Epoch {} gate loss {:.4f} fertility loss {:.4f} state loss {:.4f} \n joint_gate acc {:.4f} joint_fertility acc {:.4f} joint acc {:.4f} f1 {:.4f} turn acc {:.4f}".
          format(epoch+1, epoch_gate_loss, epoch_fertility_loss, epoch_state_loss,
                 joint_gate_acc, joint_fertility_acc, joint_acc_score, F1_score, turn_acc_score))

    with open(args['path'] + '/val_log.csv', 'a') as f:
        f.write('{},{},{},{},{},{},{},{},{}\n'.
                format(epoch+1,
                       epoch_gate_loss, epoch_fertility_loss, epoch_state_loss,
                       joint_gate_acc, joint_fertility_acc,
                       joint_acc_score, F1_score, turn_acc_score))

    return (epoch_gate_loss + epoch_fertility_loss + epoch_state_loss)/3, (joint_gate_acc + joint_fertility_acc + joint_acc_score)/3, joint_acc_score


train, dev, test, max_fertility, dictionary, domain_dicrionary, slot_dictionary, all_slots = prepare_data(
    args)

save_data = {
    'train': train,
    'dev': dev,
    'test': test,
    'dictionary': dictionary,
    'domain_dicrionary': domain_dicrionary,
    'slot_dictionary': slot_dictionary,
    'all_slots': all_slots,
    'args': args
}
if not os.path.exists(args['path']):
    os.makedirs(args['path'])
pkl.dump(save_data, open(args['path'] + '/data.pkl', 'wb'))


#pbar = tqdm(enumerate(dev), total=len(dev), desc="evaluating", ncols=0)

#print("max_fertility" + str(max_fertility))

model = create_model(
    dictionary, domain_dicrionary, slot_dictionary, max_fertility, args)

fertility_criterion = nn.CrossEntropyLoss()
gate_criterion = nn.CrossEntropyLoss()
state_criterion = LabelSmoothing(size=len(
    dictionary.word2index), padding_idx=PAD_token_id, smoothing=0.1, run_softmax=False)
opt = NoamOpt(args['d_model'], 1, args['warmup'], torch.optim.Adam(
    model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
loss_compute = LossCompute(model, fertility_criterion,
                           state_criterion, gate_criterion, opt)
evaluator = Evaluator(all_slots)

min_dev_loss = float("Inf")
max_dev_acc = -float("Inf")
max_dev_slot_acc = -float("Inf")
waiting = 0

best_modelfile = args['path'] + '/model_best.pth.tar'

for epoch in range(args['epochs']):
    model.train()
    run_train_epoch(epoch, args['epochs'], train, model)

    model.eval()
    dev_loss, dev_acc, dev_joint_acc = run_val_epoch(epoch, dev, model)
    check = (dev_loss < min_dev_loss)
    if check:
        modelfile = args['path'] + '/model_epoch{}.pth.tar'.format(epoch+1)
        torch.save(model, modelfile)
        print('Dev loss changes from {} --> {}'.format(min_dev_loss, dev_loss))
        print('Dev acc changes from {} --> {}'.format(max_dev_acc, dev_acc))
        print('Dev slot acc changes from {} --> {}'.format(max_dev_slot_acc, dev_joint_acc))
        min_dev_loss = dev_loss
        max_dev_acc = dev_acc
        max_dev_slot_acc = dev_joint_acc
        if os.path.exists(best_modelfile):
            os.remove(best_modelfile)
        os.symlink(os.path.basename('model_epoch{}.pth.tar'.format(epoch+1)), best_modelfile)
    else:
        waiting += 1
    if waiting == args["patience"]:
        print("Early stop due to patience...")
        break


# for _, data in pbar:
#     prediction = predict(data, model, dictionary, domain_dicrionary, slot_dictionary, {})
#     print(prediction)
#     break
