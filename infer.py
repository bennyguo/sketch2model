import os
from options.infer_options import InferOptions
from data import create_dataset
from models import create_model
import torch


if __name__ == '__main__':
    opt = InferOptions().parse()

    dataset_infer = create_dataset(opt, mode='infer', shuffle=False)

    model = create_model(opt)
    current_epoch = model.setup(opt)

    out_dir = os.path.join(opt.results_dir, opt.name, 'infer_{}'.format(current_epoch), opt.data_name)
    print('creating out directory', out_dir)
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        model.inference(current_epoch, dataset_infer, save_dir=out_dir)

