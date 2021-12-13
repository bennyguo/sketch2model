import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import torch


if __name__ == '__main__':
    opt = TestOptions().parse()

    dataset_test = create_dataset(opt, mode=opt.test_split, shuffle=False)

    print(f'The number of test data = {len(dataset_test)}')

    model = create_model(opt)
    current_epoch = model.setup(opt)

    out_dir = os.path.join(opt.results_dir, opt.name, '{}-{}_{}'.format(opt.dataset_mode, opt.test_split, current_epoch))
    print('creating out directory', out_dir)
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        model.test(current_epoch, dataset_test, save_dir=out_dir)
    test_losses = model.get_current_losses('test')
    print("Test losses |", ' '.join([f"{k}: {v:.3e}" for k, v in test_losses.items()]))
