import os
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import torch
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    opt = TrainOptions().parse()

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    dataset_train = create_dataset(opt, mode='train', shuffle=True)
    dataset_val = create_dataset(opt, mode='val', shuffle=False)
    dataset_test = create_dataset(opt, mode='test', shuffle=False)

    print(f'The number of training data = {len(dataset_train)}')
    print(f'The number of validation data = {len(dataset_val)}')
    print(f'The number of test data = {len(dataset_test)}')

    model = create_model(opt)
    writer = SummaryWriter(os.path.join(opt.summary_dir, opt.name))
    current_epoch = model.setup(opt)
    total_iters = current_epoch * len(dataset_train.dataloader)

    for epoch in range(current_epoch + 1, opt.n_epochs + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        print('Learning rate:', f"{model.get_learning_rate():.3e}")
        model.update_hyperparameters(epoch)
        for i, data in enumerate(dataset_train):
            iter_start_time = time.time()
            total_iters += 1
            epoch_iter += 1
            model.update_hyperparameters_step(total_iters)
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            model.train()
            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.vis_freq == 0:
                model.visualize_train(total_iters)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses('train')
                t_comp = time.time() - iter_start_time
                for loss_name, loss_val in losses.items():
                    writer.add_scalar(f"train_{loss_name}", loss_val, global_step=total_iters)
                print(f"Epoch {epoch} - Iteration {epoch_iter}/{len(dataset_train.dataloader)} (comp time {t_comp:.3f}, data time {t_data:.3f})")
                print("Training losses |", ' '.join([f"{k}: {v:.3e}" for k, v in losses.items()]))

            iter_data_time = time.time()
        
        if epoch % opt.val_epoch_freq == 0:
            model.eval()
            with torch.no_grad():
                model.validate(epoch, dataset_val, phase='val')
            val_losses = model.get_current_losses('val')
            for loss_name, loss_val in val_losses.items():
                writer.add_scalar(f"val_{loss_name}", loss_val, global_step=epoch)
            print("Validation losses |", ' '.join([f"{k}: {v:.3e}" for k, v in val_losses.items()]))

        if epoch % opt.test_epoch_freq == 0:
            model.eval()
            with torch.no_grad():
                model.validate(epoch, dataset_test, phase='test')
            val_losses = model.get_current_losses('val')
            for loss_name, loss_val in val_losses.items():
                writer.add_scalar(f"test_{loss_name}", loss_val, global_step=epoch)
            print("Test losses |", ' '.join([f"{k}: {v:.3e}" for k, v in val_losses.items()]))
        
        if epoch % opt.save_epoch_freq == 0:
            print('Saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(epoch)
            model.save_networks('latest')
    
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs, time.time() - epoch_start_time))

        model.update_learning_rate()

