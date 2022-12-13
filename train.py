from models.GAN_model import GANModel
from utiles.utiles import create_loss_metrics, update_losses, visualize, log_results
from utiles.preprocessor import sample_test_dataset, make_dataloaders
import tqdm
import torch


def train_loop_function(model, train_dataset, val_dataset, epochs, display_every=1100):
    data_validation = next(iter(val_dataset))
    for e in range(epochs):
        loss_meter_dict = create_loss_metrics()
        iteration_in_epoch = 0
        for data in tqdm.tqdm(train_dataset):
            model.setup_input(data)
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0))
            iteration_in_epoch += 1
            if iteration_in_epoch % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {iteration_in_epoch}/{len(train_dataset)}")
                log_results(loss_meter_dict)
                visualize(model, data, save=False)

        # with torch.no_grad():
        #     model.eval()
        #     for data in tqdm.tqdm(data_validation):
        #         visualize(model, data, save=False)
        #         break


def main():
    train_paths, val_paths = sample_test_dataset("/home/oleh/.fastai/data/coco_sample", 3000)
    train_dataset = make_dataloaders(paths=train_paths, split='train')
    val_dataset = make_dataloaders(paths=val_paths, split='val')
    data = next(iter(train_dataset))
    Ls, abs_ = data['L'], data['ab']
    print(Ls.shape, abs_.shape)
    print(len(train_dataset), len(val_dataset))
    model = GANModel()
    train_loop_function(model, train_dataset, val_dataset, 300)


if __name__ == "__main__":
    main()
