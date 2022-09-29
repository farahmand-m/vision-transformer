import argparse
import pathlib
import pydoc

import torch
import torch.nn as nn
import torch.utils.data as data
import yaml
from ignite.engine import create_supervised_evaluator
from ignite.handlers.checkpoint import Checkpoint
from ignite.metrics import Accuracy, Loss
from torchvision.datasets import ImageFolder

from models import from_template
from preprocessing import transform

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='name of the target architecture')
    parser.add_argument('dataset', help='name of the dataset folder'),
    parser.add_argument('experiment', help='name of the current experiment'),
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--num-classes', type=int, default=1000)
    parser.add_argument('--checkpoints-dir', default='checkpoints')
    parser.add_argument('--experiments-dir', default='experiments')
    parser.add_argument('--data-dir', type=pathlib.Path, default='data')
    parser.add_argument('--test-split', default='val')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu_only else 'cpu')

    data_dir = pathlib.Path(args.data_dir) / args.dataset
    checkpoints_dir = pathlib.Path(args.checkpoints_dir)
    experiments_dir = pathlib.Path(args.experiments_dir)

    if data_dir.exists():
        test_set = ImageFolder(data_dir / args.test_split, transform(args.image_size, False))
    else:
        DatasetClass = pydoc.locate(f'torchvision.datasets.{args.dataset}')
        test_set = DatasetClass(train=False, transform=transform(args.image_size, False))

    test_loader = data.DataLoader(test_set, args.batch_size, shuffle=False)

    model = from_template(args.name, image_size=args.image_size, num_classes=args.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    val_metrics = {'accuracy': Accuracy(), 'loss': Loss(criterion)}
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    state_dicts = {'model': model}

    experiment_fp = experiments_dir / f'{args.experiment}.yaml'
    with open(experiments_dir / f'{args.experiment}.yaml') as stream:
        di = yaml.full_load(stream)
    checkpoint_fp = checkpoints_dir / f'{args.experiment}_checkpoint_{di["last_checkpoint"]}.pt'
    checkpoint = torch.load(checkpoint_fp)
    Checkpoint.load_objects(state_dicts, checkpoint=checkpoint)
    print('Loading from', checkpoint_fp)

    evaluator.run(test_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_loss = metrics['loss']
    print(f'Loss: {avg_loss:.4f} Accuracy: {avg_accuracy:.4f}')
