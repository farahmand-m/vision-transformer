import argparse
import functools
import os
import pathlib
import pydoc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.utils.tensorboard as tb
import yaml
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import GpuInfo
from ignite.engine import Engine, Events
from ignite.engine import create_supervised_evaluator
from ignite.handlers import DiskSaver
from ignite.handlers.checkpoint import Checkpoint
from ignite.metrics import Accuracy, Loss
from ignite.metrics import RunningAverage
from torchvision.datasets import ImageFolder

from models import from_template
from preprocessing import transform

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='name of the target architecture')
    parser.add_argument('dataset', help='name of the dataset folder'),
    parser.add_argument('experiment', help='name of the current experiment'),
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--no-grad-clip', action='store_true')
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-epochs', type=int, default=90)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--num-classes', type=int, default=1000)
    parser.add_argument('--warmup-iters', type=int, default=10000)
    parser.add_argument('--logging-iters', type=int, default=50)
    parser.add_argument('--checkpoint-iters', type=int, default=1000)
    parser.add_argument('--checkpoints-keep', type=int, default=5)
    parser.add_argument('--checkpoints-dir', default='checkpoints')
    parser.add_argument('--experiments-dir', default='experiments')
    parser.add_argument('--data-dir', type=pathlib.Path, default='data')
    parser.add_argument('--logs-dir', type=pathlib.Path, default='logs')
    parser.add_argument('--train-split', default='train')
    parser.add_argument('--val-split', default='val')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu_only else 'cpu')

    logs_dir = pathlib.Path(args.logs_dir)
    data_dir = pathlib.Path(args.data_dir) / args.dataset
    checkpoints_dir = pathlib.Path(args.checkpoints_dir)
    experiments_dir = pathlib.Path(args.experiments_dir)

    for directory in (logs_dir, checkpoints_dir, experiments_dir):
        if not directory.exists():
            os.makedirs(directory)

    if data_dir.exists():
        train_set = ImageFolder(data_dir / args.train_split, transform(args.image_size, True))
        validation_set = ImageFolder(data_dir / args.val_split, transform(args.image_size, False))
    else:
        DatasetClass = pydoc.locate(f'torchvision.datasets.{args.dataset}')
        train_set = DatasetClass(train=True, transform=transform(args.image_size, True))
        validation_set = DatasetClass(train=False, transform=transform(args.image_size, False))

    train_loader = data.DataLoader(train_set, args.batch_size, shuffle=True)
    valid_loader = data.DataLoader(validation_set, args.batch_size, shuffle=False)

    model = from_template(args.name, image_size=args.image_size, num_classes=args.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)

    def train_step(engine, batch):
        optimizer.zero_grad()
        model.train()
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        if not args.no_grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if engine.state.iteration > args.warmup_iters:
            scheduler.step()
        return loss.item()

    trainer = Engine(train_step)

    GpuInfo().attach(trainer, name='gpu')
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    metric_names = ['loss'] if args.cpu_only else ['loss', 'gpu:0 mem(%)', 'gpu:0 util(%)']
    ProgressBar(persist=True).attach(trainer, metric_names=metric_names)

    writer = tb.SummaryWriter(log_dir=logs_dir)

    @trainer.on(Events.ITERATION_COMPLETED(every=args.logging_iters))
    def log_training_loss(engine):
        metrics = engine.state.metrics
        if not args.cpu_only:
            writer.add_scalar('gpu/mem', metrics['gpu:0 mem(%)'], engine.state.iteration)
            writer.add_scalar('gpu/util', metrics['gpu:0 util(%)'], engine.state.iteration)
        writer.add_scalar('training/loss', engine.state.output, engine.state.iteration)

    val_metrics = {'accuracy': Accuracy(), 'loss': Loss(criterion)}
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    def log_metrics(engine, loader, label):
        evaluator.run(loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        print(f'{label}/avg_loss: {avg_accuracy:.2f}\n{label}/avg_accuracy: {avg_loss:.2f}')
        writer.add_scalar(f'{label}/avg_loss', avg_loss, engine.state.epoch)
        writer.add_scalar(f'{label}/avg_accuracy', avg_accuracy, engine.state.epoch)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, functools.partial(log_metrics, loader=train_loader, label='train'))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, functools.partial(log_metrics, loader=valid_loader, label='validation'))

    state_dicts = {'model': model, 'optimizer': optimizer, 'scheduler': scheduler, 'trainer': trainer}
    save_handler = DiskSaver(args.checkpoints_dir, create_dir=True, require_empty=False)
    handler = Checkpoint(state_dicts, save_handler, filename_prefix=args.experiment, n_saved=args.checkpoints_keep)
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=args.checkpoint_iters), handler)

    @trainer.on(Events.ITERATION_COMPLETED(every=args.checkpoint_iters))
    def update_experiment(engine):
        with open(experiments_dir / f'{args.experiment}.yaml', 'w') as stream:
            yaml.dump({'last_checkpoint': engine.state.iteration}, stream)

    if args.cont:
        experiment_fp = experiments_dir / f'{args.experiment}.yaml'
        if os.path.exists(experiment_fp):
            with open(experiments_dir / f'{args.experiment}.yaml') as stream:
                di = yaml.full_load(stream)
            checkpoint_fp = checkpoints_dir / f'{args.experiment}_checkpoint_{di["last_checkpoint"]}.pt'
            if os.path.exists(checkpoint_fp):
                checkpoint = torch.load(checkpoint_fp)
                Checkpoint.load_objects(state_dicts, checkpoint=checkpoint)
                print('Continuing training from', checkpoint_fp)

    trainer.run(train_loader, max_epochs=args.num_epochs)
    writer.close()
