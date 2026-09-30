"""CPU-only validation of the immutable epoch-14 restart checkpoint."""
import argparse
from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    args = ap.parse_args()
    path = Path(args.checkpoint)
    if not path.is_file():
        raise SystemExit('Checkpoint missing: ' + str(path))
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict):
        raise SystemExit('Expected a full MorphFlow checkpoint dictionary')
    epoch, step = int(ckpt.get('epoch', -1)), int(ckpt.get('step', -1))
    if (epoch, step) != (14, 70000):
        raise SystemExit('Wrong checkpoint: epoch=%d step=%d; require epoch=14 step=70000' % (epoch, step))
    if ckpt.get('evaluation_only', False):
        raise SystemExit('An evaluation-only snapshot cannot be resumed')
    for field in ('model', 'optimizer', 'scheduler'):
        if ckpt.get(field) is None:
            raise SystemExit('Missing ' + field + ' state')
    config = ckpt.get('args', {})
    if not isinstance(config, dict):
        config = vars(config)
    flow = ckpt.get('flow_target', config.get('flow_target'))
    if flow != 'ss' or config.get('ss_flow_arch', 'standard') != 'standard':
        raise SystemExit('Require a standard SS checkpoint')
    if int(config.get('use_lora', 0)) != 1:
        raise SystemExit('This experiment expects the existing LoRA run')
    print('CHECKPOINT OK:', path)
    print('epoch:', epoch, '| global_step:', step, '| val_loss:', ckpt.get('val_loss'))
    print('model tensors:', len(ckpt['model']))
    for index, group in enumerate(ckpt['optimizer'].get('param_groups', [])):
        print('restored group', group.get('name', index), 'lr=', group['lr'])
    print('Resume will start at epoch 15; --train_epochs 25 means 11 remaining epochs.')


if __name__ == '__main__':
    main()
