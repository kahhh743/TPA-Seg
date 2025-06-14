import os.path as osp

from mmengine import Config
from mmengine.runner import Runner


def main():
    filepath = ('myproject/TPA-Seg.py')
    cfg = Config.fromfile(filepath)
    cfg.work_dir = osp.join('./work_dirs',
                        filepath)
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()