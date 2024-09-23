import os.path as osp
from tqdm import tqdm

import mmcv
import mmengine

from mmengine import Config
from mmengine.runner import Runner


def main():
    filepath = ('myproject/myswintf2.py')
    cfg = Config.fromfile(filepath)
    cfg.work_dir = osp.join('./work_dirs',
                        filepath)

    # register all modules in mmseg into the registries
    # do not init the default scope here because it will be init in the runner
    #register_all_modules(init_default_scope=False)
    runner = Runner.from_cfg(cfg)

    runner.train()
    #runner.val()

if __name__ == '__main__':
    main()