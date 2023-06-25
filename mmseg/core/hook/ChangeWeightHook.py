from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class ChangeWeightHook(Hook):
    def __init__(self, changeiter=5000, toweight=0.1):
        self.changeIter = changeiter
        self.toWeight = toweight

    def after_train_iter(self, runner):
        if runner.iter+1==self.changeIter:
            contrastloss = runner.model.module.decode_head.loss_decode[1]
            contrastloss.loss_weight = self.toWeight
            # assert torch.isfinite(runner.outputs['loss']), \
            #     runner.logger.info('loss become infinite or NaN!')