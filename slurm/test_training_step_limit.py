"""Exercise the real epoch/step stop branches without loading models or Torch."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest


TRAIN = Path(__file__).resolve().parents[1]/'train.py'


class Progress:
    def __init__(self, loader, **kwargs):self.loader=loader
    def __iter__(self):return iter(self.loader)
    def close(self):pass


def simulate(steps_per_epoch, epochs, limit, start_epoch=1, global_step=0):
    tree=ast.parse(TRAIN.read_text())
    train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    outer=copy.deepcopy(next(n for n in train.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch'))
    inner=next(n for n in outer.body if isinstance(n,ast.For))
    # Keep actual budget branches, step accounting and epoch mean. Replace only
    # the model/optimizer work with constant loss, then record the post-loop
    # validation/checkpoint boundary reached by the original loop.
    stop=inner.body[-1]
    assert isinstance(stop,ast.If) and any(isinstance(n,ast.Attribute) and n.attr=='max_train_steps' for n in ast.walk(stop))
    step_update=next(n for n in inner.body if isinstance(n,ast.AugAssign) and getattr(n.target,'id','')=='global_step')
    loss_update=next(n for n in inner.body if isinstance(n,ast.AugAssign) and getattr(n.target,'id','')=='running_loss')
    inner.body=[loss_update,step_update,stop]
    prefix=[]
    for node in outer.body:
        if node is inner:break
        prefix.append(node)
    mean=next(n for n in outer.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='epoch_avg' for t in n.targets))
    record=ast.parse('observations.append((epoch,batch_idx,global_step,epoch_avg))').body[0]
    outer.body=prefix+[inner,mean,record]
    module=ast.fix_missing_locations(ast.Module(body=[outer],type_ignores=[]))
    state=dict(args=SimpleNamespace(train_epochs=epochs,max_train_steps=limit),
               start_epoch=start_epoch,end_epoch=epochs,global_step=global_step,
               accelerator=SimpleNamespace(is_local_main_process=False,print=lambda *args:None),
               tqdm=Progress,loader=range(steps_per_epoch),loss_value=2.0,observations=[])
    exec(compile(module,str(TRAIN),'exec'),state)
    return state


class TrainingStepLimitTests(unittest.TestCase):
    def test_three_gpu_baseline_and_swap_finish_at_250000_with_partial_epoch_validation(self):
        for steps,epochs,last_batches in [(16667,15,16662),(33334,8,16662)]:
            state=simulate(steps,epochs,250000)
            self.assertEqual(state['global_step'],250000)
            self.assertEqual(state['observations'][-1],(epochs,last_batches,250000,2.0))
            self.assertEqual(len(state['observations']),epochs)

    def test_resume_stops_at_same_global_budget_and_completed_run_does_no_updates(self):
        state=simulate(16667,15,250000,start_epoch=11,global_step=166670)
        self.assertEqual(state['global_step'],250000)
        self.assertEqual(len(state['observations']),5)
        state=simulate(16667,15,250000,start_epoch=15,global_step=250000)
        self.assertEqual(state['global_step'],250000)
        self.assertEqual(state['observations'],[])

    def test_no_limit_preserves_complete_epochs(self):
        state=simulate(10,3,None)
        self.assertEqual(state['global_step'],30)
        self.assertEqual(state['observations'][-1],(3,10,30,2.0))


if __name__=='__main__':unittest.main()
