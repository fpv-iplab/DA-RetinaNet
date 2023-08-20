import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
import logging
import os
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.modeling import build_model
from detectron2.engine import default_writers
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
import torch
from detectron2.data.datasets import register_pascal_voc

#training set
register_pascal_voc("city_trainS", "cityscape/VOC2007/", "train_s", 2007, ['car','person','rider','truck','bus','train','motorcycle','bicycle'])
register_pascal_voc("city_trainT", "cityscape/VOC2007/", "train_t", 2007, ['car','person','rider','truck','bus','train','motorcycle','bicycle'])

#test set
register_pascal_voc("city_testT", "cityscape/VOC2007/", "test_t", 2007, ['car','person','rider','truck','bus','train','motorcycle','bicycle'])

logger = logging.getLogger("detectron2")

def do_train(cfg_source, cfg_target, model, resume = False):
    print(model)
    model.train()
    optimizer = build_optimizer(cfg_source, model)
    scheduler = build_lr_scheduler(cfg_source, optimizer)
    checkpointer = DetectionCheckpointer(model, cfg_source.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)

    start_iter = (checkpointer.resume_or_load(cfg_source.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)
    max_iter = cfg_source.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg_source.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)
    writers = default_writers(cfg_source.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    alpha3 = 0
    alpha4 = 0
    alpha5 = 0
    
    i = 1
    max_epoch = 16.86 # max iter / min(data_len(data_soruce,data_target))
    current_epoch = 0
    data_len = 2965
    
    data_loader_source = build_detection_train_loader(cfg_source)
    data_loader_target = build_detection_train_loader(cfg_target)
    logger.info("Starting training from iteration {}".format(start_iter))

    with EventStorage(start_iter) as storage:
        for data_source,data_target, iteration in zip(data_loader_source, data_loader_target, range(start_iter, max_iter)):
            storage.iter = iteration

            iteration = iteration + 1
            if (iteration % data_len) == 0:
                current_epoch += 1
                i = 1

            p = float( i + current_epoch * data_len) / max_epoch / data_len
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            i+=1

            alpha3 = alpha
            alpha4 = alpha
            alpha5 = alpha

            if alpha3 > 0.5:
                alpha3 = 0.5

            if alpha4 > 0.5:
                alpha4 = 0.5

            if alpha5 > 0.3:
                alpha5 = 0.3
            
            loss_dict = model(data_source, False, alpha3, alpha4, alpha5)
            loss_dict_target = model(data_target, True, alpha3, alpha4, alpha5)
            loss_dict["loss_r3"] += loss_dict_target["loss_r3"]
            loss_dict["loss_r4"] += loss_dict_target["loss_r4"]
            loss_dict["loss_r5"] += loss_dict_target["loss_r5"]

            loss_dict["loss_r3"] *= 0.5
            loss_dict["loss_r4"] *= 0.5
            loss_dict["loss_r5"] *= 0.5

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)
                
cfg_source = get_cfg()
cfg_source.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
cfg_source.DATASETS.TRAIN = ("city_trainS",)
cfg_source.DATALOADER.NUM_WORKERS = 0
cfg_source.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")
cfg_source.SOLVER.IMS_PER_BATCH = 2
cfg_source.SOLVER.BASE_LR = 0.001
cfg_source.SOLVER.MAX_ITER = 70000
cfg_source.SOLVER.STEPS = (50000,)
cfg_source.INPUT.MIN_SIZE_TRAIN = (600,)
cfg_source.INPUT.MIN_SIZE_TEST = 0
os.makedirs(cfg_source.OUTPUT_DIR, exist_ok=True)
cfg_source.MODEL.RETINANET.NUM_CLASSES = 8
model = build_model(cfg_source)

cfg_target = get_cfg()
cfg_target.DATASETS.TRAIN = ("city_trainT",)
cfg_target.INPUT.MIN_SIZE_TRAIN = (600,)
cfg_target.DATALOADER.NUM_WORKERS = 0
cfg_target.SOLVER.IMS_PER_BATCH = 2

do_train(cfg_source,cfg_target,model,False)

from detectron2.evaluation import inference_on_dataset, PascalVOCDetectionEvaluator
evaluator = PascalVOCDetectionEvaluator("city_testT")
val_loader = build_detection_test_loader(cfg_source, "city_testT")
res = inference_on_dataset(model, val_loader, evaluator)
print(res)
