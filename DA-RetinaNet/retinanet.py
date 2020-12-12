# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
import numpy as np
from typing import List
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn
from torch.nn import functional as F
from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..anchor_generator import build_anchor_generator
from ..backbone import build_backbone
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..postprocessing import detector_postprocess
from .build import META_ARCH_REGISTRY

__all__ = ["RetinaNet"]


def permute_to_N_HWA_K(tensor, K):
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


def permute_all_cls_and_box_to_N_HWA_K_and_concat(box_cls, box_delta, num_classes=80):
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]

    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    return box_cls, box_delta

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DiscriminatorRes3(nn.Module):

    def __init__(self):

        super(DiscriminatorRes3, self).__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = (1, 1) ,bias = False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size = (1, 1) ,bias = False),  
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(256, 1, kernel_size=(1, 1), bias = False)
        ).cuda()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x, domain_target = False, alpha = 1):
        x = GradReverse.apply(x, alpha)
        x = self.reducer(x) 
        x = torch.flatten(x,1)
        if domain_target:
            domain_t = torch.ones(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x,domain_t, alpha=0.25,gamma=2,reduction="mean")
        else:
            domain_s = torch.zeros(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x,domain_s, alpha=0.25,gamma=2,reduction="mean")
        return loss

class DiscriminatorRes4(nn.Module):

    def __init__(self):

        super(DiscriminatorRes4, self).__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size = (3, 3), stride = 2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Conv2d(512, 128, kernel_size = (3, 3), stride = 2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Conv2d(128, 128, kernel_size = (3, 3), stride = 2, bias=False), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.AdaptiveAvgPool2d((1, 1)),
        ).cuda()
        self.reducer2 = nn.Linear(128, 1, bias = False ).cuda()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x, domain_target = False, alpha = 1):
        x = GradReverse.apply(x, alpha)
        x = self.reducer(x) 
        x = torch.flatten(x,1)
        x = self.reducer2(x)
        if domain_target:
            domain_t = torch.ones(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x,domain_t,alpha=0.25,gamma=2,reduction="mean")
        else:
            domain_s = torch.zeros(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x,domain_s,alpha=0.25,gamma=2,reduction="mean")
        return loss
        
class DiscriminatorRes5(nn.Module):

    def __init__(self):
        super(DiscriminatorRes5, self).__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size = (3, 3), stride = 2, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Conv2d(1024, 256, kernel_size = (3, 3), stride = 2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Conv2d(256, 256, kernel_size = (3, 3), stride = 2, bias=False), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.AdaptiveAvgPool2d((1, 1)),
        ).cuda()
        self.reducer2 = nn.Sequential(
            nn.Linear(256, 128, bias = False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(128, 1, bias= False)
        ).cuda() 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x, domain_target = False, alpha = 1):
        x = GradReverse.apply(x, alpha)
        x = self.reducer(x) 
        x = torch.flatten(x,1)
        x = self.reducer2(x)
        if domain_target:
            domain_t = torch.ones(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x,domain_t,alpha=0.25,gamma=2,reduction="mean")
        else:
            domain_s = torch.zeros(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x,domain_s,alpha=0.25,gamma=2,reduction="mean")
        return loss


@META_ARCH_REGISTRY.register()
class RetinaNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        # fmt: off
        self.num_classes              = cfg.MODEL.RETINANET.NUM_CLASSES
        self.in_features              = cfg.MODEL.RETINANET.IN_FEATURES
        # Loss parameters:
        self.focal_loss_alpha         = cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma         = cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta      = cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA
        # Inference parameters:
        self.score_threshold          = cfg.MODEL.RETINANET.SCORE_THRESH_TEST
        self.topk_candidates          = cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST
        self.nms_threshold            = cfg.MODEL.RETINANET.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # Vis parameters
        self.vis_period               = cfg.VIS_PERIOD
        self.input_format             = cfg.INPUT.FORMAT
        # fmt: on

        # Discriminator
        self.discriminatorRes3 = DiscriminatorRes3()
        self.discriminatorRes4 = DiscriminatorRes4()
        self.discriminatorRes5 = DiscriminatorRes5()
        
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = RetinaNetHead(cfg, feature_shapes)
        self.anchor_generator = build_anchor_generator(cfg, feature_shapes)

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.matcher = Matcher(
            cfg.MODEL.RETINANET.IOU_THRESHOLDS,
            cfg.MODEL.RETINANET.IOU_LABELS,
            allow_low_quality_matches=True,
        )

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)


        self.loss_normalizer = 100  # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_momentum = 0.9

    def visualize_training(self, batched_inputs, results):

        from detectron2.utils.visualizer import Visualizer

        assert len(batched_inputs) == len(
            results
        ), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()
        max_boxes = 20

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"].cpu().numpy()
        assert img.shape[0] == 3, "Images should have 3 channels."
        if self.input_format == "BGR":
            img = img[::-1, :, :]
        img = img.transpose(1, 2, 0)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[image_index], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)

    def forward(self, batched_inputs, domain_target = False, alpha3 = 1, alpha4 = 1, alpha5 = 1):

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        features,res = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta = self.head(features)
        anchors = self.anchor_generator(features)
        if self.training:
            gt_classes, gt_anchors_reg_deltas = self.get_ground_truth(anchors, gt_instances)
            losses = self.losses(gt_classes, gt_anchors_reg_deltas, box_cls, box_delta, res["res3"], res["res4"], res["res5"], domain_target, alpha3, alpha4, alpha5)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(box_cls, box_delta, anchors, images.image_sizes)
                    self.visualize_training(batched_inputs, results)

            return losses
        else:
            results = self.inference(box_cls, box_delta, anchors, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(self, gt_classes, gt_anchors_deltas, pred_class_logits, pred_anchor_deltas, res3, res4, res5, domain_target, alpha3, alpha4, alpha5):

        if domain_target:
            loss_res3 = self.discriminatorRes3(res3, domain_target, alpha3)
            loss_res4 = self.discriminatorRes4(res4, domain_target, alpha4)
            loss_res5 = self.discriminatorRes5(res5, domain_target, alpha5)
            return {"loss_r3": loss_res3, "loss_r4": loss_res4, "loss_r5": loss_res5}
        else:
            loss_res3 = self.discriminatorRes3(res3, domain_target, alpha3)
            loss_res4 = self.discriminatorRes4(res4, domain_target, alpha4)
            loss_res5 = self.discriminatorRes5(res5, domain_target, alpha5)
        
        pred_class_logits, pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            pred_class_logits, pred_anchor_deltas, self.num_classes
        )  # Shapes: (N x R, K) and (N x R, 4), respectively.

        gt_classes = gt_classes.flatten()
        gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum().item()
        get_event_storage().put_scalar("num_foreground", num_foreground)
        self.loss_normalizer = (
            self.loss_normalizer_momentum * self.loss_normalizer
            + (1 - self.loss_normalizer_momentum) * num_foreground
        )

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1, self.loss_normalizer)

        # regression loss
        loss_box_reg = smooth_l1_loss(
            pred_anchor_deltas[foreground_idxs],
            gt_anchors_deltas[foreground_idxs],
            beta=self.smooth_l1_loss_beta,
            reduction="sum",
        ) / max(1, self.loss_normalizer)
        
        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg, "loss_r3": loss_res3, "loss_r4": loss_res4, "loss_r5": loss_res5}

    @torch.no_grad()
    def get_ground_truth(self, anchors, targets):

        gt_classes = []
        gt_anchors_deltas = []
        anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]
        # list[Tensor(R, 4)], one for each image

        for anchors_per_image, targets_per_image in zip(anchors, targets):
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, anchors_per_image)
            gt_matched_idxs, anchor_labels = self.matcher(match_quality_matrix)

            has_gt = len(targets_per_image) > 0
            if has_gt:
                # ground truth box regression
                matched_gt_boxes = targets_per_image.gt_boxes[gt_matched_idxs]
                gt_anchors_reg_deltas_i = self.box2box_transform.get_deltas(
                    anchors_per_image.tensor, matched_gt_boxes.tensor
                )

                gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_classes_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_classes_i[anchor_labels == -1] = -1
            else:
                gt_classes_i = torch.zeros_like(gt_matched_idxs) + self.num_classes
                gt_anchors_reg_deltas_i = torch.zeros_like(anchors_per_image.tensor)

            gt_classes.append(gt_classes_i)
            gt_anchors_deltas.append(gt_anchors_reg_deltas_i)

        return torch.stack(gt_classes), torch.stack(gt_anchors_deltas)

    def inference(self, box_cls, box_delta, anchors, image_sizes):

        assert len(anchors) == len(image_sizes)
        results = []

        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)

        for img_idx, anchors_per_image in enumerate(anchors):
            image_size = image_sizes[img_idx]
            box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
            box_reg_per_image = [box_reg_per_level[img_idx] for box_reg_per_level in box_delta]
            results_per_image = self.inference_single_image(
                box_cls_per_image, box_reg_per_image, anchors_per_image, tuple(image_size)
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta, anchors, image_size):

        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            # (HxWxAxK,)
            box_cls_i = box_cls_i.flatten().sigmoid_()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class RetinaNetHead(nn.Module):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        in_channels      = input_shape[0].channels
        num_classes      = cfg.MODEL.RETINANET.NUM_CLASSES
        num_convs        = cfg.MODEL.RETINANET.NUM_CONVS
        prior_prob       = cfg.MODEL.RETINANET.PRIOR_PROB
        num_anchors      = build_anchor_generator(cfg, input_shape).num_cell_anchors
        # fmt: on
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):

        logits = []
        bbox_reg = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg
