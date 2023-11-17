# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel, is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


# class ComputeLoss:
#     sort_obj_iou = False

#     # Compute losses
#     def __init__(self, model, autobalance=False):
#         device = next(model.parameters()).device  # get model device
#         h = model.hyp  # hyperparameters

#         # Define criteria
#         BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
#         BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

#         # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
#         self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

#         # Focal loss
#         g = h['fl_gamma']  # focal loss gamma
#         if g > 0:
#             BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

#         m = de_parallel(model).model[-1]  # Detect() module
#         self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
#         self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
#         self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
#         self.na = m.na  # number of anchors
#         self.nc = m.nc  # number of classes
#         self.nl = m.nl  # number of layers
#         self.anchors = m.anchors
#         self.device = device

#     def __call__(self, p, targets):  # predictions, targets
#         lcls = torch.zeros(1, device=self.device)  # class loss
#         lbox = torch.zeros(1, device=self.device)  # box loss
#         lobj = torch.zeros(1, device=self.device)  # object loss
#         tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

#         # Losses
#         for i, pi in enumerate(p):  # layer index, layer predictions
#             b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
#             tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

#             n = b.shape[0]  # number of targets
#             if n:
#                 # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
#                 pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

#                 # Regression
#                 pxy = pxy.sigmoid() * 2 - 0.5
#                 pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
#                 pbox = torch.cat((pxy, pwh), 1)  # predicted box
#                 iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
#                 lbox += (1.0 - iou).mean()  # iou loss

#                 # Objectness
#                 iou = iou.detach().clamp(0).type(tobj.dtype)
#                 if self.sort_obj_iou:
#                     j = iou.argsort()
#                     b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
#                 if self.gr < 1:
#                     iou = (1.0 - self.gr) + self.gr * iou
#                 tobj[b, a, gj, gi] = iou  # iou ratio

#                 # Classification
#                 if self.nc > 1:  # cls loss (only if multiple classes)
#                     t = torch.full_like(pcls, self.cn, device=self.device)  # targets
#                     t[range(n), tcls[i]] = self.cp
#                     lcls += self.BCEcls(pcls, t)  # BCE

#                 # Append targets to text file
#                 # with open('targets.txt', 'a') as file:
#                 #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

#             obji = self.BCEobj(pi[..., 4], tobj)
#             lobj += obji * self.balance[i]  # obj loss
#             if self.autobalance:
#                 self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

#         if self.autobalance:
#             self.balance = [x / self.balance[self.ssi] for x in self.balance]
#         lbox *= self.hyp['box']
#         lobj *= self.hyp['obj']
#         lcls *= self.hyp['cls']
#         bs = tobj.shape[0]  # batch size

#         return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

#     def build_targets(self, p, targets):
#         # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
#         na, nt = self.na, targets.shape[0]  # number of anchors, targets
#         tcls, tbox, indices, anch = [], [], [], []
#         gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
#         ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
#         targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

#         g = 0.5  # bias
#         off = torch.tensor(
#             [
#                 [0, 0],
#                 [1, 0],
#                 [0, 1],
#                 [-1, 0],
#                 [0, -1],  # j,k,l,m
#                 # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
#             ],
#             device=self.device).float() * g  # offsets

#         for i in range(self.nl):
#             anchors, shape = self.anchors[i], p[i].shape
#             gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

#             # Match targets to anchors
#             t = targets * gain  # shape(3,n,7)
#             if nt:
#                 # Matches
#                 r = t[..., 4:6] / anchors[:, None]  # wh ratio
#                 j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
#                 # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
#                 t = t[j]  # filter

#                 # Offsets
#                 gxy = t[:, 2:4]  # grid xy
#                 gxi = gain[[2, 3]] - gxy  # inverse
#                 j, k = ((gxy % 1 < g) & (gxy > 1)).T
#                 l, m = ((gxi % 1 < g) & (gxi > 1)).T
#                 j = torch.stack((torch.ones_like(j), j, k, l, m))
#                 t = t.repeat((5, 1, 1))[j]
#                 offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
#             else:
#                 t = targets[0]
#                 offsets = 0

#             # Define
#             bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
#             a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
#             gij = (gxy - offsets).long()
#             gi, gj = gij.T  # grid indices

#             # Append
#             indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
#             tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
#             anch.append(anchors[a])  # anchors
#             tcls.append(c)  # class

#         return tcls, tbox, indices, anch

class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        #self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.1, .05])  # P3-P7
        #self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.5, 0.4, .1])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        
        lcls_aux, lbox_aux, lobj_aux = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        # for i, pi in enumerate(p):  # layer index, layer predictions
        #     b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        #     tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        #     n = b.shape[0]  # number of targets
        #     if n:
        #         ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

        #         # Regression
        #         pxy = ps[:, :2].sigmoid() * 2. - 0.5
        #         pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
        #         pbox = torch.cat((pxy, pwh), 1)  # predicted box
        #         iou = bbox_iou(pbox.T, x1y1x2y2=False, tbox[i], CIoU=True)  # iou(prediction, target)
        #         lbox += (1.0 - iou).mean()  # iou loss

        #         # Objectness
        #         tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

        #         # Classification
        #         if self.nc > 1:  # cls loss (only if multiple classes)
        #             t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
        #             t[range(n), tcls[i]] = self.cp
        #             #t[t==self.cp] = iou.detach().clamp(0).type(t.dtype)
        #             lcls += self.BCEcls(ps[:, 5:], t)  # BCE

        #         # Append targets to text file
        #         # with open('targets.txt', 'a') as file:
        #         #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        #     obji = self.BCEobj(pi[..., 4], tobj)
        #     lobj += obji * self.balance[i]  # obj loss
        #     if self.autobalance:
        #         self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        
        # Losses
        # for i, pi in enumerate(p):  # layer index, layer predictions
        train_flag = len(p) == self.nl*2
        for i in range(self.nl):
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            # print(self.training)
            if train_flag:
                pi = p[self.nl + i]
                p_aux = p[i]
            else:
                pi = p[i]
            
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            
            if train_flag:
                tobj_aux = torch.zeros_like(p_aux[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
                
                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                
                if train_flag:
                    ps_aux = p_aux[b, a, gj, gi]
                    pxy_aux = ps_aux[:, :2].sigmoid() * 2. - 0.5
                    pwh_aux = (ps_aux[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                    pbox_aux = torch.cat((pxy_aux, pwh_aux), 1)  # predicted box
                    iou_aux = bbox_iou(pbox_aux.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                    lbox_aux += (1.0 - iou_aux).mean()  # iou loss
                
                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                
                if train_flag:
                    tobj_aux[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou_aux.detach().clamp(0).type(tobj.dtype)  # iou ratio
                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    #t[t==self.cp] = iou.detach().clamp(0).type(t.dtype)
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE
                    
                    if train_flag:
                        lcls_aux += self.BCEcls(ps_aux[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            
            if train_flag:
                obji_aux = self.BCEobj(p_aux[..., 4], tobj_aux)
                lobj_aux += obji_aux * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        
        if train_flag:
            lbox_aux *= self.hyp['box']
            lobj_aux *= self.hyp['obj']
            lcls_aux *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + (lbox_aux + lobj_aux + lcls_aux) * self.hyp['aux']
        return loss * bs, torch.cat((lbox, lobj, lbox_aux)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

class ComputeFeatLoss:

    # Compute losses
    def __init__(self, model, symmetrization=False, global_feat=False):
        """
        Parameters:
        +---------------------------------------------------------------+
        model: I feel this parameter is no use at all...
            But keep consistent with ComputeLoss, i keep this parameter.

        symmetrization: Super Resolution for image or target(object).
       +----------------------------------------------------------------+
        """
        self.device = next(model.parameters()).device
        m = de_parallel(model)
        # self.global_feat = global_feat

        self.projs = m.projs
        # self.preds = m.preds

        # Define criteria
        # self.loss_fn = nn.CosineSimilarity(dim=1)
        # self.pool = nn.AdaptiveAvgPool2d(1)
        # self.loss_fn = nn.SmoothL1Loss()

        m = de_parallel(model).model[-1]  # Detect() module
        self.hyp = model.hyp # hyperparameters
        # self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        # self.balance = [4., 2., 1.]
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers

    def __call__(self, feature_hr, feature_lr, pred_hr, pred_lr, targets):  # predictions, target
        # targets is a constant in the training processing, NOT update the parameters of the model.
        loss_feat_total = torch.tensor(0., device=self.device)
        indexes_hr = self.build_targets(pred_hr, targets)
        indexes_lr = self.build_targets(pred_lr, targets)
        for i in range(self.nl):
            feature_hr_p = feature_hr[i].moveaxis(1, -1)
            feature_hr_p = feature_hr_p[indexes_hr[i]]

            feature_lr_p = feature_lr[i].moveaxis(1, -1)
            feature_lr_p = feature_lr_p[indexes_lr[i]]

            # lr and hr have same point_num
            if feature_lr_p.shape[0] == feature_hr_p.shape[0]:
                feature_hr_p = self.projs[i](feature_hr_p)
                # feature_hr2lr = self.preds[i](feature_hr_p)

                feature_lr_p = self.projs[i](feature_lr_p)
                # feature_lr2hr = self.preds[i](feature_lr_p)

                # loss_feat_total -= self.loss_fn(feature_lr2hr, feature_hr_p.detach()).mean()
                # loss_feat_total -= self.loss_fn(feature_hr2lr, feature_lr_p.detach()).mean()
                loss_feat_total += vicreg_loss(feature_lr_p, feature_hr_p, self.hyp)
        return loss_feat_total * 0.5

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = 1, targets.shape[0]  # number of anchors, targets

        indices = []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices
        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            shape = p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                target_dim = t.shape[-1]
                t = t.reshape(-1, target_dim)

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                # j, k = ((gxy % 1 < g) & (gxy > 1)).T
                j, k = (gxy % 1 < g).T

                # l, m = ((gxi % 1 < g) & (gxi > 1)).T
                l, m = (gxi % 1 < g).T

                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid

        return indices
    
def vicreg_loss(pred, target, hyp):
    """
    pred: [N, feat_dims]
    target: [N, feat_dims]
    """
    batch_size, feat_dims = pred.shape
    repr_loss = F.mse_loss(pred, target)
    
    pred = pred - pred.mean(dim=0)
    target = target - target.mean(dim=0)
    
    std_pred = torch.sqrt(pred.var(dim=0)+0.0001)
    std_target = torch.sqrt(target.var(dim=0)+0.0001)
    std_loss = torch.mean(F.relu(1-std_pred))/2 + torch.mean(F.relu(1-std_target))/2
    
    cov_pred = (pred.T @ pred) / (batch_size - 1)
    cov_target = (target.T @ target) / (batch_size - 1)
    cov_loss = off_diagonal(cov_pred).pow_(2).sum().div_(
        feat_dims
    ) + off_diagonal(cov_target).pow_(2).sum().div_(feat_dims) 
    
    loss = (
        hyp['sim'] * repr_loss
        + hyp['std'] * std_loss
        + hyp['cov'] * cov_loss
    )
    return loss
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()