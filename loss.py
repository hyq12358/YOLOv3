import random
import torch
import torch.nn as nn

from utils import intersection_over_union, bbox_overlaps_ciou

class CIoULoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(IouLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, bboxes1, bboxes2):
        num = bboxes1.shape[0] 
        
        loss = torch.sum(1.0 - bbox_overlaps_ciou(bboxes1, bboxes12))            
     
        if self.reduction=='mean':
            loss = loss/num
        else:
            loss = loss
        return loss

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict)
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) 
- (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = BCEFocalLoss()
        self.sigmoid = nn.Sigmoid()
        self.ciou = CIoULoss()
        # loss weights
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        # Non object loss
        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # object loss
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # ciou loss
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        #target[..., 3:5] = torch.log(
        #    (1e-16 + target[..., 3:5] / anchors)
        #)  # width, height coordinates
        predictions[..., 3:5] = torch.exp(predictions[..., 3:5])*anchors
        predictions[..., 0:1] = predictions[..., 0:1]-(predictions[..., 2:3])/2 #x -> left
        predictions[..., 1:2] = predictions[..., 1:2]-(predictions[..., 3:4])/2 #y -> top
        predictions[..., 2:3] = predictions[..., 0:1]+(predictions[..., 2:3]) #w -> right
        predictions[..., 3:4] = predictions[..., 1:2]+(predictions[..., 3:4]) #x -> down
        box_loss = self.ciou(predictions[..., 1:5][obj], target[..., 1:5][obj])


        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.focal(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        #print("__________________________________")
        #print(self.lambda_box * box_loss)
        #print(self.lambda_obj * object_loss)
        #print(self.lambda_noobj * no_object_loss)
        #print(self.lambda_class * class_loss)
        #print("\n")

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
