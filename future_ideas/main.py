from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score
import numpy as np
import os
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn.functional as F
import torch.nn as nn
from model.model import SingleLayerModel
from util.losses import TripletLoss
from util.dataset import LFW, LFW_Train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RCF = 0.8


def CosineDistance(x1, x2):
    return 1 - F.cosine_similarity(x1, x2)


metric = nn.CosineSimilarity(eps=1e-6)
cnn = SingleLayerModel(embedding_size=1024).to(device)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def compute_f1_test(model, grt):
    model.eval()
    for x, y, label in grt:
        x = x.to(device=device)
        y = y.to(device=device)
        predictx = model(x)
        predicty = model(y)
        pred = []
        for i in range(len(predictx)):
            pred.append((1+sum(predictx[i]*predicty[i]))/2)
        pred = np.array(list(map(lambda x: int(x > RCF), pred)))
        return f1_score(label, pred)


def validation_f1(model, grt):
    model.eval()
    for x, y, z in grt:
        x = x.to(device=device)
        y = y.to(device=device)
        z = z.to(device=device)
        predictx = model(x)
        predicty = model(y)
        predictz = model(z)
        pred = []
        label = []
        for i in range(len(predictx)):
            pred.append(((1+sum(predictx[i]*predicty[i]))/2))
            #pred.append(((1+sum(predictx[i]*predictz[i]))/2))
            label.append(1.0)
            label.append(0.0)

            print(predictx[i].shape)
            print(predictx[i]*predicty[i])
            print(sum(predictx[i]*predicty[i]))
            break
        label = np.array(label)
        return f1_score(label, predicted)


def validation(val_loader):
    cnn.eval()
    scores = []
    scores_imposter = []
    i = 200
    for mask_embedding, face_embedding, negative_embedding in val_loader:
        mask_embedding = mask_embedding.to(device)
        face_embedding = face_embedding.to(device)
        negative_embedding = negative_embedding.to(device)
        with torch.no_grad():
            pred = cnn(mask_embedding)
        scores.append(metric(l2_norm(pred), l2_norm(face_embedding)).item())
        m = (metric(l2_norm(pred), l2_norm(negative_embedding)).item())
        scores_imposter.append(m)
        i -= 1
    cnn.train()


    return np.mean(scores),np.mean(scores_imposter)


def training(epoch, weights, batch_size):

    train_dataset = LFW_Train("lfw/pairsDevTrain.txt",
                              transform=A.Compose([
                                  ToTensorV2(),
                                  A.HorizontalFlip(p=0.5),
                                  A.Rotate(limit=(0, 25), p=0.5),
                              ]))

    test_dataset = LFW("lfw/pairsDevTest.txt")

    data_size = len(train_dataset)
    validation_fraction = .2

    val_split = int(np.floor((validation_fraction) * data_size))
    indices = list(range(data_size))
    np.random.seed(42)
    np.random.shuffle(indices)

    val_indices, train_indices = indices[:val_split], indices[val_split:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                             sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=float(0.1), momentum=0.9, nesterov=True, weight_decay=0.0)
    scheduler = StepLR(cnn_optimizer, gamma=0.1, step_size=3)
    criterion = TripletLoss(distance="Triplet").to(device)
    early_stopping = True
    patience = 20

    epochs_no_improvement = 0
    max_val_fscore = 0.0
    best_weights = None
    best_epoch = -1
    cnn.to(device)
    cnn.train()

    for epoch in range(1, 1 + epoch):
        loss_total = 0.0
        positive_loss_total = 0.0
        negative_loss_total = 0.0
        negative_positive_total = 0.0

        progress_bar = tqdm(train_loader)
        for i, (sample_embedding, positive_embedding, negative_embedding) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            sample_embedding = sample_embedding.to(device)
            positive_embedding = positive_embedding.to(device)
            negative_embedding = negative_embedding.to(device)
            cnn.zero_grad()
            pred = cnn(sample_embedding)
            loss, positive_loss, negative_loss, negative_positive = criterion(pred, positive_embedding,
                                                                              negative_embedding)
            loss.backward()
            cnn_optimizer.step()

            loss_total += loss.item()
            positive_loss_total += positive_loss.item()
            negative_loss_total += negative_loss.item()
            negative_positive_total += negative_positive.item()

            progress_bar.set_postfix(
                loss='%.5f' % (loss_total / (i + 1)), negative_loss='%.5f' % (negative_loss_total / (i + 1)),
                positive_loss='%.5f' % (positive_loss_total / (i + 1)),
                negative_positive='%.5f' % (negative_positive_total / (i + 1)))

        val_fscore, val_fscore_imposter = validation(val_loader)

        scheduler.step()

        do_stop = False
        if early_stopping:
            if val_fscore > max_val_fscore:
                max_val_fscore = val_fscore
                epochs_no_improvement = 0
                best_weights = cnn.state_dict()
                best_epoch = epoch
            else:
                epochs_no_improvement += 1

            if epochs_no_improvement >= patience and do_stop:
                print(f"EARLY STOPPING at {best_epoch}: {max_val_fscore}")
                break
        else:
            best_weights = cnn.state_dict()
    if not os.path.isdir(weights):
        os.makedirs(weights)
    torch.save(best_weights, f"{weights}/weights.pt")


if __name__ == '__main__':
    training(epoch=30, weights='weights/Triplet', batch_size=8)
