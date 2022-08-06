import sys
import torch
import numpy as np


##  matrix connecter (MxM),(NxN) -> ((M+N)x(M+N))
def mat_connect(mat1, mat2):
    size1 = list(mat1.shape)
    size2 = list(mat2.shape)
    size1[2] = mat2.shape[2]
    size2[2] = mat1.shape[2]
    # blank region
    blank1 = torch.zeros(size1, dtype=mat1.dtype)
    blank2 = torch.zeros(size2, dtype=mat2.dtype)
    # concat blank
    mat1_extend = torch.cat((mat1, blank1), dim=2)
    mat2_extend = torch.cat((blank2, mat2), dim=2)
    # concat mat1 & mat2
    mat = torch.cat((mat1_extend, mat2_extend), dim=1)
    # return
    return mat


##  Batch controller
class BatchLoader:
    def __init__(self, dataloader, maxsize):
        self.loader = iter(dataloader)
        self.store = next(self.loader)
        self.maxsize = maxsize
        self.counter = len(self.loader)-1
    def __iter__(self):
        return self
    def __next__(self):
        if self.counter < 0: raise StopIteration
        num = 1
        # get current data
        dat1, dat2, dat3, target, mask, name = self.store
        name = str(name)
        total_size = self.store[0].shape[1]
        if self.counter <= 0:
            self.counter = -1
            return dat1, dat2, dat3, target, mask, name, num
        # store next data
        self.store = next(self.loader)
        self.counter -= 1
        total_size = total_size + self.store[0].shape[1]
        # iteratively stacking
        while total_size < self.maxsize:
            dat1 = torch.cat((dat1, self.store[0]), 1)
            dat2 = mat_connect(dat2, self.store[1])
            dat3 = mat_connect(dat3, self.store[2])
            target = torch.cat((target, self.store[3]), 1)
            mask = torch.cat((mask, self.store[4]), 1)
            name = name + '_' + str(self.store[5])
            num += 1
            if self.counter <= 0:
                self.counter = -1
                return dat1, dat2, dat3, target, mask, name, num
            self.store = next(self.loader)
            self.counter -= 1
            total_size = total_size + self.store[0].shape[1]
        # return
        return dat1, dat2, dat3, target, mask, name, num


##  Training module
def train(model, criterion, source, train_loader, optimizer, hypara):
    model.train()
    # for transfer learning
    if source.onlypred is True:
        for params in model.embedding.parameters():
            params.requires_grad = False
    # training
    batch_loader = BatchLoader(train_loader, hypara.batchsize_cut)
    total_loss, total_count, total_correct, total_sample_count = 0, 0, 0, 0
    for batch_idx, (dat1, dat2, dat3, target, mask, name, num) in enumerate(batch_loader):
        dat1 = dat1.squeeze(0).to(source.device)
        dat2 = dat2.squeeze(0).to(source.device)
        dat3 = dat3.squeeze(0).to(source.device)
        target = target.squeeze(0).to(source.device)
        mask = mask.squeeze(0).to(source.device)
        total_sample_count += num
        optimizer.zero_grad()
        outputs = model(dat1, dat2, dat3)
        #loss = criterion(outputs*(mask.unsqueeze(1).float()), target)
        loss = criterion(outputs[mask], target[mask])
        predicted = torch.max(outputs, 1)
        count, correct = 0, 0
        for iaa in range(target.size()[0]):
            if (mask[iaa] == True):
                count = count + 1
                if (predicted[1][iaa] == target[iaa]):
                    correct = correct + 1
        total_count += count
        total_correct += correct
        total_loss += loss.item()*count
        ##  backward  ##
        loss.backward()
        optimizer.step()
        ################
        sys.stderr.write('\r\033[K' + '[{}/{}]'.format(total_sample_count, train_loader.__len__()))
        sys.stderr.flush()
    # loss & accuracy
    avg_loss = total_loss / total_count
    avg_acc = 100 * total_correct / total_count
    print(' T.Loss: {loss:.3f},  T.Acc: {acc:.3f}, '.
          format(loss=avg_loss, acc=avg_acc, file=sys.stderr), end='')
    # return
    return avg_loss, avg_acc


##  Validation module
def valid(model, criterion, source, valid_loader):
    model.eval()
    total_loss, total_count, total_correct = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (dat1, dat2, dat3, target, mask, name) in enumerate(valid_loader):
            dat1 = dat1.squeeze(0).to(source.device)
            dat2 = dat2.squeeze(0).to(source.device)
            dat3 = dat3.squeeze(0).to(source.device)
            target = target.squeeze(0).to(source.device)
            mask = mask.squeeze(0).to(source.device)
            outputs = model(dat1, dat2, dat3)
            #loss = criterion(outputs*(mask.unsqueeze(1).float()), target)
            loss = criterion(outputs[mask], target[mask])
            predicted = torch.max(outputs, 1)
            count, correct = 0, 0
            for iaa in range(target.size()[0]):
                if (mask[iaa] == True):
                    count = count + 1
                    if (predicted[1][iaa] == target[iaa]):
                        correct = correct + 1
            total_count += count
            total_correct += correct
            total_loss += loss.item()*count
    # loss & accuracy
    avg_loss = total_loss/total_count
    avg_acc = 100*total_correct/total_count
    print(' V.Loss: {loss:.3f}, V.Acc: {acc:.3f}'.
          format(loss=avg_loss, acc=avg_acc, file=sys.stderr))
    # return
    return avg_loss, avg_acc


##  Test module
def test(model, criterion, source, test_loader):
    model.eval()
    total_loss, total_count, total_correct = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (dat1, dat2, dat3, target, mask, name) in enumerate(test_loader):
            dat1 = dat1.squeeze(0).to(source.device)
            dat2 = dat2.squeeze(0).to(source.device)
            dat3 = dat3.squeeze(0).to(source.device)
            target = target.squeeze(0).to(source.device)
            mask = mask.squeeze(0).to(source.device)
            outputs = model(dat1, dat2, dat3)
            loss = criterion(outputs[mask], target[mask]).item()
            predicted = torch.max(outputs, 1)
            count, correct = 0, 0
            for iaa in range(target.size()[0]):
                if (mask[iaa] == True):
                    count = count + 1
                    if (predicted[1][iaa] == target[iaa]):
                        correct = correct + 1
            print('Loss=%7.4f   Acc=%6.2f %%  : L=%4d  (%s)' % (loss, 100*correct/count, count, name[0]))
            total_count += count
            total_loss += loss*count
            total_correct += correct
    # return
    return total_loss/total_count, 100*total_correct/total_count
