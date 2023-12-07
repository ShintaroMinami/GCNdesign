import sys
import torch
import numpy as np

def get_random_mask(ratio, size, device='cuda'):
    num_masked_aa = int(size * ratio)
    idx = np.random.choice(range(size), num_masked_aa, replace=False)
    mask = torch.zeros(size, dtype=bool)
    mask[idx] = 1
    return mask.to(device)

def get_batched_random_mask(batch, size, device='cuda'):
    mask_stack = []
    for i in range(batch):
        ratio = (i/batch)
        mask_stack.append(get_random_mask(ratio, size, device))
    return torch.stack(mask_stack)


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
def train(model, criterion, train_loader, optimizer, maxsize, device, max_ratio=0.95):
    model.train()
    mask_index = model.mask_index
    # training
    batch_loader = BatchLoader(train_loader, maxsize)
    total_loss, total_count, total_correct, total_sample_count = 0, 0, 0, 0
    for batch_idx, (dat1, dat2, dat3, target, mask, name, num) in enumerate(batch_loader):
        dat1, dat2, dat3 = dat1.to(device), dat2.to(device), dat3.to(device)
        target, mask = target.to(device), mask.to(device)
        length = dat1.shape[1]
        total_sample_count += num
        # random mask
        open_ratio = np.random.rand(1)[0] * max_ratio
        random_mask = get_random_mask(open_ratio, length, dat1.device).unsqueeze(0)
        masked_resid = random_mask * target + ~random_mask * mask_index
        # model
        optimizer.zero_grad()
        outputs = model(dat1, dat2, dat3, masked_resid)
        # loss & acc
        is_checked = mask * ~random_mask
        loss = criterion(outputs[is_checked], target[is_checked])
        prediction = torch.max(outputs, -1)[1]
        # sum
        count = (is_checked == True).to(int).sum().item()
        total_count += count
        total_correct += (prediction[is_checked] == target[is_checked]).to(int).sum().item()
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
    sys.stderr.write(f' T.Loss: {avg_loss:4.2f}  T.Acc: {avg_acc:4.1f} ')
    # return
    return avg_loss, avg_acc


##  Validation module
def valid(model, criterion, valid_loader, device, check_ratios=[0.0, 0.5, 0.95]):
    model.eval()
    mask_index = model.mask_index
    total_count = {open_ratio: 0 for open_ratio in check_ratios}
    total_loss = {open_ratio: 0 for open_ratio in check_ratios}
    total_correct = {open_ratio: 0 for open_ratio in check_ratios}
    with torch.no_grad():
        for batch_idx, (dat1, dat2, dat3, target, mask, name) in enumerate(valid_loader):
            dat1, dat2, dat3 = dat1.to(device), dat2.to(device), dat3.to(device)
            target, mask = target.to(device), mask.to(device)
            length = dat1.shape[1]
            # random mask
            for open_ratio in check_ratios:
                random_mask = get_random_mask(open_ratio, length, dat1.device).unsqueeze(0)
                masked_resid = random_mask * target + ~random_mask * mask_index
                # model
                outputs = model(dat1, dat2, dat3, masked_resid)
                # loss & acc
                is_checked = mask * ~random_mask
                loss = criterion(outputs[is_checked], target[is_checked])
                prediction = torch.max(outputs, -1)[1]
                # sum
                count = (is_checked == True).to(int).sum().item()
                correct = (prediction[is_checked] == target[is_checked]).to(int).sum().item()
                loss =  loss.item()*count
                total_count[open_ratio] = total_count[open_ratio] + count
                total_loss[open_ratio] = total_loss[open_ratio] + loss
                total_correct[open_ratio] = total_correct[open_ratio] + correct
    # loss & accuracy
    avg_loss = {key: total_loss[key]/total_count[key] for key in check_ratios}
    avg_acc = {key: 100.0*total_correct[key]/total_count[key] for key in check_ratios}
    # write    
    sys.stderr.write("  V.Loss:")
    for ratio in check_ratios:
        sys.stderr.write(f" {avg_loss[ratio]:4.2f}")
    sys.stderr.write("  V.Acc:")
    for ratio in check_ratios:
        sys.stderr.write(f" {avg_acc[ratio]:4.1f}")
    sys.stderr.write("\n")
    sys.stderr.flush()
    # return
    return avg_loss, avg_acc




##  Validation module
def valid0(model, criterion, valid_loader, device):
    model.eval()
    total_loss, total_count, total_correct = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (dat1, dat2, dat3, target, mask, name) in enumerate(valid_loader):
            dat1 = dat1.to(device)
            dat2 = dat2.to(device)
            dat3 = dat3.to(device)
            target = target.squeeze(0).to(device)
            mask = mask.squeeze(0).to(device)
            outputs = model(dat1, dat2, dat3).squeeze(0)
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
def test(model, criterion, test_loader, device):
    model.eval()
    total_loss, total_count, total_correct = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (dat1, dat2, dat3, target, mask, name) in enumerate(test_loader):
            dat1 = dat1.squeeze(0).to(device)
            dat2 = dat2.squeeze(0).to(device)
            dat3 = dat3.squeeze(0).to(device)
            target = target.squeeze(0).to(device)
            mask = mask.squeeze(0).to(device)
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
