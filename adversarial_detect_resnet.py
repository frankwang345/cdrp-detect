import torch
import datetime
import numpy as np
import torch.nn.functional as F
from torch.nn import Parameter
import torchvision.transforms as transforms
from dataset import SampleDataset
import torchvision.models as models
import os
import argparse
import misc
import types
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, roc_auc_score

print = misc.logger.info
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', help='IMAGENET_DATA_DIR')
parser.add_argument('--arch', '-a', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--start_class', default=0, type=int)
parser.add_argument('--end_class', default=1000, type=int)
parser.add_argument('--train_num_per_class', default=1, type=int)
parser.add_argument('--test_num_per_class', default=1, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--display_freq', default=100, type=int)

args = parser.parse_args()
_timenow = str(datetime.datetime.now())
args.logdir = 'adversarial-detect-%s/train_num-%d-test_num-%d_%s' %\
              (args.arch, args.train_num_per_class, args.test_num_per_class, _timenow)
misc.prepare_logging(args)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


# Datra loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    SampleDataset(
        './data/train_images_list.pkl',
        start_class=args.start_class,
        end_class=args.end_class,
        num_per_class=args.train_num_per_class,
        random_order=True,
        transform=transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    ),
    batch_size=1, shuffle=False,
    num_workers=4, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    SampleDataset(
        './data/val_images_list.pkl',
        start_class=args.start_class,
        end_class=args.end_class,
        num_per_class=args.test_num_per_class,
        random_order=True,
        transform=transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    ),
    batch_size=1, shuffle=False,
    num_workers=4, pin_memory=True)

def load_model(args):
    model = models.__dict__[args.arch](pretrained=True)
    model.eval()
    return model


def init_control_gates(m):
    name = m.__class__.__name__
    if name.find('Bottleneck') != -1:
        m.control_gates = Parameter(torch.FloatTensor(m.bn3.num_features))
        m.control_gates.data.fill_(1.0)

def reset_control_gates(m):
    name = m.__class__.__name__
    if name.find('Bottleneck') != -1:
        m.control_gates.data.fill_(1.0)
        m.control_gates.grad.data.fill_(0.0)

def new_forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    out = self.control_gates.view(1, -1, 1, 1) * out

    return out

def replace(m):
    name = m.__class__.__name__
    if name.find('Bottleneck') != -1:
        m.forward = types.MethodType(new_forward, m)

def collect_control_gates(m):
    name = m.__class__.__name__
    if name.find('Bottleneck') != -1:
        control_gates.append(m.control_gates)

def get_adv_targeted_input(inp, model, T, eps, target):
    inp_min, inp_max = inp.data.min(), inp.data.max()
    target_var = Variable(target).cuda()

    for i in range(T):
        zero_gradients(inp)
        output = model(inp)
        loss = F.cross_entropy(output, target_var)
        loss.backward()
        inp.data -= eps * torch.sign(inp.grad.data)
        inp.data = torch.clamp(inp.data, inp_min, inp_max)

def get_critical_path(data_var, model):
    self_predicted_output = model(data_var)
    self_pred = self_predicted_output.data.max(1)[1]
    self_predicted_prob = F.softmax(self_predicted_output)
    self_predicted_prob_var = Variable(self_predicted_prob.data)

    lambd = 0.05
    max_iters = 30
    min_loss = 1e10

    for j in range(max_iters):
        output = model(data_var)
        prob = F.softmax(output)

        pred = output.data.max(1)[1]

        loss = - (self_predicted_prob_var * torch.log(prob + 1e-20)).sum(1)

        for v in control_gates:
            loss += lambd * v.abs().sum()

        if pred[0] == self_pred[0]:
            if loss.data[0] < min_loss:
                cg_list = []
                for v in control_gates:
                    cg_list.append(v.data.clone())

                min_loss = loss.data[0]
                best_output = output.data.clone()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for v in control_gates:
            v.data.clamp_(0, 100)

    model.apply(reset_control_gates)

    return cg_list, best_output

base_model = load_model(args)
base_model.cuda()

adv_target_class = torch.randperm(1000)

model = load_model(args)
control_gates = []

model.apply(init_control_gates)
model.apply(replace)
model.apply(collect_control_gates)
model.cuda()

all_orig_cglist = []
all_adv_cglist = []

for i, (data, target) in enumerate(train_loader):
    optimizer = torch.optim.SGD(control_gates, lr=0.1, momentum=0.9, weight_decay=0)

    data_var = Variable(data).cuda()
    target_var = Variable(target).cuda()
    orig_cg_list, _ = get_critical_path(data_var, model)

    adv_inp = Variable(data.cuda(), requires_grad=True)
    get_adv_targeted_input(adv_inp, base_model, 10, 0.01, adv_target_class[target])

    adv_data_var = Variable(adv_inp.data)
    adv_cg_list, _ = get_critical_path(adv_inp, model)

    all_orig_cglist.append(torch.cat(orig_cg_list))
    all_adv_cglist.append(torch.cat(adv_cg_list))
    if i % args.display_freq == 0:
        print('processing [%d/%d] image...' % (i, len(train_loader)))

all_orig_cglist = torch.stack(all_orig_cglist)
all_adv_cglist = torch.stack(all_adv_cglist)


all_train_samples = torch.cat([all_orig_cglist, all_adv_cglist]).cpu().numpy()
all_train_labels = np.hstack([np.ones(len(all_orig_cglist)), np.zeros(len(all_adv_cglist))])

_idx = np.random.permutation(np.arange(len(all_train_labels)))
all_train_samples = all_train_samples[_idx]
all_train_labels = all_train_labels[_idx]

clf = RandomForestClassifier(50)
clf.fit(all_train_samples, all_train_labels)


#############

val_all_orig_cglist = []
val_all_adv_cglist = []

for i, (data, target) in enumerate(val_loader):
    optimizer = torch.optim.SGD(control_gates, lr=0.1, momentum=0.9, weight_decay=0)

    data_var = Variable(data).cuda()
    target_var = Variable(target).cuda()
    val_orig_cg_list, _ = get_critical_path(data_var, model)

    adv_inp = Variable(data.cuda(), requires_grad=True)
    get_adv_targeted_input(adv_inp, base_model, 10, 0.01, adv_target_class[target])

    adv_data_var = Variable(adv_inp.data)
    val_adv_cg_list, _ = get_critical_path(adv_inp, model)

    val_all_orig_cglist.append(torch.cat(val_orig_cg_list))
    val_all_adv_cglist.append(torch.cat(val_adv_cg_list))
    if i % args.display_freq == 0:
        print('generate [%d/%d] val image...' % (i, len(val_loader)))

val_all_orig_cglist = torch.stack(val_all_orig_cglist)
val_all_adv_cglist = torch.stack(val_all_adv_cglist)

all_val_samples = torch.cat([val_all_orig_cglist, val_all_adv_cglist]).cpu().numpy()
all_val_labels = np.hstack([np.ones(len(val_all_orig_cglist)), np.zeros(len(val_all_adv_cglist))])

_idx = np.random.permutation(np.arange(len(all_val_labels)))
all_val_samples = all_val_samples[_idx]
all_val_labels = all_val_labels[_idx]

preds = clf.predict(all_val_samples)
prec = precision_score(all_val_labels, preds)
ras = roc_auc_score(all_val_labels, preds)

print('precision = %.4f, roc_auc_score = %.4f' % (prec, ras))
misc.dump_pickle([all_train_samples, all_train_labels], os.path.join(args.logdir, 'train_infos.pkl'))
misc.dump_pickle([all_val_samples, all_val_labels], os.path.join(args.logdir, 'val_infos.pkl'))
misc.dump_pickle(clf, os.path.join(args.logdir, 'clf.pkl'))
