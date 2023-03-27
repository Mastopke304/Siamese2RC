import argparse
import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc

from Network import *
from dataset import *
from utils import *

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='PyTorch Siamese Reservoir Network')

parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--seed', type=int, default=304, help='Random seed')
parser.add_argument('--noise', type=bool, default=False)
parser.add_argument('--sigma', type=int, default=30, help='The sigma of gaussian noise, true sigma^2 = sigma**2 / (255 ** 2), cause the image is normalized into [0,1]')


# model
parser.add_argument('--filters', type=list, default=[32, 64, 96, 96], help='Filters of cnn')
parser.add_argument('--num_vec', type=int, default=16, help='The number of vectors feature maps be divided')
parser.add_argument('--outSize', type=int, default=1, help='Reservoir output size')
parser.add_argument('--resSize', type=int, default=2000, help='Reservoir capacity')
parser.add_argument('--alpha', type=float, default=0.3, help='Leaky rate')
parser.add_argument('--trainLen', type=int, default=8192, help='Number of data for training')
parser.add_argument('--batch_size_train', type=int, default=1024)
parser.add_argument('--batch_size_test', type=int, default=1000)

# dir
parser.add_argument('--save_dir', type=str, default='./result/train/weights/')
parser.add_argument('--img_save_dir', type=str, default='./result/train/')
parser.add_argument('--load_dir', type=str, default='./result/train/weights/')

args = parser.parse_args()
print(args)

myseed = args.seed
np.random.seed(myseed)
torch.manual_seed(myseed)
print("Current Seed: ", myseed)

print("Initializing...")
filters = args.filters
num_vec = args.num_vec
inSize = int((filters[-1] * 6 * 6) / num_vec * 2)
outSize = args.outSize
resSize = args.resSize
a = args.alpha
trainLen = args.trainLen
batch_size_train = args.batch_size_train
batch_size_test = args.batch_size_test
device = torch.device(args.device)

# ger_mnist(): If noise=True, then a Gaussian noise will be added to the raw images.
train_iter, test_iter = get_mnist(batch_size_train, batch_size_test, noise=args.noise, sigma=args.sigma)

loss = nn.MSELoss()
Train_MSE = []
Train_AUC = []
Test_MSE = []
Test_AUC = []

rc = ReservoirNet(inSize, resSize, a)
net2rc = Siamese2RC(inSize, filters)
print("Initialized!")


def train(net2rc, rc, train_iter, test_iter, device):
    print("train on", device)
    net2rc.apply(weight_init)
    net2rc.to(device)
    rc.to(device)
    cuda = next(net2rc.parameters()).device
    net2rc.eval()
    start = time.time()
    with torch.no_grad():
        Wouts = []
        train_simi = torch.zeros((10, batch_size_train))
        train_labels = torch.zeros((10, batch_size_train))
        test_simi = torch.zeros((10, batch_size_test))
        for times in range(10):
            train_start = time.time()
            for i, (img0, img1, label) in enumerate(train_iter):
                img0, img1, label = img0.to(cuda), img1.to(cuda), label.to(cuda)
                RCin = net2rc(img0, img1)
                gt = torch.hstack(
                    [torch.tensor([label[t] for _ in range(num_vec)]).T for t in range(batch_size_train)]).to(
                    torch.float32)
                if i == 0:
                    batch_Wout = rc(RCin, gt)
                elif i > 0 and (i + 1) * batch_size_train <= trainLen:
                    batch_Wout = batch_Wout + rc(RCin, gt)
                else:
                    break
            Wouts.append(batch_Wout / (trainLen / batch_size_train))
            pred = rc.RCPred(Wouts[times], RCin)
            for t in range(batch_size_train):
                train_simi[times, t] = torch.mean(pred[t * num_vec:(t + 1) * num_vec])
                train_labels[times, t] = gt[t * num_vec]
            l = loss(train_simi[times, :], train_labels[times, :].float())
            Train_MSE.append(l.item())
            auc = roc_auc_score(train_labels[times, :].detach().numpy(),
                                train_simi[times, :].detach().numpy())
            Train_AUC.append(auc)
            train_end = time.time() - train_start
            print('================================')
            print(f'Time cost for training: {train_end:.4f}s')
            batch_MSE = []
            batch_AUC = []
            test_start = time.time()
            for i, (img0, img1, label) in enumerate(test_iter):
                img0, img1, label = img0.to(cuda), img1.to(cuda), label.to(cuda)
                RCin = net2rc(img0, img1)
                pred = rc.RCPred(Wouts[times], RCin)
                for t in range(batch_size_test):
                    test_simi[times, t] = torch.mean(pred[t * num_vec:(t + 1) * num_vec])
                l = loss(test_simi[times], label.float())
                batch_MSE.append(l.item())
                auc = roc_auc_score(label.detach().numpy(),
                                    test_simi[times, :].detach().numpy())
                batch_AUC.append(auc)
            Test_MSE.append(np.mean(batch_MSE))
            Test_AUC.append(np.mean(batch_AUC))
            test_end = time.time() - test_start
            print(f'Times: {times + 1}, Train MSE: {Train_MSE[times]:.4f}, Train AUC: {Train_AUC[times]:.4f}')
            print(f'Times: {times + 1}, Test MSE: {Test_MSE[times]:.4f}, Test AUC: {Test_AUC[times]:.4f}')
            print(f'Time cost for testing: {test_end:.4f}s')
        trainAUC_idx = Train_AUC.index(max(Train_AUC))
        testAUC_idx = Test_AUC.index(max(Test_AUC))
        total_cost = time.time() - start
        torch.save(net2rc.state_dict(), args.save_dir + f'Siamese_{inSize}_{trainLen}_{batch_size_train}_{resSize}.pth')
        print(f'Best train AUC: {Train_AUC[trainAUC_idx]:.4f} from Weight {trainAUC_idx}')
        print(f'Best test AUC: {Test_AUC[testAUC_idx]:.4f} from Weight {testAUC_idx}')
        print(f'Total cost for 10 times training and testing: {total_cost:.4f}s')
        Wout = Wouts[testAUC_idx]
        if args.noise:
            np.save(args.save_dir + f'noiseWout_{trainLen}_{batch_size_train}_{resSize}.npy', Wout.detach().numpy())
        else:
            np.save(args.save_dir + f'Wout_{trainLen}_{batch_size_train}_{resSize}.npy', Wout.detach().numpy())

    return Wouts, Wout


def calculate_simi(model, rc, test_iter, batch_size_test, Wout, device):
    print("test on", device)
    model.to(device)
    cuda = next(model.parameters()).device
    start = time.time()
    model.eval()
    with torch.no_grad():
        similarity = torch.zeros((10000))
        labels = torch.zeros((10000))
        for i, (img0, img1, label) in enumerate(test_iter):
            img0, img1, label = img0.to(cuda), img1.to(cuda), label.to(cuda)
            labels[i * batch_size_test:(i + 1) * batch_size_test] = label
            if i == 0:
                example0 = torch.hstack([img0[t] for t in range(10)])
                example1 = torch.hstack([img1[t] for t in range(10)])
                example = torch.cat((example0, example1), dim=0)
            RCin = model(img0, img1)
            pred = rc.RCPred(Wout, RCin)
            for t in range(batch_size_test):
                similarity[i * batch_size_test + t] = torch.mean(pred[t * num_vec:(t + 1) * num_vec])
            l = loss(similarity[i * batch_size_test:(i + 1) * batch_size_test], label.float())
            MSE.append(l.item())
            auc = roc_auc_score(label.detach().numpy(),
                                similarity[i * batch_size_test:(i + 1) * batch_size_test].detach().numpy())
            AUC.append(auc)
            print(f'Batch: {i + 1}, MSE: {l.item():.4f}, AUC: {auc:.4f}')
    time_test = time.time() - start
    print(f'Time cost for testing: {time_test:.4f}s')

    return similarity, example, labels


Wouts, Wout = train(net2rc, rc, train_iter, test_iter, device)
if args.noise:
    print("Readout weight is saved at", args.save_dir + f"noiseWout_{trainLen}_{batch_size_train}_{resSize}.npy")
else:
    print("Readout weight is saved at", args.save_dir + f"Wout_{trainLen}_{batch_size_train}_{resSize}.npy")

MSE = []
AUC = []
model = Siamese2RC(inSize, filters)
model.load_state_dict(torch.load(args.load_dir + f'Siamese_{inSize}_{trainLen}_{batch_size_train}_{resSize}.pth', map_location=torch.device('cpu')))
model.eval()

print("Testing:")
similarity, example, labels = calculate_simi(model, rc, test_iter, batch_size_test, Wout, device)

similarity = similarity.detach().numpy()
labels = labels.detach().numpy()

output = pd.DataFrame({'Similarity': similarity, 'Ground Truth': labels})

if args.noise:
    output.to_csv(args.img_save_dir + f'noiseSimilarity_{trainLen}_{batch_size_train}_{resSize}.csv', index=False, sep=',')
    print("Similarity is saved at ", args.img_save_dir + f"noiseSimilarity_{trainLen}_{batch_size_train}_{resSize}.csv")
else:
    output.to_csv(args.img_save_dir + f'Similarity_{trainLen}_{batch_size_train}_{resSize}.csv', index=False, sep=',')
    print("Similarity is saved at ", args.img_save_dir + f"Similarity_{trainLen}_{batch_size_train}_{resSize}.csv")

fpr, tpr, thresh = roc_curve(labels, similarity)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(5, 5))
plt.title(f'ROC_{trainLen}_{batch_size_train}_{resSize}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(fpr, tpr, label=u'AUC = %0.4f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
if args.noise:
    plt.savefig(args.img_save_dir + 'noiseROC.svg', dpi=300)
else:
    plt.savefig(args.img_save_dir + 'ROC.svg', dpi=300)
plt.show()

x = np.round(similarity[0:10], 3)

plt.figure()
plt.title(f'Example_{trainLen}_{batch_size_train}_{resSize}')
plt.imshow(example, 'gray')
plt.xticks(range(0, 280, 28), x)
x_major_locator = plt.MultipleLocator(28)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
if args.noise:
    plt.savefig(args.img_save_dir + "noiseExample_images.svg", dpi=300)
else:
    plt.savefig(args.img_save_dir + "Example_images.svg", dpi=300)
plt.show()
