import argparse
import time

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc

from Network import *
from dataset import *

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Test')

parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--seed', type=int, default=304, help='Set the random seed the same as training')
parser.add_argument('--noise', type=bool, default=False)
parser.add_argument('--sigma', type=int, default=30, help='The sigma of gaussian noise, true sigma^2 = sigma**2 / (255 ** 2), cause the image is normalized into [0,1]')
parser.add_argument('--weight_type', type=str, default='clean', choices=['clean', 'noise'], help='Choose the weights that trained on clean images or noise images for testing')

# model
parser.add_argument('--filters', type=list, default=[32, 64, 96, 96], help='Filters of cnn')
parser.add_argument('--num_vec', type=int, default=16, help='The number of vectors feature maps be divided')
parser.add_argument('--outSize', type=int, default=1, help='Reservoir output size')
parser.add_argument('--resSize', type=int, default=2000, help='Reservoir capacity')
parser.add_argument('--alpha', type=float, default=0.3, help='Leaky rate')
parser.add_argument('--trainLen', type=int, default=8192, help='Number of data for training (max: 60000)')
parser.add_argument('--batch_size_train', type=int, default=1024, help='useless')
parser.add_argument('--batch_size_test', type=int, default=1000)

# dir
parser.add_argument('--save_dir', type=str, default='./result/test/')
parser.add_argument('--load_dir', type=str, default='./result/train/weights/')

args = parser.parse_args()
print(args)

myseed = args.seed
np.random.seed(myseed)
torch.manual_seed(myseed)
print("Current Seed: ", myseed)

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
_, test_iter = get_mnist(batch_size_train, batch_size_test, noise=args.noise, sigma=args.sigma)

if args.weight_type == 'clean':
    Woutfile = f'Wout_{trainLen}_{batch_size_train}_{resSize}.npy'
elif args.weight_type == 'noise':
    Woutfile = f'noiseWout_{trainLen}_{batch_size_train}_{resSize}.npy'
else:
    print(f"Error, weight_type must be one of [clean, noise], you entered: {args.weight_type}")

loss = nn.MSELoss()
MSE = []
AUC = []

print("Initializing...")
Wout = torch.from_numpy(np.load(args.load_dir + Woutfile))
Rc = ReservoirNet(inSize, resSize, a)
model = Siamese2RC(inSize, filters)
model.load_state_dict(torch.load(args.load_dir + f'Siamese_{inSize}_{trainLen}_{batch_size_train}_{resSize}.pth', map_location=device))
model.eval()
print("Initialized!")


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


similarity, example, labels = calculate_simi(model, Rc, test_iter, batch_size_test, Wout, device)
print("Mean Test MSE", np.mean(MSE))
print("Mean Test AUC", np.mean(AUC))

similarity = similarity.detach().numpy()
labels = labels.detach().numpy()
output = pd.DataFrame({'Similarity': similarity, 'Ground Truth': labels})
if args.noise:
    if args.weight_type == 'noise':
        output.to_csv(args.save_dir + f'noiseSimilarity_{trainLen}_{batch_size_train}_{resSize}_noiseWeight.csv', index=False, sep=',')
    else:
        output.to_csv(args.save_dir + f'noiseSimilarity_{trainLen}_{batch_size_train}_{resSize}.csv', index=False, sep=',')
else:
    output.to_csv(args.save_dir + f'Similarity_{trainLen}_{batch_size_train}_{resSize}.csv', index=False, sep=',')
print("Similarity is saved at ", args.save_dir + f"Similarity_{trainLen}_{batch_size_train}_{resSize}.csv")


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
    if args.weight_type == 'noise':
        plt.savefig(args.save_dir + 'noiseROC_noiseWeight.svg', dpi=300)
    else:
        plt.savefig(args.save_dir + 'noiseROC.svg', dpi=300)
else:
    plt.savefig(args.save_dir + 'ROC.svg', dpi=300)
plt.show()


x = np.round(similarity[0:10], 3)
plt.figure()
plt.title(f'Example__{trainLen}_{batch_size_train}_{resSize}')
plt.imshow(example, 'gray')
plt.xticks(range(0, 280, 28), x)
x_major_locator = plt.MultipleLocator(28)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
if args.noise:
    if args.weight_type == 'noise':
        plt.savefig(args.save_dir + "noiseExample_images_noiseWeight.svg", dpi=300)
    else:
        plt.savefig(args.save_dir + "noiseExample_images.svg", dpi=300)
else:
    plt.savefig(args.save_dir + "Example_images.svg", dpi=300)
plt.show()


