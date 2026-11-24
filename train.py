import os
import random
import torch
import argparse
import numpy as np
import cv2
import torch.optim as optim
import torch.nn.functional as F
import Transforms as myTransforms
from torch.utils.data import DataLoader
from model.decoder import FlickCD
from loadData import makeDataset
from tqdm import tqdm
from utils import get_logger, Evaluator

def Cal_loss(output, target):
    loss_res = 0
    for res in output:
        loss_res += F.binary_cross_entropy(res, target)
    return loss_res

class Trainer(object):
    def __init__(self, args):
        self.device = torch.device(f'cuda:{args.gpu_id}')
        self.args = args
        self.data_name = args.data_name
        self.TITLE = args.title
        self.evaluator = Evaluator(num_class=2)
        self.evaluator_train = Evaluator(num_class=2)

        # Set model parameters
        window_size=None
        stride=None
        load_pretrained = False
        if args.data_name in ['SYSU', 'CDD']:
            window_size = (8, 8, 16)
            stride = (4, 4, 8)
        elif args.data_name == 'WHU':
            window_size = (4, 4, 8)
            stride = (4, 4, 8)
        elif args.data_name == 'LEVIR+':
            window_size = (4, 8, 8)
            stride = (4, 8, 8)
        pretrained_path = args.pretrained
        if args.mode == 'train' and pretrained_path.lower() != 'none':
            if os.path.isfile(pretrained_path):
                load_pretrained = True
            else:
                print(f"RepViT checkpoint not found: {pretrained_path}; training from random initialization.")

        self.model = FlickCD(
            window_size, stride, load_pretrained,
            pretrained_path=pretrained_path if load_pretrained else None,
        )
        self.model = self.model.to(self.device)

        self.model_save_path = os.path.join(args.savedir, self.TITLE)
        self.log_dir = self.model_save_path + '/Logs/'

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.logger = get_logger(self.log_dir + self.TITLE + '.log')

        self.lr = args.learning_rate
        self.epoch = args.epochs
        self.patch_size = args.patch_size
        self.optim = optim.AdamW(self.model.parameters(), self.lr, weight_decay=args.weight_decay)

        mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
        std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

        self.trainTransform = myTransforms.Compose([
            myTransforms.Normalize(mean=mean, std=std),
            myTransforms.Scale(args.input_size, args.input_size),
            myTransforms.RandomCropResize(int(7. / 224. * args.input_size)),
            myTransforms.RandomFlip(),
            myTransforms.RandomExchange(),
            myTransforms.Rotate(),
            myTransforms.ToTensor(),
        ])

        self.valTransform = myTransforms.Compose([
            myTransforms.Normalize(mean=mean, std=std),
            myTransforms.Scale(args.input_size, args.input_size),
            myTransforms.ToTensor(),
        ])

        generator = torch.Generator().manual_seed(args.seed)
        self.train_dataset = makeDataset(self.args.train_dataset_path, self.args.train_name_list,
                                         self.trainTransform, image_dir_a=args.image_dir_a,
                                         image_dir_b=args.image_dir_b, label_dir=args.label_dir,
                                         patch_size=self.patch_size)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, generator=generator, num_workers=args.num_workers, drop_last=False)

        self.val_dataset = makeDataset(self.args.val_dataset_path, self.args.val_name_list,
                                       self.valTransform, image_dir_a=args.image_dir_a,
                                       image_dir_b=args.image_dir_b, label_dir=args.label_dir,
                                       patch_size=self.patch_size)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, num_workers=args.num_workers, drop_last=False)

        self.test_dataset = makeDataset(self.args.test_dataset_path, self.args.test_name_list,
                                        self.valTransform, image_dir_a=args.image_dir_a,
                                        image_dir_b=args.image_dir_b, label_dir=args.label_dir,
                                        patch_size=self.patch_size)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, num_workers=args.num_workers, drop_last=False)
        self.logger.info(
            'Dataset samples after on-the-fly patching: train=%d val=%d test=%d patch_size=%d',
            len(self.train_dataset), len(self.val_dataset), len(self.test_dataset), self.patch_size
        )

        self.best_f1 = 0.0
        self.best_epoch = 0
        self.best_f1_train = 0.0
        self.best_epoch_train = 0
        self.start_epoch = 0
        self.top_models = []

    def training(self):
        self.logger.info('Net: ' + self.TITLE)
        mean_loss = 0.0

        if self.args.resume is None:
            self.args.resume = self.model_save_path + '/checkpoint.pth.tar'
        if os.path.isfile(self.args.resume):
            print("=> loading checkpoint '{}'".format(self.args.resume))
            checkpoint = torch.load(self.args.resume, weights_only=False)
            self.start_epoch = checkpoint['epoch']
            self.best_f1 = checkpoint['best_f1']
            self.best_epoch = checkpoint['best_epoch']
            self.best_f1_train = checkpoint['best_f1_train']
            self.best_epoch_train = checkpoint['best_epoch_train']
            self.top_models = checkpoint.get('top_models', [])
            if not self.top_models and os.path.isfile(os.path.join(self.model_save_path, 'best_model.pth')):
                self.top_models = [{'score': float(self.best_f1), 'epoch': int(self.best_epoch), 'rank': 1}]
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(self.args.resume))

        torch.cuda.empty_cache()
        total_step = len(self.train_dataloader)

        for e in range(self.start_epoch, self.epoch):
            for iter, data in enumerate(self.train_dataloader):
                pre_img, post_img, gt, data_idx = data
                pre_img = pre_img.to(self.device).float()
                post_img = post_img.to(self.device).float()
                label = gt.unsqueeze(dim=1).to(self.device).float()

                output, output2, output3= self.model(pre_img, post_img)

                loss = Cal_loss([output, output2, output3], label)

                mean_loss += loss.item()
                pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                pred = pred.cpu().numpy()
                label = label.cpu().numpy()
                self.evaluator_train.add_batch(label, pred)

                if (iter + 1) % 5 == 0:
                    print('%d/%d, Epoch:%d, mean_loss:%f' % (iter+1, total_step, e+1, mean_loss / 5))
                    mean_loss = 0

                if (iter + 1) == total_step:
                    self.val_phase(e, 'val')

    def save_top_models(self, f1_score, epoch):
        entries = []
        for entry in self.top_models:
            rank_path = os.path.join(
                self.model_save_path,
                'best_model.pth' if entry['rank'] == 1 else f"best_model_{entry['rank']}.pth",
            )
            if os.path.isfile(rank_path):
                entries.append({
                    'score': float(entry['score']),
                    'epoch': int(entry['epoch']),
                    'state_dict': torch.load(rank_path, map_location='cpu', weights_only=True),
                })
        entries.append({
            'score': float(f1_score),
            'epoch': int(epoch + 1),
            'state_dict': {key: value.detach().cpu().clone()
                           for key, value in self.model.state_dict().items()},
        })
        entries.sort(key=lambda item: (-item['score'], item['epoch']))
        selected = entries[:3]

        for rank, entry in enumerate(selected, start=1):
            filename = 'best_model.pth' if rank == 1 else f'best_model_{rank}.pth'
            target_path = os.path.join(self.model_save_path, filename)
            temporary_path = target_path + '.tmp'
            torch.save(entry['state_dict'], temporary_path)
            os.replace(temporary_path, target_path)
            entry['rank'] = rank
            del entry['state_dict']

        for rank in range(len(selected) + 1, 4):
            stale_path = os.path.join(self.model_save_path, f'best_model_{rank}.pth')
            if os.path.exists(stale_path):
                os.remove(stale_path)
        self.top_models = selected
        self.best_f1 = selected[0]['score']
        self.best_epoch = selected[0]['epoch']

    def val_phase(self, epoch, type):
        f1_train = self.evaluator_train.Pixel_F1_score()
        oa_train = self.evaluator_train.Pixel_Accuracy()
        rec_train = self.evaluator_train.Pixel_Recall_Rate()
        pre_train = self.evaluator_train.Pixel_Precision_Rate()
        iou_train = self.evaluator_train.Intersection_over_Union()
        kc_train = self.evaluator_train.Kappa_coefficient()
        if f1_train > self.best_f1_train:
            self.best_f1_train = f1_train
            self.best_epoch_train = epoch + 1
        self.evaluator_train.reset()
        self.logger.info(
            'Epoch:[{}/{}]  train_Pre={:.4f}  train_Rec={:.4f}  train_OA={:.4f}  train_F1={:.4f}  train_IoU={:.4f}  train_KC={:.4f}  best_F1_train:[{:.4f}/{}]'.format(
                epoch + 1, self.epoch, pre_train, rec_train, oa_train, f1_train, iou_train, kc_train, self.best_f1_train,
                self.best_epoch_train))

        self.model.eval()
        rec, pre, oa, f1_score, iou, kc = self.validation(type)

        self.save_top_models(f1_score, epoch)

        checkpoint = {
            'epoch': epoch + 1,
            'best_f1': self.best_f1,
            'best_epoch': self.best_epoch,
            'best_f1_train': self.best_f1_train,
            'best_epoch_train': self.best_epoch_train,
            'top_models': self.top_models,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
        }
        checkpoint_path = self.model_save_path + '/checkpoint.pth.tar'
        torch.save(checkpoint, checkpoint_path + '.tmp')
        os.replace(checkpoint_path + '.tmp', checkpoint_path)

        self.logger.info(
            'Epoch:[{}/{}]  val_Pre={:.4f}  val_Rec={:.4f}  val_OA={:.4f}  val_F1={:.4f}  val_IoU={:.4f}  val_KC={:.4f} best_F1:[{:.4f}/{}]'.format(
                epoch + 1, self.epoch, pre, rec, oa, f1_score, iou, kc, self.best_f1, self.best_epoch))
        self.model.train()

    def validation(self, type):
        self.evaluator.reset()
        if type == 'val':
            data_loader = self.val_dataloader
        elif type == 'test':
            data_loader = self.test_dataloader

        torch.cuda.empty_cache()

        with torch.no_grad():
            stitched_name = None
            stitched_pred = None
            stitched_label = None
            for iter, data in enumerate(tqdm(data_loader)):
                pre_img, post_img, gt, metadata = data
                pre_img = pre_img.to(self.device).float()
                post_img = post_img.to(self.device).float()

                output, output2, output3 = self.model(pre_img, post_img)
                pred = (output > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)
                for index in range(pred.shape[0]):
                    name = metadata['name'][index]
                    height = int(metadata['height'][index])
                    width = int(metadata['width'][index])
                    top = int(metadata['top'][index])
                    left = int(metadata['left'][index])
                    region_h = int(metadata['crop_height'][index])
                    region_w = int(metadata['crop_width'][index])
                    if name != stitched_name:
                        if stitched_name is not None:
                            self.evaluator.add_batch(stitched_label, stitched_pred)
                        stitched_name = name
                        stitched_pred = np.zeros((height, width), dtype=np.uint8)
                        label_path = os.path.join(data_loader.dataset.data_path,
                                                 data_loader.dataset.label_dir, name)
                        stitched_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                        if stitched_label is None or stitched_label.shape != (height, width):
                            raise ValueError(f'Could not read original-resolution label: {label_path}')
                        stitched_label = np.ceil(stitched_label / 255).astype(np.uint8)
                    pred_patch = cv2.resize(pred[index], (region_w, region_h), interpolation=cv2.INTER_NEAREST)
                    stitched_pred[top:top + region_h, left:left + region_w] = pred_patch

            if stitched_name is not None:
                self.evaluator.add_batch(stitched_label, stitched_pred)

        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()
        return rec, pre, oa, f1_score, iou, kc

    def test(self):
        print("----------Starting Test!----------")
        model_file_name = self.args.test_checkpoint or os.path.join(self.model_save_path, 'best_model.pth')
        state_dict = torch.load(model_file_name, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        rec, pre, oa, f1_score, iou, kc  = self.validation(type='test')
        self.logger.info(
            'Test\t test_Pre={:.4f}\t test_Rec:{:.4f}\t test_OA={:.4f}\t test_F1={:.4f}\t test_IoU={:.4f}\t test_KC={:.4f}'.format(
                pre, rec, oa, f1_score, iou, kc))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

def main():
    parser = argparse.ArgumentParser(description="Argument for training")
    parser.add_argument('--title', type=str)

    # set data path
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--train_dataset_path', type=str)
    parser.add_argument('--train_list_path', type=str, default=None)
    parser.add_argument('--val_dataset_path', type=str)
    parser.add_argument('--val_list_path', type=str, default=None)
    parser.add_argument('--test_dataset_path', type=str)
    parser.add_argument('--test_list_path', type=str, default=None)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--savedir', default='./result/', type=str)
    parser.add_argument('--image_dir_a', type=str, default='A')
    parser.add_argument('--image_dir_b', type=str, default='B')
    parser.add_argument('--label_dir', type=str, default='label')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--test_checkpoint', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default='./model/repvit_m0_6.pth',
                        help="RepViT encoder checkpoint for training, or 'none' for random initialization")
    
    # Choose GPU
    parser.add_argument('--gpu_id', type=int, default=0)

    # Hyper-parameter
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--patch_size', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)

    parser.add_argument('--mode', choices=["train", "test"])
    parser.add_argument('--seed', type=int, default=2333)

    args = parser.parse_args()

    # load the name list of the data
    def read_names(list_path, split_path):
        if list_path is None:
            split_name = os.path.basename(os.path.normpath(split_path))
            dataset_root = os.path.dirname(os.path.normpath(split_path))
            candidates = [os.path.join(dataset_root, split_name + '.txt')]
            candidates.append(os.path.join(os.path.dirname(dataset_root), split_name + '.txt'))
            list_path = next((candidate for candidate in candidates if os.path.isfile(candidate)), None)
        if list_path:
            with open(list_path, "r", encoding="utf-8") as name_file:
                return [name.strip() for name in name_file if name.strip()]
        label_path = os.path.join(split_path, args.label_dir)
        return sorted(name for name in os.listdir(label_path) if not name.startswith('.'))

    args.train_name_list = read_names(args.train_list_path, args.train_dataset_path)
    args.val_name_list = read_names(args.val_list_path, args.val_dataset_path)
    args.test_name_list = read_names(args.test_list_path, args.test_dataset_path)

    set_seed(args.seed)

    trainer = Trainer(args)
    if args.mode == 'train':
        trainer.training()
    trainer.test()

if __name__ == "__main__":
    main()
