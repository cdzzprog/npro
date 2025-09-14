# import argparse
# import os
# import time
# import platform
# import shutil
# from pathlib import Path
# import cv2
# import torch
# import torch.backends.cudnn as cudnn
# import numpy as np
# from models.experimental import attempt_load
# from utils.datasets import LoadStreams, LoadImages
# from utils.general import (
#     check_img_size, non_max_suppression, apply_classifier, scale_coords,
#     xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
# from utils.torch_utils import select_device, load_classifier, time_synchronized
# def set_parser():
#     parser = argparse.ArgumentParser()
#     # parser.add_argument('--source', type=str, default='/media/zengwb/PC/Dataset/ReID-dataset/channel1/1.mp4',
#     #                     help='source')  # file/folder, 0 for webcam
#     # parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
#     # parser.add_argument('--img-size', type=int, default=960, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
#     parser.add_argument('--view-img', default=True, help='display results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     return parser.parse_args()
# def bbox_r(width, height, *xyxy):
#     """" Calculates the relative bounding box from absolute pixel values. """
#     bbox_left = min([xyxy[0].item(), xyxy[2].item()])
#     bbox_top = min([xyxy[1].item(), xyxy[3].item()])
#     bbox_w = abs(xyxy[0].item() - xyxy[2].item())
#     bbox_h = abs(xyxy[1].item() - xyxy[3].item())
#     x_c = (bbox_left + bbox_w / 2)
#     y_c = (bbox_top + bbox_h / 2)
#     w = bbox_w
#     h = bbox_h
#     return x_c, y_c, w, h
# class Person_detect():
#     def __init__(self, opt, source):
#         # Initialize
#         self.device = opt.device if torch.cuda.is_available() else 'cpu'
#         self.half = self.device != 'cpu'  # half precision only supported on CUDA
#         self.augment = opt.augment
#         self.conf_thres = opt.conf_thres
#         self.iou_thres = opt.iou_thres
#         self.classes = opt.classes
#         self.agnostic_nms = opt.agnostic_nms
#         self.webcam = opt.cam
#         # Load model
#         self.model = attempt_load(opt.weights, map_location=self.device)  # load FP32 model
#         print('111111111111111111111111111111111111111', self.model.stride.max())
#         if self.half:
#             self.model.half()  # to FP16
#         # Get names and colors
#         self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
#         self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
#     def detect(self, path, img, im0s, vid_cap):
#         half = self.device != 'cpu'  # half precision only supported on CUDA
#         # print('444444444444444444444444444444444')
#         # Run inference
#         # print('55555555555555555555555555555')
#         img = torch.from_numpy(img).to(self.device)
#         img = img.half() if half else img.float()  # uint8 to fp16/32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)
#         # Inference
#         t1 = time_synchronized()
#         pred = self.model(img, augment=self.augment)[0]
#         # Apply NMS
#         pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
#                                    agnostic=self.agnostic_nms)
#         # Process detections
#         bbox_xywh = []
#         confs = []
#         clas = []
#         xy = []
#         for i, det in enumerate(pred):  # detections per image
#             # if self.webcam:  # batch_size >= 1
#             #     p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
#             # else:
#             #     p, s, im0 = path, '', im0s
#             if det is not None and len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
#                 # Write results
#                 for *xyxy, conf, cls in reversed(det):
#                     img_h, img_w, _ = im0s.shape  # get image shape
#                     x_c, y_c, bbox_w, bbox_h = bbox_r(img_w, img_h, *xyxy)
#                     obj = [x_c, y_c, bbox_w, bbox_h]
#                     # if cls == opt.classes:  # detct classes id
#                     if not conf.item() > 0.3:
#                         continue
#                     bbox_xywh.append(obj)
#                     confs.append(conf.item())
#                     clas.append(cls.item())
#                     xy.append(xyxy)
#                     # print('jjjjjjjjjjjjjjjjjjjj', confs)
#         return np.array(bbox_xywh), confs, clas, xy
# if __name__ == '__main__':
#     person_detect = Person_detect(source='/media/zengwb/PC/Dataset/ReID-dataset/channel1/1.mp4')
#     with torch.no_grad():
#             person_detect.detect()
# import time
# import cv2
# import numpy as np
# from retinaface import Retinaface
# if __name__ == "__main__":
#     retinaface = Retinaface()
#     #----------------------------------------------------------------------------------------------------------#
#     #   mode用于指定测试的模式：
#     #   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
#     #   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
#     #   'fps'表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
#     #   'dir_predict'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
#     #----------------------------------------------------------------------------------------------------------#
#     mode = "predict"
#     #----------------------------------------------------------------------------------------------------------#
#     #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
#     #   想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
#     #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
#     #   想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
#     #   video_fps用于保存的视频的fps
#     #   video_path、video_save_path和video_fps仅在mode='video'时有效
#     #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
#     #----------------------------------------------------------------------------------------------------------#
#     video_path      = 0
#     video_save_path = ""
#     video_fps       = 25.0
#     #-------------------------------------------------------------------------#
#     #   test_interval用于指定测量fps的时候，图片检测的次数
#     #   理论上test_interval越大，fps越准确。
#     #-------------------------------------------------------------------------#
#     test_interval   = 100
#     #-------------------------------------------------------------------------#
#     #   dir_origin_path指定了用于检测的图片的文件夹路径
#     #   dir_save_path指定了检测完图片的保存路径
#     #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
#     #-------------------------------------------------------------------------#
#     dir_origin_path = "img/"
#     dir_save_path   = "img_out/"
#     if mode == "predict":
#         '''
#         predict.py有几个注意点
#         1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用cv2.imread打开图片文件进行预测。
#         2、如果想要保存，利用cv2.imwrite("img.jpg", r_image)即可保存。
#         3、如果想要获得框的坐标，可以进入detect_image函数，读取(b[0], b[1]), (b[2], b[3])这四个值。
#         4、如果想要截取下目标，可以利用获取到的(b[0], b[1]), (b[2], b[3])这四个值在原图上利用矩阵的方式进行截取。
#         5、在更换facenet网络后一定要重新进行人脸编码，运行encoding.py。
#         '''
#         while True:
#             img = input('Input image filename:')
#             image = cv2.imread(img)
#             if image is None:
#                 print('Open Error! Try again!')
#                 continue
#             else:
#                 image   = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#                 r_image = retinaface.detect_image(image)
#                 r_image = cv2.cvtColor(r_image,cv2.COLOR_RGB2BGR)
#                 cv2.imshow("after",r_image)
#                 cv2.waitKey(0)
#                 save_path = input('Input save path for the result image:')
#                 cv2.imwrite(save_path, r_image)
#                 print(f"Image saved to {save_path}")
#     elif mode == "video":
#         capture = cv2.VideoCapture(video_path)
#         if video_save_path!="":
#             fourcc = cv2.VideoWriter_fourcc(*'XVID')
#             size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#             out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
#         ref, frame = capture.read()
#         if not ref:
#             raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
#         fps = 0.0
#         while(True):
#             t1 = time.time()
#             # 读取某一帧
#             ref, frame = capture.read()
#             if not ref:
#                 break
#             # 格式转变，BGRtoRGB
#             frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#             # 进行检测
#             frame = np.array(retinaface.detect_image(frame))
#             # RGBtoBGR满足opencv显示格式
#             frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
#             fps  = ( fps + (1./(time.time()-t1)) ) / 2
#             print("fps= %.2f"%(fps))
#             frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.imshow("video",frame)
#             c= cv2.waitKey(1) & 0xff 
#             if video_save_path!="":
#                 out.write(frame)
#             if c==27:
#                 capture.release()
#                 break
#         print("Video Detection Done!")
#         capture.release()
#         if video_save_path!="":
#             print("Save processed video to the path :" + video_save_path)
#             out.release()
#         cv2.destroyAllWindows()
#     elif mode == "fps":
#         img = cv2.imread('img/obama.jpg')
#         img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#         tact_time = retinaface.get_FPS(img, test_interval)
#         print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
#     elif mode == "dir_predict":
#         import os
#         from tqdm import tqdm
#         img_names = os.listdir(dir_origin_path)
#         for img_name in tqdm(img_names):
#             if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
#                 image_path  = os.path.join(dir_origin_path, img_name)
#                 image       = cv2.imread(image_path)
#                 image       = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#                 r_image     = retinaface.detect_image(image)
#                 r_image     = cv2.cvtColor(r_image,cv2.COLOR_RGB2BGR)
#                 if not os.path.exists(dir_save_path):
#                     os.makedirs(dir_save_path)
#                 cv2.imwrite(os.path.join(dir_save_path, img_name), r_image)
#     else:
#         raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
# import argparse
# import math
# import os
# import random
# import time
# import logging
# from pathlib import Path
# import numpy as np
# import torch.distributed as dist
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
# import torch.utils.data
# import yaml
# from torch.cuda import amp
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm
# import test  # import test.py to get mAP after each epoch
# from models.yolo import Model
# from utils.datasets import create_dataloader
# from utils.general import (
#     torch_distributed_zero_first, labels_to_class_weights, plot_labels, check_anchors, labels_to_image_weights,
#     compute_loss, plot_images, fitness, strip_optimizer, plot_results, get_latest_run, check_dataset, check_file,
#     check_git_status, check_img_size, increment_dir, print_mutation, plot_evolution, set_logging)
# from utils.google_utils import attempt_download
# from utils.torch_utils import init_seeds, ModelEMA, select_device, intersect_dicts
# logger = logging.getLogger(__name__)
# def train(hyp, opt, device, tb_writer=None):
#     logger.info(f'Hyperparameters {hyp}')
#     log_dir = Path(tb_writer.log_dir) if tb_writer else Path(opt.logdir) / 'evolve'  # logging directory
#     wdir = str(log_dir / 'weights') + os.sep  # weights directory
#     os.makedirs(wdir, exist_ok=True)
#     last = wdir + 'last.pt'
#     best = wdir + 'best.pt'
#     results_file = str(log_dir / 'results.txt')
#     epochs, batch_size, total_batch_size, weights, rank = \
#         opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank
#     # TODO: Use DDP logging. Only the first process is allowed to log.
#     # Save run settings
#     with open(log_dir / 'hyp.yaml', 'w') as f:
#         yaml.dump(hyp, f, sort_keys=False)
#     with open(log_dir / 'opt.yaml', 'w') as f:
#         yaml.dump(vars(opt), f, sort_keys=False)
#     # Configure
#     cuda = device.type != 'cpu'
#     init_seeds(2 + rank)
#     with open(opt.data) as f:
#         data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
#     with torch_distributed_zero_first(rank):
#         check_dataset(data_dict)  # check
#     train_path = data_dict['train']
#     test_path = data_dict['val']
#     nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
#     assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check
#     # Model
#     pretrained = weights.endswith('.pt')
#     if pretrained:
#         with torch_distributed_zero_first(rank):
#             attempt_download(weights)  # download if not found locally
#         ckpt = torch.load(weights, map_location=device)  # load checkpoint
#         model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)  # create
#         exclude = ['anchor'] if opt.cfg else []  # exclude keys
#         state_dict = ckpt['model'].float().state_dict()  # to FP32
#         state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
#         model.load_state_dict(state_dict, strict=False)  # load
#         logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
#     else:
#         model = Model(opt.cfg, ch=3, nc=nc).to(device)  # create
#     # Freeze
#     freeze = ['', ]  # parameter names to freeze (full or partial)
#     if any(freeze):
#         for k, v in model.named_parameters():
#             if any(x in k for x in freeze):
#                 print('freezing %s' % k)
#                 v.requires_grad = False
#     # Optimizer
#     nbs = 64  # nominal batch size
#     accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
#     hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
#     pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
#     for k, v in model.named_parameters():
#         v.requires_grad = True
#         if '.bias' in k:
#             pg2.append(v)  # biases
#         elif '.weight' in k and '.bn' not in k:
#             pg1.append(v)  # apply weight decay
#         else:
#             pg0.append(v)  # all else
#     if opt.adam:
#         optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
#     else:
#         optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
#     optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
#     optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
#     logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
#     del pg0, pg1, pg2
#     # Scheduler https://arxiv.org/pdf/1812.01187.pdf
#     # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
#     lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
#     scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
#     # plot_lr_scheduler(optimizer, scheduler, epochs)
#     # Resume
#     start_epoch, best_fitness = 0, 0.0
#     if pretrained:
#         # Optimizer
#         if ckpt['optimizer'] is not None:
#             optimizer.load_state_dict(ckpt['optimizer'])
#             best_fitness = ckpt['best_fitness']
#         # Results
#         if ckpt.get('training_results') is not None:
#             with open(results_file, 'w') as file:
#                 file.write(ckpt['training_results'])  # write results.txt
#         # Epochs
#         start_epoch = ckpt['epoch'] + 1
#         if epochs < start_epoch:
#             logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
#                   (weights, ckpt['epoch'], epochs))
#             epochs += ckpt['epoch']  # finetune additional epochs
#         del ckpt, state_dict
#     # Image sizes
#     gs = int(max(model.stride))  # grid size (max stride)
#     imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
#     # DP mode
#     if cuda and rank == -1 and torch.cuda.device_count() > 1:
#         model = torch.nn.DataParallel(model)
#     # SyncBatchNorm
#     if opt.sync_bn and cuda and rank != -1:
#         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
#         logger.info('Using SyncBatchNorm()')
#     # Exponential moving average
#     ema = ModelEMA(model) if rank in [-1, 0] else None
#     # DDP mode
#     if cuda and rank != -1:
#         model = DDP(model, device_ids=[opt.local_rank], output_device=(opt.local_rank))
#     # Trainloader
#     dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt, hyp=hyp, augment=True,
#                                             cache=opt.cache_images, rect=opt.rect, rank=rank,
#                                             world_size=opt.world_size, workers=opt.workers)
#     mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
#     nb = len(dataloader)  # number of batches
#     assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)
#     # Testloader
#     if rank in [-1, 0]:
#         # local_rank is set to -1. Because only the first process is expected to do evaluation.
#         testloader = create_dataloader(test_path, imgsz_test, total_batch_size, gs, opt, hyp=hyp, augment=False,
#                                        cache=opt.cache_images, rect=True, rank=-1, world_size=opt.world_size,
#                                        workers=opt.workers)[0]
#     # Model parameters
#     hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
#     model.nc = nc  # attach number of classes to model
#     model.hyp = hyp  # attach hyperparameters to model
#     model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
#     model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
#     model.names = names
#     # Class frequency
#     if rank in [-1, 0]:
#         labels = np.concatenate(dataset.labels, 0)
#         c = torch.tensor(labels[:, 0])  # classes
#         # cf = torch.bincount(c.long(), minlength=nc) + 1.
#         # model._initialize_biases(cf.to(device))
#         plot_labels(labels, save_dir=log_dir)
#         if tb_writer:
#             # tb_writer.add_hparams(hyp, {})  # causes duplicate https://github.com/ultralytics/yolov5/pull/384
#             tb_writer.add_histogram('classes', c, 0)
#         # Check anchors
#         if not opt.noautoanchor:
#             check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
#     # Start training
#     t0 = time.time()
#     nw = max(3 * nb, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
#     # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
#     maps = np.zeros(nc)  # mAP per class
#     results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
#     scheduler.last_epoch = start_epoch - 1  # do not move
#     scaler = amp.GradScaler(enabled=cuda)
#     logger.info('Image sizes %g train, %g test' % (imgsz, imgsz_test))
#     logger.info('Using %g dataloader workers' % dataloader.num_workers)
#     logger.info('Starting training for %g epochs...' % epochs)
#     # torch.autograd.set_detect_anomaly(True)
#     for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
#         model.train()
#         # Update image weights (optional)
#         if dataset.image_weights:
#             # Generate indices
#             if rank in [-1, 0]:
#                 w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
#                 image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
#                 dataset.indices = random.choices(range(dataset.n), weights=image_weights,
#                                                  k=dataset.n)  # rand weighted idx
#             # Broadcast if DDP
#             if rank != -1:
#                 indices = torch.zeros([dataset.n], dtype=torch.int)
#                 if rank == 0:
#                     indices[:] = torch.from_tensor(dataset.indices, dtype=torch.int)
#                 dist.broadcast(indices, 0)
#                 if rank != 0:
#                     dataset.indices = indices.cpu().numpy()
#         # Update mosaic border
#         # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
#         # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders
#         mloss = torch.zeros(4, device=device)  # mean losses
#         if rank != -1:
#             dataloader.sampler.set_epoch(epoch)
#         pbar = enumerate(dataloader)
#         logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
#         if rank in [-1, 0]:
#             pbar = tqdm(pbar, total=nb)  # progress bar
#         optimizer.zero_grad()
#         for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
#             ni = i + nb * epoch  # number integrated batches (since train start)
#             imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
#             # Warmup
#             if ni <= nw:
#                 xi = [0, nw]  # x interp
#                 # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
#                 accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
#                 for j, x in enumerate(optimizer.param_groups):
#                     # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
#                     x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
#                     if 'momentum' in x:
#                         x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])
#             # Multi-scale
#             if opt.multi_scale:
#                 sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
#                 sf = sz / max(imgs.shape[2:])  # scale factor
#                 if sf != 1:
#                     ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
#                     imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
#             # Autocast
#             with amp.autocast(enabled=cuda):
#                 # Forward
#                 pred = model(imgs)
#                 # Loss
#                 loss, loss_items = compute_loss(pred, targets.to(device), model)  # scaled by batch_size
#                 if rank != -1:
#                     loss *= opt.world_size  # gradient averaged between devices in DDP mode
#                 # if not torch.isfinite(loss):
#                 #     logger.info('WARNING: non-finite loss, ending training ', loss_items)
#                 #     return results
#             # Backward
#             scaler.scale(loss).backward()
#             # Optimize
#             if ni % accumulate == 0:
#                 scaler.step(optimizer)  # optimizer.step
#                 scaler.update()
#                 optimizer.zero_grad()
#                 if ema is not None:
#                     ema.update(model)
#             # Print
#             if rank in [-1, 0]:
#                 mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
#                 mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
#                 s = ('%10s' * 2 + '%10.4g' * 6) % (
#                     '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
#                 pbar.set_description(s)
#                 # Plot
#                 if ni < 3:
#                     f = str(log_dir / ('train_batch%g.jpg' % ni))  # filename
#                     result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
#                     if tb_writer and result is not None:
#                         tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
#                         # tb_writer.add_graph(model, imgs)  # add model to tensorboard
#             # end batch ------------------------------------------------------------------------------------------------
#         # Scheduler
#         scheduler.step()
#         # DDP process 0 or single-GPU
#         if rank in [-1, 0]:
#             # mAP
#             if ema is not None:
#                 ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
#             final_epoch = epoch + 1 == epochs
#             if not opt.notest or final_epoch:  # Calculate mAP
#                 results, maps, times = test.test(opt.data,
#                                                  batch_size=total_batch_size,
#                                                  imgsz=imgsz_test,
#                                                  model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
#                                                  single_cls=opt.single_cls,
#                                                  dataloader=testloader,
#                                                  save_dir=log_dir)
#             # Write
#             with open(results_file, 'a') as f:
#                 f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
#             if len(opt.name) and opt.bucket:
#                 os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))
#             # Tensorboard
#             if tb_writer:
#                 tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
#                         'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
#                         'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
#                 for x, tag in zip(list(mloss[:-1]) + list(results), tags):
#                     tb_writer.add_scalar(tag, x, epoch)
#             # Update best mAP
#             fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
#             if fi > best_fitness:
#                 best_fitness = fi
#             # Save model
#             save = (not opt.nosave) or (final_epoch and not opt.evolve)
#             if save:
#                 with open(results_file, 'r') as f:  # create checkpoint
#                     ckpt = {'epoch': epoch,
#                             'best_fitness': best_fitness,
#                             'training_results': f.read(),
#                             'model': ema.ema.module if hasattr(ema, 'module') else ema.ema,
#                             'optimizer': None if final_epoch else optimizer.state_dict()}
#                 # Save last, best and delete
#                 torch.save(ckpt, last)
#                 if best_fitness == fi:
#                     torch.save(ckpt, best)
#                 del ckpt
#         # end epoch ----------------------------------------------------------------------------------------------------
#     # end training
#     if rank in [-1, 0]:
#         # Strip optimizers
#         n = ('_' if len(opt.name) and not opt.name.isnumeric() else '') + opt.name
#         fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
#         for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
#             if os.path.exists(f1):
#                 os.rename(f1, f2)  # rename
#                 ispt = f2.endswith('.pt')  # is *.pt
#                 strip_optimizer(f2) if ispt else None  # strip optimizer
#                 os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload
#         # Finish
#         if not opt.evolve:
#             plot_results(save_dir=log_dir)  # save as results.png
#         logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
#     dist.destroy_process_group() if rank not in [-1, 0] else None
#     torch.cuda.empty_cache()
#     return results
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
#     parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
#     parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
#     parser.add_argument('--hyp', type=str, default='', help='hyperparameters path, i.e. data/hyp.scratch.yaml')
#     parser.add_argument('--epochs', type=int, default=300)
#     parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
#     parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
#     parser.add_argument('--rect', action='store_true', help='rectangular training')
#     parser.add_argument('--resume', nargs='?', const='get_last', default=False,
#                         help='resume from given path/last.pt, or most recent run if blank')
#     parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
#     parser.add_argument('--notest', action='store_true', help='only test final epoch')
#     parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
#     parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
#     parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
#     parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
#     parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
#     parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
#     parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
#     parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
#     parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
#     parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
#     parser.add_argument('--workers', type=int, default=8, help='maximum number of workers for dataloader')
#     opt = parser.parse_args()
#     # Set DDP variables
#     opt.total_batch_size = opt.batch_size
#     opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
#     opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
#     set_logging(opt.global_rank)
#     # Resume
#     if opt.resume:
#         last = get_latest_run() if opt.resume == 'get_last' else opt.resume  # resume from most recent run
#         if last and not opt.weights:
#             logger.info(f'Resuming training from {last}')
#         opt.weights = last if opt.resume and not opt.weights else opt.weights
#     if opt.global_rank in [-1,0]:
#         check_git_status()
#     opt.hyp = opt.hyp or ('data/hyp.finetune.yaml' if opt.weights else 'data/hyp.scratch.yaml')
#     opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
#     assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
#     opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
#     device = select_device(opt.device, batch_size=opt.batch_size)
#     # DDP mode
#     if opt.local_rank != -1:
#         assert torch.cuda.device_count() > opt.local_rank
#         torch.cuda.set_device(opt.local_rank)
#         device = torch.device('cuda', opt.local_rank)
#         dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
#         assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
#         opt.batch_size = opt.total_batch_size // opt.world_size
#     logger.info(opt)
#     with open(opt.hyp) as f:
#         hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
#     # Train
#     if not opt.evolve:
#         tb_writer = None
#         if opt.global_rank in [-1, 0]:
#             logger.info('Start Tensorboard with "tensorboard --logdir %s", view at http://localhost:6006/' % opt.logdir)
#             tb_writer = SummaryWriter(log_dir=increment_dir(Path(opt.logdir) / 'exp', opt.name))  # runs/exp
#         train(hyp, opt, device, tb_writer)
#     # Evolve hyperparameters (optional)
#     else:
#         # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
#         meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
#                 'momentum': (0.1, 0.6, 0.98),  # SGD momentum/Adam beta1
#                 'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
#                 'giou': (1, 0.02, 0.2),  # GIoU loss gain
#                 'cls': (1, 0.2, 4.0),  # cls loss gain
#                 'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
#                 'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
#                 'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
#                 'iou_t': (0, 0.1, 0.7),  # IoU training threshold
#                 'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
#                 'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
#                 'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
#                 'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
#                 'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
#                 'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
#                 'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
#                 'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
#                 'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
#                 'perspective': (1, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
#                 'flipud': (0, 0.0, 1.0),  # image flip up-down (probability)
#                 'fliplr': (1, 0.0, 1.0),  # image flip left-right (probability)
#                 'mixup': (1, 0.0, 1.0)}  # image mixup (probability)
#         assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
#         opt.notest, opt.nosave = True, True  # only test/save final epoch
#         # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
#         yaml_file = Path('runs/evolve/hyp_evolved.yaml')  # save best result here
#         if opt.bucket:
#             os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists
#         for _ in range(100):  # generations to evolve
#             if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
#                 # Select parent(s)
#                 parent = 'single'  # parent selection method: 'single' or 'weighted'
#                 x = np.loadtxt('evolve.txt', ndmin=2)
#                 n = min(5, len(x))  # number of previous results to consider
#                 x = x[np.argsort(-fitness(x))][:n]  # top n mutations
#                 w = fitness(x) - fitness(x).min()  # weights
#                 if parent == 'single' or len(x) == 1:
#                     # x = x[random.randint(0, n - 1)]  # random selection
#                     x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
#                 elif parent == 'weighted':
#                     x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination
#                 # Mutate
#                 mp, s = 0.9, 0.2  # mutation probability, sigma
#                 npr = np.random
#                 npr.seed(int(time.time()))
#                 g = np.array([x[0] for x in meta.values()])  # gains 0-1
#                 ng = len(meta)
#                 v = np.ones(ng)
#                 while all(v == 1):  # mutate until a change occurs (prevent duplicates)
#                     v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
#                 for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
#                     hyp[k] = float(x[i + 7] * v[i])  # mutate
#             # Constrain to limits
#             for k, v in meta.items():
#                 hyp[k] = max(hyp[k], v[1])  # lower limit
#                 hyp[k] = min(hyp[k], v[2])  # upper limit
#                 hyp[k] = round(hyp[k], 5)  # significant digits
#             # Train mutation
#             results = train(hyp.copy(), opt, device)
#             # Write mutation results
#             print_mutation(hyp.copy(), results, yaml_file, opt.bucket)
#         # Plot results
#         plot_evolution(yaml_file)
#         print('Hyperparameter evolution complete. Best results saved as: %s\nCommand to train a new model with these '
#               'hyperparameters: $ python train.py --hyp %s' % (yaml_file, yaml_file))
# import argparse
# import glob
# import json
# import os
# import shutil
# from pathlib import Path
# import numpy as np
# import torch
# import yaml
# from tqdm import tqdm
# from models.experimental import attempt_load
# from utils.datasets import create_dataloader
# from utils.general import (
#     coco80_to_coco91_class, check_dataset, check_file, check_img_size, compute_loss, non_max_suppression, scale_coords, 
#     xyxy2xywh, clip_coords, plot_images, xywh2xyxy, box_iou, output_to_target, ap_per_class, set_logging)
# from utils.torch_utils import select_device, time_synchronized
# def test(data,
#          weights=None,
#          batch_size=16,
#          imgsz=640,
#          conf_thres=0.001,
#          iou_thres=0.6,  # for NMS
#          save_json=False,
#          single_cls=False,
#          augment=False,
#          verbose=False,
#          model=None,
#          dataloader=None,
#          save_dir='',
#          merge=False,
#          save_txt=False):
#     # Initialize/load model and set device
#     training = model is not None
#     if training:  # called by train.py
#         device = next(model.parameters()).device  # get model device
#     else:  # called directly
#         set_logging()
#         device = select_device(opt.device, batch_size=batch_size)
#         merge, save_txt = opt.merge, opt.save_txt  # use Merge NMS, save *.txt labels
#         if save_txt:
#             out = Path('inference/output')
#             if os.path.exists(out):
#                 shutil.rmtree(out)  # delete output folder
#             os.makedirs(out)  # make new output folder
#         # Remove previous
#         for f in glob.glob(str(Path(save_dir) / 'test_batch*.jpg')):
#             os.remove(f)
#         # Load model
#         model = attempt_load(weights, map_location=device)  # load FP32 model
#         imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
#         # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
#         # if device.type != 'cpu' and torch.cuda.device_count() > 1:
#         #     model = nn.DataParallel(model)
#     # Half
#     half = device.type != 'cpu'  # half precision only supported on CUDA
#     if half:
#         model.half()
#     # Configure
#     model.eval()
#     with open(data) as f:
#         data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
#     check_dataset(data)  # check
#     nc = 1 if single_cls else int(data['nc'])  # number of classes
#     iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
#     niou = iouv.numel()
#     # Dataloader
#     if not training:
#         img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
#         _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
#         path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
#         dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt,
#                                        hyp=None, augment=False, cache=False, pad=0.5, rect=True)[0]
#     seen = 0
#     names = model.names if hasattr(model, 'names') else model.module.names
#     coco91class = coco80_to_coco91_class()
#     s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
#     p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
#     loss = torch.zeros(3, device=device)
#     jdict, stats, ap, ap_class = [], [], [], []
#     for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
#         img = img.to(device, non_blocking=True)
#         img = img.half() if half else img.float()  # uint8 to fp16/32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         targets = targets.to(device)
#         nb, _, height, width = img.shape  # batch size, channels, height, width
#         whwh = torch.Tensor([width, height, width, height]).to(device)
#         # Disable gradients
#         with torch.no_grad():
#             # Run model
#             t = time_synchronized()
#             inf_out, train_out = model(img, augment=augment)  # inference and training outputs
#             t0 += time_synchronized() - t
#             # Compute loss
#             if training:  # if model has loss hyperparameters
#                 loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # GIoU, obj, cls
#             # Run NMS
#             t = time_synchronized()
#             output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, merge=merge)
#             t1 += time_synchronized() - t
#         # Statistics per image
#         for si, pred in enumerate(output):
#             labels = targets[targets[:, 0] == si, 1:]
#             nl = len(labels)
#             tcls = labels[:, 0].tolist() if nl else []  # target class
#             seen += 1
#             if pred is None:
#                 if nl:
#                     stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
#                 continue
#             # Append to text file
#             if save_txt:
#                 gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
#                 txt_path = str(out / Path(paths[si]).stem)
#                 pred[:, :4] = scale_coords(img[si].shape[1:], pred[:, :4], shapes[si][0], shapes[si][1])  # to original
#                 for *xyxy, conf, cls in pred:
#                     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                     with open(txt_path + '.txt', 'a') as f:
#                         f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
#             # Clip boxes to image bounds
#             clip_coords(pred, (height, width))
#             # Append to pycocotools JSON dictionary
#             if save_json:
#                 # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
#                 image_id = Path(paths[si]).stem
#                 box = pred[:, :4].clone()  # xyxy
#                 scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
#                 box = xyxy2xywh(box)  # xywh
#                 box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
#                 for p, b in zip(pred.tolist(), box.tolist()):
#                     jdict.append({'image_id': int(image_id) if image_id.isnumeric() else image_id,
#                                   'category_id': coco91class[int(p[5])],
#                                   'bbox': [round(x, 3) for x in b],
#                                   'score': round(p[4], 5)})
#             # Assign all predictions as incorrect
#             correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
#             if nl:
#                 detected = []  # target indices
#                 tcls_tensor = labels[:, 0]
#                 # target boxes
#                 tbox = xywh2xyxy(labels[:, 1:5]) * whwh
#                 # Per target class
#                 for cls in torch.unique(tcls_tensor):
#                     ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
#                     pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices
#                     # Search for detections
#                     if pi.shape[0]:
#                         # Prediction to target ious
#                         ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices
#                         # Append detections
#                         for j in (ious > iouv[0]).nonzero(as_tuple=False):
#                             d = ti[i[j]]  # detected target
#                             if d not in detected:
#                                 detected.append(d)
#                                 correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
#                                 if len(detected) == nl:  # all targets already located in image
#                                     break
#             # Append statistics (correct, conf, pcls, tcls)
#             stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
#         # Plot images
#         if batch_i < 1:
#             f = Path(save_dir) / ('test_batch%g_gt.jpg' % batch_i)  # filename
#             plot_images(img, targets, paths, str(f), names)  # ground truth
#             f = Path(save_dir) / ('test_batch%g_pred.jpg' % batch_i)
#             plot_images(img, output_to_target(output, width, height), paths, str(f), names)  # predictions
#     # Compute statistics
#     stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
#     if len(stats) and stats[0].any():
#         p, r, ap, f1, ap_class = ap_per_class(*stats)
#         p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
#         mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
#         nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
#     else:
#         nt = torch.zeros(1)
#     # Print results
#     pf = '%20s' + '%12.3g' * 6  # print format
#     print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
#     # Print results per class
#     if verbose and nc > 1 and len(stats):
#         for i, c in enumerate(ap_class):
#             print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
#     # Print speeds
#     t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
#     if not training:
#         print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)
#     # Save JSON
#     if save_json and len(jdict):
#         f = 'detections_val2017_%s_results.json' % \
#             (weights.split(os.sep)[-1].replace('.pt', '') if isinstance(weights, str) else '')  # filename
#         print('\nCOCO mAP with pycocotools... saving %s...' % f)
#         with open(f, 'w') as file:
#             json.dump(jdict, file)
#         try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
#             from pycocotools.coco import COCO
#             from pycocotools.cocoeval import COCOeval
#             imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]
#             cocoGt = COCO(glob.glob('../coco/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
#             cocoDt = cocoGt.loadRes(f)  # initialize COCO pred api
#             cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
#             cocoEval.params.imgIds = imgIds  # image IDs to evaluate
#             cocoEval.evaluate()
#             cocoEval.accumulate()
#             cocoEval.summarize()
#             map, map50 = cocoEval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
#         except Exception as e:
#             print('ERROR: pycocotools unable to run: %s' % e)
#     # Return results
#     model.float()  # for training
#     maps = np.zeros(nc) + map
#     for i, c in enumerate(ap_class):
#         maps[c] = ap[i]
#     return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(prog='test.py')
#     parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
#     parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
#     parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
#     parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
#     parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
#     parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--merge', action='store_true', help='use Merge NMS')
#     parser.add_argument('--verbose', action='store_true', help='report mAP by class')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     opt = parser.parse_args()
#     opt.save_json |= opt.data.endswith('coco.yaml')
#     opt.data = check_file(opt.data)  # check file
#     print(opt)
#     if opt.task in ['val', 'test']:  # run normally
#         test(opt.data,
#              opt.weights,
#              opt.batch_size,
#              opt.img_size,
#              opt.conf_thres,
#              opt.iou_thres,
#              opt.save_json,
#              opt.single_cls,
#              opt.augment,
#              opt.verbose)
#     elif opt.task == 'study':  # run over a range of settings and save/plot
#         for weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
#             f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem)  # filename to save to
#             x = list(range(352, 832, 64))  # x axis
#             y = []  # y axis
#             for i in x:  # img-size
#                 print('\nRunning %s point %s...' % (f, i))
#                 r, _, t = test(opt.data, weights, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json)
#                 y.append(r + t)  # results and times
#             np.savetxt(f, y, fmt='%10.4g')  # save
#         os.system('zip -r study.zip study_*.txt')
#         # plot_study_txt(f, x)  # plot
# import base64
# import os
# import time
# import datetime
# from urllib.parse import urlparse
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# from PIL import Image, ImageDraw, ImageFont
# from tqdm import tqdm
# from demo_face.backbone.model_irse import IR_50
# from demo_face.nets_retinaface.retinaface import RetinaFace
# from demo_face.util.anchors import Anchors
# from demo_face.util.config import cfg_mnet, cfg_re50
# from demo_face.util.utils2 import (Alignment_1, compare_faces, letterbox_image,
#                                    preprocess_input)
# from demo_face.util.utils_bbox import (decode, decode_landm, non_max_suppression,
#                                        retinaface_correct_boxes)
# import requests
# import json
# import urllib.parse
# # from settings import user_data
# # from torch2trt import TRTModule
# def cv2ImgAddText(img, label, left, top, textColor=(255, 255, 255)):
#     img = Image.fromarray(np.uint8(img))
#     # ---------------#
#     #   设置字体
#     # ---------------#
#     font = ImageFont.truetype(font='model/simhei.ttf', size=20)
#     draw = ImageDraw.Draw(img)
#     label = label.encode('utf-8')
#     draw.text((left, top), str(label, 'UTF-8'), fill=textColor, font=font)
#     return np.asarray(img)
# #  注意backbone和model_path的对应
# class Retinaface(object):
#     _defaults = {
#         # ----------------------------------------------------------------------#
#         #   retinaface训练完的权值路径
#         # ----------------------------------------------------------------------#
#         "retinaface_model_path": 'D:/runcode/Yolov5-Deepsort-Fastreid-main/demo_face/model/Retinaface_resnet50.pth',
#         # ----------------------------------------------------------------------#
#         #   retinaface所使用的主干网络，有mobilenet和resnet50
#         # ----------------------------------------------------------------------#
#         "retinaface_backbone": "resnet50",
#         # ----------------------------------------------------------------------#
#         #   retinaface中只有得分大于置信度的预测框会被保留下来
#         # ----------------------------------------------------------------------#
#         "confidence": 0.25,
#         # ----------------------------------------------------------------------#
#         #   retinaface中非极大抑制所用到的nms_iou大小
#         # ----------------------------------------------------------------------#
#         "nms_iou": 0.4,
#         # ----------------------------------------------------------------------#
#         #   是否需要进行图像大小限制。
#         #   输入图像大小会大幅度地影响FPS，想加快检测速度可以减少input_shape。
#         #   开启后，会将输入图像的大小限制为input_shape。否则使用原图进行预测。
#         #   会导致检测结果偏差，主干为resnet50不存在此问题。
#         #   可根据输入图像的大小自行调整input_shape，注意为32的倍数，如[640, 640, 3]
#         # ----------------------------------------------------------------------#
#         "retinaface_input_shape": [640, 640, 3],
#         # ----------------------------------------------------------------------#
#         #   是否需要进行图像大小限制。
#         # ----------------------------------------------------------------------#
#         "letterbox_image": True,
#         # ----------------------------------------------------------------------#
#         #   训练完的权值路径
#         # ----------------------------------------------------------------------#
#         "Re_model_path": 'D:/runcode/Yolov5-Deepsort-Fastreid-main/demo_face/model/backbone_ir50_asia.pth',
#         # ----------------------------------------------------------------------#
#         #   使用的主干网络
#         # ----------------------------------------------------------------------#
#         "backbone": "Ir_50",
#         # ----------------------------------------------------------------------#
#         #  输入图片大小
#         # ----------------------------------------------------------------------#
#         "Re_input_shape": [112, 112],
#         # ----------------------------------------------------------------------#
#         #   阈值
#         # ----------------------------------------------------------------------#
#         "threhold": 70,
#         # --------------------------------#
#         #   是否使用Cuda
#         #   没有GPU可以设置成False
#         # --------------------------------#
#         "cuda": True
#     }
#     @classmethod
#     def get_defaults(cls, n):
#         if n in cls._defaults:
#             return cls._defaults[n]
#         else:
#             return "Unrecognized attribute name '" + n + "'"
#     # ---------------------------------------------------#
#     #   初始化Retinaface
#     # ---------------------------------------------------#
#     def __init__(self, encoding=0, **kwargs):
#         self.__dict__.update(self._defaults)
#         for name, value in kwargs.items():
#             setattr(self, name, value)
#         self.net = None
#         self.model = None
#         # ---------------------------------------------------#
#         #   不同主干网络的config信息
#         # ---------------------------------------------------#
#         if self.retinaface_backbone == "resnet50":
#             self.cfg = cfg_re50
#         else:
#             self.cfg = cfg_mnet
#         # ---------------------------------------------------#
#         #   先验框的生成
#         # ---------------------------------------------------#
#         self.anchors = Anchors(self.cfg, image_size=(
#             self.retinaface_input_shape[0], self.retinaface_input_shape[1])).get_anchors()
#         # trt #
#         # trt_retinaface_model_path = '/home/admin2/PycharmProjects/pythonProject/demo_face/model/retinaface_model_trt.pth'
#         # trt_ir_50_model_path = '/home/admin2/PycharmProjects/pythonProject/demo_face/model/ir_50_model_trt.pth'
#         trt_retinaface_model_path = ''
#         trt_ir_50_model_path = ''
#         self.trt_retinaface_model_path = trt_retinaface_model_path
#         self.trt_ir_50_model_path = trt_ir_50_model_path
#     def load_face_features(self, encoding):
#         # try:
#         #     self.known_face_encodings = np.load(
#         #         f"demo_face/model/{self.backbone}_face_encoding.npy")
#         #     self.known_face_names = np.load(
#         #         f"demo_face/model/{self.backbone}_names.npy")
#         # except:
#         #     if not encoding:
#         #         print("载入已有人脸特征失败，请检查model下面是否生成了相关的人脸特征文件。")
#         try:
#             self.known_face_encodings = np.load(f"demo_face/model/{self.backbone}_face_encoding.npy")
#             self.known_face_names = np.load(f"demo_face/model/{self.backbone}_names.npy")
#         except:
#             if not encoding:
#                 print("载入已有人脸特征失败，请检查model下面是否生成了相关的人脸特征文件。")
#         else:
#             if not hasattr(self, 'known_face_encodings') or self.known_face_encodings is None:
#                 self.known_face_encodings = np.load(f"demo_face/model/{self.backbone}_face_encoding.npy")
#                 self.known_face_names = np.load(f"demo_face/model/{self.backbone}_names.npy")
#     # ---------------------------------------------------#
#     #   获得所有的分类
#     # ---------------------------------------------------#
#     def load_detection_model(self):
#         # -------------------------------#
#         #   载入检测模型与权值
#         # -------------------------------#
#         device = torch.device('cuda' if self.cuda else 'cpu')
#         if self.trt_retinaface_model_path:
#             self.net = TRTModule()
#             self.net.load_state_dict(torch.load(self.trt_retinaface_model_path, map_location=device))
#         else:
#             self.net = RetinaFace(cfg=self.cfg, phase='eval', pre_train=False).eval()
#             state_dict = torch.load(self.retinaface_model_path, map_location=device)
#             self.net.load_state_dict(state_dict)
#             if self.cuda:
#                 self.net = nn.DataParallel(self.net)
#                 self.net = self.net.cuda()
#         print('检测模型加载完成!')
#     def load_comparison_model(self):
#         # -------------------------------#
#         #   载入比对模型与权值
#         # -------------------------------#
#         device = torch.device('cuda' if self.cuda else 'cpu')
#         if self.trt_ir_50_model_path:
#             self.model = TRTModule()
#             self.model.load_state_dict(torch.load(self.trt_ir_50_model_path, map_location=device))
#             print('1')
#         else:
#             self.model = IR_50([112, 112]).eval()
#             state_dict = torch.load(self.Re_model_path, map_location=device)
#             self.model.load_state_dict(state_dict, strict=False)
#             if self.cuda:
#                 if torch.cuda.is_available():
#                     print("CUDA is available!")
#                 else:
#                     print("CUDA is not available.")
#                 self.model = nn.DataParallel(self.model)
#                 self.model = self.model.cuda()
#         print('比对模型加载完成!')
#     def encode_face_dataset(self, image_paths, names):
#         if self.net is None:
#             self.load_detection_model()
#         if self.model is None:
#             self.load_comparison_model()
#         encoding_file = 'model/{backbone}_face_encoding.npy'.format(backbone=self.backbone)
#         names_file = 'model/{backbone}_names.npy'.format(backbone=self.backbone)
#         if os.path.isfile(encoding_file) and os.path.isfile(names_file):
#             face_encodings = np.load(encoding_file, allow_pickle=True).tolist()
#             known_names = np.load(names_file, allow_pickle=True).tolist()
#             for name in names:
#                 known_names.append(name)
#         else:
#             face_encodings = []
#             known_names = names
#         for index, path in enumerate(tqdm(image_paths)):
#             # ---------------------------------------------------#
#             #   打开人脸图片
#             # ---------------------------------------------------#
#             image = np.array(Image.open(path), np.float32)
#             # 转换成3通道
#             image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
#             # print(path)
#             # ---------------------------------------------------#
#             #   对输入图像进行一个备份
#             # ---------------------------------------------------#
#             old_image = image.copy()
#             # ---------------------------------------------------#
#             #   计算输入图片的高和宽
#             # ---------------------------------------------------#
#             im_height, im_width, _ = np.shape(image)
#             # ---------------------------------------------------#
#             #   计算scale，用于将获得的预测框转换成原图的高宽
#             # ---------------------------------------------------#
#             scale = [
#                 np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
#             ]
#             scale_for_landmarks = [
#                 np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
#                 np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
#                 np.shape(image)[1], np.shape(image)[0]
#             ]
#             if self.letterbox_image:
#                 image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
#                 anchors = self.anchors
#             else:
#                 anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
#             # ---------------------------------------------------#
#             #   将处理完的图片传入Retinaface网络当中进行预测
#             # ---------------------------------------------------#
#             with torch.no_grad():
#                 # -----------------------------------------------------------#
#                 #   图片预处理，归一化。
#                 # -----------------------------------------------------------#
#                 image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(
#                     torch.FloatTensor)
#                 if self.cuda:
#                     image = image.cuda()
#                     anchors = anchors.cuda()
#                 loc, conf, landms = self.net(image)
#                 # -----------------------------------------------------------#
#                 #   对预测框进行解码
#                 # -----------------------------------------------------------#
#                 boxes = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])
#                 # -----------------------------------------------------------#
#                 #   获得预测结果的置信度
#                 # -----------------------------------------------------------#
#                 conf = conf.data.squeeze(0)[:, 1:2]
#                 # -----------------------------------------------------------#
#                 #   对人脸关键点进行解码
#                 # -----------------------------------------------------------#
#                 landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])
#                 # -----------------------------------------------------------#
#                 #   对人脸检测结果进行堆叠
#                 # -----------------------------------------------------------#
#                 boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
#                 boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
#                 if len(boxes_conf_landms) <= 0:
#                     print(known_names[index], "：未检测到人脸")
#                     continue
#                 # ---------------------------------------------------------#
#                 #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
#                 # ---------------------------------------------------------#
#                 if self.letterbox_image:
#                     boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
#                                                                  np.array([self.retinaface_input_shape[0],
#                                                                            self.retinaface_input_shape[1]]),
#                                                                  np.array([im_height, im_width]))
#             boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
#             boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks
#             # ---------------------------------------------------#
#             #   选取最大的人脸框。
#             # ---------------------------------------------------#
#             best_face_location = None
#             biggest_area = 0
#             for result in boxes_conf_landms:
#                 left, top, right, bottom = result[0:4]
#                 w = right - left
#                 h = bottom - top
#                 if w * h > biggest_area:
#                     biggest_area = w * h
#                     best_face_location = result
#             # ---------------------------------------------------#
#             #   截取图像
#             # ---------------------------------------------------#
#             crop_img = old_image[int(best_face_location[1]):int(best_face_location[3]),
#                        int(best_face_location[0]):int(best_face_location[2])]
#             landmark = np.reshape(best_face_location[5:], (5, 2)) - np.array(
#                 [int(best_face_location[0]), int(best_face_location[1])])
#             crop_img, _ = Alignment_1(crop_img, landmark)
#             crop_img = np.array(
#                 letterbox_image(np.uint8(crop_img), (self.Re_input_shape[1], self.Re_input_shape[0]))) / 255
#             crop_img = crop_img.transpose(2, 0, 1)
#             crop_img = np.expand_dims(crop_img, 0)
#             # ---------------------------------------------------#
#             #   利用图像算取特征向量
#             # ---------------------------------------------------#
#             with torch.no_grad():
#                 crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
#                 if self.cuda:
#                     crop_img = crop_img.cuda()
#                 face_encoding = self.model(crop_img)[0].cpu().numpy()
#                 # print(face_encoding)
#                 face_encodings.append(face_encoding)
#                 # print(face_encodings)
#         np.save("demo_face/model/{backbone}_face_encoding.npy".format(backbone=self.backbone), face_encodings)
#         np.save("demo_face/model/{backbone}_names.npy".format(backbone=self.backbone), known_names)
#     # ---------------------------------------------------#
#     #   检测图片
#     # ---------------------------------------------------
#     def detect_image(self, image, timestamp):
#         if self.net is None:
#             self.load_detection_model()
#         if self.model is None:
#             self.load_comparison_model()
#         # ---------------------------------------------------#
#         #   对输入图像进行一个备份，后面用于绘图
#         # ---------------------------------------------------#
#         results = []
#         face_result = []
#         old_image = image.copy()
#         # ---------------------------------------------------#
#         #   把图像转换成numpy的形式
#         # ---------------------------------------------------#
#         image = np.array(image, np.float32)
#         # ---------------------------------------------------#
#         #   Retinaface检测部分-开始
#         # ---------------------------------------------------#
#         # ---------------------------------------------------#
#         #   计算输入图片的高和宽
#         # ---------------------------------------------------#
#         im_height, im_width, _ = np.shape(image)
#         # ---------------------------------------------------#
#         #   计算scale，用于将获得的预测框转换成原图的高宽
#         # ---------------------------------------------------#
#         scale = [
#             np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
#         ]
#         scale_for_landmarks = [
#             np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
#             np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
#             np.shape(image)[1], np.shape(image)[0]
#         ]
#         # ---------------------------------------------------------#
#         #   letterbox_image可以给图像增加灰条，实现不失真的resize
#         # ---------------------------------------------------------#
#         if self.letterbox_image:
#             image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
#             anchors = self.anchors
#         else:
#             anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
#         # ---------------------------------------------------#
#         #   将处理完的图片传入Retinaface网络当中进行预测
#         # ---------------------------------------------------#
#         with torch.no_grad():
#             # -----------------------------------------------------------#
#             #   图片预处理，归一化。
#             # -----------------------------------------------------------#
#             image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)
#             if self.cuda:
#                 anchors = anchors.cuda()
#                 image = image.cuda()
#             # ---------------------------------------------------------#
#             #   传入网络进行预测
#             # ---------------------------------------------------------#
#             loc, conf, landms = self.net(image)
#             boxes = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])
#             conf = conf.data.squeeze(0)[:, 1:2]
#             landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])
#             boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
#             boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
#             if len(boxes_conf_landms) <= 0:
#                 return results
#             if self.letterbox_image:
#                 boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms,
#                                                              np.array([self.retinaface_input_shape[0],
#                                                                        self.retinaface_input_shape[1]]),
#                                                              np.array([im_height, im_width]))
#             boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
#             boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks
#         face_encodings = []
#         for boxes_conf_landm in boxes_conf_landms:
#             # ----------------------#
#             #   图像截取，人脸矫正
#             # ----------------------#
#             boxes_conf_landm = np.maximum(boxes_conf_landm, 0)
#             crop_img = np.array(old_image)[int(boxes_conf_landm[1]):int(boxes_conf_landm[3]),
#                        int(boxes_conf_landm[0]):int(boxes_conf_landm[2])]
#             landmark = np.reshape(boxes_conf_landm[5:], (5, 2)) - np.array(
#                 [int(boxes_conf_landm[0]), int(boxes_conf_landm[1])])
#             crop_img, _ = Alignment_1(crop_img, landmark)
#             # ----------------------#
#             #   人脸编码
#             # ----------------------#
#             crop_img = np.array(
#                 letterbox_image(np.uint8(crop_img), (self.Re_input_shape[1], self.Re_input_shape[0]))) / 255
#             crop_img = np.expand_dims(crop_img.transpose(2, 0, 1), 0)
#             with torch.no_grad():
#                 crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
#                 if self.cuda:
#                     crop_img = crop_img.cuda()
#                 face_encoding = self.model(crop_img)[0].cpu().numpy()
#                 face_encodings.append(face_encoding)
#         face_names = []
#         self.load_face_features(encoding=True)
#         for face_encoding in face_encodings:
#             #   取出一张脸并与数据库中所有的人脸进行对比，计算得分
#             matches, face_similarities = compare_faces(self.known_face_encodings, face_encoding,
#                                                        tolerance=self.threhold)
#             name = "Unknown"
#             #   取出这个最近人脸的评分
#             #   取出当前输入进来的人脸，最接近的已知人脸的序号
#             best_match_index = np.argmax(face_similarities)
#             best_score = face_similarities[best_match_index]
#             if best_score >= self.threhold and matches[best_match_index]:
#                 name = self.known_face_names[best_match_index]
#             face_names.append(name)
#         # -----------------------------------------------#
#         #   人脸特征比对-结束
#         # -----------------------------------------------#
#         for i, b in enumerate(boxes_conf_landms):
#             b = list(map(int, b))
#             name = face_names[i]
#             if name != "Unknown":
#                 x1 = max(0, b[0])
#                 y1 = max(0, b[1])
#                 x2 = min(old_image.shape[1], b[2])
#                 y2 = min(old_image.shape[0], b[3])
#                 if x2 > x1 and y2 > y1:
#                     crop_img = old_image[y1:y2, x1:x2]
#                     _, buffer = cv2.imencode('.jpg', crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
#                     base64_encoded_img = base64.b64encode(buffer).decode('utf-8')
#                     x1 = float(b[0])
#                     y1 = float(b[1])
#                     x2 = float(b[2])
#                     y2 = float(b[3])
#                     w = x2 - x1
#                     h = y2 - y1
#                     face_result = [x1, y1, x2, y2]
#                     results.append(face_result)
#                     # result_dict = {
#                     #     "time": timestamp,
#                     #     "tid": name,
#                     #     "x": x1,
#                     #     "y": y1,
#                     #     "w": w,
#                     #     "h": h,
#                     #     "score": float(best_score),
#                     #     "image_tid": base64_encoded_img
#                     # }
#                     # results.append(result_dict)
#         return results
#     def photograph(self, image, timestamp):
#         results = []
#         old_image = image.copy()
#         image = np.array(image, np.float32)
#         # ---------------------------------------------------#
#         #   Retinaface检测部分-开始
#         # ---------------------------------------------------#
#         im_height, im_width, _ = np.shape(image)
#         # ---------------------------------------------------#
#         #   计算scale，用于将获得的预测框转换成原图的高宽
#         # ---------------------------------------------------#
#         scale = [
#             np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
#         ]
#         scale_for_landmarks = [
#             np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
#             np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
#             np.shape(image)[1], np.shape(image)[0]
#         ]
#         # ---------------------------------------------------------#
#         #   letterbox_image可以给图像增加灰条，实现不失真的resize
#         # ---------------------------------------------------------#
#         if self.letterbox_image:
#             image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
#             anchors = self.anchors
#         else:
#             anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
#         # ---------------------------------------------------#
#         #   将处理完的图片传入Retinaface网络当中进行预测
#         # ---------------------------------------------------#
#         with torch.no_grad():
#             # -----------------------------------------------------------#
#             #   图片预处理，归一化。
#             # -----------------------------------------------------------#
#             image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)
#             if self.cuda:
#                 anchors = anchors.cuda()
#                 image = image.cuda()
#             # ---------------------------------------------------------#
#             #   传入网络进行预测
#             # ---------------------------------------------------------#
#             loc, conf, landms = self.net(image)
#             boxes = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])
#             conf = conf.data.squeeze(0)[:, 1:2]
#             landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])
#             boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
#             boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
#             if len(boxes_conf_landms) <= 0:
#                 return results
#             if self.letterbox_image:
#                 boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms,
#                                                              np.array([self.retinaface_input_shape[0],
#                                                                        self.retinaface_input_shape[1]]),
#                                                              np.array([im_height, im_width]))
#             boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
#             boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks
#         face_encodings = []
#         for boxes_conf_landm in boxes_conf_landms:
#             # ----------------------#
#             #   图像截取，人脸矫正
#             # ----------------------#
#             boxes_conf_landm = np.maximum(boxes_conf_landm, 0)
#             crop_img = np.array(old_image)[int(boxes_conf_landm[1]):int(boxes_conf_landm[3]),
#                        int(boxes_conf_landm[0]):int(boxes_conf_landm[2])]
#             landmark = np.reshape(boxes_conf_landm[5:], (5, 2)) - np.array(
#                 [int(boxes_conf_landm[0]), int(boxes_conf_landm[1])])
#             crop_img, _ = Alignment_1(crop_img, landmark)
#             crop = crop_img
#             # ----------------------#
#             #   人脸编码
#             # ----------------------#
#             crop_img = np.array(
#                 letterbox_image(np.uint8(crop_img), (self.Re_input_shape[1], self.Re_input_shape[0]))) / 255
#             crop_img = np.expand_dims(crop_img.transpose(2, 0, 1), 0)
#             with torch.no_grad():
#                 crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
#                 if self.cuda:
#                     crop_img = crop_img.cuda()
#                 face_encoding = self.model(crop_img)[0].cpu().numpy()
#                 face_encodings.append(face_encoding)
#         for i, b in enumerate(boxes_conf_landms):
#             b = list(map(int, b))
#             x1 = max(0, b[0])
#             y1 = max(0, b[1])
#             x2 = min(old_image.shape[1], b[2])
#             y2 = min(old_image.shape[0], b[3])
#             if x2 > x1 and y2 > y1:
#                 _, buffer = cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
#                 base64_encoded_img = base64.b64encode(buffer).decode('utf-8')
#                 x1 = float(b[0])
#                 y1 = float(b[1])
#                 x2 = float(b[2])
#                 y2 = float(b[3])
#                 w = x2 - x1
#                 h = y2 - y1
#                 result_dict = {
#                     "time": timestamp,
#                     "tid": i,
#                     "x": x1,
#                     "y": y1,
#                     "w": w,
#                     "h": h,
#                     "encoding": face_encodings[i].tolist(),
#                     "image_tid": base64_encoded_img
#                 }
#                 results.append(result_dict)
#         return results
#     def compare_face_to_images(self, reference_image, image_list, threshold=0.5):
#         """
#         比对一张已知的人脸图片与一系列图片，返回相似度高于阈值的匹配结果。
#         :param reference_image: 已知的人脸图片（numpy数组）
#         :param image_list: 要比对的图片列表（numpy数组的列表）
#         :param threshold: 相似度阈值，越大表示越相似
#         :return: 匹配结果列表，每个元素是一个包含(图片索引, 相似度得分)的元组
#         """
#         # 确保比对模型已加载
#         if self.model is None:
#             self.load_comparison_model()
#         reference_detections = self.photograph(reference_image, "timestamp_not_used")
#         if not reference_detections:
#             raise ValueError("No face detected in the reference image.")
#         reference_encoding = reference_detections[0]["encoding"]
#         matches = []
#         for idx, image in enumerate(image_list):
#             detections = self.photograph(image, "timestamp_not_used")
#             for detection in detections:
#                 face_encoding = detection["encoding"]
#                 similarity_score = 1 - cosine(reference_encoding, face_encoding)
#                 if similarity_score > threshold:
#                     matches.append((idx, similarity_score))
#         return matches
# import os
# from retinaface import Retinaface
# retinaface_ec = Retinaface(1)
# list_dir = os.listdir("face")
# image_paths = []
# names = []
# for name in list_dir:
#     image_paths.append("face/" + name)
#     name_without_extension = os.path.splitext(name)[0]
#     names.append(name_without_extension)
# retinaface_ec.encode_face_dataset(image_paths, names)
# import cv2
# import time
# import numpy as np
# import multiprocessing as mp
# from retinaface import Retinaface
# from settings import cam_addres, img_shape
# import ffmpeg
# import logging
# # 捕获视频流
# # def push_image(raw_q, cam_addr):
# #     cap = cv2.VideoCapture(cam_addr, cv2.CAP_FFMPEG)
# #     while True:
# #         t1 = time.time()
# #         is_opened, frame = cap.read()
# #
# #         if is_opened:
# #             raw_q.put((frame, cam_addr, t1))
# #         else:
# #             cap = cv2.VideoCapture(cam_addr, cv2.CAP_FFMPEG)
# #         if raw_q.qsize() > 2:
# #             # 删除旧图片
# #             raw_q.get()
# #         else:
# #             # 等待
# #             time.sleep(0.01)
# def detect_stream_decoder(source):
#     probe = ffmpeg.probe(source)
#     video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
#     codec_name = video_info['codec_name']
#     if codec_name == 'h264':
#         return "h264_cuvid"
#     elif codec_name == 'hevc' or codec_name == 'h265':
#         return "hevc_cuvid"
#     else:
#         raise ValueError(f'Unsupported codec: {codec_name}')
# def push_image(raw_q, source):
#     decoder = detect_stream_decoder(source)
#     args = {
#         "rtsp_transport": "tcp",
#         "fflags": "nobuffer",
#         "flags": "low_delay",
#         "hwaccel": "cuda",
#         "c:v": decoder
#     }
#     probe = ffmpeg.probe(source)
#     cap_info = next(x for x in probe['streams'] if x['codec_type'] == 'video')
#     print("fps: {}".format(cap_info['r_frame_rate']))
#     width = cap_info['width']
#     height = cap_info['height']
#     up, down = str(cap_info['r_frame_rate']).split('/')
#     fps = eval(up) / eval(down)
#     print("fps: {}".format(fps))
#     process1 = (
#         ffmpeg
#         .input(source, **args)
#         .output('pipe:', format='rawvideo', pix_fmt='rgb24')
#         .overwrite_output()
#         .run_async(pipe_stdout=True)
#     )
#     frame_count = 0
#     while True:
#         t1 = time.time()
#         in_bytes = process1.stdout.read(width * height * 3)
#         if not in_bytes:
#             break
#         in_frame = (
#             np
#             .frombuffer(in_bytes, np.uint8)
#             .reshape([height, width, 3])
#         )
#         if raw_q.qsize() > 2:
#             raw_q.get()
#         logging.debug('Pushing frame into queue. Queue size: {}'.format(raw_q.qsize()))
#         raw_q.put((in_frame, source, t1))
#         frame_count += 1
#         print(frame_count)
# def predict(raw_q, pred_q):
#     retinaface = Retinaface()
#     while True:
#         raw_img, cam_address, t1 = raw_q.get()
#         pred_img = np.array(retinaface.detect_image(raw_img, 'output/result', t1, cam_address))
#         # RGBtoBGR满足opencv显示格式
#         pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
#         pred_q.put(pred_img)
#         toc = time.time()
#         time_cost = toc - t1
#         print('Processed in %.3fs FPS = %.3f' % (time_cost, 1 / time_cost))
# def pop_image(pred_q, window_name, img_shape):
#     cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
#     while True:
#         frame = pred_q.get()
#         frame = cv2.resize(frame, img_shape)
#         cv2.imshow(window_name, frame)
#         cv2.waitKey(1)
# # 显示
# def display(cam_addrs, window_names, img_shape=(300, 300)):
#     raw_queues = [mp.Queue(maxsize=2) for _ in cam_addrs]
#     pred_queues = [mp.Queue(maxsize=4) for _ in cam_addrs]
#     processes = []
#     for raw_q, pred_q, cam_addr, window_name in zip(raw_queues, pred_queues, cam_addrs, window_names):
#         processes.append(mp.Process(target=push_image, args=(raw_q, cam_addr)))
#         processes.append(mp.Process(target=predict, args=(raw_q, pred_q)))
#         processes.append(mp.Process(target=pop_image, args=(pred_q, window_name, img_shape)))
#     [setattr(process, "daemon", True) for process in processes]
#     [process.start() for process in processes]
#     [process.join() for process in processes]
# if __name__ == '__main__':
#     mp.set_start_method(method='spawn')
#     num_cameras = len(cam_addres)
# display(cam_addres[:num_cameras], ['camera' for _ in cam_addres], img_shape)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# input = torch.tensor([[0.9, 0.1], [0.2, 0.8]])  # 预测输出 (batch_size=2, num_classes=2)
# target = torch.tensor([[1, 0], [0, 1]])
# class DiceLoss(nn.Module):
# 	def __init__(self):
# 		super(DiceLoss, self).__init__()
# 	def	forward(self, input, target):
# 		N = target.size(0)
# 		smooth = 1
# 		input_flat = input.view(N, -1)
# 		target_flat = target.view(N, -1)
# 		intersection = input_flat * target_flat
# 		loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
# 		loss = 1 - loss.sum() / N
# 		return loss
# class MulticlassDiceLoss(nn.Module):
# 	"""
# 	requires one hot encoded target. Applies DiceLoss on each class iteratively.
# 	requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
# 	  batch size and C is number of classes
# 	"""
# 	def __init__(self):
# 		super(MulticlassDiceLoss, self).__init__()
# 	def forward(self, input, target, weights=None):
# 		C = target.shape[1]
# 		# if weights is None:
# 		# 	weights = torch.ones(C) #uniform weights for all classes
# 		dice = DiceLoss()
# 		totalLoss = 0
# 		for i in range(C):
# 			diceLoss = dice(input[:,i], target[:,i])
# 			if weights is not None:
# 				diceLoss *= weights[i]
# 			totalLoss += diceLoss
# 		return totalLoss