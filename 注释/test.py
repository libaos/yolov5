# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset
Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""
'''===============================================一、导入包==================================================='''
'''======================1.导入安装好的python库====================='''
 
import argparse # 解析命令行参数的库
import json # 实现字典列表和JSON字符串之间的相互解析
import os  # 与操作系统进行交互的文件库 包含文件路径操作与解析
import sys # sys系统模块 包含了与Python解释器和它的环境有关的函数
from pathlib import Path # Path将str转换为Path对象 使字符串路径易于操作的模块
from threading import Thread # python中处理多线程的库
 
import numpy as np # 矩阵计算基础库
import torch # pytorch 深度学习库
from tqdm import tqdm  # 用于直观显示进度条的一个库
 
'''===================2.获取当前文件的绝对路径========================'''
FILE = Path(__file__).resolve()# __file__指的是当前文件(即val.py),FILE最终保存着当前文件的绝对路径,比如D://yolov5/val.py
ROOT = FILE.parents[0]  # YOLOv5 root directory ROOT保存着当前项目的父目录,比如 D://yolov5
if str(ROOT) not in sys.path: # sys.path即当前python环境可以运行的路径,假如当前项目不在该路径中,就无法运行其中的模块,所以就需要加载路径
    sys.path.append(str(ROOT))  # add ROOT to PATH 把ROOT添加到运行路径上
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative ROOT设置为相对路径
 
'''===================3..加载自定义模块============================'''
from models.common import DetectMultiBackend # yolov5的网络结构(yolov5)
from utils.callbacks import Callbacks # 和日志相关的回调函数
from utils.datasets import create_dataloader # 加载数据集的函数
from utils.general import (LOGGER, NCOLS, box_iou, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)  # 定义了一些常用的工具函数
from utils.metrics import ConfusionMatrix, ap_per_class # 在YOLOv5中，fitness函数实现对 [P, R, mAP@.5, mAP@.5-.95] 指标进行加权
from utils.plots import output_to_target, plot_images, plot_val_study # 定义了Annotator类，可以在图像上绘制矩形框和标注信息
from utils.torch_utils import select_device, time_sync  # 定义了一些与PyTorch有关的工具函数
 
'''===============================================二、保存信息==================================================='''
'''======================1.保存预测信息到txt文件====================='''
def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    # gn = [w, h, w, h] 对应图片的宽高  用于后面归一化
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    # 将每个图片的预测信息分别存入save_dir/labels下的xxx.txt中 每行: class_id + score + xywh
    for *xyxy, conf, cls in predn.tolist():
        # 将xyxy(左上角+右下角)格式转为xywh(中心点+宽长)格式，并归一化，转化为列表再保存
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        # line的形式是： ”类别 x y w h“，若save_conf为true，则line的形式是：”类别 x y w h 置信度“
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        # 保存预测类别和坐标值到对应图片image_name.txt文件中
        with open(file, 'a') as f:
            # 写入对应的文件夹里，路径默认为“runs\detect\exp*\labels”
            f.write(('%g ' * len(line)).rstrip() % line + '\n')
 
'''======================2.保存预测信息到coco格式的json字典====================='''
def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    # 获取图片id
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    # 获取预测框 并将xyxy转为xywh格式
    box = xyxy2xywh(predn[:, :4])  # xywh
    # 之前的的xyxy格式是左上角右下角坐标  xywh是中心的坐标和宽高
    # 而coco的json格式的框坐标是xywh(左上角坐标 + 宽高)
    # 所以这行代码是将中心点坐标 -> 左上角坐
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    # 序列解包
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})
    '''
     image_id: 图片id 即属于哪张图片
     category_id: 类别 coco91class()从索引0~79映射到索引0~90
     bbox: 预测框坐标
     score: 预测得分
    '''
'''===============================================三、计算指标==================================================='''
def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix.
    返回每个预测框在10个IoU阈值上是TP还是FP
    Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    # 构建一个[pred_nums, 10]全为False的矩阵
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    # 计算每个gt与每个pred的iou，shape为: [gt_nums, pred_nums]
    iou = box_iou(labels[:, 1:], detections[:, :4])
    '''
    首先iou >= iouv[0]：挑选出iou>0.5的所有预测框，进行筛选,shape为: [gt_nums, pred_nums]
    同时labels[:, 0:1] == detections[:, 5]：构建出一个预测类别与真实标签是否相同的矩阵表, shape为: [gt_nums, pred_nums]
    只有同时符合以上两点条件才被赋值为True，此时返回当前矩阵的一个行列索引，x是两个元祖x1,x2
    点(x[0][i], x[1][i])就是符合条件的预测框
    '''
    # iou超过阈值而且类别正确，则为True，返回索引
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    # 如果存在符合条件的预测框
    if x[0].shape[0]: # 至少有一个TP
        # 将符合条件的位置构建成一个新的矩阵，第一列是行索引（表示gt索引），第二列是列索引（表示预测框索引），第三列是iou值
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            # argsort获得有小到大排序的索引, [::-1]相当于取反reserve操作，变成由大到小排序的索引，对matches矩阵进行排序
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            '''
            参数return_index=True：表示会返回唯一值的索引，[0]返回的是唯一值，[1]返回的是索引
            matches[:, 1]：这里的是获取iou矩阵每个预测框的唯一值，返回的是最大唯一值的索引，因为前面已由大到小排序
            这个操作的含义：每个预测框最多只能出现一次，如果有一个预测框同时和多个gt匹配，只取其最大iou的一个
            '''
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            '''
            matches[:, 0]：这里的是获取iou矩阵gt的唯一值，返回的是最大唯一值的索引，因为前面已由大到小排序
            这个操作的含义: 每个gt也最多只能出现一次，如果一个gt同时匹配多个预测框，只取其匹配最大的那一个预测框
            '''
            # 以上操作实现了为每一个gt分配一个iou最高的类别的预测框，实现一一对应
 
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
        '''
         当前获得了gt与预测框的一一对应，其对于的iou可以作为评价指标，构建一个评价矩阵
         需要注意，这里的matches[:, 1]表示的是为对应的预测框来赋予其iou所能达到的程度，也就是iouv的评价指标
        '''
        # 在correct中，只有与gt匹配的预测框才有对应的iou评价指标，其他大多数没有匹配的预测框都是全部为False
    return correct
 
'''===============================================四、run()函数==================================================='''
'''======================1.设置参数====================='''
@torch.no_grad()
def run(data, # 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息 train.py时传入data_dict
        weights=None,  # 模型的权重文件地址 运行train.py=None 运行test.py=默认weights/yolov5s
        batch_size=32,  # 前向传播的批次大小 运行test.py传入默认32 运行train.py则传入batch_size // WORLD_SIZE * 2
        imgsz=640,  # 输入网络的图片分辨率 运行test.py传入默认640 运行train.py则传入imgsz_test
        conf_thres=0.001,  # object置信度阈值 默认0.001
        iou_thres=0.6,  # 进行NMS时IOU的阈值 默认0.6
        task='val',  # 设置测试的类型 有train, val, test, speed or study几种 默认val
        device='',  # 执行 val.py 所在的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # dataloader中的最大 worker 数（线程个数）
        augment=False,  # 数据集是否只有一个类别 默认False
        verbose=False,  # 是否打印出每个类别的mAP 运行test.py传入默认Fasle 运行train.py则传入nc < 50 and final_epoch
        save_txt=False,  # 是否以txt文件的形式保存模型预测框的坐标 默认True
        save_hybrid=False,  # 是否保存预测每个目标的置信度到预测txt文件中 默认True
        save_conf=False,  # 保存置信度
        save_json=False,  # 是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签）,
                      #运行test.py传入默认Fasle 运行train.py则传入is_coco and final_epoch(一般也是False)
        project=ROOT / 'runs/val',  # 验证结果保存的根目录 默认是 runs/val
        name='exp',  # 验证结果保存的目录 默认是exp  最终: runs/val/exp
        exist_ok=False,  # 如果文件存在就increment name，不存在就新建  默认False(默认文件都是不存在的)
        half=True,  # 使用 FP16 的半精度推理
        dnn=False,  # 在 ONNX 推理时使用 OpenCV DNN 后段端
        model=None,  # 如果执行val.py就为None 如果执行train.py就会传入( model=attempt_load(f, device).half() )
        dataloader=None, # 数据加载器 如果执行val.py就为None 如果执行train.py就会传入testloader
        save_dir=Path(''), # 文件保存路径 如果执行val.py就为‘’ , 如果执行train.py就会传入save_dir(runs/train/expn)
        plots=True, # 是否可视化 运行val.py传入，默认True
        callbacks=Callbacks(),
        compute_loss=None, # 损失函数 运行val.py传入默认None 运行train.py则传入compute_loss(train)
        ):
    '''======================2.初始化/加载模型以及设置设备====================='''
    # Initialize/load model and set device
    training = model is not None
    if training:  # 通过 train.py 调用的run函数
        # 获得记录在模型中的设备 next为迭代器
        device, pt = next(model.parameters()).device, True
 
        # 精度减半
        # 如果设备类型不是cpu 则将模型由32位浮点数转换为16位浮点数
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
 
        # 如果不是train.py调用run函数(执行val.py脚本)就调用select_device选择可用的设备
        # 并生成save_dir + make dir + 加载模型model + check imgsz + 加载data配置信息
    else:  # 直接通过 val.py 调用 run 函数
        # 调用torch_utils中select_device来选择执行程序时的设备
        device = select_device(device, batch_size=batch_size)
 
        # 路径
        # 调用genera.py中的increment_path函数来生成save_dir文件路径  run\test\expn
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        # mkdir创建路径最后一级目录
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
 
        # 加载模型  只在运行test.py才需要自己加载model
        # 加载模型为32位浮点数模型（权重参数） 调用experimental.py文件中的attempt_load函数
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, pt = model.stride, model.pt
        # 调用general.py中的check_img_size函数来检查图像分辨率能否被32整除
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        # 如果不是CPU，使用半进度(图片半精度/模型半精度)
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()
        else:
            half = False
            batch_size = 1  # export.py models default to batch-size 1
            device = torch.device('cpu')
            # 打印耗时
            LOGGER.info(f'Forcing --batch-size 1 square inference shape(1,3,{imgsz},{imgsz}) for non-PyTorch backends')
 
        # Data
        # 调用general.py中的check_dataset函数来检查数据文件是否正常
        data = check_dataset(data)  # check
 
    '''======================3.加载配置====================='''
    # Configure
    # 将模型转换为测试模式 固定住dropout层和Batch Normalization层
    model.eval()
    # 通过 COCO 数据集的文件夹组织结构判断当前数据集是否为 COCO 数据集
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    # 确定检测的类别数目
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    # 计算mAP相关参数
    # 设置iou阈值 从0.5-0.95取10个(0.05间隔)   iou vector for mAP@0.5:0.95
    # iouv: [0.50000, 0.55000, 0.60000, 0.65000, 0.70000, 0.75000, 0.80000, 0.85000, 0.90000, 0.95000]
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # mAP@0.5:0.95 的iou向量
    # numel为pytorch预置函数 用来获取张量中的元素个数
    niou = iouv.numel()
 
    '''======================4.加载val数据集====================='''
    # Dataloader
    if not training:
        if pt and device.type != 'cpu':
            # 创建一张全为0的图片（四维张量）
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        # 调用datasets.py文件中的create_dataloader函数创建dataloader
        dataloader = create_dataloader(data[task], imgsz, batch_size, stride, single_cls, pad=pad, rect=pt,
                                       prefix=colorstr(f'{task}: '))[0]
 
    '''======================5.初始化====================='''
    # 初始化已完成测试的图片数量
    seen = 0
    # 调用matrics中函数 存储混淆矩阵
    confusion_matrix = ConfusionMatrix(nc=nc)
    # 获取数据集所有类别的类名
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    # 调用general.py中的函数  获取coco数据集的类别索引
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    # 设置tqdm进度条的显示信息
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    # 初始化detection中各个指标的值
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # 初始化网络训练的loss
    loss = torch.zeros(3, device=device)
    # 初始化json文件涉及到的字典、统计信息、AP、每一个类别的AP、图片汇总
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
 
    '''======================6.开始验证====================='''
    '''===6.1 开始验证前的预处理==='''
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        # 获取当前时间
        t1 = time_sync()
        if pt:
            # 将图片数据拷贝到device（GPU）上面
            im = im.to(device, non_blocking=True)
            #对targets也做同样拷贝的操作
            targets = targets.to(device)
        # 将图片从64位精度转换为32位精度
        im = im.half() if half else im.float()  # uint8 to fp16/32
        # 将图像像素值0-255的范围归一化到0-1的范围
        im /= 255  # 0 - 255 to 0.0 - 1.0
        # 四个变量分别代表batchsize、通道数目、图像高度、图像宽度
        nb, _, height, width = im.shape  # batch size, channels, height, width
        # 获取当前时间
        t2 = time_sync()
        # dt[0]: 累计处理数据时间
        dt[0] += t2 - t1
 
        '''===6.2 前向推理==='''
        # Inference
        out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
        # 累计前向推理时间  dt[1]
        dt[1] += time_sync() - t2
 
        '''===6.3 计算损失==='''
        # Loss
        # compute_loss不为空 说明正在执行train.py  根据传入的compute_loss计算损失值
        if compute_loss:
            # loss 包含bounding box 回归的GIoU、object和class 三者的损失
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls
 
        '''===6.4 NMS获得预测框==='''
        # NMS
        # 运行NMS 目标检测的后处理模块 用于删除冗余的bounding box
        # targets: [num_target, img_index+class_index+xywh] = [31, 6]
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        # 提取bach中每一张图片的目标的label
        # lb: {list: bs} 第一张图片的target[17, 5] 第二张[1, 5] 第三张[7, 5] 第四张[6, 5]
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        # 计算NMS过程所需要的时间
        t3 = time_sync()
        # 调用general.py中的函数 进行非极大值抑制操作
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        # 累计NMS时间
        dt[2] += time_sync() - t3
 
        '''===6.5 统计真实框、预测框信息==='''
        # Metrics
        # 为每张图片做统计，将写预测信息到txt文件，生成json文件字典，统计tp等
        # out: list{bs}  [300, 6] [42, 6] [300, 6] [300, 6]  [:, image_index+class+xywh]
        # si代表第si张图片，pred是对应图片预测的label信息
        for si, pred in enumerate(out):
            # 获取第si张图片的gt标签信息 包括class, x, y, w, h    target[:, 0]为标签属于哪张图片的编号
            labels = targets[targets[:, 0] == si, 1:]
            # nl为图片检测到的目标个数
            nl = len(labels)
            # tcls为检测到的目标的类别 label矩阵的第一列
            tcls = labels[:, 0].tolist() if nl else []  # target class
            # 第si张图片对应的文件路径
            path, shape = Path(paths[si]), shapes[si][0]
            # 统计测试图片数量 +1
            seen += 1
 
            # 如果预测为空，则添加空的信息到stats里
            if len(pred) == 0:
                if nl: # 预测为空但同时有label信息
                    # stats初始化为一个空列表[] 此处添加一个空信息
                    # 添加的每一个元素均为tuple 其中第二第三个变量为一个空的tensor
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
 
            # Predictions
            # 预测
            if single_cls:
                pred[:, 5] = 0
            # 对pred进行深复制
            predn = pred.clone()
            # 调用general.py中的函数 将图片调整为原图大小
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred
 
            # Evaluate
            # 预测框评估
            if nl:
                # 获得xyxy格式的框
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                # 调用general.py中的函数 将图片调整为原图大小
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                # 处理完gt的尺寸信息，重新构建成 (cls, xyxy)的格式
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space label
                # 对当前的预测框与gt进行一一匹配，并且在预测框的对应位置上获取iou的评分信息，其余没有匹配上的预测框设置为False
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    # 计算混淆矩阵 confusion_matrix
                    confusion_matrix.process_batch(predn, labelsn)
            else:
                # 返回一个形状为为pred.shape[0, 类型为torch.dtype，里面的每一个值都是0的tensor
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            # 每张图片的结果统计到stats里
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)
 
            # Save/log
            # 保存预测信息到txt文件
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            # 保存预测信息到json字典
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])
 
        '''===6.6 画出前三个batch图片的 gt 和 pred 框==='''
        # Plot images
        # 画出前三个batch的图片的ground truth和预测框predictions(两个图)一起保存
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(im, targets, paths, f, names), daemon=True).start()
            '''
              Thread()函数为创建一个新的线程来执行这个函数 函数为plots.py中的plot_images函数
              target: 执行的函数  args: 传入的函数参数  daemon: 当主线程结束后, 由他创建的子线程Thread也已经自动结束了
              .start(): 启动线程  当thread一启动的时候, 就会运行我们自己定义的这个函数plot_images
              如果在plot_images里面打开断点调试, 可以发现子线程暂停, 但是主线程还是在正常的训练(还是正常的跑)
            '''
            # 传入plot_images函数之前需要改变pred的格式  target则不需要改
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(im, output_to_target(out), paths, f, names), daemon=True).start()
 
    '''===6.7 计算指标==='''
    # Compute metrics
    # 将stats列表的信息拼接到一起
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # 转换为对应格式numpy
    # stats[0].any(): stats[0]是否全部为False, 是则返回 False, 如果有一个为 True, 则返回 True
    if len(stats) and stats[0].any():
        # 计算上述测试过程中的各种性能指标
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        '''
        根据上面的统计预测结果计算p, r, ap, f1, ap_class（ap_per_class函数是计算每个类的mAP等指标的）等指标
        p: [nc] 最大平均f1时每个类别的precision
        r: [nc] 最大平均f1时每个类别的recall
        ap: [71, 10] 数据集每个类别在10个iou阈值下的mAP
        f1 [nc] 最大平均f1时每个类别的f1
        ap_class: [nc] 返回数据集中所有的类别index
        '''
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        '''
        ap50: [nc] 所有类别的mAP@0.5   
        ap: [nc] 所有类别的mAP@0.5:0.95 
        '''
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        '''
         mp: [1] 所有类别的平均precision(最大f1时)
         mr: [1] 所有类别的平均recall(最大f1时)
         map50: [1] 所有类别的平均mAP@0.5
         map: [1] 所有类别的平均mAP@0.5:0.95
        '''
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        '''
         nt: [nc] 统计出整个数据集的gt框中数据集各个类别的个数
        '''
    else:
        nt = torch.zeros(1)
 
    '''===6.8 打印日志==='''
    # Print results
    # 按照以下格式来打印测试过程的指标
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
 
    # Print results per class
    # 打印每一个类别对应的性能指标
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
 
    # Print speeds
    # 打印 推断/NMS过程/总过程 的在每一个batch上面的时间消耗
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)
 
    '''===6.9 保存验证结果==='''
    # Plots
    # 绘图
    if plots:
        # confusion_matrix.plot（）函数绘制混淆矩阵
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        # 调用Loggers中的on_val_end方法，将日志记录并生成一些记录的图片
        callbacks.run('on_val_end')
 
    # Save JSON
    # 采用之前保存的json文件格式预测结果 通过coco的api评估各个指标
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        # 注释的json格式
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        # 预测的json格式
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        # 在控制台打印coco的api评估各个指标，保存到json文件
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        # 打开pred_json文件只用于写入
        with open(pred_json, 'w') as f: # w:打开一个文件只用于写入。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
            # 测试集的标签也需要转成coco的json格式。将 dict==>json 序列化，用json.dumps()
            json.dump(jdict, f)
 
        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            # 以下过程为利用官方coco工具进行结果的评测
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
 
            # 获取并初始化测试集标签的json文件
            anno = COCO(anno_json)  # init annotations api
            # 初始化预测框的文件
            pred = anno.loadRes(pred_json)  # init predictions api
            # 创建评估器
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            # 评估
            eval.evaluate()
            eval.accumulate()
            # 展示结果
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')
 
    '''===6.10 返回结果==='''
    # Return results
    # 返回测试指标结果
    model.float() # 将模型转换为适用于训练的状态
    if not training:# 如果不是训练过程则将结果保存到对应的路径
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # 在控制台中打印保存结果
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    # 返回对应的测试结果
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
'''
 (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()): {tuple:7}
          0: mp [1] 所有类别的平均precision(最大f1时)
          1: mr [1] 所有类别的平均recall(最大f1时)
          2: map50 [1] 所有类别的平均mAP@0.5
          3: map [1] 所有类别的平均mAP@0.5:0.95
          4: val_box_loss [1] 验证集回归损失
          5: val_obj_loss [1] 验证集置信度损失
          6: val_cls_loss [1] 验证集分类损失
     maps: [80] 所有类别的mAP@0.5:0.95
     t: {tuple: 3} 0: 打印前向传播耗费的总时间   1: nms耗费总时间   2: 总时间
'''
 
'''===============================================五、设置opt参数==================================================='''
def parse_opt():
    parser = argparse.ArgumentParser()
    # 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    # 模型的权重文件地址yolov5s.pt
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    # 前向传播的批次大小 默认32
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    # 输入网络的图片分辨率 默认640
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    # object置信度阈值 默认0.001
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    # 进行NMS时IOU的阈值 默认0.6
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    # 设置测试的类型 有train, val, test, speed or study几种 默认val
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    # 测试的设备
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 数据集是否只用一个类别 默认False
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    # 测试是否使用TTA Test Time Augment 默认False
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # 是否打印出每个类别的mAP 默认False
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    # 是否以txt文件的形式保存模型预测的框坐标, 默认False
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # 保存label+prediction杂交结果到对应.txt，默认False
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    # 保存置信度
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # 是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签） 默认False
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    # 测试保存的源文件 默认runs/val
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    # 测试保存的文件地址 默认exp  保存在runs/val/exp下
    parser.add_argument('--name', default='exp', help='save to project/name')
    # 是否存在当前文件 默认False 一般是 no exist-ok 连用  所以一般都要重新创建文件夹
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # 是否使用半精度推理 默认False
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # 是否使用 OpenCV DNN对ONNX 模型推理
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
 
    # 解析上述参数
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)
    # |或 左右两个变量有一个为True 左边变量就为True
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt
 
'''===============================================六、执行main（）函数==================================================='''
def main(opt):
    # 检测requirements文件中需要的包是否安装好了
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
 
    # 如果task in ['train', 'val', 'test']就正常测试 训练集/验证集/测试集
    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
        run(**vars(opt))
 
    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        # 如果opt.task == 'speed' 就测试yolov5系列和yolov3-spp各个模型的速度评估
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)
 
        # 如果opt.task = ['study']就评估yolov5系列和yolov3-spp各个模型在各个尺度下的指标并可视化
        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                # 保存的文件名
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                # x坐标轴和y坐标
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    # 返回相关结果和时间
                    y.append(r + t)  # results and times
                # 将y输出保存
                np.savetxt(f, y, fmt='%10.4g')  # save
            # 命令行执行命令将study文件进行压缩
            os.system('zip -r study.zip study_*.txt')
            # 调用plots.py中的函数 可视化各个指标
            plot_val_study(x=x)  # plot
 
# python val.py --data data/mask_data.yaml --weights runs/train/exp_yolov5s/weights/best.pt --img 640
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)