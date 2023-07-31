# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules
Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""
 
'''===============================================一、导入包==================================================='''
'''======================1.导入安装好的python库====================='''
import argparse  # 解析命令行参数模块
import sys  # sys系统模块 包含了与Python解释器和它的环境有关的函数
from copy import deepcopy  # 数据拷贝模块 深拷贝
from pathlib import Path  # Path将str转换为Path对象 使字符串路径易于操作的模块
 
'''===================2.获取当前文件的绝对路径========================'''
FILE = Path(__file__).resolve() # __file__指的是当前文件(即val.py),FILE最终保存着当前文件的绝对路径,比如D://yolov5/modles/yolo.py
ROOT = FILE.parents[1]  # YOLOv5 root directory 保存着当前项目的父目录,比如 D://yolov5
if str(ROOT) not in sys.path:  # sys.path即当前python环境可以运行的路径,假如当前项目不在该路径中,就无法运行其中的模块,所以就需要加载路径
    sys.path.append(str(ROOT))  # add ROOT to PATH  把ROOT添加到运行路径上
# ROOT = ROOT.relative_to(Path.cwd())  # relative  ROOT设置为相对路径
 
'''===================3..加载自定义模块============================'''
from models.common import * # yolov5的网络结构(yolov5)
from models.experimental import *   # 导入在线下载模块
from utils.autoanchor import check_anchor_order   # 导入检查anchors合法性的函数
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args   # 定义了一些常用的工具函数
from utils.plots import feature_visualization  # 定义了Annotator类，可以在图像上绘制矩形框和标注信息
from utils.torch_utils import (copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device,
                               time_sync)   # 定义了一些与PyTorch有关的工具函数
 
# 导入thop包 用于计算FLOPs
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None
 
'''===============================================二、Detect模块==================================================='''
'''
   Detect模块是用来构建Detect层的,将输入feature map 通过一个卷积操作和公式计算到我们想要的shape, 为后面的计算损失或者NMS后处理作准备
'''
class Detect(nn.Module):
    stride = None  # 特征图的缩放步长
    onnx_dynamic = False  # ONNX动态量化
 
    '''===================1.获取预测得到的参数============================'''
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        """
        detection layer 相当于yolov3中的YOLOLayer层
          :params nc: 数据集类别数量
          :params anchors: 传入3个feature map上的所有anchor的大小(P3、P4、P5)
          :params ch: [128, 256, 512] 3个输出feature map的channel
          """
        super().__init__()
 
        # nc: 数据集类别数量
        self.nc = nc
        # no: 表示每个anchor的输出数，前nc个01字符对应类别，后5个对应：是否有目标，目标框的中心，目标框的宽高
        self.no = nc + 5  # nc+5=nc+(x,y,w,h,conf)
        # nl: 表示预测层数，yolov5是3层预测
        self.nl = len(anchors)
        # na: 表示anchors的数量，除以2是因为[10,13, 16,30, 33,23]这个长度是6，对应3个anchor
        self.na = len(anchors[0]) // 2
        # grid: 表示初始化grid列表大小，下面会计算grid，grid就是每个格子的x，y坐标（整数，比如0-19），左上角为(1,1),右下角为(input.w/stride,input.h/stride)
        self.grid = [torch.zeros(1)] * self.nl
        # anchor_grid: 表示初始化anchor_grid列表大小，空列表
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        # 注册常量anchor，并将预选框（尺寸）以数对形式存入，并命名为anchors
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2) 注意后面就可以通过self.anchors来访问它了
        # 每一张进行三次预测，每一个预测结果包含nc+5个值
        # (n, 255, 80, 80),(n, 255, 40, 40),(n, 255, 20, 20) --> ch=(255, 255, 255)
        # 255 -> (nc+5)*3 ===> 为了提取出预测框的位置信息以及预测框尺寸信息
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv 3个输出层最后的1乘1卷积
       # inplace: 一般都是True，默认不使用AWS，Inferentia加速
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
    # 如果模型不训练那么将会对这些预测得到的参数进一步处理,然后输出,可以方便后期的直接调用
	# 包含了三个信息pred_box [x,y,w,h] pred_conf[confidence] pre_cls[cls0,cls1,cls2,...clsn]
 
    '''===================2.向前传播============================'''
    def forward(self, x):
        z = []  # inference output
        # 对三个feature map分别进行处理
        # (n, 255, 80, 80),(n, 255, 40, 40),(n, 255, 20, 20)
        for i in range(self.nl):
 
            # 下面3行代码的工作：
            # (n, 255, _, _) -> (n, 3, nc+5, ny, nx) -> (n, 3, ny, nx, nc+5)
            # 相当于三层分别预测了80*80、40*40、20*20次，每一次预测都包含3个框
 
            # 进行1*1卷积，经过这个卷积就变成（5+分类数）个通道了
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
           # 维度重排列: bs, 先验框组数, 检测框行数, 检测框列数, 属性数 + 分类数
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  # contiguous 将数据保证内存中位置连续
 
            '''
            向前传播时需要将相对坐标转换到grid绝对坐标系中
            '''
            if not self.training:  # inference
                '''
                生成坐标系
                grid[i].shape = [1,1,ny,nx,2]
                                [[[[1,1],[1,2],...[1,nx]],
                                [[2,1],[2,2],...[2,nx]],
                                ...,
                                [[ny,1],[ny,2],...[ny,nx]]]]
                '''
                # 换输入后重新设定锚框
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    # 加载网格点坐标 先验框尺寸
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
 
                '''
                按损失函数的回归方式来转换坐标
                '''
                y = x[i].sigmoid()
                # 改变原数据 计算定位参数
                if self.inplace:
                    # grid: 位置基准 或者理解为 cell的预测初始位置，而y[..., 0:2]是作为在grid坐标基础上的位置偏移
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    # anchor_grid: 预测框基准 或者理解为 预测框的初始位置，而 y[..., 2:4]是作为预测框位置的调整
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
 
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    # stride: 是一个grid cell的实际尺寸
                    # 经过sigmoid, 值范围变成了(0-1),下一行代码将值变成范围（-0.5，1.5）
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    # 范围变成(0-4)倍，设置为4倍的原因是下层的感受野是上层的2倍
                    # 因下层注重检测大目标，相对比上层而言，计算量更小，4倍是一个折中的选择
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                # 存储每个特征图检测框的信息
                z.append(y.view(bs, -1, self.no))
        # 训练阶段直接返回x
        # 预测阶段返回3个特征图拼接的结果
        return x if self.training else (torch.cat(z, 1), x)
 
    '''===================3.相对坐标转换到grid绝对坐标系============================'''
    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            # 网格标尺坐标
            # indexing='ij' 表示的是i是同一行，j表示同一列
            # indexing='xy' 表示的是x是同一列，y表示同一行
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        # grid --> (20, 20, 2), 复制成3倍，因为是三个框 -> (3, 20, 20, 2)
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        # anchor_grid即每个格子对应的anchor宽高，stride是下采样率，三层分别是8，16，32，这里为啥要乘呢，因为在外面已经把anchors给除了对应的下采样率，这里再乘回来
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid
 
'''===============================================三、Model模块==================================================='''
class Model(nn.Module):
    '''===================1.__init__函数==========================='''
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        """
           :params cfg:YOLO v5模型配置文件 这里使用yolov5s模型
           :params ch: 输入图片的通道数 默认为3
           :params nc: 数据集的类别个数
           :anchors: 表示anchor框, 一般是None
        """
        # 父类的构造方法
        super().__init__()
        # 检查传入的参数格式，如果cfg是加载好的字典结果
        if isinstance(cfg, dict):
            # 直接保存到模型中
            self.yaml = cfg  # model dict
        # 若不是字典 则为yaml文件路径
        else:  # is *.yaml 一般执行这里
            # 导入yaml文件
            import yaml  # for torch hub
            # 保存文件名：cfg file name = yolov5s.yaml
            self.yaml_file = Path(cfg).name
            # 如果配置文件中有中文，打开时要加encoding参数
            with open(cfg, encoding='ascii', errors='ignore') as f:
                # 将yaml文件加载为字典
                self.yaml = yaml.safe_load(f)  # model dict 取到配置文件中每条的信息（没有注释内容）
 
        '''===================2.获取输入通道============================'''
        # Define model
        # 搭建模型
        # yaml.get('ch', ch)表示若不存在键'ch',则返回值ch
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
 
        # 判断类的通道数和yaml中的通道数是否相等，一般不执行，因为nc=self.yaml['nc']恒成立
        if nc and nc != self.yaml['nc']:
            # 在终端给出提示
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            # 将yaml中的值修改为构造方法中的值
            self.yaml['nc'] = nc  # override yaml value
 
        # 重写anchor，一般不执行, 因为传进来的anchors一般都是None
        if anchors:
            # 在终端给出提示
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            # 将yaml中的值改为构造方法中的值
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # 解析模型，self.model是解析后的模型 self.save是每一层与之相连的层
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # deepcopy()复杂产生一个新的对象
        # 加载每一类的类别名
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # inplace指的是原地操作 如x+=1 有利于节约内存
        # self.inplace=True  默认True  不使用加速推理
        self.inplace = self.yaml.get('inplace', True)
 
        '''===================3.获取Detect输出模块============================'''
        # Build strides, anchors
        # 构造步长、先验框
        m = self.model[-1]  # Detect()
        # 判断最后一层是否为Detect层
        if isinstance(m, Detect):
            # 定义一个256 * 256大小的输入
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # 保存特征层的stride,并且将anchor处理成相对于特征层的格式
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            # 原始定义的anchor是原始图片上的像素值，要将其缩放至特征图的大小
            m.anchors /= m.stride.view(-1, 1, 1)
            # 检查anchor顺序与stride顺序是否一致
            check_anchor_order(m)
            # 将步长保存至模型
            self.stride = m.stride
            # 初始化bias
            self._initialize_biases()  # only run once
 
        # Init weights, biases
        # 初始化权重
        initialize_weights(self)
        # 打印模型信息
        self.info()
        LOGGER.info('')
 
    '''===================4.数据增强============================'''
 
    # ===1.forward():管理前向传播函数=== #
    def forward(self, x, augment=False, profile=False, visualize=False):
        # 是否在测试时也使用数据增强
        if augment:
            # 增强训练，对数据采取了一些了操作
            return self._forward_augment(x)  # augmented inference, None
        # 默认执行，正常前向推理
        return self._forward_once(x, profile, visualize)  # single-scale inference, train
 
    # ===2._forward_augment():推理的forward=== #
    # 将图片进行裁剪,并分别送入模型进行检测
    def _forward_augment(self, x):
        # 获得图像的高和宽
        img_size = x.shape[-2:]  # height, width
        # s是规模
        s = [1, 0.83, 0.67]  # scales
        # flip是翻转，这里的参数表示沿着哪个轴翻转
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            # scale_img函数的作用就是根据传入的参数缩放和翻转图像
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            # 模型前向传播
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            #  恢复数据增强前的模样
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        # 对不同尺寸进行不同程度的筛选
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train
 
    # ===3._forward_once():训练的forward=== #
    def _forward_once(self, x, profile=False, visualize=False):
        """
        :params x: 输入图像
        :params profile: True 可以做一些性能评估
        :params feature_vis: True 可以做一些特征可视化
        :return train: 一个tensor list 存放三个元素   [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                        分别是 [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
                 inference: 0 [1, 19200+4800+1200, 25] = [bs, anchor_num*grid_w*grid_h, xywh+c+20classes]
                                   1 一个tensor list 存放三个元素 [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                                     [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
        """
        # 各网络层输出, 各网络层推导耗时
        # y: 存放着self.save=True的每一层的输出，因为后面的层结构concat等操作要用到
        # dt: 在profile中做性能评估时使用
        y, dt = [], []  # outputs
        # 遍历model的各个模块
        for m in self.model:
            # m.f 就是该层的输入来源，如果不为-1那就不是从上一层而来
            if m.f != -1:  # if not from previous layer
                # from 参数指向的网络层输出的列表
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            # 测试该网络层的性能
            if profile:
                self._profile_one_layer(m, x, dt)
            # 使用该网络层进行推导, 得到该网络层的输出
            x = m(x)  # run
            # 存放着self.save的每一层的输出，因为后面需要用来作concat等操作要用到  不在self.save层的输出就为None
            y.append(x if m.i in self.save else None)  # save output
            # 将每一层的输出结果保存到y
            if visualize:
                # 绘制该 batch 中第一张图像的特征图
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x
 
    # ===4._descale_pred():将推理结果恢复到原图尺寸(逆操作)=== #
    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        ''' 用在上面的__init__函数上
           将推理结果恢复到原图图片尺寸  Test Time Augmentation(TTA)中用到
           :params p: 推理结果
           :params flips: 翻转标记(2-ud上下, 3-lr左右)
           :params scale: 图片缩放比例
           :params img_size: 原图图片尺寸
        '''
        if self.inplace:
            # 把x,y,w,h恢复成原来的大小
            p[..., :4] /= scale  # de-scale
            # bs c h w  当flips=2是对h进行变换，那就是上下进行翻转
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            # 同理flips=3是对水平进行翻转
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p
 
    # ===5._clip_augmented（）:TTA的时候对原图片进行裁剪=== #
    # 也是一种数据增强方式，用在TTA测试的时候
    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y
 
    # ===6._profile_one_layer（）:打印日志信息=== #
    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")
 
    # ===7._initialize_biases（）:初始化偏置biases信息=== #
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
 
    # ===8._print_biases（）:打印偏置biases信息=== #
    def _print_biases(self):
        """
        打印模型中最后Detect层的偏置bias信息(也可以任选哪些层bias信息)
        """
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))
 
    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights
 
    # ===9.fuse（）:将Conv2d+BN进行融合=== #
    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        """用在detect.py、val.py
        fuse model Conv2d() + BatchNorm2d() layers
        调用oneflow_utils.py中的fuse_conv_and_bn函数和common.py中Conv模块的fuseforward函数
        """
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            # 如果当前层是卷积层Conv且有bn结构, 那么就调用fuse_conv_and_bn函数讲conv和bn进行融合, 加速推理
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                # 更新卷积层
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                # 移除bn
                delattr(m, 'bn')  # remove batchnorm
                # 更新前向传播
                m.forward = m.forward_fuse  # update forward
        # 打印conv+bn融合后的模型信息
        self.info()
        return self
 
    # ===10.autoshape（）:扩展模型功能=== #
    def autoshape(self):  # add AutoShape module
        """
        add AutoShape module  直接调用common.py中的AutoShape模块  也是一个扩展模型功能的模块
        """
        LOGGER.info('Adding AutoShape... ')
        #  此时模型包含前处理、推理、后处理的模块(预处理 + 推理 + nms)
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m
 
    # ===11.info():打印模型结构信息=== #
    def info(self, verbose=False, img_size=640):  # print model information
        """
        用在上面的__init__函数上调用torch_utils.py下model_info函数打印模型信息
        """
        model_info(self, verbose, img_size)
 
    # ===12._apply():将模块转移到 CPU/ GPU上=== #
    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self
 
'''===============================================四、parse_model模块==================================================='''
def parse_model(d, ch):  # model_dict, input_channels(3)
 
    '''===================1. 获取对应参数============================'''
 
    # 使用 logging 模块输出列标签
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # 获取anchors，nc，depth_multiple，width_multiple，这些参数在介绍yaml时已经介绍过
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    # na: 每组先验框包含的先验框数
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # no: na * 属性数 (5 + 分类数)
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
 
    '''===================2. 开始搭建网络============================'''
 
    # 网络单元列表, 网络输出引用列表, 当前的输出通道数
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # 读取 backbone, head 中的网络单元
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # 利用 eval 函数, 读取 model 参数对应的类名
        m = eval(m) if isinstance(m, str) else m  # eval strings
        # 使用 eval 函数将字符串转换为变量
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass
 
        '''===================3. 更新当前层的参数，计算c2============================'''
        # depth gain: 控制深度，如yolov5s: n*0.33，n: 当前模块的次数(间接控制深度)
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        # 当该网络单元的参数含有: 输入通道数, 输出通道数
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
            # c1: 当前层的输入channel数; c2: 当前层的输出channel数(初定); ch: 记录着所有层的输出channel数
            c1, c2 = ch[f], args[0]
            # no=75，只有最后一层c2=no，最后一层不用控制宽度，输出channel必须是no
            if c2 != no:  # if not output
                # width gain: 控制宽度，如yolov5s: c2*0.5; c2: 当前层的最终输出channel数(间接控制宽度)
                c2 = make_divisible(c2 * gw, 8)
 
            '''===================4.使用当前层的参数搭建当前层============================'''
            # 在初始args的基础上更新，加入当前层的输入channel并更新当前层
            # [in_channels, out_channels, *args[1:]]
            args = [c1, c2, *args[1:]]
            # 如果当前层是BottleneckCSP/C3/C3TR/C3Ghost/C3x，则需要在args中加入Bottleneck的个数
            # [in_channels, out_channels, Bottleneck个数, Bool(shortcut有无标记)]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                # 在第二个位置插入bottleneck个数n
                args.insert(2, n)  # number of repeats
                # 恢复默认值1
                n = 1
        # 判断是否是归一化模块
        elif m is nn.BatchNorm2d:
            # BN层只需要返回上一层的输出channel
            args = [ch[f]]
        # 判断是否是tensor连接模块
        elif m is Concat:
            # Concat层则将f中所有的输出累加得到这层的输出channel
            c2 = sum(ch[x] for x in f)
        # 判断是否是detect模块
        elif m is Detect:
            # 在args中加入三个Detect层的输出channel
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors 几乎不执行
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract: # 不怎么用
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:  # 不怎么用
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]  # args不变
 
        '''===================5.打印和保存layers信息============================'''
        # m_: 得到当前层的module，将n个模块组合存放到m_里面
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
 
        # 打印当前层结构的一些基本信息
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        # 计算这一层的参数量
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
 
        # 把所有层结构中的from不是-1的值记下 [6,4,14,10,17,20,23]
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
 
        # 将当前层结构module加入layers中
        layers.append(m_)
        if i == 0:
            ch = [] # 去除输入channel[3]
        # 把当前层的输出channel数加入ch
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser() # 创建解析器
    # --cfg: 模型配置文件
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    # --device: 选用设备
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # --profile: 用户配置文件
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    # --test: 测试
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()# 增加后的属性赋值给args
    opt.cfg = check_yaml(opt.cfg)  # 检查YAML文件
    print_args(FILE.stem, opt) # 检测YOLO v5的github仓库是否更新,若已更新,给出提示
    device = select_device(opt.device) # 选择设备
 
    # Create model
    # 构造模型
    model = Model(opt.cfg).to(device)
    model.train()
 
    # Profile
    #用户自定义配置
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)
 
    # Test all models
    # 测试所有的模型
    if opt.test:
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')
 
    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph