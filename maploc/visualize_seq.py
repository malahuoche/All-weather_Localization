from pprint import pprint
from pathlib import Path
import torch
import yaml
from torchmetrics import MetricCollection
from omegaconf import OmegaConf as OC
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pytorch_lightning import seed_everything
import maploc
from maploc.data import BoreasDataModule
from maploc.module import GenericModule
from maploc.evaluation.run import resolve_checkpoint_path
from maploc.utils.viz_2d import plot_images, features_to_RGB, save_plot, add_text
from maploc.utils.viz_localization import likelihood_overlay, plot_pose, plot_dense_rotations, add_circle_inset
from maploc.data.torch import unbatch_to_device
from maploc.osm.viz import Colormap, plot_nodes
from maploc.models.metrics import Location2DError, AngleError
from maploc.models.voting import argmax_xyr, fuse_gps

torch.set_grad_enabled(False);
plt.rcParams.update({'figure.max_open_warning': 0})
experiment = "/home/classlab2/root/OrienterNet/experiments/OrienterNet_MGL_0207/last.ckpt"
# experiment = "experiment_name"  # find the best checkpoint
# experiment = "experiment_name/checkpoint-step=N.ckpt"  # a given checkpoint
path = resolve_checkpoint_path(experiment)
print(path)
override = {'model': {"num_rotations": 256, "apply_map_prior": True}}
model = GenericModule.load_from_checkpoint(
    path, strict=True, find_best=not experiment.endswith('.ckpt'), cfg=override)
model = model.eval().cuda()
conf = OC.load(Path(maploc.__file__).parent / 'conf/data/boreas.yaml')
conf = OC.merge(conf, OC.create(yaml.full_load("""
data_dir: "/home/classlab2/16T/datasets/boreas."
max_init_error: 0
loading:
    val: {batch_size: 1, num_workers: 0}
    train: ${.val}
random: false
return_gps: true
init_from_gps: true
augmentation: {rot90: false, flip: false, image: {apply: false}}
""")))
OC.resolve(conf)
assert model.cfg.data.resize_image == conf.resize_image
dataset =BoreasDataModule(conf)
dataset.prepare_data()
dataset.setup()
dset, chunk2idx = dataset.sequence_dataset("val", max_length=100000, max_delay_s=0.01, max_inter_dist=10000)
for c in chunk2idx:
    print(c)
chunk_key = ('boreas-2020-12-01-13-26/radar', 0)
indices = chunk2idx[chunk_key]
batches = [dset[i] for i in indices]
plot_images([b['image'].permute(1, 2, 0) for b in batches], dpi=25)

from maploc.data.sequential import unpack_batches
images, canvas, maps, yaws_gt, uv_gt, xy_gt, xy_gps = unpack_batches(batches)
maps = list(map(Colormap.apply, maps))

location = dset.names[indices][0][0]
latlon_gps = dset.data['gps_position'][indices]
# print("latlon_gps 的形状:", latlon_gps.shape)
# print(latlon_gps)
xy_gps = dset.tile_managers[location].projection.project(latlon_gps)
from maploc.utils.geo import BoundaryBox
bbox_seq = BoundaryBox(xy_gps.min(0), xy_gps.max(0)) + dset.cfg.crop_size_meters + 16
canvas_total = dset.tile_managers[location].query(bbox_seq)
map_total = Colormap.apply(canvas_total.raster)
plot_images([map_total], dpi=50)
plt.scatter(*canvas_total.to_uv(xy_gps).T, c='black', lw=0)
from maploc.data.torch import collate, unbatch_to_device
from tqdm import tqdm
from maploc.models.sequential import RigidAligner, log_softmax_spatial
aligner = RigidAligner(num_rotations=model.model.conf.num_rotations)
aligner2 = RigidAligner(num_rotations=model.model.conf.num_rotations)
preds = []
for data in tqdm(batches):
    batch = model.transfer_batch_to_device(collate([data]), model.device, 0)
    pred = model(batch)
    xy = data['canvas'].to_xy(batch["uv"].squeeze(0).double())
    yaw = batch["roll_pitch_yaw"].squeeze(0)[-1].double()
    aligner.update_with_ref(pred['scores'][0], data['canvas'], xy, yaw)
#     aligner2.update(pred['log_probs'][0], canvas, xy, yaw)
    pred = unbatch_to_device(pred)
    pred['log_probs_seq'] = log_softmax_spatial(aligner.belief).cpu()
    pred['xyr_max_seq'] = argmax_xyr(pred['log_probs_seq'])
    preds.append(pred)
    del pred
uvt_p = torch.stack([p["uvr_max"] for p in preds])
# uvt_seq = torch.stack([p["uvr_max_seq"] for p in preds])
xy_p = torch.stack([c.to_xy(uv) for c, uv in zip(canvas, uvt_p[:,:2])])
# xy_seq = torch.stack([c.to_xy(uv) for c, uv in zip(canvas, uvt_seq[:,:2])])
logprobs = torch.stack([p["log_probs"] for p in preds])
# logprobs_seq = torch.stack([p["log_probs_seq"] for p in preds])

from maploc.models.metrics import angle_error
err_xy = torch.norm(uv_gt - uvt_p[:, :2], dim=1)/dataset.cfg.pixel_per_meter
err_yaw = angle_error(yaws_gt, uvt_p[:, 2])
# err_xy_seq = torch.norm(uv_gt - uvt_seq[:, :2], dim=1)/dataset.cfg.pixel_per_meter
# err_yaw_seq = angle_error(yaws_gt, uvt_seq[:, 2])

plot_images([lp.max(-1).values for lp in logprobs], cmaps="jet", dpi=25)
# plot_images([lp.max(-1).values for lp in logprobs_seq], cmaps="jet", dpi=25)
print('xy error single vs seq:')
# pprint(torch.stack([err_xy, err_xy_seq], -1).numpy().tolist())
print('yaw error single vs seq:')
# pprint(torch.stack([err_yaw, err_yaw_seq], -1).numpy().tolist())
def bbox_to_extent(bbox):
    return np.r_[bbox.min_, bbox.max_][[0,2,1,3]]

show_heatmap = lambda p: likelihood_overlay(p.max(-1).values.exp().numpy(), p_rgb=0.02, p_alpha=1/50)

map_gray = map_total.mean(-1)
xy_gt = dset.data['t_c2w'][indices][:, :2]
# for idx, (im, xy_gt_, b, lp, lps) in enumerate(zip(images, xy_gt, batches, logprobs, logprobs_seq)):
#     plot_images([im]+[map_gray]*2, titles=['input image', 'single image', 'sequential'])
#     axes = plt.gcf().axes
#     for ax in axes[1:]:
#         ax.plot(*xy_gt.T, c='blue', lw=1, marker='o', ms=5, mfc='none')
#         ax.images[0].set_extent(bbox_to_extent(canvas_total.bbox))
#         ax.autoscale(enable=False)
#     for ax, xy_ in zip(axes[1:], [xy_p, xy_seq]):
#         ax.plot(*xy_[:idx+1].T, c='red', lw=1, marker='o', ms=5)
#     bbox = b['canvas'].bbox
#     axes[1].imshow(show_heatmap(lp), alpha=1.0, cmap='jet', extent=bbox_to_extent(bbox), zorder=10)
#     axes[2].imshow(show_heatmap(lps), alpha=1.0, cmap='jet', extent=bbox_to_extent(bbox), zorder=10)
#     axins = add_circle_inset(axes[2], xy_gt_);
#     axins.scatter(*xy_gt_, c='none', zorder=10, ec='blue', s=70)
import matplotlib.pyplot as plt
accumulated_xy_gt = []
import csv
output_file_path = '/home/classlab2/root/OrienterNet/notebooks/fusion_predict_snow.csv'
with open(output_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Index', 'X', 'Y'])  # 写入表头
    for idx, xy in enumerate(xy_p):
        writer.writerow([idx, xy[0].item(), xy[1].item()]) 
output_file_path = '/home/classlab2/root/OrienterNet/notebooks/gt_predict.csv'
# with open(output_file_path, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Index', 'X', 'Y'])  # 写入表头
#     for idx, xy in enumerate(xy_gt):
#         xy=xy.T
#         writer.writerow([idx, xy[0].item(), xy[1].item()]) 
for idx, (im, xy_gt_, b, lp) in enumerate(zip(images, xy_gt, batches, logprobs)):
    # 绘制输入图像和一个灰度图，设置对应的标题
    accumulated_xy_gt.append(xy_gt_)
    plot_images([im] + [map_gray], titles=['input image', 'single image'])
    axes = plt.gcf().axes  # 获取当前图形中的所有轴

    # 在第二个轴上（单图）绘制实际位置的轨迹
    ax = axes[1]  # 只关注单图部分
    ax.plot(*xy_gt_.T, c='blue', lw=0, marker='o', ms=1)  # 绘制实际位置
    # if accumulated_xy_gt:  # 检查是否有累积的实际位置点
    #     ax.plot(*np.array(accumulated_xy_gt).T, c='blue', lw=0, marker='o', ms=1, mfc='none')
    ax.images[0].set_extent(bbox_to_extent(canvas_total.bbox))  # 设置图像范围
    ax.autoscale(enable=False)  # 禁用自动缩放
    ax.plot(*xy_p[:idx + 1].T, c='red', lw=0, marker='o', ms=1, mfc='none')  # 绘制预测位置
    # 在单图上显示概率热图
    # bbox = b['canvas'].bbox  # 获取当前批次的边界框
    # ax.imshow(show_heatmap(lp), alpha=1.0, cmap='jet', extent=bbox_to_extent(bbox), zorder=10)  # 显示概率热图

    # 添加圆形放大镜（如果需要）
    # axins = add_circle_inset(ax, xy_gt_)  # 在单图上添加圆形放大镜
    # axins.scatter(*xy_gt_, c='none', zorder=10, ec='blue', s=70)  # 在放大镜中绘制实际位置标记

    # 保存图像
    plt.savefig(f"/home/classlab2/root/OrienterNet/notebooks/viz/output_image_{idx}.png")  # 保存图像到当前目录，可以根据需要修改路径和文件名
    plt.close()  # 关闭当前图形，以便开始绘制下一个