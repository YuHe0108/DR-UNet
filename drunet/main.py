import argparse
import os
import pathlib
import time

import SegModels
import cv2 as cv
import data
import losses
import matplotlib.pyplot as plt
import module
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from tensorflow import keras
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tf_package import utils as tool_utils
from tf_package.CV import cv_utils
# 自定义包
from tf_package.Segment import performance

config = ConfigProto()
config.gpu_options.allow_growth = True

# GPUS = tf.config.experimental.list_physical_devices('GPU')
# print(GPUS)
# tf.config.experimental.set_memory_growth(GPUS[0], True)
print(tf.__version__)

# 1. 参数设置
parser = argparse.ArgumentParser(description="Segment Use Args")
parser.add_argument('--model-name', default='my_model', type=str)
parser.add_argument('--dims', default=64, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch-size', default=16, type=int)
parser.add_argument('--lr', default=2e-4, type=float)

# 训练数据、测试、验证参数设置
parser.add_argument('--height', default=128, type=int)
parser.add_argument('--width', default=128, type=int)
parser.add_argument('--channel', default=1, type=int)
parser.add_argument('--pred-height', default=4 * 128, type=int)
parser.add_argument('--pred-width', default=4 * 128, type=int)
parser.add_argument('--total-samples', default=7331, type=int)
parser.add_argument('--invalid-samples', default=1024, type=int)
parser.add_argument('--regularizer', default=False, type=bool)
parser.add_argument('--record-dir', default=r'data\\tf_records', type=str,
                    help='the save dir of tfrecord')
parser.add_argument('--train-record-name', type=str,
                    default=r'ct_128_channel_1_7331_no_augment_train',
                    help='the train record save name')
parser.add_argument('--invalid-record-name', type=str,
                    default=r'ct_128_channel_1_1024_no_augment_test',
                    help='the invalid record save name')
parser.add_argument('--test-image-dir', default=r'data\\test_images', type=str)
parser.add_argument('--gt-mask-dir', default=r'data\\invalid\\invalid_data\\mask', type=str)
parser.add_argument('--invalid-volume-dir', default=r'data\\invalid\\invalid_volume\\images',
                    type=str, help='测试模型的对患者出血量体积的估计')
args = parser.parse_args()


class Segmentation:
    def __init__(self, params):
        self.params = params
        self.input_shape = [params.height, params.width, params.channel]
        self.mask_shape = [params.height, params.width, 1]
        self.model_name = params.model_name
        self.crop_height = params.pred_height
        self.crop_width = params.pred_width
        self.regularizer = params.regularizer

        # 1. 获取图像分割模型
        self.seg_model = SegModels.get_seg_model(
            model_name=params.model_name, input_shape=self.input_shape, dims=params.dims)
        self.seg_model.summary()

        # 2、优化函数
        self.optim = tf.keras.optimizers.Adam(lr=params.lr)

        # 3、训练模型权重、推断图像的保存地址
        self.save_dir = str(params.model_name).upper()
        # 保存模型的训练权重
        self.weight_save_dir = os.path.join(self.save_dir, 'checkpoint')
        # 每隔 1 个epoch， 预测invalid-images，用于检验模型的分割性能
        self.pred_invalid_save_dir = os.path.join(self.save_dir, 'invalid_pred')
        # 对预测的invalid-images 裁剪成单个image，于gt一起用于计算分割参数
        self.invalid_crop_save_dir = os.path.join(self.save_dir, 'invalid_pred_crop')
        # 每隔固定的训练批次，预测test image
        self.pred_test_save_dir = os.path.join(self.save_dir, 'test_pred')
        tool_utils.check_file([
            self.save_dir, self.weight_save_dir, self.pred_invalid_save_dir,
            self.pred_test_save_dir, self.invalid_crop_save_dir]
        )

        # 4. 需要保存参数
        train_steps = tf.Variable(0, tf.int32)
        self.save_ckpt = tf.train.Checkpoint(
            train_steps=train_steps, seg_model=self.seg_model, model_optimizer=self.optim)
        self.save_manger = tf.train.CheckpointManager(
            self.save_ckpt, directory=self.weight_save_dir, max_to_keep=1)

        # 5. 设置损失函数
        self.loss_fun = losses.loss_fun(params.model_name)

    def load_model(self):
        if self.save_manger.latest_checkpoint:
            self.save_ckpt.restore(self.save_manger.latest_checkpoint)
            print('加载模型: {}'.format(self.save_manger.latest_checkpoint))
            print('模型之前已经经过{}次训练，训练继续！'.format(self.save_ckpt.train_steps.numpy()))
        else:
            print('重新训练模型！')
        return

    @tf.function
    def train_step(self, inputs, target):
        tf.keras.backend.set_learning_phase(True)

        with tf.GradientTape() as tape:
            pred_mask = self.seg_model(inputs)
            loss = self.loss_fun(target, pred_mask)
            if self.regularizer:
                loss = tf.reduce_sum(loss) + tf.reduce_sum(self.seg_model.losses)
        gradient = tape.gradient(loss, self.seg_model.trainable_variables)
        self.optim.apply_gradients(zip(gradient, self.seg_model.trainable_variables))
        return tf.reduce_mean(loss)

    @tf.function
    def inference(self, inputs):
        tf.keras.backend.set_learning_phase(True)
        pred = self.seg_model(inputs)
        return pred

    @staticmethod
    def calculate_volume_by_mask(mask_dir, save_dir, model_name, dpi=96, thickness=0.45):
        all_mask_file_paths = tool_utils.list_file(mask_dir)

        pd_record = pd.DataFrame(columns=['file_name', 'Volume'])
        for file_dir in tqdm.tqdm(all_mask_file_paths):
            file_name = pathlib.Path(file_dir).stem  # 测试文件夹的名称
            # 结算每一个mask目录下的血肿体积
            each_blood_volume = module.calculate_volume(file_dir, thickness=thickness, dpi=dpi)
            # 根据每层血肿的面积计算出血量
            pd_record = pd_record.append({'file_name': file_name, 'Volume': each_blood_volume}, ignore_index=True)
            pd_record.to_csv(
                os.path.join(save_dir, '{}_{}.csv'.format(model_name, file_name)), index=True, header=True)
        return

    def predict_blood_volume(self, input_dir, save_dir, calc_nums=-1, dpi=96, thickness=0.45):
        """
        :param input_dir: 测试出血量图像的目录, 目录下存放多个文件夹，每个文件夹表示一位患者的CT图像
        :param save_dir: 预测的分割图像保存目录
        :param calc_nums: 预测文件夹中的多少张图像
        :param dpi: 图像参数
        :param thickness: 切片厚度
        :return:
        """
        # 加载权重
        self.load_model()
        save_pred_images_dir = os.path.join(save_dir, 'pred_images')  # 保存预测的图像
        save_pred_csv_dir = os.path.join(save_dir, 'pred_csv')  # 保存每一位患者血肿体积和面积的csv文件
        tool_utils.check_file([save_pred_images_dir, save_pred_csv_dir])
        # 包含了目录下所有的文件夹路径
        all_file_dirs = tool_utils.list_file(input_dir)

        cost_time_list = []
        total_images = 0
        for file_dir in tqdm.tqdm(all_file_dirs):
            file_name = pathlib.Path(file_dir).stem  # 测试文件夹的名称

            # 加载invalid图像: 图像文件的名字、原始0-255 uint8图像、0-1 norm图像
            image_names, ori_images, normed_images = data.get_test_data(
                test_data_path=file_dir, image_shape=self.input_shape, image_nums=calc_nums)
            total_images += len(image_names)
            print(normed_images.shape)

            start_time = time.time()
            pred_mask = self.inference(normed_images)
            end_time = time.time()
            print('FPS: {}'.format(pred_mask.shape[0] / (end_time - start_time)),
                  pred_mask.shape, end_time - start_time)

            denorm_pred_mask = module.reverse_pred_image(pred_mask.numpy())  # (image_nums, 256, 256, 1)
            # 如果只有一张图像，则需要在第一个维度上expand_dims()
            if denorm_pred_mask.ndim == 2 and self.input_shape[-1] == 1:
                denorm_pred_mask = np.expand_dims(denorm_pred_mask, 0)

            drawed_images = []
            blood_areas = []
            pd_record = pd.DataFrame(columns=['image_name', 'Square Centimeter', 'Volume'])
            for index in range(denorm_pred_mask.shape[0]):
                drawed_image, blood_area = module.draw_contours(ori_images[index], denorm_pred_mask[index], dpi=dpi)
                drawed_images.append(drawed_image)
                blood_areas.append(blood_area)
                pd_record = pd_record.append({'image_name': image_names[index], 'Square Centimeter': blood_area},
                                             ignore_index=True)

            # 输入了原始原图图像、原始图像的文件名、绘制了出血轮廓的image、预测分割结果
            one_pred_save_dir = os.path.join(save_pred_images_dir, file_name)
            module.save_invalid_data(ori_images, drawed_images, denorm_pred_mask,
                                     image_names, reshape=True, save_dir=one_pred_save_dir)
            # 根据每层血肿的面积计算出血量
            blood_volume = module.count_volume(blood_areas, thickness=thickness)  # 计算出血量
            pd_record = pd_record.append({'Volume': blood_volume}, ignore_index=True)
            pd_record.to_csv(os.path.join(save_pred_csv_dir, '{}_{}.csv'.format(self.model_name, file_name)),
                             index=True, header=True)
            cost_time_list.append(end_time - start_time)
            print('FileName: {} time: {}'.format(file_name, end_time - start_time))
        print('total_time: {:.2f}, mean_time: {:.2f}, total_images: {}'.format(
            np.sum(cost_time_list), np.mean(cost_time_list), total_images))
        return

    def predict_and_save(self, input_dir, save_dir, calc_nums=-1, batch_size=16):
        """ predict 出血图像并保存
        :param input_dir: input_dir 文件夹下存放了若干等待测试的图像
        :param save_dir:  保存模型预测的分割图像的文件目录
        :param calc_nums: 从所在目录下的取多少张图像参与计算
        :param batch_size: 每次测试多少个图像
        :return:
        """
        mask_save_dir = os.path.join(save_dir, 'pred_mask')
        drawed_save_dir = os.path.join(save_dir, 'drawed_image')
        tool_utils.check_file([mask_save_dir, drawed_save_dir])
        self.load_model()

        # 加载图像
        test_image_list = tool_utils.list_file(input_dir)
        # if len(test_image_list) > 128:
        for index in range(len(test_image_list) // 128 + 1):
            input_test_list = test_image_list[index * 128:(index + 1) * 128]

            # 如果测试数据量太大，则分批进行
            image_names, ori_images, normed_images = data.get_test_data(
                test_data_path=input_test_list, image_shape=self.input_shape, image_nums=-1,
            )
            if calc_nums != -1:
                ori_images = ori_images[:calc_nums]
                normed_images = normed_images[:calc_nums]
                image_names = image_names[:calc_nums]

            inference_times = normed_images.shape[0] // batch_size + 1
            for inference_time in range(inference_times):
                # 分批次
                this_normed_images = normed_images[
                                     inference_time * batch_size:(inference_time + 1) * batch_size, ...]
                this_ori_images = ori_images[
                                  inference_time * batch_size:(inference_time + 1) * batch_size, ...]
                this_image_names = image_names[
                                   inference_time * batch_size:(inference_time + 1) * batch_size]

                # 对每批次进行训练
                start_time = time.time()
                this_pred_mask = self.inference(this_normed_images)
                print('cost time: {}'.format(time.time() - start_time))
                this_denorm_pred_mask = module.reverse_pred_image(this_pred_mask.numpy())
                if ori_images.shape[0] == 1:
                    # 如果输入只有一张图像
                    this_denorm_pred_mask = np.expand_dims(this_denorm_pred_mask, 0)

                for i in range(this_denorm_pred_mask.shape[0]):
                    if str(self.model_name).lower() == 'fcn':
                        bin_denorm_pred_mask = module.binary_image(this_denorm_pred_mask[i])
                    else:
                        bin_denorm_pred_mask = this_denorm_pred_mask[i]
                    # 将mask绘制在原始图像上
                    this_drawed_image, this_blood_area = module.draw_contours(
                        this_ori_images[i], bin_denorm_pred_mask, dpi=96)
                    cv.imwrite(os.path.join(
                        mask_save_dir, '{}'.format(this_image_names[i])), bin_denorm_pred_mask)
                    cv.imwrite(os.path.join(
                        drawed_save_dir, '{}'.format(this_image_names[i])), this_drawed_image)
        return

    def train(self, start_epoch=1):
        # 获取训练和验证数据集
        train_data = data.get_tfrecord_data(
            self.params.record_dir, self.params.train_record_name,
            self.input_shape, batch_size=self.params.batch_size)
        self.load_model()

        # 记录模型的训练
        pd_record = pd.DataFrame(columns=['Epoch', 'Iteration', 'Loss', 'Time'])

        # 测试图像的name、0-255的测试图像、归一化（0-1）的测试图像
        data_name, original_test_image, norm_test_image = data.get_test_data(
            test_data_path=self.params.test_image_dir, image_shape=self.input_shape, image_nums=-1)

        start_time = time.time()
        best_dice = 0.0
        for epoch in range(start_epoch, self.params.epochs):
            for train_image, gt_mask in tqdm.tqdm(
                    train_data, total=self.params.total_samples // self.params.batch_size):
                self.save_ckpt.train_steps.assign_add(1)
                iteration = self.save_ckpt.train_steps.numpy()

                # 训练步骤
                train_loss = self.train_step(train_image, gt_mask)
                if iteration % 100 == 0:
                    print('Epoch: {}, Iteration: {}, Loss: {:.2f}, Time: {:.2f} s'.format(
                        epoch, iteration, train_loss, time.time() - start_time))

                    # 测试步骤
                    test_pred = self.inference(norm_test_image)
                    module.save_images(image_shape=self.mask_shape, pred=test_pred,
                                       save_path=self.pred_test_save_dir, index=iteration, split=False)

                    # 保存训练的参数
                    pd_record = pd_record.append({
                        'Epoch': epoch, 'Iteration': iteration, 'Loss': train_loss.numpy(),
                        'Time': time.time() - start_time}, ignore_index=True)
                    pd_record.to_csv(os.path.join(
                        self.save_dir, '{}_record.csv'.format(self.params.model_name)), index=True, header=True)

            m_dice = self.invalid(epoch)  # 预测验证数据集步骤
            if m_dice > best_dice:
                best_dice = m_dice
                print('最好的Dice:{}'.format(best_dice))
                self.save_manger.save(checkpoint_number=epoch)
        return

    def invalid(self, epoch):
        invalid_data = data.get_tfrecord_data(
            self.params.record_dir, self.params.invalid_record_name,
            self.input_shape, batch_size=self.params.batch_size, shuffle=False)

        for index, (invalid_image, invalid_mask) in enumerate(
                tqdm.tqdm(invalid_data, total=self.params.invalid_samples // self.params.batch_size + 1)):
            invalid_pred = self.inference(invalid_image)
            epoch_pred_save_dir = os.path.join(
                self.pred_invalid_save_dir, f'epoch_{epoch}'
            )
            module.save_images(image_shape=self.mask_shape, pred=invalid_pred,
                               save_path=epoch_pred_save_dir, index=f'{index}', split=False)

        # 测试模型性能
        epoch_croped_save_dir = os.path.join(
            self.invalid_crop_save_dir, f'epoch_{epoch}'
        )
        cv_utils.crop_image(
            epoch_pred_save_dir, epoch_croped_save_dir,
            self.crop_width, self.crop_height,
            self.input_shape[0], self.input_shape[1]
        )
        m_dice, m_iou, m_precision, m_recall = performance.save_performace_to_csv(
            pred_dir=epoch_croped_save_dir,
            gt_dir=self.params.gt_mask_dir,
            img_resize=(self.params.height, self.params.width),
            csv_save_name=f'{self.model_name}_epoch_{epoch}',
            csv_save_path=epoch_croped_save_dir,
        )
        return m_dice


if __name__ == '__main__':
    TRAIN = True
    segment = Segmentation(args)
    if TRAIN:
        segment.train(start_epoch=1)
    else:
        # segment.predict_blood_volume(r'data\invalid\invalid_volume\images',
        #                              r'data\fcn_predict_volume')
        # segment.calculate_volume_by_mask(r'data\invalid\invalid_volume\masks', '', 'ground_truth')
        segment.predict_and_save(r'D:\Users\Ying\Desktop\Uneven\image',
                                 r'D:\Users\Ying\Desktop\Uneven\predict')
