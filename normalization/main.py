import blur as blur
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import os.path as osp
from cv2.mat_wrapper import Mat
from torch import Size
from torchvision.transforms import GaussianBlur

from macenko import MacenkoNormalizer
from vahadane import VahadaneNormalizer
import mmcv
# # import histomicstk.preprocessing.color_normalization.deconvolution_based_normalization as deconvolution_based_normalization
train_ann = '/root/autodl-tmp/data/myData/images/single/'
val_ann = 'data/myData/images/validation1'
train_results = 'results/single'
val_results = 'results/validation1'

# def standard_transfrom(standard_img,method = 'V'):
#     if method == 'V':
#         stain_method = VahadaneNormalizer()
#         stain_method.fit(standard_img)
#     else:
#         stain_method = MacenkoNormalizer()
#         stain_method.fit(standard_img)
#     return stain_method
#
# def read_image(path):
#     img = cv2.imread(path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # opencv default color space is BGR, change it to RGB
#     p = np.percentile(img, 90)
#     img = np.clip(img * 255.0 / p, 0, 255).astype(np.uint8)
#     return img
#
#
# def main(stain_method):
# #     path='data/myDa):
#
#     # 处理训练集的标注图片
#     for img_name in os.listdir(train_ann):
#
#         print(img_name)
#         if osp.splitext(img_name)[1] == '.png':
#             img = cv2.imread(osp.join(train_ann, img_name))
#
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img2 = stain_method.transform(img)
#             img2 = img2[:, :, [2, 1, 0]]
#             cv2.imwrite(osp.join(train_results, osp.splitext(img_name)[0] + '.png'), img2, [cv2.IMWRITE_PNG_COMPRESSION, 0])
#
#
#     # # 处理测试集的标注图片
#     # for img_name in os.listdir(val_ann):
#     #
#     #     print(img_name)
#     #     if osp.splitext(img_name)[1] == '.png':
#     #         img = cv2.imread(osp.join(val_ann, img_name))
#     #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     #         img2 = stain_method.transform(img)
#     #         img2 = img2[:, :, [2, 1, 0]]
#     #         cv2.imwrite(osp.join(val_results, osp.splitext(img_name)[0] + '.png'), img2,
#     #                     [cv2.IMWRITE_PNG_COMPRESSION, 0])
#
#     print('Done!')
#
#
# if __name__ == '__main__':
#     path = '/root/autodl-tmp/159.png'
#     sttd = read_image(path)
#     stain_method = standard_transfrom(sttd, method='V')
#     main(stain_method)
#
#     # # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # path = '/tmp/pycharm_project_892/data/myData/images/training/012.png'
#     # sttd = read_image(path)
#     # # img = mmcv.imread(osp.join(train_ann, '012.png'))
#     # img = cv2.imread(osp.join(train_ann, '001.png'))
#     # # img = cv2.imread(osp.join(train_ann, '162.png'))
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # stain_method = standard_transfrom(sttd, method='V')
#     # img2 = stain_method.transform(img)
#     # img2 = img2[:,:,[2,1,0]]
#     # plt.imshow(img2)
#     # plt.show()
#     # # r,g,b =cv2.split(img2)
#     # # img2 =cv2.merge(b,g,r)
#     # # plt.figure(figsize=(512.0/144, 512.0/144),dpi=144)
#     # # plt.axis('off')  # 去坐标轴
#     # # plt.xticks([])  # 去 x 轴刻度
#     # # plt.yticks([])  # 去 y 轴刻度    # plt.imshow(img)   , [cv2.IMWRITE_PNG_COMPRESSION, 0]
#     # cv2.imwrite(osp.join('/tmp/pycharm_project_892',  '1.png'),img2, [cv2.IMWRITE_PNG_COMPRESSION, 0])








# import staintools
#
#     # Read data
# target = staintools.read_image("./data/my_target_image.png")
# to_transform = staintools.read_image("./data/my_image_to_transform.png")
#
#     # Standardize brightness (optional, can improve the tissue mask calculation)
# target = staintools.LuminosityStandardizer.standardize(target)
# to_transform = staintools.LuminosityStandardizer.standardize(to_transform)
#
#     # Stain normalize
# normalizer = staintools.StainNormalizer(method='vahadane')
# normalizer.fit(target)
# transformed = normalizer.transform(to_transform)












    # img = mmcv.imread(osp.join(train_ann, '001.png'))
    # main()
    # path='data/myData/images/training/012.png'
    # sttd=read_image(path)
#     plt.imshow(sttd)
#     plt.show()

    # stain_method = standard_transfrom(sttd, method='M')
    # img2 = stain_method.transform(img)
    # plt.imshow(img2)
    # plt.show()

#     img=cv2.imread('/tmp/pycharm_project_892/data/myData/images/training/012.png')
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #     Mat blur,usm;
# #     GaussianBlur(img, blur, Size(0, 0), 25);
# #     addWeighted(img, 1.5, blur, -0.5, 0, usm);
# #     plt.imshow("usm", usm);
#     img2 = stain_method.transform(img)
#     plt.figure()
#     # plt.subplot(1,2,1)
#     # plt.imshow(img)
#     # plt.subplot(1,2,2)
#     plt.imshow(img2)
#     plt.axis('off')  # 去坐标轴
#     plt.xticks([])  # 去 x 轴刻度
#     plt.yticks([])  # 去 y 轴刻度
#     plt.savefig("test.png", bbox_inches='tight', pad_inches=0)
#     plt.show()
#
#
#
# # if __name__ == '__main__':
# #     path='training/121.png'
# #     sttd=read_image(path)
# #     # plt.imshow(sttd)
# #     # plt.show()
# #     # stain_method = standard_transfrom(sttd, method='M')
# #     img=cv2.imread('training/068.png')
# #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # #     Mat blur,usm;
# # #     GaussianBlur(img, blur, Size(0, 0), 25);
# # #     addWeighted(img, 1.5, blur, -0.5, 0, usm);
# # #     plt.imshow("usm", usm);
# #     img2 = stain_normalization(img,sttd)
# #     plt.figure()
# #     # plt.subplot(1,2,1)
# #     # plt.imshow(img)
# #     # plt.subplot(1,2,2)
# #     plt.imshow(img2)
# #     plt.axis('off')  # 去坐标轴
# #     plt.xticks([])  # 去 x 轴刻度
# #     plt.yticks([])  # 去 y 轴刻度
# #     plt.savefig("test.png", bbox_inches='tight', pad_inches=0)
# #     plt.show()



























import cv2
import numpy as np
import matplotlib.pyplot as plt

def main(kernel2):
#     path='data/myDa):

    # 处理训练集的标注图片
    for img_name in os.listdir(train_ann):

        print(img_name)
        if osp.splitext(img_name)[1] == '.png':
            img = cv2.imread(osp.join(train_ann, img_name))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            Laplace(img, kernel2,osp.join(train_results, osp.splitext(img_name)[0] + '.png'))
            # img2 = img2[:, :, [2, 1, 0]]
            # cv2.imwrite(osp.join(train_results, osp.splitext(img_name)[0] + '.png'), img2, [cv2.IMWRITE_PNG_COMPRESSION, 0])


    # # 处理测试集的标注图片
    # for img_name in os.listdir(val_ann):
    #
    #     print(img_name)
    #     if osp.splitext(img_name)[1] == '.png':
    #         img = cv2.imread(osp.join(val_ann, img_name))
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         Laplace(img, kernel2,osp.join(val_results, osp.splitext(img_name)[0] + '.png'))
    #         # img2 = img2[:, :, [2, 1, 0]]
    #         # cv2.imwrite(osp.join(val_results, osp.splitext(img_name)[0] + '.png'), img2,
    #         #             [cv2.IMWRITE_PNG_COMPRESSION, 0])

    print('Done!')

def Laplace(img, kernel,path):

    des_8U = cv2.filter2D(img, -1, kernel=kernel, borderType=cv2.BORDER_DEFAULT)
    des_16S = cv2.filter2D(img, ddepth=cv2.CV_16SC1, kernel=kernel, borderType=cv2.BORDER_DEFAULT)

    g = img - des_16S
    g[g<0] = 0
    g[g>255] = 255

    g = g[:, :, [2, 1, 0]]
    # plt.imshow(g)
    # plt.show()
    cv2.imwrite(path, g, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    # plt.figure(figsize=(10,14))
    #
    # # origin, des_8U, des_16S, filtered
    # plt.subplot(221)
    # plt.imshow(img)
    # plt.title('origin')
    #
    # plt.subplot(222)
    # plt.imshow(des_8U)
    # plt.title('des-8U')
    #
    # plt.subplot(223)
    # plt.imshow(des_16S)
    # plt.title('des-16S')
    #
    # plt.subplot(224)
    # plt.imshow(g)
    # plt.title('g')
    # plt.show()

path='/root/autodl-tmp/data/清晰化1.png'
f = cv2.imread(path)
f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB) # opencv default color space is BGR, change it to RGB

kernel1 = np.asarray([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])

# Laplace(f, kernel1)

kernel2 = np.asarray([[1, 1, 1],
                      [1, -8, 1],
                      [1, 1, 1]])



main(kernel2)
