
class Args():
    # For training
    path_ir = ''
    cuda = 1
    lr = 1e-3
    epochs = 2
    batch_size = 4
    device = 0;

    # Network Parameters
    # Height = 128
    # Width = 128

    n = 64  # number of filters
    channel = 1  # 1 - gray, 3 - RGB
    s = 3  # filter size
    stride = 1
    num_block = 4  
    train_num = 20000

    resume_model = None
    save_fusion_model = "./model"
    save_loss_dir = "./model/loss_v1"

    # model=pcf
    #         dataset=seg_args.dataset,
    #         backbone=seg_args.backbone,
    #         aux=seg_args.aux,

    # model = 'fcn'        # 分割模型名称（如 'fcn', 'deeplab'）
    # # dataset = 'custom'    # 数据集名称
    # backbone = 'resnet50'# 主干网络
    # aux = False          # 是否使用辅助损失
    # num_classes = 9     # 类别数





