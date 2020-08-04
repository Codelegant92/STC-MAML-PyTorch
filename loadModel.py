import torch
model_file = 'logs/miniimagenet1shot/cls_5.tskn_4.spttrain_1.qrytrain_15.numstep5.updatelr0.01/model-100.pth'
model = torch.load(model_file)
model.eval()
#for param_tensor in model.state_dict():
#	print(param_tensor, model.state_dict()[param_tensor].size())
print(model)