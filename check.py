# import torch
# import torch_ema
# from torch.distributed.checkpoint import load_state_dict
#
# context = 'Pre_train_model/CelebA'
# generator = torch.load('Output/Ear_256_16_4_20230413_1927_autodl/models/Ear_256_16_4_20230413_1927/model_26.pt')
# generator.load_state_dict(torch.load('Output/Ear_256_16_4_20230413_1927_autodl/models/Ear_256_16_4_20230413_1927/model_26.pt', map_location='cuda')) # 加载.pt文件
# # generator = torch.load('Output/EarOutputDir3/EarOutputDir3/ema.pth')
# # discriminator = torch.load(context + '/discriminator.pth')
# print(generator)
# # print(discriminator)

# import torch
# import torch.nn as nn
# from torchsummary import summary
#
# generator = torch.load('Output/Ear_256_16_4_20230413_1927_autodl/models/Ear_256_16_4_20230413_1927/model_26.pt')
# # model_without_classifier = nn.Sequential(*list(generator.children())[:-1])
# summary(generator, input_size=(3, 224, 224)) # 打印模型概述

# import torch.onnx
#
# #Function to Convert to ONNX
# def Convert_ONNX():
#
#     # set the model to inference mode
#     model.eval()
#
#     # Let's create a dummy input tensor
#     dummy_input = torch.randn(1, 256, requires_grad=True)
#
#     # Export the model
#     torch.onnx.export(model,         # model being run
#          dummy_input,       # model input (or a tuple for multiple inputs)
#          "ImageClassifier.onnx",       # where to save the model
#          export_params=True,  # store the trained parameter weights inside the model file
#          opset_version=10,    # the ONNX version to export the model to
#          do_constant_folding=True,  # whether to execute constant folding for optimization
#          input_names = ['modelInput'],   # the model's input names
#          output_names = ['modelOutput'], # the model's output names
#          dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes
#                                 'modelOutput' : {0 : 'batch_size'}})
#     print(" ")
#     print('Model has been converted to ONNX')
#
#
# if __name__ == "__main__":
#     # Let's build our model
#     # train(5)
#     # print('Finished Training')
#
#     # Test which classes performed well
#     # testAccuracy()
#
#     # Let's load the model we just created and test the accuracy per label
#     model = Network()
#     path = "myFirstModel.pth"
#     model.load_state_dict(torch.load(path))
#
#     # Test with batch of images
#     # testBatch()
#     # Test how the classes performed
#     # testClassess()
#
#     # Conversion to ONNX
#     Convert_ONNX()

import torch
import torch.nn
import onnx

model = torch.load('Output/Ear_256_16_4_20230413_1927_autodl/models/Ear_256_16_4_20230413_1927/model_26.pt')
model.eval()

input_names = ['input']
output_names = ['output']

x = torch.randn(1, 3, 32, 32, requires_grad=True)

torch.onnx.export(model, x, 'model_26.onnx', input_names=input_names, output_names=output_names, verbose='True')