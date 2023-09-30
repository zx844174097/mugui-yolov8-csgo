# 输入库
import torch

# 查看版本
print(torch.__version__)
# 查看是否支持GPU
print(torch.cuda.is_available())
# 查看GPU数量
print(torch.cuda.device_count())
# 查看当前GPU索引号，索引号从0开始

i = torch.cuda.current_device()

print(i)
# 根据索引号查看GPU名字
print(torch.cuda.get_device_name(i))
# 查看GPU属性
print(torch.cuda.get_device_properties(i))
# 查看GPU当前使用情况
print(torch.cuda.memory_allocated())
# 查看对应CUDA版本号
print(torch.backends.cudnn.version())
print(torch.version.cuda)
# 查看CUDA是否可用
print(torch.backends.cudnn.enabled)
quit(1)
