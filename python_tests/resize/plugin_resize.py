from pathlib import WindowsPath
import tensorrt as trt
from torch._C import dtype
from torch2trt.plugins import *


size = (224,224)
mode = "bilinear"
align_corners = False

from torch2trt.plugins import InterpolatePlugin
PLUGIN_NAME = 'interpolate'
registry = trt.get_plugin_registry()
for c in registry.plugin_creator_list:
    print("plugin name:", c.name, "plugin namespace:", c.plugin_namespace)
    if c.name == PLUGIN_NAME and c.plugin_namespace == 'torch2trt':
        creator = c
        break
torch2trt_plugin = InterpolatePlugin(size=size, mode=mode, align_corners=align_corners)        # python 封装的 c++ 类, <class 'torch2trt.plugins.InterpolatePlugin'>
data_string = torch2trt_plugin.serializeToString()
intert = creator.deserialize_plugin(PLUGIN_NAME, data_string)         # 这两应该是一样的，为啥要用后面的, <class 'tensorrt.tensorrt.IPluginV2'>


# 属性设置
shape = (1, 3, 112, 112)
        

# 创建网络
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# 添加输入
input_trt = network.add_input(
        name = "input0",
        shape = shape,
        dtype = trt.float32
)

# 添加resize, 采用插件
layer = network.add_plugin_v2([input_trt], intert)
layer.name = "interprate"
output = layer.get_output(0)

output.name = "output"
# output.location = trt.TensorLocation.DEVICE
# output.dtype = trt.float32
network.mark_output(output)

# 添加resize，采用内置
# layer = network.add_resize(input_trt)
# layer.scales = [1] + [2, 2]
# layer.resize_mode = trt.ResizeMode.LINEAR
# output = layer.get_output(0)

# output.name = "output"
# output.location = trt.TensorLocation.DEVICE
# output.dtype = trt.float32
# network.mark_output(output)

# 创建engine,context
builder.max_workspace_size = 1 << 25
builder.fp16_mode = False
builder.max_batch_size = 1
builder.strict_type_constraints = False

engine = builder.build_cuda_engine(network)

# 属性设置
from PIL import Image
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

img_path = "../../imgs/1.png"

# 读入图片数据
img = Image.open(img_path)
img = img.resize(shape[2:])
img = np.array(img, dtype=np.float32)

# 申请内存
def allocate_buffers(engine):
    # host buffer
    h_input  = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype( engine.get_binding_dtype(0) ))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype( engine.get_binding_dtype(1) ))
    # device buffer
    d_input  = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()
    return h_input, h_output, d_input, d_output, stream

h_input, h_output, d_input, d_output, stream = allocate_buffers(engine)

np.copyto(h_input, img.transpose([2, 0, 1]).ravel())


# 推理
# cuda.memcpy_htod_async(d_input, h_input, stream)      # 类似异步的 c++ 中需要处理 stream 事件
cuda.memcpy_htod(d_input, h_input)                      # 类似同步的
print("python in int", int(d_input))                    # int(input), 数据的地址，c++ cout<< intput[0]
print("python out int", int(d_output))                  # 输出数据的地址
with engine.create_execution_context() as context:
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

out_img = np.array(h_output).reshape((3, 224, 224))
out_img = out_img.transpose([1,2,0])

import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.imshow(img.astype(np.uint8))
plt.subplot(1,2,2)
plt.imshow(out_img.astype(np.uint8))
plt.savefig("./inter.jpg")
plt.show()


