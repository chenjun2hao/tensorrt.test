# data:
# version:
#   - tensorrt==7.2.2.3

import tensorrt as trt


# 属性设置
shape = (3, 112, 112)

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


# 添加resize，采用内置
layer = network.add_resize(input_trt)
layer.scales = [1] + [2, 2]
layer.resize_mode = trt.ResizeMode.LINEAR
output = layer.get_output(0)

output.name = "output"
output.location = trt.TensorLocation.DEVICE
output.dtype = trt.float32
network.mark_output(output)

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

img_path = "./imgs/1.png"

# 读入图片数据
img = Image.open(img_path)
img = img.resize(shape[1:])
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
cuda.memcpy_htod_async(d_input, h_input, stream)
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
plt.show()


