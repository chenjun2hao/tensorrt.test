#include <torch/extension.h>
#include <torch/script.h>
#include <iostream>
#include <string>
#include <sstream>
#include <NvInfer.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/torch.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;
using namespace std;

namespace torch2trt {

    
// class InterpolatePlugin : public IPluginV2 {
class InterpolatePlugin : public IPluginV2DynamicExt {
// class InterpolatePlugin : public IPluginV2Ext {

private:
    
  // configured by class
  at::TensorOptions tensor_options;
  std::vector<int64_t> input_sizes;
  std::vector<int64_t> output_sizes;
  DataType dtype;
    
  // configured by user
  std::vector<int64_t> size;
  std::string mode;
  bool align_corners;

public:
  
  ////////////////////////////////// initial ///////////////////////////
  // initial
  InterpolatePlugin(std::vector<int64_t> size, std::string mode, bool align_corners) :
    size(size), mode(mode), align_corners(align_corners)
  {
    cout << "print test....................." << endl;
  }
    
  InterpolatePlugin(const char *data, size_t length) : InterpolatePlugin(std::string(data, length)) {}
    
  // create from serialized data
  InterpolatePlugin(const std::string &data) {
      deserializeFromString(data);
  }


  ////////////////////////////////// IPluginv2 methods ///////////////////////////
  const char* getPluginType() const override {
    cout << "in getplugintype......." << endl;
    return "interpolate";
  };

  const char* getPluginVersion() const override {
    cout << "in getpluginversion......." << endl;
    return "1";
  }

  int getNbOutputs() const override {
    cout << "in getnboutputs......." << endl;
    return 1;
  } 

  int initialize() override {
    cout << "in initialize......." << endl;
    // set device
    tensor_options = tensor_options.device(c10::kCUDA);

    cout << "dtype2" << int(dtype) << endl;
      
    // set data type
    if (dtype == DataType::kFLOAT) {
        tensor_options = tensor_options.dtype(c10::kFloat);
    } else if (dtype == DataType::kHALF) {
        tensor_options = tensor_options.dtype(c10::kHalf);
    }
      
    return 0;
  }

  void terminate() override { 
    cout << "in terminate......." << endl;    
  }

    // 返回序列化时需要的字节数
  size_t getSerializationSize() const override {
    return serializeToString().size();
  }
  
  // 把参数和权重按照顺序序列化到 buffer 
  void serialize(void* buffer) const override {
      cout << "in serialize......." << endl;
      std::string data = serializeToString();
      size_t size = getSerializationSize();
      data.copy((char *) buffer, size);
  }

  void destroy() override {
    cout << "in destroy......." << endl;
  }

    // 为插件设置namespace名字
  void setPluginNamespace(const char* pluginNamespace) override {}

  const char *getPluginNamespace() const override {
    return "torch2trt";
  }


  ////////////////////////////////// IPluginV2Ext methods ///////////////////////////
  nvinfer1::DataType getOutputDataType(int index,
                                      const nvinfer1::DataType *inputTypes,
                                      int nbInputs) const override {
    cout << "in getoutput datatype......." << endl;
    return inputTypes[0];
  }

  // 主要用于传递不变的权重和参数，复制多份
  IPluginV2DynamicExt *clone() const  {
    cout << "in clone......." << endl;
    cout << "mode" << mode << endl;
    for (size_t i = 0; i < size.size(); i++){
      cout << "size: " << size[i] << endl;
    }
    
    return new InterpolatePlugin(size, mode, align_corners);
  }

  // 3维[3,-1,-1]变成[1,3,-1,-1],"内定"
  DimsExprs getOutputDimensions(int outputIndex,
                              const DimsExprs* inputs, 
                              int nbInputs, 
                              IExprBuilder& exprBuilder) override{
    cout << "in get output dimensions......." << endl;
    DimsExprs out_dims;
    out_dims.nbDims = inputs->nbDims;

    // debug
    // for (size_t i = 0; i < inputs->nbDims; i++)
    // {
    //   auto tmp = inputs->d[i];
    //   cout <<"input dims:" << tmp->getConstantValue() << endl;
    // }

    out_dims.d[0] = inputs->d[0];           // batch
    out_dims.d[1] = inputs->d[1];           // channel
    for (int i = 0; i < size.size(); i++) {
      out_dims.d[i + 2] = exprBuilder.constant(size[i]);
    }

    return out_dims;
  };

  // ???
  bool supportsFormatCombination(int pos,
                                const nvinfer1::PluginTensorDesc *inOut,
                                int nbInputs, int nbOutputs) override{
    cout << "in support format combination......." << endl;
    return true;
  }

  // ???
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in,
                      int nbInputs,
                      const nvinfer1::DynamicPluginTensorDesc *out,
                      int nbOutputs) override {
    cout << "in configurePlugin......." << endl;
    // debug show
    // auto tmp = in->desc;
    // for (size_t i = 0; i < tmp.dims.nbDims; i++)
    // {
    //   cout << "in size: " << tmp.dims.d[i] << endl;
    // }
    // tmp = out->desc;
    // for (size_t i = 0; i < tmp.dims.nbDims; i++)
    // {
    //   cout << "out size: " << tmp.dims.d[i] << endl;
    // }

    // set input size
    input_sizes.resize(in->desc.dims.nbDims);
    for (size_t i = 0; i < in->desc.dims.nbDims; i++) {
      input_sizes[i] = in->desc.dims.d[i];
    }

    // set out size
    output_sizes.resize(out->desc.dims.nbDims);
    for (size_t i = 0; i < out->desc.dims.nbDims ; i++)
    {
      output_sizes[i] = out->desc.dims.d[i];
    }

    // set dtype
    dtype = in->desc.type;

  }

  // 中间数据的显存大小
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc *outputs,
                          int nbOutputs) const override {
    cout << "in get workspace size......." << endl;
    return 0;
  }

  // op前向推理函数
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
              const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace,
              cudaStream_t stream) override {
    cout << "in enqueue......." << endl;
    // int batchSize = input_sizes.size();
    // int batchSize = 1;
    // cout << batchSize << endl;
    // get input / output dimensions
    std::vector<long> batch_input_sizes = input_sizes;
    std::vector<long> batch_output_sizes = output_sizes;
    // batch_input_sizes.insert(batch_input_sizes.begin(), batchSize);
    // batch_output_sizes.insert(batch_output_sizes.begin(), batchSize);
    // debug
    for (size_t i = 0; i < batch_input_sizes.size(); i++){
      cout << "batch_input_sizes is:" << batch_input_sizes[i] << endl;
    }

    for (size_t i = 0; i < batch_output_sizes.size(); i++){
      cout << "batch_out_sizes is:" << batch_output_sizes[i] << endl;
    }

    cout << "input desc: " << int(inputDesc->type) << endl;
    cout << "output desc: " << int(outputDesc->type) << endl;

    for (size_t i = 0; i < inputDesc->dims.nbDims; i++) {
      cout << "input desc2:" << inputDesc->dims.d[i] << endl;
      cout << "output desc2:" << outputDesc->dims.d[i] << endl;
    }
    
    tensor_options = tensor_options.device(c10::kCUDA);
    tensor_options = tensor_options.dtype(c10::kFloat);

    cout << "part 1......." << endl;
    cout << "input: " << inputs[0] << endl;
    cout << "output: " << outputs[0] << endl;
    cout << "tensor device" << tensor_options.device() << endl;
    
    // create tensor wrappers
    at::Tensor input = at::from_blob((void*) inputs[0], batch_input_sizes, [](void*){}, tensor_options);
    at::Tensor output = at::from_blob(outputs[0], batch_output_sizes, [](void*){}, tensor_options);
    
    at::Tensor tm;
    if (input.is_cuda()){
      tm = input.cpu();
    }
    else{
      cout << "input is cpu" << endl;
      tm = input;
    }
    
    float* data = (float*)tm.data_ptr();
    cout << "get intput data" << endl;
    for (size_t i = 0; i < 10; i++)  {
      cout << "input data:" << data[i] << endl;
    }

    cout << "part 2......." << endl;
    // // create new torch cuda stream
    // at::cuda::CUDAStream torch_stream = at::cuda::getStreamFromPool();
    // at::cuda::CUDAStreamGuard torch_guard(torch_stream);

    // // capture current work on tensorrt cuda stream
    // cudaEvent_t event;
    // cudaEventCreate(&event);
    // cudaEventRecord(event, stream);

    // // make torch cuda stream wait on tensorrt work
    // cudaStreamWaitEvent(torch_stream.stream(), event, 0);

    cout << "part 3......." << endl;
    cout << mode << endl;
    for (size_t i = 0; i < size.size(); i++)
    {
      cout << "size is:" << size[i] << endl;
    }
    
    // enqueue work
    if (mode == "bilinear") {
      cout << "before......." << endl;
      cout << "size: " << size[0] << "," << size[1] << endl;
      cout << "in tensor dim: " << input.numel() << endl;
      cout << "out tensor dim:" << output.numel() << endl;
      cout << "in tensor nbytes: " << input.is_cuda() << endl;
      cout << "out tensor nbytes:" << output.is_cuda() << endl;     
      at::upsample_bilinear2d_out(output, input, {size[0], size[1]}, align_corners);
      cout << "after......." << endl;
    } else if (mode == "nearest") {
      at::upsample_nearest2d_out(output, input, {size[0], size[1]});
    } else if (mode == "area") {
      at::adaptive_avg_pool2d_out(output, input, {size[0], size[1]});
    } else if (mode == "bicubic") {
      at::upsample_bicubic2d_out(output, input, {size[0], size[1]}, align_corners);
    }

    cout << "part 4......." << endl;
    // // capture event on enqueued stream
    // cudaEvent_t torch_event;
    // cudaEventCreate(&torch_event);
    // cudaEventRecord(torch_event, torch_stream.stream());

    // cudaStreamWaitEvent(stream, torch_event, 0);

    // cudaEventDestroy(event);
    // cudaEventDestroy(torch_event);

    cout << "part 5......." << endl;
    return 0;
  }


////////////////////////////////// OWN methods ///////////////////////////
  void deserializeFromString(const std::string &data) {
      cout << "in deserialize from string......." << endl;
      std::istringstream data_stream(data);
      torch::serialize::InputArchive input_archive;
      input_archive.load_from(data_stream);
      {
          torch::IValue value;
          input_archive.read("size", value);
#ifdef USE_DEPRECATED_INTLIST
          size = value.toIntListRef().vec();
#else
          size = value.toIntVector();
#endif
      }
      {
          torch::IValue value;
          input_archive.read("mode", value);
          mode = value.toStringRef();
      }
      {
          torch::IValue value;
          input_archive.read("align_corners", value);
          align_corners = value.toBool();
      }
      {
          torch::IValue value;
          input_archive.read("dtype", value);
          dtype = (DataType) value.toInt();
          cout << "dtype: " << value.toInt() << endl;
      }
      {
          torch::IValue value;
          input_archive.read("input_sizes", value);
#ifdef USE_DEPRECATED_INTLIST
          input_sizes = value.toIntListRef().vec();
#else
          input_sizes = value.toIntVector();
#endif
      }
      {
          torch::IValue value;
          input_archive.read("output_sizes", value);
#ifdef USE_DEPRECATED_INTLIST
          output_sizes = value.toIntListRef().vec();
#else
          output_sizes = value.toIntVector();
#endif
      }
  }
    
  std::string serializeToString() const {
      cout << "in serialize to string......." << endl;
      torch::serialize::OutputArchive output_archive;
      output_archive.write("size", torch::IValue(size));
      output_archive.write("mode", torch::IValue(mode));
      output_archive.write("align_corners", torch::IValue(align_corners));
      output_archive.write("dtype", torch::IValue((int) dtype));
      output_archive.write("input_sizes", torch::IValue(input_sizes));
      output_archive.write("output_sizes", torch::IValue(output_sizes));
      std::ostringstream data_str;
      output_archive.save_to(data_str);
      return data_str.str();
  }

};




class InterpolatePluginCreator : public IPluginCreator {
public:
  InterpolatePluginCreator() {}

  const char *getPluginNamespace() const override {
    return "torch2trt";
  }

  const char *getPluginName() const override {
    return "interpolate";
  }

  const char *getPluginVersion() const override {
    return "1";
  }

  // 什么时候调用？？？
  IPluginV2 *deserializePlugin(const char *name, const void *data, size_t length) override {
    return new InterpolatePlugin((const char*) data, length);
  }

  void setPluginNamespace(const char *N) override {}

  // 通过工厂函数 creator 创建插件 
  const PluginFieldCollection *getFieldNames() override { return nullptr; }

  IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) override { return nullptr; }

};


REGISTER_TENSORRT_PLUGIN(InterpolatePluginCreator);       // 注册
    
} // namespace torch2trt
