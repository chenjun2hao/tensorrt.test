#include <torch/extension.h>
#include <torch/script.h>
#include <iostream>
#include <string>
#include <sstream>
#include <NvInfer.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/torch.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;

namespace torch2trt {


// class LReLU : public BasePlugin
class InterpolatePlugin : public IPluginV2 
{
public:
    InterpolatePlugin(std::vector<int64_t> size, std::string mode, bool align_corners) {};
      
    InterpolatePlugin(const char *data, size_t length){};
      
    // create from serialized data
    InterpolatePlugin(const std::string &data) {  };

    int getNbOutputs() const override {};

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override{    };

    int initialize() override{};

    void terminate() override{};

    size_t getWorkspaceSize(int maxBatchSize) const override{};

    int enqueue(
        int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override{        };

    size_t getSerializationSize() const override{};

    void serialize(void* buffer) const override{};

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type,
        PluginFormat format, int maxBatchSize) override{};

    bool supportsFormat(DataType type, PluginFormat format) const override{};

    const char* getPluginType() const override{};

    const char* getPluginVersion() const override{};

    void destroy() override{};

    IPluginV2* clone() const override{};

    void setPluginNamespace(const char* libNamespace) override{}

    const char* getPluginNamespace() const override{}

private:
    // float mNegSlope;
    // int mBatchDim;
      // configured by user
    std::vector<int64_t> size;
    std::string mode;
    bool align_corners;
};


    
// class InterpolatePlugin : public IPluginV2 {
// // class InterpolatePlugin : public IPluginV2DynamicExt {
// private:
    
//   // configured by class
//   at::TensorOptions tensor_options;
//   std::vector<int64_t> input_sizes;
//   std::vector<int64_t> output_sizes;
//   DataType dtype;
    
//   // configured by user
//   std::vector<int64_t> size;
//   std::string mode;
//   bool align_corners;

// public:
    
//   // create from arguments
//   InterpolatePlugin(std::vector<int64_t> size, std::string mode, bool align_corners)
//   {};
    
//   InterpolatePlugin(const char *data, size_t length){};
    
//   // create from serialized data
//   InterpolatePlugin(const std::string &data) {
//   };
    
//   const char* getPluginType() const override {
//     return "interpolate";
//   };

//   const char* getPluginVersion() const override {
//     return "1";
//   }

//   int getNbOutputs() const override {
//     return 1;
//   } 

//   // 3维[3,-1,-1]变成[1,3,-1,-1],"内定"
//   Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
//     Dims dims;
//     return dims;
//   }

//   bool supportsFormat(DataType type, PluginFormat format) const override {
//     return true;
//   }

//   void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims,
//       int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override {
    
//   }

//   int initialize() override {
//     return 0;
//   }

//   void terminate() override {}

//   // 中间数据的显存大小
//   size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }

//   // op前向推理函数
//   int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override {
//     return 0;
//   }

//   // 返回序列化时需要的字节数
//   size_t getSerializationSize() const override {  }
  
//   // 把参数和权重按照顺序序列化到 buffer 
//   void serialize(void* buffer) const override {  }

//   void destroy() override {}

//   // 主要用于传递不变的权重和参数，复制多份
//   IPluginV2* clone() const override {
//     return new InterpolatePlugin(size, mode, align_corners);
//   }

//   // 为插件设置namespace名字
//   void setPluginNamespace(const char* pluginNamespace) override {}

//   const char *getPluginNamespace() const override {
//     return "torch2trt";
//   }

// };

// class InterpolatePluginCreator : public IPluginCreator {
// public:
//   InterpolatePluginCreator() {}

//   const char *getPluginNamespace() const override {
//     return "torch2trt";
//   }

//   const char *getPluginName() const override {
//     return "interpolate";
//   }

//   const char *getPluginVersion() const override {
//     return "1";
//   }

//   // 什么时候调用？？？
//   IPluginV2 *deserializePlugin(const char *name, const void *data, size_t length) override {
//     return new InterpolatePlugin((const char*) data, length);
//   }

//   void setPluginNamespace(const char *N) override {}

//   // 通过工厂函数 creator 创建插件 
//   const PluginFieldCollection *getFieldNames() override { return nullptr; }

//   IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) override { return nullptr; }

// };


// REGISTER_TENSORRT_PLUGIN(InterpolatePluginCreator);       // 注册
    
} // namespace torch2trt
