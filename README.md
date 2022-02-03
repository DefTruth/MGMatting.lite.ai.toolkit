# MGMatting.lite.ai.toolkit
使用 🍅🍅 Lite.AI.ToolKit C++工具箱来跑MGMatting人像抠图的一些案例(https://github.com/DefTruth/lite.ai.toolkit) , 包含ONNXRuntime C++、MNN、TNN版本。

<div align='center'>
  <img src='examples/resources/input.jpg' height="150px" width="150px">
  <img src='examples/resources/mask.png' height="150px" width="150px">
  <img src='resources/pha.jpg' height="150px" width="150px">
  <img src='resources/fgr.jpg' height="150px" width="150px">
  <img src='resources/merge.jpg' height="150px" width="150px">
</div>    

如果觉得有用，不妨给个Star⭐️🌟支持一下吧~ 🙃🤪🍀

## 2. C++版本源码

MGMatting C++ 版本的源码包含ONNXRuntime、MNN和TNN三个版本，源码可以在 [lite.ai.toolkit](https://github.com/DefTruth/lite.ai.toolkit) 工具箱中找到。本项目主要介绍如何基于 [lite.ai.toolkit](https://github.com/DefTruth/lite.ai.toolkit) 工具箱，直接使用MGMatting来跑人像抠图。需要说明的是，本项目是基于MacOS下编译的 [liblite.ai.toolkit.v0.1.0.dylib](https://github.com/DefTruth/yolox.lite.ai.toolkit/blob/main/lite.ai.toolkit/lib) 来实现的，对于使用MacOS的用户，可以直接下载本项目包含的*liblite.ai.toolkit.v0.1.0*动态库和其他依赖库进行使用。而非MacOS用户，则需要从[lite.ai.toolkit](https://github.com/DefTruth/lite.ai.toolkit) 中下载源码进行编译。[lite.ai.toolkit](https://github.com/DefTruth/lite.ai.toolkit) c++工具箱目前包含80+流行的开源模型，就不多介绍了，只是平时顺手捏的，整合了自己学习过程中接触到的一些模型，感兴趣的同学可以去看看。
* [mgmatting.cpp](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/ort/cv/mgmatting.cpp)
* [mgmatting.h](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/ort/cv/mgmatting.h)
* [mnn_mgmatting.cpp](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/mnn/cv/mnn_mgmatting.cpp)
* [mnn_mgmatting.h](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/mnn/cv/mnn_mgmatting.h)
* [tnn_mgmatting.cpp](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/tnn/cv/tnn_mgmatting.cpp)
* [tnn_mgmatting.h](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/tnn/cv/tnn_mgmatting.h)

ONNXRuntime C++、MNN和TNN版本的推理实现均已测试通过，欢迎白嫖~  


## 3. 模型文件

### 3.1 ONNX模型文件
可以从我提供的链接下载 ([Baidu Drive](https://pan.baidu.com/s/1elUGcx7CZkkjEoYhTMwTRQ) code: 8gin) 。


|                 Class                 |      Pretrained ONNX Files      |              Rename or Converted From (Repo)              | Size  |
| :-----------------------------------: | :-----------------------------: | :-------------------------------------------------------: | :---: |  
| *lite::cv::matting::MGMatting* |   MGMatting-DIM-100k.onnx   | [MGMatting](https://github.com/yucornetto/MGMatting) | 113Mb |
| *lite::cv::matting::MGMatting* |   MGMatting-RWP-100k.onnx   | [MGMatting](https://github.com/yucornetto/MGMatting) | 113Mb |

### 3.2 MNN模型文件
MNN模型文件下载地址，([Baidu Drive](https://pan.baidu.com/s/1KyO-bCYUv6qPq2M8BH_Okg) code: 9v63) 。

|                 Class                 |      Pretrained MNN Files      |              Rename or Converted From (Repo)              | Size  |
| :-----------------------------------: | :-----------------------------: | :-------------------------------------------------------: | :---: |
| *lite::mnn::cv::matting::MGMatting* |   MGMatting-DIM-100k.mnn   | [MGMatting](https://github.com/yucornetto/MGMatting) | 113Mb |
| *lite::mnn::cv::matting::MGMatting* |   MGMatting-RWP-100k.mnn   | [MGMatting](https://github.com/yucornetto/MGMatting) | 113Mb |


### 3.3 TNN模型文件
TNN模型文件下载地址，([Baidu Drive](https://pan.baidu.com/s/1lvM2YKyUbEc5HKVtqITpcw) code: 6o6k) 。

|                 Class                 |      Pretrained TNN Files      |              Rename or Converted From (Repo)              | Size  |
| :-----------------------------------: | :-----------------------------: | :-------------------------------------------------------: | :---: |
| *lite::tnn::cv::matting::MGMatting* |   MGMatting-DIM-100k.opt.tnnproto&tnnmodel   | [MGMatting](https://github.com/yucornetto/MGMatting) | 113Mb |
| *lite::tnn::cv::matting::MGMatting* |   MGMatting-RWP-100k.opt.tnnproto&tnnmodel   | [MGMatting](https://github.com/yucornetto/MGMatting) | 113Mb |


## 4. 接口文档

在[lite.ai.toolkit](https://github.com/DefTruth/lite.ai.toolkit) 中，MGMatting的实现类为：

```c++
class LITE_EXPORTS lite::cv::face::detect::MGMatting;
class LITE_EXPORTS lite::mnn::cv::face::detect::MGMatting;
class LITE_EXPORTS lite::tnn::cv::face::detect::MGMatting;
```  

该类型目前包含1公共接口`detect`用于进行目标检测。
```c++
public:
    /**
     * Image Matting Using MGMatting(https://github.com/yucornetto/MGMatting)
     * @param mat: cv::Mat BGR HWC, source image
     * @param mask: cv::Mat Gray, guidance mask.
     * @param guidance_threshold: int, guidance threshold..
     * @param content: types::MattingContent to catch the detected results.
     */
    void detect(const cv::Mat &mat, cv::Mat &mask, types::MattingContent &content,
                bool remove_noise = false, unsigned int guidance_threshold = 128);
```
`detect`接口的输入参数说明：
* mat: cv::Mat类型，BGR格式。
* mask: cv::Mat类型，Gray格式，指导抠图的mask，可以是coarse-binary-map/trimap/coarse-matte中的任意一种; 
* guidance_threshold：guidance mask阈值，参考MGMatting论文和官方仓库，使用默认的128即可；
* remove_noise：是否移检测到的除小的连通区域，默认true；
* content: types::MattingContent类型，用来保存检测的结果，包含类型为cv::Mat的三个成员，分别是
    * `fgr_mat`: `cv::Mat (H,W,C=3) BGR` 格式，值范围为0~255 的 `CV_8UC3`, 用于保存估计的前景
    * `pha_mat`:` cv::Mat (H,W,C=1)` 值范围为0.~1.的 `CV_32FC1`, 用于保存估计的alpha(matte)值
    * `merge_mat`: `cv::Mat (H,W,C=3) BGR` 格式，值范围为0~255 的 `CV_8UC3`, 用于保存根据pha融合前景背景的合成图像
    * `flag`: bool 类型标志位，表示是否检测成功

## 5. 使用案例
这里测试使用的是MGMatting-DIM-100k版本的模型，你可以尝试使用其他版本的模型。

### 5.1 ONNXRuntime版本
```c++
#include "lite/lite.h"

static void test_default()
{
    std::string onnx_path = "../hub/onnx/cv/MGMatting-DIM-100k.onnx";
    std::string test_img_path = "../resources/input.jpg";
    std::string test_mask_path = "../resources/mask.png";
    std::string save_fgr_path = "../logs/fgr.jpg";
    std::string save_pha_path = "../logs/pha.jpg";
    std::string save_merge_path = "../logs/merge.jpg";
    
    auto *mgmatting = new lite::cv::matting::MGMatting(onnx_path, 16); // 16 threads
    
    lite::types::MattingContent content;
    cv::Mat img_bgr = cv::imread(test_img_path);
    cv::Mat mask = cv::imread(test_mask_path, cv::IMREAD_GRAYSCALE);
    
    // 1. image matting.
    mgmatting->detect(img_bgr, mask, content, true);
    
    if (content.flag)
    {
        if (!content.fgr_mat.empty()) cv::imwrite(save_fgr_path, content.fgr_mat);
        if (!content.pha_mat.empty()) cv::imwrite(save_pha_path, content.pha_mat * 255.);
        if (!content.merge_mat.empty()) cv::imwrite(save_merge_path, content.merge_mat);
        std::cout << "Default Version MGMatting Done!" << std::endl;
    }
    
    delete mgmatting;
}
```  

### 5.2 MNN版本
```c++
#include "lite/lite.h"

static void test_mnn()
{
#ifdef ENABLE_MNN
    std::string mnn_path = "../hub/mnn/cv/MGMatting-DIM-100k.mnn";
    std::string test_img_path = "../resources/input.jpg";
    std::string test_mask_path = "../resources/mask.png";
    std::string save_fgr_path = "../logs/fgr_mnn.jpg";
    std::string save_pha_path = "../logs/pha_mnn.jpg";
    std::string save_merge_path = "../logs/merge_mnn.jpg";
    
    auto *mgmatting = new lite::mnn::cv::matting::MGMatting(mnn_path, 16); // 16 threads
    
    lite::types::MattingContent content;
    cv::Mat img_bgr = cv::imread(test_img_path);
    cv::Mat mask = cv::imread(test_mask_path, cv::IMREAD_GRAYSCALE);
    
    // 1. image matting.
    mgmatting->detect(img_bgr, mask, content, true);
    
    if (content.flag)
    {
        if (!content.fgr_mat.empty()) cv::imwrite(save_fgr_path, content.fgr_mat);
        if (!content.pha_mat.empty()) cv::imwrite(save_pha_path, content.pha_mat * 255.);
        if (!content.merge_mat.empty()) cv::imwrite(save_merge_path, content.merge_mat);
        std::cout << "MNN Version MGMatting Done!" << std::endl;
    }
    
    delete mgmatting;
#endif
}
```  

### 5.3 TNN版本
```c++
#include "lite/lite.h"

static void test_tnn()
{
#ifdef ENABLE_TNN
    std::string proto_path = "../hub/tnn/cv/MGMatting-DIM-100k.opt.tnnproto";
    std::string model_path = "../hub/tnn/cv/MGMatting-DIM-100k.opt.tnnmodel";
    std::string test_img_path = "../resources/input.jpg";
    std::string test_mask_path = "../resources/mask.png";
    std::string save_fgr_path = "../logs/fgr_tnn.jpg";
    std::string save_pha_path = "../logs/pha_tnn.jpg";
    std::string save_merge_path = "../logs/merge_tnn.jpg";
    
    auto *mgmatting = new lite::tnn::cv::matting::MGMatting(proto_path, model_path, 16); // 16 threads
    
    lite::types::MattingContent content;
    cv::Mat img_bgr = cv::imread(test_img_path);
    cv::Mat mask = cv::imread(test_mask_path, cv::IMREAD_GRAYSCALE);
    
    // 1. image matting.
    mgmatting->detect(img_bgr, mask, content, true);
    
    if (content.flag)
    {
        if (!content.fgr_mat.empty()) cv::imwrite(save_fgr_path, content.fgr_mat);
        if (!content.pha_mat.empty()) cv::imwrite(save_pha_path, content.pha_mat * 255.);
        if (!content.merge_mat.empty()) cv::imwrite(save_merge_path, content.merge_mat);
        std::cout << "TNN Version MGMatting Done!" << std::endl;
    }
    
    delete mgmatting;
#endif
}
```  

* 输出结果为:

<div align='center'>
  <img src='examples/resources/input.jpg' height="150px" width="150px">
  <img src='examples/resources/mask.png' height="150px" width="150px">
  <img src='resources/pha.jpg' height="150px" width="150px">
  <img src='resources/fgr.jpg' height="150px" width="150px">
  <img src='resources/merge.jpg' height="150px" width="150px">
</div>    

## 6. 编译运行
在MacOS下可以直接编译运行本项目，无需下载其他依赖库。其他系统则需要从[lite.ai.toolkit](https://github.com/DefTruth/lite.ai.toolkit) 中下载源码先编译*lite.ai.toolkit.v0.1.0*动态库。
```shell
git clone --depth=1 https://github.com/DefTruth/MGMatting.lite.ai.toolkit.git
cd MGMatting.lite.ai.toolkit 
sh ./build.sh
```  

* CMakeLists.txt设置

```cmake
cmake_minimum_required(VERSION 3.17)
project(MGMatting.lite.ai.toolkit)

set(CMAKE_CXX_STANDARD 11)

# setting up lite.ai.toolkit
set(LITE_AI_DIR ${CMAKE_SOURCE_DIR}/lite.ai.toolkit)
set(LITE_AI_INCLUDE_DIR ${LITE_AI_DIR}/include)
set(LITE_AI_LIBRARY_DIR ${LITE_AI_DIR}/lib)
include_directories(${LITE_AI_INCLUDE_DIR})
link_directories(${LITE_AI_LIBRARY_DIR})

set(OpenCV_LIBS
        opencv_highgui
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        opencv_video
        opencv_videoio
        )
# add your executable
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/examples/build)

add_executable(lite_mgmatting examples/test_lite_mgmatting.cpp)
target_link_libraries(lite_mgmatting
        lite.ai.toolkit
        onnxruntime
        MNN  # need, if built lite.ai.toolkit with ENABLE_MNN=ON,  default OFF
        ncnn # need, if built lite.ai.toolkit with ENABLE_NCNN=ON, default OFF
        TNN  # need, if built lite.ai.toolkit with ENABLE_TNN=ON,  default OFF
        ${OpenCV_LIBS})  # link lite.ai.toolkit & other libs.
```

* building && testing information:
```shell
[ 50%] Building CXX object CMakeFiles/lite_mgmatting.dir/examples/test_lite_mgmatting.cpp.o
[100%] Linking CXX executable lite_mgmatting
[100%] Built target lite_mgmatting
Testing Start ...
LITEORT_DEBUG LogId: ../hub/onnx/cv/MGMatting-DIM-100k.onnx
=============== Inputs ==============
Dynamic Input: image Init [1,3,512,512]
Dynamic Input: mask Init [1,1,512,512]
=============== Outputs ==============
Dynamic Output 0: alpha_os1
Dynamic Output 1: alpha_os4
Dynamic Output 2: alpha_os8
Default Version MGMatting Done!
LITEORT_DEBUG LogId: ../hub/onnx/cv/MGMatting-DIM-100k.onnx
=============== Inputs ==============
Dynamic Input: image Init [1,3,512,512]
Dynamic Input: mask Init [1,1,512,512]
=============== Outputs ==============
Dynamic Output 0: alpha_os1
Dynamic Output 1: alpha_os4
Dynamic Output 2: alpha_os8
ONNXRuntime Version MGMatting Done!
Compute Shape Error for 598
LITEMNN_DEBUG LogId: ../hub/mnn/cv/MGMatting-DIM-100k.mnn
=============== Input-Dims ==============
        **Tensor shape**: 1, 3, 0, 0, 
        **Tensor shape**: 1, 1, 0, 0, 
Dimension Type: (CAFFE/PyTorch/ONNX)NCHW
=============== Output-Dims ==============
getSessionOutputAll done!
Output: alpha_os1:      **Tensor shape**: 0, 0, 0, 0, 
Output: alpha_os4:      **Tensor shape**: 0, 0, 0, 0, 
Output: alpha_os8:      **Tensor shape**: 0, 0, 0, 0, 
========================================
MNN Version MGMatting Done!
LITETNN_DEBUG LogId: ../hub/tnn/cv/MGMatting-DIM-100k.opt.tnnproto
=============== Input-Dims ==============
image: [1 3 1024 1024 ]
mask: [1 1 1024 1024 ]
Input Data Format: NCHW
=============== Output-Dims ==============
alpha_os1: [1 1 1024 1024 ]
alpha_os4: [1 1 1024 1024 ]
alpha_os8: [1 1 1024 1024 ]
========================================
TNN Version MGMatting Done!
Testing Successful !
```  

![](resources/10.jpg)
