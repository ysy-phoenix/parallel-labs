#import "template.typ": *

#show: project.with(
  title: "并行计算实验一报告\n——OpenMP 及 CUDA 实验环境的搭建",
  authors: (
    "PB20111701 叶升宇",
    "PB20111689 蓝俊玮",
  ),
)

#info[
  以下为 PB20111701 叶升宇的实验环境。
]
= OpenmMP 环境搭建
== 下载 CLion
本学期并行计算实验使用的 IDE 为 CLion，下载链接如下 
#emph(text(blue)[
  #link("https://www.jetbrains.com/clion/")[CLion]
])。

== 下载 MinGW64
使用 MinGW64 作为工具集，下载链接如下
#emph(text(blue)[
  #link("https://nchc.dl.sourceforge.net/project/mingw-w64/Toolchains%20targetting%20Win32/Personal%20Builds/mingw-builds/installer/mingw-w64-install.exe")[MinGW64]
])。

== 配置 LLVM + clang
前往 
#emph(text(blue)[
  #link("https://releases.llvm.org/download.html")[LLVM]
])
官网下载，c 编译器使用 LLVM clang，c++ 编译器使用 LLVM clang++，构建工具使用 Ninja(而非通常的 Make)。整体工具链配置完毕后如下：
#align(center)[
  #image("fig1.png", height: 33%, width: 90%)
]
== 配置 CMake  
编写 CMakeLists.txt 如下：
- 使用 c++ 14 标准；
- 用 `-fopenmp` 选项开启 OpenMP 支持
```Cpp
cmake_minimum_required(VERSION 3.10)
project(sort)
set(CMAKE_CXX_STANDARD 14)
find_package(OpenMP REQUIRED)
set(SOURCE_FILES main.cpp)
add_executable(sort ${SOURCE_FILES})
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
```
Cmake 相关配置如下：
#align(center)[
  #image("fig2.png", height: 37%, width: 90%)
]

== 验证配置成功
打印简单的 OpenMP 相关参数， 来验证配置成功：
#align(center)[
  #image("fig3.png", height: auto, width: 90%)
]
  
= CUDA 环境搭建
== NVIDIA 驱动的安装
依然使用 CLion 作为 IDE，配 OpenMP 时已经完成，不再赘述 。
下面检查驱动是否安装：在 Windows Terminal 中输入 nvidia-smi：
#align(center)[
  #image("fig4.png", height: auto, width: 90%)
]
#info[
  这里我是已经配好了 CUDA 环境，所以驱动和 CUDA 版本都有显示 。
]

没有安装驱动的情况下，需要到
#emph(text(blue)[
  #link("https://www.nvidia.cn/geforce/drivers/")[英伟达官网]
])
选取对应显卡的版本下载安装。

== CUDA 安装
进入 
#emph(text(blue)[
  #link("https://developer.nvidia.com/cuda-toolkit")[CUDA Toolkit 官网]
])
进行下载并安装，我选择的是 12.1 版本，同时在系统变量中加入 CUDA 路径：
#align(center)[
  #image("fig5.png", height: auto, width: 90%)
]
以及在 PATH 中加入 CUDA 的 BIN 路径：
#align(center)[
  #image("fig6.png", height: auto, width: 90%)
]

== 验证安装成功
下面通过运行 `deviceQuery.exe`、`bandwidthTest.exe`，`result = PASS` 说明安装成功： 
#align(center)[
  #image("fig7.png", height: auto, width: 90%)
  #image("fig8.png", height: auto, width: 90%)
]

== 配置 CMake
在 CLion 中，选择项目为 CUDA 可执行文件，然后编写 CMakeLists.txt 如下：
```cpp
cmake_minimum_required(VERSION 3.10)
project(cuda_examples CUDA)
set(CMAKE_CUDA_STANDARD 14)
find_package(CUDA REQUIRED)
add_executable(cuda_examples main.cu)
```
这里以老师 PPT 上的乘积求和程序为例，输出结果如下，也验证了 CUDA 配置成功：
#align(center)[
  #image("fig9.png", height: 42%, width: 90%)
]
= 其它设备参数
== CPU 参数
通过任务管理器来查看， 我的电脑是 6 核 12 线程的：
#align(center)[
  #image("fig10.png", height: 38%, width: 90%)
]
== GPU 参数
利用课程群中提供的 GPU-Z 查看 GPU 参数如下：
#align(center)[
  #image("fig11.png", height: auto, width: 90%)
]
#info[
  实际上，在先前配置 CUDA 时候通过 `deviceQuery.exe` 也可以来获取相关参数 。
]

#info[
  以下为 PB20111689 蓝俊玮的实验环境。
]

= OpenmMP 环境搭建
通过 Visual Studio 2022 进行环境搭建，在 Visual Studio 2022 可以通过配置 OpenMP Support 开启 OpenMP 2.0 的拓展支持。

可以在 Visual Studio 2022 内右键点击创建的项目，或者上方任务栏中的项目，选择属性。在打开属性页之后，点击 C/C++ —> 语言，更改如下配置如图所示，然后点应用，确定。

#align(center)[
  #image("openmp-settings.png", height: auto, width: 90%)
]

测试效果如下：

#align(center)[
  #image("openmp-running.png", height: auto, width: 90%)
]

= MPI 环境搭建
通过 Visual Studio 2022 进行环境搭建，在 #link("https://www.microsoft.com/en-us/download/details.aspx?id=57467")[Microsoft MPI v10.0] 下载 MPI，将 `msmpisetup.exe` 和 `msmpisdk.msi` 都安装下来。

接下来在 Visual Studio 2022 的项目内，打开项目属性：

1. VC++目录 -> 包含目录 -> 编辑：
添加 `D:\mpisdk\Include`（这个与个人安装路径有关）
2. VC++目录 -> 库目录 -> 编辑：
添加 `D:\mpisdk\Lib\x64`（x64 与 Visual Studio 2022 的平台活动一致）
3. C/C++-> 预处理器 -> 预处理器定义:
添加：MPICH_SKIP_MPICXX
4. C/C++ -> 代码生成 -> 运行库:
选择：多线程调试（/MTd）
5. 链接器 -> 输入 -> 附加依赖项:
添加：msmpi.lib, msmpifec.lib, msmpifmc.lib（即 Lib/X64 下的 lib 文件）

编译和运行时通过 Visual Studio 2022 的菜单栏中的“生成”的“生成解决方案”可以生成可执行文件，然后使用 Windows Terminal，在生成的可执行程序的目录下运行它，运行命令为：`mpiexec -n 6 test.exe`(其中 -n 后面跟着的是进程数量)

#info[
  参考链接：#link("https://blog.csdn.net/m0_58064058/article/details/129543203")[VS2022配置MPI环境]\
]

测试效果如下：

#align(center)[
  #image("mpi-running.png", height: auto, width: 90%)
]


= OpenCV 4.7.0 安装
使用 GitCode.net 的 OpenCV 镜像安装 OpenCV

```sh
git clone https://github.com/opencv/opencv
cmake -B opencv-build -D OPENCV_GENERATE_PKGCONFIG=ON opencv
```

#info(type: "注意", fg: red, bg: rgb("F5A7A7"))[
  `-D OPENCV_GENERATE_PKGCONFIG=ON` 这个选项十分重要，控制是否生成 pkg_config，opencv4 中如果不加这个命令，就不会生成 pkgconfig，就会导致安装后找不到 opencv4 文件。
]

`cmake` 结束后执行 `make -j8` 指令，然后执行 `make install`，由于我使用的是服务器，没有足够的权限，所以修改 `cmake_install.cmake` 中的 `CMAKE_INSTALL_PREFIX` 为 `/amax/home/junwei/opencv_usr_local` (原来为 `/usr/local`)

安装后，添加环境变量，在 `/amax/home/junwei/.bashrc` 中添加如下:

```sh
export PKG_CONFIG_PATH=/amax/home/junwei/opencv_usr_local/lib/pkgconfig
export LD_LIBRARY_PATH=/amax/home/junwei/opencv_usr_local/lib
```

添加完成后：

```sh
source ~/.bashrc
```

验证是否安装成功，查看opencv的版本：
```sh
pkg-config --modversion opencv4
```

当使用 opencv 时，可以采用下面的方法编译运行：
```sh
nvcc test.cu -o test.out -I/amax/home/junwei/opencv_usr_local/include/opencv4 -L/amax/home/junwei/opencv_usr_local/lib `pkg-config --cflags --libs opencv4`
```

其中 `-I` 表示了添加额外的搜索库的路径，`-L` 表示了加载了额外的库

#info[
  参考链接：#link("https://blog.csdn.net/qq_41340996/article/details/121319056")[Linux下安装opencv(root身份和非root普通用户安装)]\
  参考链接：#link("https://blog.csdn.net/csdn_codechina/article/details/123678909")[下载不再卡顿，OpenCV 中国镜像仓库正式启用]\
  参考链接：#link("https://blog.csdn.net/WindSunLike/article/details/104231160")[GCC中 -I、-L、-l 选项的作用]
]

= Cuda 环境
由于我使用的是实验室的服务器，因此 Cuda 的驱动都是已经安装好的，这里就不多说。

= 设备参数
我的个人电脑 CPU(AMD Ryzen 7 4800H with Radeon Graphics            2.90 GHz) 是 8 核 16 线程的，下面展示我个人电脑的 CPU 设备性能（没有 GPU）

#align(center)[
  #image("gpu-z.png", height: auto, width: 90%)
  #image("prop.png", height: auto, width: 90%)
]

下面展示服务器 GPU 设备的性能：

#align(center)[
  #image("smi.png", height: auto, width: 90%)
  #image("prop.png", height: auto, width: 90%)
]