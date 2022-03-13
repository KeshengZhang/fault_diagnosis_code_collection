# fault_diagnosis_code_collection
A collection of Fault Diagnosis python codes


### (一)CPLS1.0
#### 安装相关依赖包
~~~python
pip install -r requirements.txt
~~~
#### 使用
~~~ python
python CPLS_demo.py
~~~

### 参考
> Qin, S. J. , & Zheng, Y. . (2013). Quality‐relevant and process‐relevant fault monitoring with concurrent projection to latent structures. *AIChE Journal**,* *59*.

### (二)DiCCA1.0
#### 安装环境

在文件目录下 

~~~python
pip install -r requirements.txt
~~~

### 使用
python DiCCA_demo.py

#### 参考
> Yining Dong ∗, ∗∗ S. Joe Qin ∗∗, & ∗∗∗. (2018). Dynamic-inner canonical correlation and causality analysis for high dimensional time series data. IFAC-PapersOnLine, 51(18), 476-481.


### (三)DiPCAv1

#### 安装环境

在文件目录下 

~~~python
pip install -r requirements.txt
~~~

#### 使用

~~~ python
python Dipca_demo.py
~~~

#### 参考

> Dong, Y. , & Qin, S. J. . (2017). A novel dynamic pca algorithm for dynamic data modeling and process monitoring. *Journal of Process Control*, S095915241730094X.



### (四)DiPLS1.0

#### 安装相关依赖包
~~~python
pip install -r requirements.txt
~~~
#### 使用
~~~ python
python DiPLS_demo.py
~~~



### (五)DiPLS2.0

在DiPLS1.0基础上，将DiPLS封装成类

#### 安装相关依赖包
~~~python
pip install -r requirements.txt
~~~

#### 使用
~~~ python
python DiPLS_demo.py
~~~

### 参考

> Dong, Y. , & Qin, S. J. . (2015). Dynamic-inner partial least squares for dynamic data modeling. IFAC-PapersOnLine, 48(8), 117-122.



##  (六)PCA for Fault Diagnosis

### 描述

程序包括两个应用案例, 一个是数值仿真案例，另外一个TE过程.

### 主要功能

- 利用累积方差贡献率选取主元的数量
- 贡献图
- 重构贡献图

### 运行方法

对于数值仿真案例, 运行demo_numerical_example.m

对于TE过程, 运行demo_TE.m

### 参考文献

待补充



### (七)TPLS1.0
#### 安装相关依赖包
~~~python
pip install -r requirements.txt
~~~
#### 使用
~~~ python
python TPLS_demo.py
~~~

### 参考
> Zhou, Donghua, Li, Gang, Qin, & S., et al. (2009). Total projection to latent structures for process monitoring. AIChE Journal.

### (八)TPLS2.0

#### 安装相关依赖包

~~~python
pip install -r requirements.txt
~~~

#### 使用

~~~ python
python TPLS_demo.py
~~~

#### 相比于1.0主要改进

将TPLS相关函数封装成类



### 参考

> Zhou, Donghua, Li, Gang, Qin, & S., et al. (2009). Total projection to latent structures for process monitoring. AIChE Journal.


Code From：LeiHu
