# 可信来源

本迁移包的项目约束来自当前项目已确认的设计决策；下列外部资源用于核对傅里叶传播、FFT实现、自动微分、优化与相位恢复背景。

## 物理与数学

- [MIT OpenCourseWare：Fraunhofer Diffraction; Fourier Transforms and Theorems](https://ocw.mit.edu/courses/2-71-optics-spring-2009/resources/mit2_71s09_lec17/)
- [Optica教程：Fundamentals to Emerging Concepts and Applications of Metasurfaces for Flat Optics](https://doi.org/10.1364/AOP.541854)
- [Phase Retrieval Algorithms: A Comparison — Fienup, 1982](https://doi.org/10.1364/AO.21.002758)
- [Phase Retrieval: An Overview of Recent Developments](https://arxiv.org/abs/1510.07713)
- [Multidimensional-Encrypted Meta-Optics Storage Empowered by Diffraction-Order Decoupling](https://doi.org/10.1002/adma.202419322)

## 软件实现

- [PyTorch：torch.fft.fft2](https://docs.pytorch.org/docs/stable/generated/torch.fft.fft2.html)
- [PyTorch：torch.fft.fftshift](https://docs.pytorch.org/docs/stable/generated/torch.fft.fftshift.html)
- [PyTorch：Automatic Differentiation with torch.autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [PyTorch：Tensor.detach](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html)
- [PyTorch：torch.optim.Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)

## 迁移时的使用原则

- 外部资料用于解释和核对，不得被用作静默修改本项目固定物理前向的理由；
- 若发现当前前向与新物理证据冲突，应暂停本重建项目并单独立项；
- 所有正式结果仍须通过本包定义的固定前向回归测试和原始强度验收。
