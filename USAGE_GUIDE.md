# TGM-DLM Usage Guide
**Last Updated:**: 2025-05-20 16:48
## TensorBoard
手动启动来查看训练进度：
`tensorboard --logdir=../../checkpoints/tensorboard_logs`
启动后，通过浏览器访问默认地址`http://localhost:6006`来查看训练指标、损失曲线等信息

在远程服务器上运行，可能需要设置转发端口或指定主机