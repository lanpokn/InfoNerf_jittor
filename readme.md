该工程是清华研究生课程《计算机图形学》的大作业，主要参考了：
https://github.com/itoshiko/InfoNeRF-jittor

主要改动为：
1.读取时不进行存储工作。可能本人所用的3050ti性能较差，多余的存储一定导致崩溃
2.读取ckpt的model文件指定文件名而不是文件夹
3.软件工程相关的大量优化，使得代码结构和文件结构更清晰
4.显示优化，可以实时得知迭代次数
5.删除除psnr外的metric，因为大作业并没有要求相关指标
#TODO
6.类型的全面升级。原本的代码有nerf，有train，就是没有infonerf，让人摸不到头脑。因此新版代码在此处进行了全面升级，将原本的nerf和新版infonerf都抽象出来，重写几乎整个train流程