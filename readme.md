This project is the final assignment for the Tsinghua University graduate course "Computer Graphics," with significant reference to the following repository: [https://github.com/itoshiko/InfoNeRF-jittor](https://github.com/itoshiko/InfoNeRF-jittor).

Jittor is used in its latest version, as using version 1.3.6.3 led to unspecified bugs.

Algorithm-related modifications include:
1. Streamlined and removed caching in code reading. Caching may cause crashes due to the lower performance of the used graphics card.
2. Completely rewrote the code for reading the model file from the checkpoint (ckpt). The original code only looked for folders and read the last one, rendering it meaningless if training crashed.
3. Revamped display, allowing real-time tracking of iteration counts.
4. Removed metrics other than PSNR since the assignment did not require additional indicators.
5. Rewrote `run_network`, removing unnecessary outer loops for speed improvement (accessing shape is O(1), as opposed to O(N) in the referenced project).
6. Significantly modified `entropyloss`.
7. After testing, found that `infoloss` had no impact in this assignment and removed related code. This made the entire project lightweight. This correction is crucial, allowing successful training of InfoNeRF with just 10,000 iterations, achieving a PSNR of 18.8.

Regarding software engineering changes:
1. Upgraded types comprehensively. The original code had "nerf" and "train" but lacked "infonerf," causing confusion. The new code underwent a complete upgrade in this regard, abstracting out both "nerf" and the new version of "infonerf." The training process was rewritten extensively, leaving "train" and "test" as external interfaces.
2. Deleted unclear code with unclear meanings (potentially related to other datasets).
3. Merged short functions with only one reference (e.g., test).
4. Restructured the file organization and clarified function references. For example, if function B is only referenced in function A, it should be defined in function A rather than externally or in other files.

Note: The interfaces for image claims have not been opened up. Only "train" and "test" are retained, but if needed, simply make "__gen_nerf_imgs" public..


该工程是清华研究生课程《计算机图形学》的大作业，主要参考了：
https://github.com/itoshiko/InfoNeRF-jittor

jittor直接用的最新版，用1.3.6.3反而有不明bug

算法相关改动为：
1.读取代码删减，cach相关完全删除。可能本人所用的显卡性能较差，多余的存储一定导致崩溃
2.读取ckpt的model文件完全重写（原本只找文件夹并且只读最后一个，一旦训练崩了就没有任何意义）
3.显示改版，可以实时得知迭代次数
4.删除除psnr外的metric，因为大作业并没有要求相关指标
5.run_network重写，如删除多余外循环来提速（访问shape是o（1），参考的工程里是O(N)
6.entropyloss大改
7.经过测试，infoloss在本次作业中完全是副作用，因此相关代码完全删除，从而使得整个工程非常轻量化。
此更正非常关键，使得infonerf可以轻松训练成功，仅仅1万轮就有18.8
（而且本来也很扯，对于边界较多的3D scene,进行这样的优化完全没有意义，只会导致整个图片模糊化。除非复现的场景特别单一
换言之，如果ray动一个小角度结果可以平滑的话，光追就不痛一个pixel内采一堆了。。。）
而信息熵就很好，让没有训练的地方，nerf的云朵也趋向于集中，还不会彻底消失

软件工程改动为：
1.类型的全面升级。原本的代码有nerf，有train，就是没有infonerf，让人摸不到头脑。因此新版代码在此处进行了全面升级，将原本的nerf和新版infonerf都抽象出来，重写几乎整个train流程。infonerf类别经过了全面整合与更新，只留了train和test作为外部接口
2.删除原本工程中的各种意义不明代码（可能与其他数据集有关）
3.将过短且只有一次引用的函数(如test等)全部合并
4.重构文件结构，并使函数引用更加清晰。比如，若函数B只在函数A中被引用，则其应该被定义在函数A中，而不是外部或其他文件

注意并没有开放图像声称的接口函数，只留了train和test,但如果有需要的话只需要将__gen_nerf_imgs改成公有即可。