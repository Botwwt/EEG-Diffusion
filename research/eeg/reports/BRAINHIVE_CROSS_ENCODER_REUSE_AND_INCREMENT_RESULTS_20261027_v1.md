# D15：完整复用旧ATMS，原生坐标下有小幅预测增量，未放行出图

本轮已恢复十被试**132300条真实折外T预测和16500条full-D预测**，复用了原30个ATMS模型，没有重训旧编码器。两次固定线性探针均在完整T拟合、全部8200条Dfit评价；原生CLIP坐标版本的MSE从c-only的0.941208降至0.937393，相对改善**0.4053%**，低于事前0.5%准入门槛。没有四舍五入、降低门槛或直接启动新图片。

三次GPU包装合计**76.83秒**。首次控制器启动至最后运行终态实际**1002.52秒，约16分42秒**，包含初次CPU预算保护退出、阶段间实现/审查与等待；不能把整轮说成只用了77秒。启动前实现和最终报告另计。正式D仍停止。

## 1. 本轮改变的问题与既有数据复用

D14显式c证据有配对作用，却使图片远弱于三个固定强分支。继续改变条件随机采样并没有带来新信息，因此本轮转查**不同EEG编码器是否提供可复用的互补刺激信息**。

旧ATMS输入为63×250的R4 EEG，输出为原生pooled CLIP-H/1024；BrainMLP输入为17×250，输出在HIVE F对齐链中使用。不能因两者都是1024维就当作同一坐标。

原30份逐图预测缓存已不在原路径，但full/oof0/oof1共30个3,200,413参数检查点均保留。严格CPU加载全部模型，原63通道R4输入与训练时缓存SHA完全相同；核验两折训练/排除概念集合，T的每条预测来自没有拟合其概念的模型。原ATMS曾用V选择checkpoint的历史保留，没有据此恢复独立V权限或宣称全系统独立确认。

此次只恢复必要T/D预测，不生成官方测试预测：132300条T和16500条D完整覆盖，各query一次。60条固定原始输入的CPU/GPU数值回放通过，最大相对L2为3.88e−7。原逐图缓存已缺失，所以**不声称重导文件与旧预测缓存逐位一致**。新银行保留在持久盘，以后直接复用，不再逐轮重跑30个旧模型。[输入与来源核验](../audit/cross_encoder_20261027_v1/actual/candidate/inputs_v2/completion.json)、[完整实际重导](../audit/cross_encoder_20261027_v1/actual/candidate/export/completion.json)

## 2. 固定线性条件增量探针

训练使用全部132300条真实OOF-T条件，13230个T图像F目标；F采用D12固定T均值、标量RMS和原输出affine，不作PCA或截断。两个视图分别使用T内、按已知被试计算的均值和单一RMS。第一版本撤销两编码器各自输出affine；下节解释原生ATMS坐标复核。

记标准化BrainMLP视图为C、ATMS视图为A、完整标准化视觉目标为Y：

\[
B_c=\operatorname{ridge}(C,Y),\quad B_{ca}=\operatorname{ridge}(C,A),
\quad E=A-CB_{ca},\quad R=Y-CB_c,
\]
\[
W=\operatorname{ridge}(E,R),\qquad
\hat Y=CB_c+(A-CB_{ca})W.
\]

这里的innovation只指该固定线性读出的剩余量，不是独立神经信息、误差置信度或精确条件密度。所有五个系统均固定`XᵀX/n + 0.1·mean(diag(XᵀX/n))·I`，FP32 Cholesky；没有正则网格或Dfit拟合。另有ATMS-only和同维错排innovation控制，训练/评价都按被试作固定异概念组置换；该控制不是精确条件独立性检验。

五个系统共5,242,880个线性系数，在任何Dfit目标读取前保存固定模型。全部82概念×10图×10被试＝8200 Dfit query评价完整1024维预测MSE、固定温度0.1的同概念十图log loss、Top1及余弦间隔。Top1是**视觉特征预测的十图区分**，不是图片对象准确率。

准入事前固定：相对c-only及错排控制均须MSE相对改善至少0.5%、subject/concept bootstrap95%下界>0；十图log loss也须改善且下界>0；十图Top1均值不下降。1000次、seed1729，重采样被试和概念并保留单元内全部十刺激。只通过才支持另登记图片测试；本轮没有Dcheck效果读取。[首版冻结协议](../configs/cross_encoder_20261027_v1/probe_spec.json)

## 3. 第一版失败后的坐标审计与一次固定复核

第一版joint MSE为0.963028，劣于c-only 0.941208；c-only减joint为−0.021820，描述性95%区间[−0.035774,−0.008948]。同概念log loss也由1.782682升至1.817918。不是只因一个指标不显著而停止。

CPU复核发现：使用T统计后，c的Dfit特征RMS约1.000；撤销ATMS各模型的输出affine后，其Dfit RMS却为**1.12–1.68**，innovation RMS为0.900–1.509，训练innovation RMS为0.7255。

ATMS原输出本来受监督于共同CLIP-H坐标；撤销不同模型各自的affine会改变这一共同输出坐标，不能照搬HIVE球面修正的内部坐标理由。只读保留原生ATMS输出时，Dfit RMS为**0.897–1.121**。因此另登记一次原生坐标复核：只保留ATMS自己的输出affine，其他目标、正则、C处理、样本、控制和门槛完全不变；不在Dfit重估均值、尺度或校准。[坐标审计](../audit/cross_encoder_20261027_v1/actual/server_audit/postresult_native_coordinate_audit.json)、[复核前冻结协议](../configs/cross_encoder_20261027_v1/probe_native_spec_v2.json)

这是在已看首版Dfit后的开发复核，不是新的独立确认。首版模型、原预测和负结果完整保留。两版本c-only系数、归一化及8200条c-only预测逐位一致，证明本次单项ATMS坐标变化没有替换基准。

## 4. 原生坐标下的完整Dfit结果

| 固定读出 | 标准化F预测MSE↓ | 同概念10图log loss↓ | 十图Top1↑ | 预测/目标余弦↑ |
|---|---:|---:|---:|---:|
| c-only | 0.941208 | 1.782682 | 36.2317% | 0.230765 |
| ATMS-only | 0.944709 | 1.920269 | 30.2268% | 0.223211 |
| c＋ATMS innovation | **0.937393** | **1.770729** | **37.2317%** | **0.241802** |
| c＋错排innovation | 0.944564 | 1.791880 | 36.1098% | 0.225273 |

相对c-only，MSE改善为**+0.003814789，[+0.002096631,+0.005337190]**，57/82概念正向；相对错排为+0.007171274，[+0.005293193,+0.008915667]。十图log loss相对c-only改善+0.01195299，[+0.00372977,+0.02030578]；相对错排改善+0.02115132，[+0.01190154,+0.03050363]。

这支持**本固定读出在已曝光Dfit上有小幅预测增量**，不能继续写成“ATMS完全没有互补信息”。但相对c-only的MSE均值只改善**0.4053%**，没有达到登记的0.5%；十图Top1虽升1个百分点，其增量区间[−0.2323,+2.2686]个百分点仍跨零。保留正向线索，同时维持STOP，不改门槛放行图片。[完整读数](../audit/cross_encoder_20261027_v1/actual/candidate/probe_native_v2/readout.json)、[全部32800条逐query值](../audit/cross_encoder_20261027_v1/actual/candidate/probe_native_v2/per_query.csv.gz)

![原生坐标复核，全部被试与82概念](../audit/cross_encoder_20261027_v1/actual/server_audit/CPU_analysis_native_v2/actual_Dfit_effects.png)

## 5. 误差分解与独立复核

对辅助增量δ和原残差r，逐query核验：

\[
\mathrm{MSE}_{after}-\mathrm{MSE}_{before}
=\operatorname{mean}\delta^2-2\operatorname{mean}(r\delta).
\]

首版增量能量0.03736634、两倍残差对齐0.01554617，净MSE损害0.02182017；原生坐标版分别为0.01414933和0.01796412，净改善0.00381479。恒等式逐query最大误差小于1.8e−8。这说明坐标处理与辅助更新的幅度/对齐确实发生改变，不等于已经唯一确定所有泛化误差的原因，也不用于事后选择新的强度。

两个版本都用原输入、原模型及冻结统计完成**8200×4预测的完整CPU回放**，最大相对L2分别9.63e−7、1.01e−6；全部MSE/余弦/十图指标CPU重算，最大绝对差1.91e−6。每版本32800条CSV身份核对，按真实逐query值恢复原dtype后，全部汇总和subject/concept bootstrap逐项精确重算。不是仅测小型schema或少量拒绝分支。[首版CPU审计](../audit/cross_encoder_20261027_v1/actual/server_audit/CPU_analysis/summary.json)、[原生坐标CPU审计](../audit/cross_encoder_20261027_v1/actual/server_audit/CPU_analysis_native_v2/summary.json)

## 6. 实际成本、失败记录与资源

| 实际范围 | 秒 |
|---|---:|
| 成功的30模型/原始输入CPU准入 | 25.32 |
| 完整148800预测重导GPU主体 | 45.71 |
| 首版五系统求解／完整GPU探针 | 0.152／13.05 |
| 原生坐标五系统求解／完整GPU探针 | 0.152／12.43 |
| 三次GPU包装合计 | **76.83** |
| 三次成功控制器合计，含源核验 | 165.72 |
| 首次GPU启动前控制器失败 | 31.83 |
| 首次控制器启动至最后终态，含阶段间实现 | **1002.52，约16分42秒** |
| 两次完整CPU后验审计 | 7.23／7.24 |

开始有一次缺`einops`的CPU导入失败，复用原纯Python addon路径修复，没有修改原ATMS源/权重或启动GPU。第一次导出控制器因原始大文件SHA校验超过预留CPU时间，在任何GPU包装启动前退出；增加的是控制器校验余量，导出GPU时限一直为180秒。失败目录和实际事件均保留，严格复核该失败的`completed_stages=[]`且没有`STAGE_STARTED`。三次真正GPU作业均成功，没有失败GPU重试。[失败范围及下一接口复核](../audit/cross_encoder_20261027_v1/actual/server_audit/pre_GPU_failure_and_next_profile_recheck.json)

峰值额外CPU为2.037GiB，GPU allocated为3.706GiB，没有提高10/20GiB上限。全部旧模型/图片、恢复银行、两个线性模型及预测保留；仅同步小证据和gzip逐query表，大数组保留持久盘SHA/shape。原D与本轮所有PID退出，GPU及锁释放，最新56个科学源不变。[完整真实闭合](../audit/cross_encoder_20261027_v1/actual/server_audit/actual_scientific_closure.json)

## 7. 已实际实现的下一接口，尚无模型收益

下一问题从“两个最终向量怎样线性混合”转向**编码器内部变化是否保留最终CLIP向量遗漏的刺激信息**。已实现并CPU实际检验30个旧模型、60条事前固定原输入：捕获原ATMS的36×40内部特征，分别将四个九位置块替换为该query的36位置均值，交回该模型**完整原生投影头与输出affine**；保存`a−a_changed`四个1024维响应。

这样得到的响应仍在各模型原有监督CLIP输出坐标中，不直接混合独立模型未对齐的隐藏通道。它是内部特征干预，不是物理EEG时间因果消融；四个额外小投影前向也应计入推理成本。

30模型的原输出/头重放均与原CPU参考逐位一致，四响应均有限且非退化，范数范围1.244–7.570；恒定token的响应最大1.43e−6。接口profile用时13.23秒，没有F目标/图像读取、新GPU、拟合或图片。[真实CPU接口profile](../audit/cross_encoder_20261027_v1/actual/server_audit/next_encoder_response_profile/completion.json)

后续拟合将优先检验这种信息是否改善完整T训练后的留出预测，结合此前D10/D12/D14的同概念刺激损失设计约束；只有新模型达到固定预测准入，才登记实际图片比较。**当前尚未构建完整T响应银行、训练新响应模型或产生新的图片收益**。不以接口非零、数学恒等式、内部logit或这0.405%的预测改善替代真正的图片胜出；D仍停止，持续研究目标未完成。
