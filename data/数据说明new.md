**数据说明**

narriative和squad这两个分别是所使用的新数据集，文件里分别都还是有一个train、valid和test代表训练集、验证集和测试集。这两个数据集简化了分类的数目，现在只有三种情况了。



这两个数据集中，测试集只随机留下了一类学生的回答答案（即response字段）。故模型预测的时候，只需要给出一类的预测结果即可。



.json文件加载后是一个列表，列表中每一个元素都是一个字典。

字典中字段说明：

- **context**：String 类型，表示阅读理解文章
- **qas**：List 类型，表示该 context 下的阅读理解问题和参考答案
- **abstract_30**：String 类型，表示将原阅读理解文章压缩成 30% 的摘要内容，在 “少量阅读” 时作为阅读理解的文章提供给模型回答

fully_response、partially_response、blank_response 三个字段表示三种不同模拟情况下的回答记录，其数量与 qas 数量是对应的。

比方说 qas 中第一个问题，对应的三个不同类型学生回答记录分别是 fully_response 的第一条记录；partially_response 的第一条记录；blank_response 的第一条记录。

- **fully_response**：List 类型，表示仔细且完全阅读情况下，学生的作答答案
- **partially_response**：List 类型，表示少量阅读情况下，学生的作答答案
- **blank_response**：List 类型，表示完全未阅读情况下，学生的作答答案