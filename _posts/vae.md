降维是减少描述数据的特征数量的过程
从初始空间到编码空间，也称为隐空间，latent sapce

降维算法 对于给定一组可选的编码器和解码器，我们希望编码时保持信息量最大，从而在解码时具有尽可能小的重构误差

自编码器用神经网络来作为编码器和解码器，并使用迭代优化学习最佳的编码-解码方案

当编码器和解码器都是深度非线性网络时，自编码器可以实现更有效的降维同时保持较低的重构损失

在没有重建损失的情况下进行重要的降维通常会带来一个代价：隐空间中缺乏可解释和可利用的结构（缺乏规则性，lack of regularity）。其次，大多数时候，降维的最终目的不仅是减少数据的维数，而是要在减少维数的同时将数据主要的结构信息保留在简化的表示中。出于这两个原因，必须根据降维的最终目的来仔细控制和调整隐空间的大小和自编码器的“深度”（深度定义压缩的程度和质量）。

自编码器的高自由度使得可以在没有信息损失的情况下进行编码和解码（尽管隐空间的维数较低）但会导致严重的过拟合

自编码器仅以尽可能少的损失为目标进行训练，而不管隐空间如何组织

为了能够将我们的自编码器的解码器用于生成目的，我们必须确保隐空间足够规则。获得这种规律性的一种可能方案是在训练过程中引入显式的正规化（regularisation）

变分自编码器可以定义为一种自编码器，其训练经过正规化以避免过度拟合，并确保隐空间具有能够进行数据生成过程的良好属性。

不是将输入编码为隐空间中的单个点，而是将其编码为隐空间中的概率分布


VAE使用了KL散度正则化，强制编码后的分布接近标准正态分布（均值为0，方差为1）。这样做的目的是：

防止方差过小：确保每个点周围有足够的“空间”，从而保持生成样本的多样性和连续性。
控制均值分散：避免隐空间中的点彼此相距过远，确保可以学习到这些点之间有效的连续过渡，提升生成样本的连贯性和质量。