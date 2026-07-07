This post is an appendix to [main post]. This is for people who have read the main post and want more details. 



# Appendix A: More on what parts of the model matter
![Maximum facts memorized per architectural configuration.](table_E5.png)

### Mixing
![](table_attention.png)
*Table A.1: Dual embedding vs. uniform attention vs. full attention.*

### MLP
![](table_ff.png)
*Table A.2: MLP*

### Norms
![](table_norm.png)
*Table A.3: Norms vs. no norms.*

### Residual connection around the MLP
![](table_residual.png)
*Table A.4: With vs. without residual connection around the MLP.*

### MLP Bias
![](table_bias.png)
*Table A.5: With vs. Bias in the MLP.*

### MLP Acctivation function
![](table_activation.png)
*Table A.6: GELU vs. ReLU.*

# Results 1: Which parts of the network matters [Move all of this to appendix]
The point of this experiment is to figure out what parts of the network is important for the sequence memorization task, so that I know what parts are safe to ignore or even remove, in order to make understanding the model easier.

What I found was that having an MLP is the most important part (approximately responsible for learning half of the facts), and everything else matter a bit.

I trained all different versions of the toy model, to see how many facts each of them could learn. There are some patterns, but unfortunately for most of them, I can't separate what is due to expressibility of the model and what is due to learnability.

For this experiment, I used a single model size across all architectures:
- $n_{input\_vocab} = 32$
- $d_{residual} = 16$
- $d_{MLP} = 16$
- $n_{output\_vocab} = 16$ 


I claim that a model has "learned a fact" if: When the model is given the first two tokens of this sequence, it correctly predicts the correct third token. And by "correctly predicts" I mean that argmaxing over the logits locates the correct output token.

To find out the maximum number of facts a model can learn, I performed a binary search over number of facts, to find the highest number of facts such that the model learned all of them.

For each number of facts I trained 11 models in parallel, with the exact same facts, but different random initialized weights. I used three different success criterions "Any", "Most" and "All", meaning that I said the model succeeded at learning all the facts, if it succeeded in any, most or all of the 11 trials.[^most]

[^most]: "Most" mean that there are more successes than failure, i.e. at least 6 out of 10.

For each architecture and each of Any/Most/All I ran a binary search to find the maximum number of facts it could learn. Furthermore, I repeated each such binary search 4 times, to check for stability. The "Any" setting had the highest stability (similar max number of fact over all 4 duplicate experiments), and "All" had the worst stability.[^4]

[^4]: It's not surprising that "All" had bad stability, since it only takes one bad run to throw off the entire batch. But it was not a priori obvious to me that "Any" would be more stable than "Most".

All the results for all the experiments are shown in the table below 

![Maximum facts memorized per architectural configuration.](table_E5.png)

*Table 1: Maximum facts memorized for each model architecture. Within each group the row-wise maximum is shown in bold and values more than 20% from the group's median are boxed as outliers. Note that 1024 is the dataset ceiling, so configurations reaching it have saturated the data rather than the model.*

To see the effect of each of Mixing, Norms, Res, Bias and Act, we'll look at pairs (or triples for Mixing) of model architectures that are the same except for this variable. E.g. for Norms, we look at every pair that differ only in terms of it has norms or not, to see how much models with norms typically outperforms the ones without norms. We're doing this analysis on the Any runs since these have the most stable outcomes.

However before doing all that, it's worth noting one major outlier.

### MLP + Norms + No Residual around MLP + No Bias + ReLU

For any form or attention (or dual embedding) networks with **MLP**=✅, **Norm**=✅, **Res**=❌, **Bias**=❌ and **Act**=**ReLU**, does really badly. Almost as bad (and in one case slightly worse) than removing the MLP.

This combination is extra bad for some reason, that isn't just an effect of the sum of it's part. I don't know why. Specifically, I don't know if the limitation is due to training dynamics or due to what is possible for this architecture.

Instead of writing "except for **MLP**=✅, **Norm**=✅, **Res**=❌, **Bias**=❌ **Act**=**ReLU**", in every subsection below. I'll just point out this here. Having pointed this out, this data will be excluded from the below triple or pairwise comparisons.


## Triple or pairwise comparisons.
Now back to triple and pairwise comparisons. I.e. we're compare outcomes (number of learned facts) for pairwise (or triples when varying Mixing) model architecture, where the only difference is a single setting o

### Mixing 

- When **Norms**=❌ and **MLP**=❌, i.e. there is nothing but embedding, possibly attention, and unembedding, in this case only, 2Emb, Unif Attn and Lrn Attn does equally well.[^M]

[^M]: **2Emb** and **Unif Attn** learn **208** facts in each of the four repeated experiments. Lrn Attn learn **208** in two of the replications, slgtly more in one, and slightly less in one.

- For all other settings **2Emb** does better than **Unif Attn** wich does better than **Lrn Attn**. 
- Going from **2Emb** to **Unif Attn** lets the network learn **74 - 176 (7.8% - 23.9%)** more facts. 
- Going from **Unif Attn** to **Lrn Attn** lets the network learn **46 - 222 (7.9% - 39%)** more facts.

It is notable that when everything else is turned off, i.e. there is only embedding maybe attention and unembedding, is when Mixing (the setting that determine the embedding and attention) has the least effect.


It's not surprising that dual embedding does the best, since this architect is (arguably) the more powerful. Because attention is non-linear, and the dual embedding is linear, there are things that the attention can express that the dual embedding can't. But on the other hand, the dual embedding gets to encode the input for each token position entirely separately, which gives the network more freedom. Additionally, this no attention setup should be easier to train, since it's simpler.

More surprising is that uniform attention is outperforming learned attention, given that learned attention is strictly more powerful. Therefore, this has to be because of ease of training. This interpretation is also supported by the observation that the number of fact network with learned attention mange to learn, is unstable. You can see this in that in the number of outliers in the Table 1 and also how much number of facts drop from Any to Most to All.

### MLP

- Adding an **MLP** block lets the network learn **324 - 794 (60% - 373%)** more facts.
- Adding an **MLP** block makes the **biggest** diffrence when **Mixing**={**2Emb, Unif Atten**} and **Norms**=❌. This is proabbly becasue in this setting the MLP's ReLU or GELU neruons are the only non-lineareties. 
- Adding an MLP block makes the **smallest** diffreince when **Mixing**=**2Emb**, and **Norms**=✅. However this is proabblay just a cealing effect, since the with MLP model maxed out for this setting.

Not supprisingly, adding the MLP makes the biggest diffrence for number of learnable facts, out of any of the setting. 

### Norms

- When **MLP**=❌ then adding norms lets the network learn **162 - 362 (79% - 174%)** more facts. This effect is *largest* for **Mixing** = **2Emb**.
- When **MLP**=✅, then adding norms makes the network able to learn **22-240 (2.2% - 42%)**. The effect is *largest* for **Mixing** = **Lrn Attn**.

Norms are generally useful for learning more facts. Norms make a bigger difference if there is attention, and if there is no MLP. Possible this means that the norms in front of the attention and unembedding are helpful, while the norm in front of the MLP is anti-helpful. Or possible the MLP and norms overlap somewhat in function, such that the MLP makes the norms less useful. 

The fact that MLP has the biggest effect when there are no other non-lineareties in the network, points to the overlaping function hypothesis. However the unsuall crappyness of **MLP**=✅, **Norms**=✅, **Res**=❌, **Bias**=❌, **Act**=**ReLU**" might be a sign that adding norms may be bad for the performance of the MLP.



### Residual Connection around the MLP

- Adding this residual connection lets the network learn **-8 to 244 (-0.7% to 34%)** more facts.
- There only setting where adding this residual is bad for the network is **Mixing**=**Lrn Attn**, **Norms**=❌, **Bias**=✅, **Act**=**ReLU**. But the effect is tiny, **8 facts (0.7%)**, so it's probably just a fluke.

### MLP Bias
Adding a bias ought to be strictly helpful, but for some reason it's anti-helpful in a few cases.

- Adding a bias to the MLP lets the network learn **-52 to 156 (-5.1% to 22%)** more facts.
- Adding an MLP Bias performs at its worst when **Norms**=❌, **Res**=✅. Given this setting adding the bias lets the network learn **-52 to 14 (-5.1% to 1.4%)** more facts.

### Activation Function

**GELU** is typically better than **ReLU**, but difference is small. 
- Changing from **ReLU** to **GELU** lets the network learn **-6 to 78 (-0.2% to +12.7%)** more facts.



# Appendix B: Best S (number neurons per label) for the hand-coded and hybrid models



$S$ is (at least for the hand-coded model) the number of neurons used buy any label. I'm interested in how this scale with various model parameters, since this might give us some clue about what we should expect superpossiton to look like over ReLU and ReLU-like neurons. To be clear, anything we see here is at best a small hint, with no guarantee to have anything do to with how computations are distributed in fully trained models. But it's still a little bit of Baisian evidence, and maybe if it gets to meet up with other evidence later on, it will tell me something. This is why I think this is worth recording.

In the experiments in the main post, in order to get the best version of my hand-coded model, I did a hyperparameter sweep over $S$ and $top\_fraction$. From this I can extract the optimal $S$ for different model sizes by looking at $S$ from the winning ($S$, $top\_fraction$) pair.
 
In the scaling experiments (see main post) I investigated models with dimensions $n_{input\_vocab}=2d$, $d_{MLP}=d$, and $n_{output\_vocab}=d$ for $d\in\{16,32,64,128,256\}$. Looking at the best performing $S$ from these runs we find that $best\_S\approx\sqrt{d}$

![](appendixB_S_vs_d.png)

However, in these experiments the relation between $n_{input\_vocab}$, $d_{MLP}$, and $n_{output\_vocab}$ is locked, i.e. we can't tell from the data which of these influences the ideal value of $S$.

In the next experiment I did a binary search for max number of facts for every combination of the following parameters:

- $n_{input\_vocab} \in \{16, 32, 64\}$
- $d_{MLP} \in \{8,16,32,64\}$
- $n_{output\_vocab} \in \{8, 16, 32, 64\}$
- accuracy requirement $\in \{90\%, 100\%\}$
- success aggregation $\in$ {*any, most, all*}
- model type $\in$ {*hand-coded, hybrid*}

Below you can see how the optimal $S$ depends on all of them.

Note that I sweped over $S \in \{1,2,3, \dots, 22\}$. A small number of runs may have hit the sealing, i.e. the optimal value for $S$ is acctualy something above $22$.

![](appendixB_S_vs_d_hc.png)

![](appendixB_S_vs_all_hc.png)

![](appendixB_S_vs_d_hy.png)

![](appendixB_S_vs_all_hy.png)

As you can see, the picture is less clean when the diffrent model dimenstions are varried independently. Just scaling up the hidden layer increases the optimal $S$ faster than $\sqrt{d}$. But $S$ also decreses with $n_{output_vocab}$ just enough to add up to the patehrn we see in the first figure (of this appendix section), when they increase toghther.

In the hybrid model $n_{input\_vocab}$ also plays a role. 

# Appendix C: Best "top_fraction" for the hand-coded and hybrid models 

The other hyperparameter (other than $S$), used when creating the embedding matrix for the handcoded and hybrid models, is a variable that I dubbed $top\_fraction$. See main post for definition. 

***I don't expect looking into this variable will tell you anything interesting. I don't recomend that you pay attention to this section unless you know something I don't.***

But for compleetion, and because it's cost me almost no extra work to add this, here are the same plot as in Appendix B, but for $top\_fraction$ instead of $S$.

Note that I only sweped over $top\_fraction\in\{0.00, 0.02, 0.04 \dots 0.38\}$ some runs with the hybrid model seems to have hit the cealing, i.e the real best $top\_fraction$ is somethig above $0.38$.

![](appendixC_f_vs_d.png)

![](appendixC_f_vs_d_hc.png)

![](appendixC_f_vs_all_hc.png)

![](appendixC_f_vs_d_hy.png)

![](appendixC_f_vs_all_hy.png)