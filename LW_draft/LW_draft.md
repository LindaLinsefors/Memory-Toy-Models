

# Why sequence memorisation

We expect that a lot of the information stored in the weights of a transformer is memorised facts, rather than general circuits.

# The training data

The training data are seqences of three token, one two input tokens and one output token. Given an input of tow tokens, the network is trained to predict the next token (i.e. the output token).

Hyper parameters for the data generation:

- input token vocabulary size
- output token vocabulary size

In our experiments, the input token vocabulary size is always twise the size as the output token vocabulary size.[^1]

[^1]: We started out the experiments, having them the same size, but the network learned all the facts to easaly, so we dubbled the input vocabulary size in order to have more possible facts.

## Generating the training data

The code for generating the training data is not very long, and is quoted below if you prefeer to read code

The inputs are

- `n_facts` -- Number of facts.
- `input_len` -- Number of input tokens. *This value is aways 2.*
- `input_vocal_size`
- `output_vocab_size`
- `seed` -- Random seed. *This value is always 42.*[^2]

[^2]: We used a fixed random seed to avoid some runs getting lucky and getting easier fracts, and to specifically have the same facts for trained networks and hand-coded networks. However this last aim failed becasuse torch random functions give diffrent results diffrent when run on CPU vs GPU, even with the seed is the same. However, this probably isn't significant a significant concern.

The code first generates a list of every possible input combination. Then this list is shuffled, and the first `n_fact` pairs from the shuffled list are used as the inputs for the `n_fact` facts. These facts are then devided as equal as is possible among the `output_vocab_size` target labels.

*[Make a colapsable box for the code below]*

```python
def generate_facts(n_facts: int, # of facts to generate,
                   input_len: int, # numer of input tokens per fact   
                   input_vocab_size: int, # of unique tokens in the vocabulary
                   output_vocab_size: int, # of unique targets
                   seed: int = 42
                  ) -> dict[str, torch.Tensor]:
    
    if n_facts > input_vocab_size ** input_len:
        raise ValueError(f"Cannot generate {n_facts} unique facts with a vocabulary of size {input_vocab_size} and input length {input_len}. Maximum unique facts: {input_vocab_size ** input_len}")
    
    device = torch.tensor(0).device  # respect default device
    generator = torch.Generator(device=device).manual_seed(seed)

    targets = torch.arange(n_facts) % output_vocab_size

    if input_len == 1:
        inputs = torch.randperm(input_vocab_size, generator=generator)[:n_facts].unsqueeze(1)
    elif input_len == 2:
        all_possible_inputs = torch.cartesian_prod(torch.arange(input_vocab_size), torch.arange(input_vocab_size))
        inputs = all_possible_inputs[torch.randperm(all_possible_inputs.size(0), generator=generator)[:n_facts]]
    else:
        inputs = torch.randint(0, input_vocab_size, (n_facts, input_len), generator=generator)

    sorted_indices = torch.argsort(targets)    
    return {"inputs": inputs[sorted_indices], "targets": targets[sorted_indices]}
```

# Model Architecture

The full toy model is a small one layer tranformer. In addition to training the full model, we also try truning off various parts in various combination, to see what parts of the model is importnat for the sequence memorisation task.

The full toy model consist of:

- Token embedding
- Positional embedded
- A single full width attention head
- A MLP layer (on the last token possiton only since we're not trying to predict intermediet tokens)
- Two recidual connections, one passed the attention, and one passed the MLP.
- Token unembedding to create the logits for the target tokens.
- Three RSMSNorms, one applied to the input to the attention, one to the input to the MLP and one to the input to the unembeding.

![The full toy transformer model, with all the diffrent parts pressent.](Memory%20Toy%20Model%20-%20Full.png)

*Figure 1: The full toy transformer model, with all the diffrent parts pressent.*

## Model variations

### Attention

There are the variants when it comes to the attention.

- **None:**
  There is no attention head, and no possitional embedding. Instead there is two diffrent token embeddings, one for each possition. These are simply added toghether to make the first recidual stream activation.

- **Uniform:**
  We remove the attention pathern $\mathrm{softmax}(QK^\top)$ and relace it with a uniform $\frac{1}{2}$.

- **Full:**
  Just norma attention.

## MLP

There are a number of variants regarnding the MLP. Firstly the MLP can either be pressent or be missing. Secondly if there is an MLP layer, each of the follwing can be varried

- **Activation Function** can be either $\mathrm{GELU}$ or $\mathrm{ReLU}$
- **Bias** can exist or not.[^3]
- **Recidual connection** around the MLP can exist or not.

[^3]: If the bias is pressent that means both the linear readout and the linaer projection from the ReLU or GELU neurons, have bias. (Making them actualy not linear functions but affine fucntion, in strict math therminology.) If there is no bias, this means nether of these have bias. All other linear connections in the rest of the network (e.g. embeddings, etc) are alwatys bias free.

## Norms

The norms can also be turned on and off. Each of the norms for the readin to the attention and MLP only exist if both that part of the network is pressent (Uniform or Full for the attention), and Norms are turned on. The last norm, just before the unembedding only depends on the norm setting, and are there if norms are turned on and not there if norms are turned off.

![A simplified version of the toy model. The MLP is pressent but everything else (attention, nomrs, recidual connection around the MLP) is turned off.](Memory%20Toy%20Model%20-%20Simple.png)

*Figure 2: A simplified version of the toy model. The MLP is pressent but everything else (attention, nomrs, recidual connection around the MLP) is turned off.*

# First experiment and result: What parts of the network matters?

We trained all diffrent versions of the toy model, to see how many facts each of them could learn. There are some patherns, but unfortunatly for most of them, we can't sepperate what is due to expresability of the model and what is due to leanability.

We say that a model has "learned a fact" if: When the model is given the first two tokens of this sequence, it correctly predicts the correct fird token. And by "correctly predics" we mean that argmaxing over the logits locates the correct ouput token.

To find out the maximum number of facts a model can learn, we performed a binary search over number of facts, to find the highest number of facts such that the model leaned all of them.

For each number of facts we trained 11 models in paralell, with the exact same facts, but diffrent random initialised weights. We used three diffrent success criterions "Any", "Most" and "All", meaning that we said the the model succeeded at learning all the facts, if it succedeed in any, most or all of the the 11 trials.

We then ran each of these experiments 4 times, to check for stability. The "Any" setting had highest stabiltity (similar max numer of fact over all 4 duplicate experiments), and "All" had the worst stability.[^4]

[^4]: It's not supprising that "All" had bad stability, since it only takes one bad run to though off the entire bathc. But it was not apriory obvious to us that "Any" would be more stable than "Most".

All the results for all the experiemnst are shown in the table below (see *Table 1*).

## Norms + No Recidual around MLP + No Bias + ReLU = Bad

This combination is bad for some reason. We don't know why.

## No Attention > Uniform Attention > Full Attention

It's not supprising that no attention does the best, since the dual embedding that we use to replace attention is (arguably) more powerfull. Becasuse attention is non-linear, and the dual embedding is linear, there are thing that the attention can express that the dual embdding can't. But on the other hand, the dual embedding get's to encode the input for each token possition entirely seperatly, which gives the netowrk more freedom. Addtionaly, the no attention setup should be easier to train, since it's simpler.

More supprising is that uniform attention is out performing full attenion, given that full attention is strictly more powerfull. Therefore this has to be becasue of ease of training. This interpetation is also suppored by the fact that the number of fact these netorks can learn is very varaible. We can see this in that in the number of outliers in the table below and also how much number of facts dropp from Any to Majority to All.

## MLP

Removing the MLP approximaly cutts the numner of leanable facts in half.

## Norms

Norms are generaly usfull for learning more facts, with one exception. If the recidual stream around the MLP is turned off, and the MLP bias is turned of, and the activation function is GELU, then and only then, does the norm make the network performe worse. This is true for all trhee settings of the attention.

If ther is no MLP the norm inceases the number of leanable facts with 79% - 191.7%[^5]

[^5]: Based on data from "Any" the numbers are similar for "Majority" and larger for "All"

In the rest of the settings, having norms is increasing the number of learnable facts with 2.2% - 42.2%[^6]

[^6]: Same as last footnote

## Recidual Connection around the MLP

Adding this recidual connection increases the number of leanable facts with 3.1% - 34%[^7]

[^7]: Based on data from "Any" the numbers are typially larger for "Majority" and larger for "All"

The one outlier is the case Norm + No Bias + GeLU, where adding a Norm makes a much larger diffrence, becasue of the extra bad synergy of Norms + No Recidual around MLP + No Bias

## Activation Function

GELU is typically better than ReLU. The diffrence is typically small -0.2% to +12.7% from chaning from ReLU to GELU.

Again the exception is Norm + No Recidual around the MLP + No Bias, where swiching from ReLU to GELU boosts the number of leanable facts with 58% - 79%.



![Maximum facts memorised per architectural configuration.](table_E5.png)

**Table 1:** Maximum facts memorised per architectural configuration ($d_\text{residual}=16$, $d_\text{ff}=16$, CE loss). **Attn** is the token-mixing mechanism: *None* (additive per-position embeddings), *Unif* (uniform/averaging attention, `qk_is_one = True`), or *Full* (learned attention). The *Any* / *Majority* / *All* column groups apply the corresponding criterion (a fact count counts as learned if at least one / more than half / all of the 11 attempts reach perfect accuracy); the four numbered columns within each group are independent repeats. Within each group the row-wise maximum is shown in **bold** and values more than 20% from the group's median are boxed as outliers. For `ff = False` rows the residual, bias and activation flags do not apply (N/A). Note that 1024 is the dataset ceiling, so configurations reaching it have saturated the data rather than the model.

# Challange / Benchmark for understanding

Can you or me, write down weights for the memory toy model, eitgher by hand or some algorithm that isn't gradient decent, such that our resulting model match the performance of the leaned model?

This chalance is benchmark for how well we understand how the model store the facts. There are two reasons why this is a usefull frameing.

- If we understand how the facts are embedded, we should be able to replicate this, whithout gadient decaent.
- Fhinkging about "How would I do this?" can be a usefull framing for mech-interp.

We think that our current best attempt (which will be pressented soon) is some non-zero progress on this challange, but there are still far to go. We encurrage all readers go give it a try.

## Model architecture

Firstly, we're not trying (yet) to produce weights for the full tranformer model, but instead aiming to find functional weights for the simplified toy model shown in *Figure 2*.

Seconldy, the version shown in *Figure 2* has unessearely many weights, wich is a legacy from being a cutt down version of the full version shown in *Figure 1*. Because the MLP is sandwich between two linear operations, we can skipp the weight matrices of the MLP.

We can simplify it further down to this: 

![This model is equivalent to the toy model configuration with settings Attn=None, FF=ON, Norm=OFF, Res=OFF, Bias=OFF, Act=ReLU.](Memory%20Toy%20Model%20-%20Hand%20Coded.png)

*Figure 3: This model architecture is equivalent to the toy model configuration with settings Attn=None, MLP=✅, Norm=❌, Res=❌, Bias=❌, Act=ReLU.*

If you want to give the challage a go, feel free to use this architecture, of any other that you find easier to work with. The final goal is to be able to write down functional weights for the full transformer model, but we think it's ok to start with a simpler case.

# Our attempt

Our algorithm has three steps

- Assign $S$ number ReLU neurons to each label. This means that each neuron will be assigned to several labels. We try to generate an assigment that accives both: Each neuron should have approximatly the same labels assigned to it as any other neuron; The max neuron overlap between any pair of labels, should be as small as possible.
- Chose the embeding weighs such that each ReLU neuron assigned to label $l$ will output zero for all facts with label $l$.
- Assigned a lage negativ weiths going from ReLU neurons assigned to label $l$, to the logit for $l$.

### Assigning neurons to labels
There are $d_{MLP}$ ReLU neurons, and $n_{output\_vocab}$ labels. Eeach label get's assigned $S\geq1$ neurons. In most of our experiments $d_{MLP}=n_{output\_vocab}$, which means for any $S>1$, the assigments will overlap.

One problem our network needs to solve is that there will likely be some pathern of facts
- $a,b$ -> $l$
- $c,d$ -> $l$
- $a,d$ -> not $l$

Any weight alocation, on this model architecture, where the logit for some label only depends on a single ReLU neuron, will fail at encoding this pathern. Therefore, the netowrk either needs more ReLU neurons than labels (not realistic) or the labels will have to some how share neurons, i.e. some sort of superpossition encoding. 

I don't know apriory what is the best value of $S$, therefore this is given as a hyperparameter for the algorithm that creates the weights. Given $S$, $d_MLP$ and $n_{output\_vocab}$, we want to find an allocation where the alocaltions are spread out nicely. I.e. we want all neurons to be used by approximatly the same number of labels, and we want to minimize the max neuron overlap between any pair of labels. 

I had Claud Code write script that does this, and verified that the outputs looks good. There are probaly many ways to accive similar allocations, and I can't think of any reason why the exact method matters, so I will not go into this further.

### Embedding weights
Next stepp is to make sure that any ReLU neurons allways output zero on all facts with a label assigned to that fact. 

The algorithm in broad stroks:

1. For each neuron, we list all facts with a label that is assigned to that neuron. 
2. For the first input token, we count how many times each token apear in this list of facts, whe then take $top\_fraction$ of these input tokens and assign them the $weight = -1$, to that neuron.
3. Repeat step 2 for the second input token.
4. Find any fact that is not covered by step 2 and 3, and assign $weight = 0$ to both first and second input tokens for all such facts.
5. Assign $weight = 1$ to all remaining input tokens.


### Unembedding weights
The last step is simple. Each unebedding weigt is $-1$ from ReLU neurons to labels assined to that neuron, and $0$ everywhere else. 

We did also try assigning possitive values everywhere else, but for the success criteria we use (looking at arg_max of the logits), adding these possitive values makes no diffrence.

## Results
As to be expected, our hand coded model is not as good as trained model, and it expecially sgruggel to reach full accuracy. But if we accept 90% accuracy fot the hand coded model, it scales almost as well as the trained model, but with a singificanlty worse pre-factor.

The plot below shoes our data from binary seach to find maximum number of fact a model can learn. 
- The trained models are using the arcitectures "full" (Figure 1, GELU and bias) and "simple" (Figure 2, with ReLU and no bias). These models are trained on Cross-Entopy loss.
- The handcoded models are generated according to the algorithm described in the prevous section. For each datapoint we selected the best result from a hyper parameter sweep over $S$ and $top\_fraction$
- All models are evaluated on accuracy, by which we mean perecentage of facts it's calculated as $mean(argmax(logits)==labels)$
- For the trained models, the required accuracy is alway 100%, for the handcoded models, we show the result for both required accuracy 90% and 100%. 
- Each experiment is run 11 times with the same facts but diffrent random initialisations (for the trained models) or diffrent random suffels of neuron alocations, and diffrent shuffelings as tie breaker in step 2 Embdding weights (for the hand coded models). "any"/"most"/"all" indicate if the success critera is that any most or all of the runs needs to reach the desired accuracy.
- Best fit lines are over aggeregations of any/most/all.
- $d$ is the dimension of the model. $n_{input\_vocab}=2d$, $d_{MLP}=d$, $n_{ouptput\_vocab}=d$

![](hc_vs_learned.png)
***Figure 4:** Maxumum number of facts learnable by diffrent models*

### Optimal S is the square root of number of ReLU neuons
To get the best version of our handcoded model, we do a hyper parameter sweep over $S$ and $top\_fraction$. From this we can extract the optimal $S$ for diffrent model sizes by looking at $S$ from the wining ($S$, $top\_fraction$) pair.

Doing so for the data from used in Figure 4, showed that $S\approx\sqrt{d}$. However this does not tell us if $S$ depends on $n_{input\_vocab}$, $d_{MLP}$ or $n_{ouptput\_vocab}$ since these all varry toghetehr in that eperiment. 

In the next experiment we did a binary search for max number of facts for every combination of the follping parameters:

- $n_{input\_vocab} = 16, 32, 64$
- $d_{MLP} = 8,16,32,64$
- $n_{output\_vocab} = 8, 16, 32, 64$
- accuracy requirement = 90%, 100%
- success agregation = any, most, all

Here's how the optimal $S$ depends on all of them.

![](best_S_vs_d_MLP.png)
***Figure 5***

![](best_S_vs.png)
***Figure 6***



