# IDL Project

Currently just a repository to share papers, references, etc! 

The descriptions probably aren't really accurate nor helpful, just jotting down thoughts.

# Ideas and References

## Safety / Interpretability / Alignment
The basic target is to learn how deep learning models work, how we can 'interpret' them, or how to 'align' them with our goals. 

1. **Interpretability** Playing around with toy monosemanticity models, it could be cool to gain a bit more insights on how neurons (weights) store information. [Paper](https://transformer-circuits.pub/2022/toy_model/index.html)
2. **Interpretability** Playing around with circuits / neurons / visualizing neurons [Distill](https://distill.pub/2020/circuits/zoom-in/)
3. **Interpretability** Exploring something  in or [Anthropic's Paper](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
4. **Safety** Reproducing results from removing fine tuning from open source models, such as: [Llama 3](https://arxiv.org/abs/2407.01376), [Llama 2](https://arxiv.org/abs/2311.00117)
5. **Intepretability** Some proposals by [Apollo](https://www.lesswrong.com/posts/KfkpgXdgRheSRWDy8/a-list-of-45-mech-interp-project-ideas-from-apollo-research)
    - This one in particular seems doable. [Apply SAEs/transcoders to a small conv net (e.g. Alex Net) and study it in depth](https://www.lesswrong.com/posts/KfkpgXdgRheSRWDy8/a-list-of-45-mech-interp-project-ideas-from-apollo-research#:~:text=Apply%20SAEs/transcoders%20to%20a%20small%20conv%20net%20(e.g.%20Alex%20Net)%20and%20study%20it%20in%20depth). Someone already attempted it on InceptionV1 [Paper](https://arxiv.org/pdf/2406.03662)  - don't really get whats being found in the inception paper though
    - and several others under Applied Interpretability section.


## NLP and LLMs  
1. **NLP / LLMs** Titans architecture - Google new architecture that mentioned has large capacity of memory during inference Not sure what can be done yet. [Paper](https://arxiv.org/abs/2501.00663)
2. **NLP / LLMs** Explore how to develop models that 'truthfully' references documents provided and ways to evaluate. See perhaps: [Citations by Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/citations), [Hallucinations benchmark](https://arxiv.org/pdf/2501.08292)
3. **NLP / LLMs** Explore something about the mechanisms used by Deepseek (e.g. quantization to 8bits, mixture of experts), perform comparisons? Although they aren't exactly new either (afaik) [Github](https://github.com/deepseek-ai/DeepSeek-V3).

