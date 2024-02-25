# Sentiment Analysis 
In this project, I aim to harness the power of natural language processing to evaluate the emotional tone expressed in textual data from popular review websites. Although this can be adapted to any website, I have used Tripadvisor in this particular use case.

## Technology Used
> Libraries
```
torch
transformers
requests
beautifulsoup4
pandas
numpy
```

> Model
    Pretrained language model: BERT Multilingual Model

## Next Steps:
* Adding support for multiple languages: 
    While I was building this project, I came across a few reviews in Hindi (with Roman script), which made me curious about the extended application to reviews in languages other than the ones currently supported.

* Detecting sarcasm: 
    Currently, the model does not do great with sarcastic reviews. It is an evolving area of research, made especially difficult by contextual and cultural differences. However, I am highly motivated to explore how this nuance of natural language can be understood and used by AI.

    [Could there be a better cliffhanger?](https://youtu.be/YCbhnuhjVGA?feature=shared)