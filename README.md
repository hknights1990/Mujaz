# CVE-Profiling

CVE and vulnerability profiling aim to create abstractive summary from features included in the original CVE description by extracting differetn features and use pre-trained model on Those features. 

## Pre-trained Models

### T5
We fine-tune T5 on single task using one feature and multiple tasks on more features and compare the summarization results. We build four different models using T5. 

### BART
We fine-tune BART as we did with T5, but since it does not supprort multi-task training, we fine-tune it on two models. 

### Metrics
We used ROUGE as our computational metric and asked 3 evaluators to evaluate 5 differnt human metrics to judge the content of the summary.   
