# [serverless-bert-huggingface-aws-lambda-docker](https://www.philschmid.de/serverless-bert-with-huggingface-aws-lambda-docker)

It's the most wonderful time of the year. Of course, I'm not talking about Christmas but re:Invent. It is re:Invent
time.

photo from the keynote by Andy Jassy, rights belong to Amazon

In the opening keynote, Andy Jassy presented the AWS Lambda Container Support, which allows us to use custom container
(docker) images up to 10GB as a runtime for AWS Lambda. With that, we can build runtimes larger than the previous 250 MB
limit, be it for "State-of-the-Art" NLP APIs with BERT or complex processing.

Furthermore, you can now configure AWS Lambda functions with up to
[10 GB of Memory and 6 vCPUs](https://aws.amazon.com/de/blogs/aws/new-for-aws-lambda-functions-with-up-to-10-gb-of-memory-and-6-vcpus/?nc1=b_rp).

For those who are not that familiar with BERT was published in 2018 by Google and stands for
Bidirectional Encoder Representations from Transformers and is designed to learn word representations or embeddings from
an unlabeled text by jointly conditioning on both left and right context. Transformers are since that the
"State-of-the-Art" Architecture in NLP.

Google Search started using BERT end of 2019 in
[1 out of 10](https://www.blog.google/products/search/search-language-understanding-bert/) English searches, since then
the usage of BERT in Google Search increased to almost
[100% of English-based queries](https://searchon.withgoogle.com/). But that's not it. Google powers now over
[70 languages with BERT for Google Search](https://twitter.com/searchliaison/status/1204152378292867074).

[https://youtu.be/ZL5x3ovujiM?t=484](https://youtu.be/ZL5x3ovujiM?t=484)

We are going to use the newest cutting edge computing power of AWS with the benefits of serverless architectures to
leverage Google's "State-of-the-Art" NLP Model.

We deploy a BERT Question-Answering API in a serverless AWS Lambda environment. Therefore we use the
[Transformers](https://github.com/huggingface/transformers) library by HuggingFace,
the [Serverless Framework](https://serverless.com/), AWS Lambda, and Amazon ECR.

Before we start i wanted to encourage you to read my blog [philschmid.de](https://www.philschmi.de) where i have already
wrote several blog post about [Serverless](https://www.philschmid.de/aws-lambda-with-custom-docker-image) or
[How to fine-tune BERT models](https://www.philschmid.de/bert-text-classification-in-a-different-language).

You find the complete code for it in this
[Github repository](https://github.com/philschmid/serverless-bert-huggingface-aws-lambda-docker).

---

# Services included in this tutorial

## Transformers Library by Huggingface

The [Transformers library](https://github.com/huggingface/transformers) provides state-of-the-art machine learning
architectures like BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet, T5 for Natural Language Understanding (NLU) and Natural
Language Generation (NLG). It also provides thousands of pre-trained models in 100+ different languages.

## AWS Lambda

[AWS Lambda](https://docs.aws.amazon.com/lambda/latest/dg/welcome.html) is a serverless computing service that lets you
run code without managing servers. It executes your code only when required and scales automatically, from a few
requests per day to thousands per second.

## Amazon Elastic Container Registry

[Amazon Elastic Container Registry (ECR)](https://aws.amazon.com/ecr/?nc1=h_ls) is a fully managed container registry.
It allows us to store, manage, share docker container images. You can share docker containers privately within your
organization or publicly worldwide for anyone.

## Serverless Framework

[The Serverless Framework](https://www.serverless.com/) helps us develop and deploy AWS Lambda functions. It’s a CLI
that offers structure, automation, and best practices right out of the box.

---

# Tutorial

Before we get started, make sure you have the [Serverless Framework](https://serverless.com/) configured and set up. You
also need a working `docker` environment. We use `docker` to create our own custom image including all needed `Python`
dependencies and our `BERT` model, which we then use in our AWS Lambda function. Furthermore, you need access to an AWS
Account to create an IAM User, an ECR Registry, an API Gateway, and the AWS Lambda function.

We design the API like that we send a context (small paragraph) and a question to it and respond with the answer to the
question.

```python
context = """We introduce a new language representation model called BERT, which stands for
Bidirectional Encoder Representations from Transformers. Unlike recent language
representation models (Peters et al., 2018a; Radford et al., 2018), BERT is
designed to pretrain deep bidirectional representations from unlabeled text by
jointly conditioning on both left and right context in all layers. As a result,
the pre-trained BERT model can be finetuned with just one additional output
layer to create state-of-the-art models for a wide range of tasks, such as
question answering and language inference, without substantial taskspecific
architecture modifications. BERT is conceptually simple and empirically
powerful. It obtains new state-of-the-art results on eleven natural language
processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute
improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1
question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD
v2.0 Test F1 to 83.1 (5.1 point absolute improvement)."""

question_one = "What is BERTs best score on Squadv2 ?"
# 83 . 1

question_two = "What does the 'B' in BERT stand for?"
# 'bidirectional encoder representations from transformers'
```

**What are we going to do:**

- create a `Python` Lambda function with the Serverless Framework.
- add the `BERT`model to our function and create an inference pipeline.
- Create a custom `docker` image
- Test our function locally with LRIE
- Deploy a custom `docker` image to ECR
- Deploy AWS Lambda function with a custom `docker` image
- Test our Serverless `BERT` API

You can find the complete code in this
[Github repository](https://github.com/philschmid/serverless-bert-huggingface-aws-lambda-docker).

---

# Create a `Python` Lambda function with the Serverless Framework.

First, we create our AWS Lambda function by using the Serverless CLI with the `aws-python3` template.

```bash
serverless create --template aws-python3 --path serverless-bert
```

This CLI command will create a new directory containing a `handler.py`, `.gitignore`, and `serverless.yaml` file. The
`handler.py` contains some basic boilerplate code.

```python
import json

def hello(event, context):
    body = {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "input": event
    }
    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }
    return response
```

---

# Add the `BERT`model to our function and create an inference pipeline.

To add our `BERT` model to our function we have to load it from the
[model hub of HuggingFace](https://huggingface.co/models). For this, I have created a python script. Before we can
execute this script we have to install the `transformers` library to our local environment and create a `model`
directory in our `serverless-bert/` directory.

```yaml
mkdir model & pip3 install torch==1.5.0 transformers==3.4.0
```

After we installed `transformers` we create `get_model.py` file in the `function/` directory and include the script
below.

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

def get_model(model):
  """Loads model from Hugginface model hub"""
  try:
    model = AutoModelForQuestionAnswering.from_pretrained(model,use_cdn=True)
    model.save_pretrained('./model')
  except Exception as e:
    raise(e)

def get_tokenizer(tokenizer):
  """Loads tokenizer from Hugginface model hub"""
  try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    tokenizer.save_pretrained('./model')
  except Exception as e:
    raise(e)

get_model('mrm8488/mobilebert-uncased-finetuned-squadv2')
get_tokenizer('mrm8488/mobilebert-uncased-finetuned-squadv2')
```

To execute the script we run `python3 get_model.py` in the `serverless-bert/` directory.

```python
python3 get_model.py
```

_**Tip**: add the `model` directory to gitignore._

The next step is to adjust our `handler.py` and include our `serverless_pipeline()`, which initializes our model and
tokenizer and returns a `predict` function, we can use in our `handler`.

```python
import json
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoConfig

def encode(tokenizer, question, context):
    """encodes the question and context with a given tokenizer"""
    encoded = tokenizer.encode_plus(question, context)
    return encoded["input_ids"], encoded["attention_mask"]

def decode(tokenizer, token):
    """decodes the tokens to the answer with a given tokenizer"""
    answer_tokens = tokenizer.convert_ids_to_tokens(
        token, skip_special_tokens=True)
    return tokenizer.convert_tokens_to_string(answer_tokens)

def serverless_pipeline(model_path='./model'):
    """Initializes the model and tokenzier and returns a predict function that ca be used as pipeline"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    def predict(question, context):
        """predicts the answer on an given question and context. Uses encode and decode method from above"""
        input_ids, attention_mask = encode(tokenizer,question, context)
        start_scores, end_scores = model(torch.tensor(
            [input_ids]), attention_mask=torch.tensor([attention_mask]))
        ans_tokens = input_ids[torch.argmax(
            start_scores): torch.argmax(end_scores)+1]
        answer = decode(tokenizer,ans_tokens)
        return answer
    return predict

# initializes the pipeline
question_answering_pipeline = serverless_pipeline()

def handler(event, context):
    try:
        # loads the incoming event into a dictonary
        body = json.loads(event['body'])
        # uses the pipeline to predict the answer
        answer = question_answering_pipeline(question=body['question'], context=body['context'])
        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps({'answer': answer})
        }
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }
```

---

# Create a custom `docker` image

Before we can create our `docker` we need to create a `requirements.txt` file with all the dependencies we want to
install in our `docker.`

We are going to use a lighter Pytorch Version and the transformers library.

```bash
https://download.pytorch.org/whl/cpu/torch-1.5.0%2Bcpu-cp38-cp38-linux_x86_64.whl
transformers==3.4.0
```

To containerize our Lambda Function, we create a `dockerfile` in the same directory and copy the following content.

```bash
FROM public.ecr.aws/lambda/python:3.8

# Copy function code and models into our /var/task
COPY ./* ${LAMBDA_TASK_ROOT}/

# install our dependencies
RUN python3 -m pip install -r requirements.txt --target ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "handler.handler" ]
```

Additionally we can add a `.dockerignore` file to exclude files from your container image.

```bash
README.md
*.pyc
*.pyo
*.pyd
__pycache__
.pytest_cache
serverless.yaml
get_model.py
```

To build our custom `docker` image we run.

```bash
docker build -t bert-lambda .
```

---

# Test our function locally

AWS also released the [Lambda Runtime Interface Emulator](https://github.com/aws/aws-lambda-runtime-interface-emulator/)
that enables us to perform local testing of the container image and check that it will run when deployed to Lambda.

We can start our `docker` by running.

```bash
docker run -p 8080:8080 bert-lambda
```

Afterwards, in a separate terminal, we can then locally invoke the function using `curl` or a REST-Client.

```bash
curl --request POST \
  --url http://localhost:8080/2015-03-31/functions/function/invocations \
  --header 'Content-Type: application/json' \
  --data '{"body":"{\"context\":\"We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial taskspecific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).\",\n\"question\":\"What is the GLUE score for Bert?\"\n}"}'

# {"statusCode": 200, "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Credentials": true}, "body": "{\"answer\": \"80 . 5 %\"}"}%
```

_Beware we have to `stringify` our body since we passing it directly into the function (only for testing)._

---

# Deploy a custom `docker` image to ECR

Since we now have a local `docker` image we can deploy this to ECR. Therefore we need to create an ECR repository with
the name `bert-lambda`.

```bash
aws ecr create-repository --repository-name bert-lambda > /dev/null
```

To be able to push our images we need to login to ECR. We are using the `aws` CLI v2.x. Therefore we need to define some
environment variables to make deploying easier.

```bash
aws_region=eu-central-1
aws_account_id=891511646143

aws ecr get-login-password \
    --region $aws_region \
| docker login \
    --username AWS \
    --password-stdin $aws_account_id.dkr.ecr.$aws_region.amazonaws.com
```

Next we need to `tag` / rename our previously created image to an ECR format. The format for this is
`{AccountID}.dkr.ecr.{region}.amazonaws.com/{repository-name}`

```bash
docker tag bert-lambda $aws_account_id.dkr.ecr.$aws_region.amazonaws.com/bert-lambda
```

To check if it worked we can run `docker images` and should see an image with our tag as name

Finally, we push the image to ECR Registry.

```bash
 docker push $aws_account_id.dkr.ecr.$aws_region.amazonaws.com/bert-lambda
```

---

# Deploy AWS Lambda function with a custom `docker` image

I provide the complete `serverless.yaml` for this example, but we go through all the details we need for our `docker`
image and leave out all standard configurations. If you want to learn more about the `serverless.yaml`, I suggest you
check out
[Scaling Machine Learning from ZERO to HERO](https://www.philschmid.de/scaling-machine-learning-from-zero-to-hero). In
this article, I went through each configuration and explain the usage of them.

```yaml
service: serverless-bert-lambda-docker

provider:
  name: aws # provider
  region: eu-central-1 # aws region
  memorySize: 5120 # optional, in MB, default is 1024
  timeout: 30 # optional, in seconds, default is 6

functions:
  questionanswering:
    image: 891511646143.dkr.ecr.eu-central-1.amazonaws.com/bert-lambda:latest #ecr url
    events:
      - http:
          path: qa # http path
          method: post # http method
```

To use a `docker` image in our `serverlss.yaml` we have to `image` and in our `function` section. The `image` has the
URL to our `docker` image also value.

For an ECR image, the URL should look like this `{AccountID}.dkr.ecr.{region}.amazonaws.com/{repository-name}@{digest}`

In order to deploy the function, we run `serverless deploy`.

```yaml
serverless deploy
```

After this process is done we should see something like this.

---

# Test our Serverless `BERT` API

To test our Lambda function we can use Insomnia, Postman, or any other REST client. Just add a JSON with a `context` and
a `question` to the body of your request. Let´s try it with our example from the colab notebook.

```json
{
  "context": "We introduce a new language representation model called BERT, which stands for idirectional Encoder Representations from Transformers. Unlike recent language epresentation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial taskspecific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).",
  "question": "What is BERTs best score on Squadv2 ?"
}
```

Our `serverless_pipeline()` answered our question correctly with `83.1`.

The first request after we deployed our `docker` based Lambda function took 27,8s. The reason is that AWS apparently
saves the `docker` container somewhere on the first initial call to provide it suitably.

I waited extra more than 15 minutes and tested it again. The cold start now took 6,7s and a warm request around 220ms

---

# Conclusion

The release of the AWS Lambda Container Support enables much wider use of AWS Lambda and Serverless. It fixes many
existing problems and gives us greater scope for the deployment of serverless applications.

We were able to deploy a "State-of-the-Art" NLP model without the need to manage any server. It will automatically scale
up to thousands of parallel requests without any worries. The increase of configurable Memory and vCPUs boosts this cold
start even more.

The future looks more than golden for AWS Lambda and Serverless.

---

You can find the [GitHub repository](https://github.com/philschmid/serverless-bert-huggingface-aws-lambda-docker) with
the complete code [here](https://github.com/philschmid/serverless-bert-huggingface-aws-lambda-docker).

Thanks for reading. If you have any questions, feel free to contact me or comment on this article. You can also connect
with me on [Twitter](https://twitter.com/_philschmid) or
[LinkedIn](https://www.linkedin.com/in/philipp-schmid-a6a2bb196/).
