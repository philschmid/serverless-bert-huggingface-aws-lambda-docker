FROM public.ecr.aws/lambda/python:3.8

# Copy function code and models into our /var/task
COPY ./ ${LAMBDA_TASK_ROOT}/

# install our dependencies
RUN python3 -m pip install -r requirements.txt --target ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "handler.handler" ]