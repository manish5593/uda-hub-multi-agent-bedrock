"""Deploy, check and delete the SageMaker embedding endpoint used for knowledge-base retrieval.

Run from the solution folder:

    python -m deploy.sagemaker_embeddings deploy --role-arn arn:aws:iam::<account>:role/<sagemaker-role>
    python -m deploy.sagemaker_embeddings status
    python -m deploy.sagemaker_embeddings test
    python -m deploy.sagemaker_embeddings delete

The endpoint serves a Hugging Face embedding model (config.EMBEDDING_MODEL when UDAHUB_EMBEDDING_PROVIDER is
"sagemaker", otherwise BAAI/bge-small-en-v1.5) with Text Embeddings Inference on a CPU instance. A real-time
endpoint is billed for every hour it exists, so run `delete` when you are done.
"""
import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from agentic import config  # noqa: E402  (reads the .env values loaded above)

# Text Embeddings Inference 1.8.2 CPU image, from the SageMaker Python SDK's huggingface-tei-cpu config.
# The image lives in a different AWS account per region; pass --image-uri for a region not listed here.
TEI_CPU_TAG = "tei-cpu:2.0.1-tei1.8.2-cpu-py310-ubuntu22.04"
TEI_CPU_REGISTRIES = {
    "us-east-1": "683313688378", "us-east-2": "257758044811", "us-west-1": "746614075791",
    "us-west-2": "246618743249", "ca-central-1": "341280168497", "eu-west-1": "141502667606",
    "eu-west-2": "764974769150", "eu-west-3": "659782779980", "eu-central-1": "492215442770",
    "eu-north-1": "662702820516", "ap-south-1": "720646828776", "ap-southeast-1": "121021644041",
    "ap-southeast-2": "783357654285", "ap-northeast-1": "354813040037", "ap-northeast-2": "366743142698",
    "sa-east-1": "737474898029",
}


def default_image_uri(region: str) -> str:
    if region not in TEI_CPU_REGISTRIES:
        sys.exit(f"No TEI image account known for {region}; pass --image-uri")
    return f"{TEI_CPU_REGISTRIES[region]}.dkr.ecr.{region}.amazonaws.com/{TEI_CPU_TAG}"


def resource_names(endpoint: str) -> dict:
    return {"endpoint": endpoint, "config": f"{endpoint}-config", "model": f"{endpoint}-model"}


def sagemaker_client(region: str):
    import boto3
    return boto3.client("sagemaker", region_name=region)


def deploy(args) -> None:
    role_arn = args.role_arn or os.getenv("SAGEMAKER_ROLE_ARN")
    if not role_arn:
        sys.exit("Pass --role-arn or set SAGEMAKER_ROLE_ARN: an IAM role SageMaker can assume")
    names = resource_names(args.endpoint)
    sm = sagemaker_client(args.region)
    image = args.image_uri or default_image_uri(args.region)
    print(f"Deploying {args.model} as {names['endpoint']} on {args.instance_type} in {args.region}\n  image: {image}")
    sm.create_model(ModelName=names["model"], ExecutionRoleArn=role_arn,
                    PrimaryContainer={"Image": image, "Environment": {"HF_MODEL_ID": args.model}})
    sm.create_endpoint_config(EndpointConfigName=names["config"], ProductionVariants=[{
        "VariantName": "AllTraffic", "ModelName": names["model"],
        "InstanceType": args.instance_type, "InitialInstanceCount": 1}])
    sm.create_endpoint(EndpointName=names["endpoint"], EndpointConfigName=names["config"])
    print("Waiting for the endpoint to come into service (usually 5-10 minutes)...")
    sm.get_waiter("endpoint_in_service").wait(EndpointName=names["endpoint"],
                                               WaiterConfig={"Delay": 30, "MaxAttempts": 40})
    print(f"In service. Set UDAHUB_EMBEDDING_PROVIDER=sagemaker and UDAHUB_SAGEMAKER_ENDPOINT={names['endpoint']}")


def status(args) -> None:
    endpoint = sagemaker_client(args.region).describe_endpoint(EndpointName=args.endpoint)
    print(f"{args.endpoint}: {endpoint['EndpointStatus']}", endpoint.get("FailureReason", ""))


def smoke_test(args) -> None:
    from agentic.sagemaker_embeddings import SageMakerEndpointEmbeddings
    embeddings = SageMakerEndpointEmbeddings(args.endpoint, region_name=args.region)
    query, related, unrelated = embeddings.embed_documents([
        "How do I cancel my reservation?",
        "To cancel a booking, open My Reservations and choose Cancel.",
        "Our museum partners include modern art galleries.",
    ])
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # vectors are normalized, so this is cosine similarity
    print(f"{len(query)}-dimensional vectors")
    print(f"similarity to a related sentence:   {dot(query, related):.3f}")
    print(f"similarity to an unrelated sentence: {dot(query, unrelated):.3f}")


def delete(args) -> None:
    names = resource_names(args.endpoint)
    sm = sagemaker_client(args.region)
    for kind, call, key in [("endpoint", sm.delete_endpoint, "EndpointName"),
                            ("config", sm.delete_endpoint_config, "EndpointConfigName"),
                            ("model", sm.delete_model, "ModelName")]:
        try:
            call(**{key: names[kind]})
            print(f"Deleted {kind} {names[kind]}")
        except sm.exceptions.ClientError as error:  # already gone, or never created
            print(f"Skipped {kind} {names[kind]}: {error.response['Error']['Message']}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["deploy", "status", "test", "delete"])
    parser.add_argument("--endpoint", default=config.SAGEMAKER_ENDPOINT)
    parser.add_argument("--region", default=config.AWS_REGION)
    parser.add_argument("--model", default=config.EMBEDDING_MODEL if config.EMBEDDING_PROVIDER == "sagemaker"
                        else "BAAI/bge-small-en-v1.5", help="Hugging Face model id")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    parser.add_argument("--role-arn", help="IAM role SageMaker assumes (or set SAGEMAKER_ROLE_ARN)")
    parser.add_argument("--image-uri", help="override the Text Embeddings Inference image")
    args = parser.parse_args(argv)
    {"deploy": deploy, "status": status, "test": smoke_test, "delete": delete}[args.command](args)


if __name__ == "__main__":
    main()
