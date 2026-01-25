"""
Test script to verify all imports in the diagrams cheat sheet are valid.

Run this inside the visual-generator container:
    docker compose exec visual-generator python tests/test_cheat_sheet_imports.py

Or locally if diagrams is installed:
    pip install diagrams
    python tests/test_cheat_sheet_imports.py
"""

import sys

def test_all_imports():
    """Test all imports from the cheat sheet."""
    errors = []
    success = []

    # ============================================
    # AWS
    # ============================================
    try:
        from diagrams.aws.compute import EC2, Lambda, ECS, EKS, Fargate, Batch
        success.append("diagrams.aws.compute")
    except ImportError as e:
        errors.append(f"diagrams.aws.compute: {e}")

    try:
        from diagrams.aws.database import RDS, Aurora, DynamoDB, ElastiCache, Redshift
        success.append("diagrams.aws.database")
    except ImportError as e:
        errors.append(f"diagrams.aws.database: {e}")

    try:
        from diagrams.aws.network import APIGateway, CloudFront, ELB, ALB, NLB, Route53, VPC
        success.append("diagrams.aws.network")
    except ImportError as e:
        errors.append(f"diagrams.aws.network: {e}")

    try:
        from diagrams.aws.storage import S3, EBS, EFS
        success.append("diagrams.aws.storage")
    except ImportError as e:
        errors.append(f"diagrams.aws.storage: {e}")

    try:
        from diagrams.aws.integration import SQS, SNS, Eventbridge, StepFunctions
        success.append("diagrams.aws.integration")
    except ImportError as e:
        errors.append(f"diagrams.aws.integration: {e}")

    try:
        from diagrams.aws.analytics import Kinesis, Glue, Athena, EMR
        success.append("diagrams.aws.analytics")
    except ImportError as e:
        errors.append(f"diagrams.aws.analytics: {e}")

    try:
        from diagrams.aws.ml import Sagemaker, Comprehend, Rekognition
        success.append("diagrams.aws.ml")
    except ImportError as e:
        errors.append(f"diagrams.aws.ml: {e}")

    try:
        from diagrams.aws.security import IAM, Cognito, WAF, Shield, KMS
        success.append("diagrams.aws.security")
    except ImportError as e:
        errors.append(f"diagrams.aws.security: {e}")

    # ============================================
    # Azure
    # ============================================
    try:
        from diagrams.azure.compute import VM, FunctionApps, ContainerInstances, AKS
        success.append("diagrams.azure.compute")
    except ImportError as e:
        errors.append(f"diagrams.azure.compute: {e}")

    try:
        from diagrams.azure.database import SQLDatabases, CosmosDB, BlobStorage
        success.append("diagrams.azure.database")
    except ImportError as e:
        errors.append(f"diagrams.azure.database: {e}")

    try:
        from diagrams.azure.network import LoadBalancers, ApplicationGateway, VirtualNetworks
        success.append("diagrams.azure.network")
    except ImportError as e:
        errors.append(f"diagrams.azure.network: {e}")

    try:
        from diagrams.azure.integration import ServiceBus, EventGrid
        success.append("diagrams.azure.integration")
    except ImportError as e:
        errors.append(f"diagrams.azure.integration: {e}")

    try:
        from diagrams.azure.ml import MachineLearningServiceWorkspaces
        success.append("diagrams.azure.ml")
    except ImportError as e:
        errors.append(f"diagrams.azure.ml: {e}")

    # ============================================
    # GCP
    # ============================================
    try:
        from diagrams.gcp.compute import ComputeEngine, Functions, Run, GKE
        success.append("diagrams.gcp.compute")
    except ImportError as e:
        errors.append(f"diagrams.gcp.compute: {e}")

    try:
        from diagrams.gcp.database import SQL, Spanner, Bigtable, Firestore
        success.append("diagrams.gcp.database")
    except ImportError as e:
        errors.append(f"diagrams.gcp.database: {e}")

    try:
        from diagrams.gcp.network import LoadBalancing, CDN, DNS
        success.append("diagrams.gcp.network")
    except ImportError as e:
        errors.append(f"diagrams.gcp.network: {e}")

    try:
        from diagrams.gcp.storage import GCS
        success.append("diagrams.gcp.storage")
    except ImportError as e:
        errors.append(f"diagrams.gcp.storage: {e}")

    try:
        from diagrams.gcp.analytics import BigQuery, Dataflow, PubSub
        success.append("diagrams.gcp.analytics")
    except ImportError as e:
        errors.append(f"diagrams.gcp.analytics: {e}")

    try:
        from diagrams.gcp.ml import AIHub, AutoML
        success.append("diagrams.gcp.ml")
    except ImportError as e:
        errors.append(f"diagrams.gcp.ml: {e}")

    # ============================================
    # Kubernetes
    # ============================================
    try:
        from diagrams.k8s.compute import Pod, Deployment, ReplicaSet, StatefulSet, DaemonSet
        success.append("diagrams.k8s.compute")
    except ImportError as e:
        errors.append(f"diagrams.k8s.compute: {e}")

    try:
        from diagrams.k8s.network import Service, Ingress, NetworkPolicy
        success.append("diagrams.k8s.network")
    except ImportError as e:
        errors.append(f"diagrams.k8s.network: {e}")

    try:
        from diagrams.k8s.storage import PV, PVC, StorageClass
        success.append("diagrams.k8s.storage")
    except ImportError as e:
        errors.append(f"diagrams.k8s.storage: {e}")

    try:
        from diagrams.k8s.rbac import ServiceAccount, Role, ClusterRole
        success.append("diagrams.k8s.rbac")
    except ImportError as e:
        errors.append(f"diagrams.k8s.rbac: {e}")

    try:
        from diagrams.k8s.group import Namespace
        success.append("diagrams.k8s.group")
    except ImportError as e:
        errors.append(f"diagrams.k8s.group: {e}")

    # ============================================
    # On-Premise (CRITICAL - Redis fix here!)
    # ============================================
    try:
        from diagrams.onprem.compute import Server, Nomad
        success.append("diagrams.onprem.compute")
    except ImportError as e:
        errors.append(f"diagrams.onprem.compute: {e}")

    try:
        from diagrams.onprem.database import PostgreSQL, MySQL, MongoDB, Cassandra, Couchdb, Mariadb
        success.append("diagrams.onprem.database")
    except ImportError as e:
        errors.append(f"diagrams.onprem.database: {e}")

    # CRITICAL TEST: Redis must be in inmemory, NOT database!
    try:
        from diagrams.onprem.inmemory import Redis, Memcached
        success.append("diagrams.onprem.inmemory (Redis, Memcached) - CRITICAL FIX VERIFIED!")
    except ImportError as e:
        errors.append(f"diagrams.onprem.inmemory: {e} - CRITICAL: Redis import still broken!")

    try:
        from diagrams.onprem.network import Nginx, HAProxy, Traefik, Kong
        success.append("diagrams.onprem.network")
    except ImportError as e:
        errors.append(f"diagrams.onprem.network: {e}")

    try:
        from diagrams.onprem.queue import Kafka, RabbitMQ, Celery
        success.append("diagrams.onprem.queue")
    except ImportError as e:
        errors.append(f"diagrams.onprem.queue: {e}")

    try:
        from diagrams.onprem.container import Docker
        success.append("diagrams.onprem.container")
    except ImportError as e:
        errors.append(f"diagrams.onprem.container: {e}")

    try:
        from diagrams.onprem.ci import Jenkins, GitlabCI, GithubActions, CircleCI
        success.append("diagrams.onprem.ci")
    except ImportError as e:
        errors.append(f"diagrams.onprem.ci: {e}")

    try:
        from diagrams.onprem.monitoring import Prometheus, Grafana, Datadog
        success.append("diagrams.onprem.monitoring")
    except ImportError as e:
        errors.append(f"diagrams.onprem.monitoring: {e}")

    try:
        from diagrams.onprem.logging import Fluentd, Loki, Graylog
        success.append("diagrams.onprem.logging")
    except ImportError as e:
        errors.append(f"diagrams.onprem.logging: {e}")

    try:
        from diagrams.onprem.mlops import Mlflow, Kubeflow
        success.append("diagrams.onprem.mlops")
    except ImportError as e:
        errors.append(f"diagrams.onprem.mlops: {e}")

    try:
        from diagrams.onprem.client import User, Users, Client
        success.append("diagrams.onprem.client")
    except ImportError as e:
        errors.append(f"diagrams.onprem.client: {e}")

    # ============================================
    # Elastic Stack (CRITICAL - Elasticsearch fix here!)
    # ============================================
    try:
        from diagrams.elastic.elasticsearch import Elasticsearch, Kibana, Logstash, Beats
        success.append("diagrams.elastic.elasticsearch (Elasticsearch, Kibana, Logstash, Beats) - CRITICAL FIX VERIFIED!")
    except ImportError as e:
        errors.append(f"diagrams.elastic.elasticsearch: {e} - CRITICAL: Elasticsearch import still broken!")

    # ============================================
    # Generic
    # ============================================
    try:
        from diagrams.generic.compute import Rack
        success.append("diagrams.generic.compute")
    except ImportError as e:
        errors.append(f"diagrams.generic.compute: {e}")

    try:
        from diagrams.generic.database import SQL as GenericSQL
        success.append("diagrams.generic.database")
    except ImportError as e:
        errors.append(f"diagrams.generic.database: {e}")

    try:
        from diagrams.generic.network import Firewall, Router, Switch
        success.append("diagrams.generic.network")
    except ImportError as e:
        errors.append(f"diagrams.generic.network: {e}")

    try:
        from diagrams.generic.os import Linux, Windows
        success.append("diagrams.generic.os")
    except ImportError as e:
        errors.append(f"diagrams.generic.os: {e}")

    try:
        from diagrams.generic.device import Mobile, Tablet
        success.append("diagrams.generic.device")
    except ImportError as e:
        errors.append(f"diagrams.generic.device: {e}")

    # ============================================
    # Programming Languages
    # ============================================
    try:
        from diagrams.programming.language import Python, Java, Go, Rust, JavaScript, TypeScript
        success.append("diagrams.programming.language")
    except ImportError as e:
        errors.append(f"diagrams.programming.language: {e}")

    # ============================================
    # SaaS
    # ============================================
    try:
        from diagrams.saas.chat import Slack, Teams
        success.append("diagrams.saas.chat")
    except ImportError as e:
        errors.append(f"diagrams.saas.chat: {e}")

    try:
        from diagrams.saas.cdn import Cloudflare
        success.append("diagrams.saas.cdn")
    except ImportError as e:
        errors.append(f"diagrams.saas.cdn: {e}")

    try:
        from diagrams.saas.identity import Auth0, Okta
        success.append("diagrams.saas.identity")
    except ImportError as e:
        errors.append(f"diagrams.saas.identity: {e}")

    # ============================================
    # Core
    # ============================================
    try:
        from diagrams.custom import Custom
        from diagrams import Diagram, Cluster, Edge
        success.append("diagrams (Diagram, Cluster, Edge, Custom)")
    except ImportError as e:
        errors.append(f"diagrams core: {e}")

    # ============================================
    # Report
    # ============================================
    print("=" * 60)
    print("DIAGRAMS CHEAT SHEET IMPORT VALIDATION")
    print("=" * 60)
    print()

    print(f"SUCCESS: {len(success)} modules")
    print("-" * 40)
    for s in success:
        print(f"  OK: {s}")
    print()

    if errors:
        print(f"ERRORS: {len(errors)} modules")
        print("-" * 40)
        for e in errors:
            print(f"  FAIL: {e}")
        print()
        print("=" * 60)
        print("VALIDATION FAILED")
        print("=" * 60)
        return False
    else:
        print("=" * 60)
        print("ALL IMPORTS VALIDATED SUCCESSFULLY!")
        print("=" * 60)
        return True


def test_old_wrong_imports():
    """Verify that the OLD wrong imports would fail."""
    print()
    print("=" * 60)
    print("VERIFYING OLD WRONG IMPORTS WOULD FAIL")
    print("=" * 60)
    print()

    # Test 1: Redis in database (WRONG - should fail)
    try:
        from diagrams.onprem.database import Redis
        print("UNEXPECTED: diagrams.onprem.database.Redis exists (library might have changed)")
    except ImportError:
        print("CONFIRMED: diagrams.onprem.database.Redis does NOT exist (expected)")

    # Test 2: Elasticsearch in onprem.database (WRONG - should fail)
    try:
        from diagrams.onprem.database import Elasticsearch
        print("UNEXPECTED: diagrams.onprem.database.Elasticsearch exists (library might have changed)")
    except ImportError:
        print("CONFIRMED: diagrams.onprem.database.Elasticsearch does NOT exist (expected)")

    print()


if __name__ == "__main__":
    # Run the validation
    all_ok = test_all_imports()
    test_old_wrong_imports()

    # Exit with appropriate code
    sys.exit(0 if all_ok else 1)
