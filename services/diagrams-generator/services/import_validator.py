"""
Import validator for the diagrams library.

This module provides comprehensive validation and auto-correction of imports
for the Python diagrams library. It maintains a complete mapping of all valid
imports and can automatically fix common mistakes made by LLMs.
"""

import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ImportFix:
    """Represents a fix for an invalid import."""
    original_module: str
    original_name: str
    correct_module: str
    correct_name: str


class ImportValidator:
    """Validates and corrects imports for the diagrams library."""

    # Complete mapping of all valid imports from the diagrams library
    # Format: module_path -> set of valid class names
    VALID_IMPORTS: Dict[str, Set[str]] = {
        # Core
        "diagrams": {"Diagram", "Cluster", "Node", "Edge"},

        # ==========================================================================
        # AWS
        # ==========================================================================
        "diagrams.aws.compute": {
            "EC2", "EC2Instance", "EC2Instances", "EC2Ami", "EC2AutoScaling",
            "EC2ContainerRegistry", "EC2ElasticIpAddress", "EC2ImageBuilder",
            "EC2Rescue", "EC2SpotInstance", "ElasticBeanstalk", "ElasticContainerService",
            "ElasticKubernetesService", "Fargate", "Lambda", "LambdaFunction",
            "Lightsail", "LocalZones", "Outposts", "ServerlessApplicationRepository",
            "ThinkboxDeadline", "ThinkboxDraft", "ThinkboxFrost", "ThinkboxKrakatoa",
            "ThinkboxSequoia", "ThinkboxStoke", "ThinkboxXmesh", "VmwareCloudOnAWS",
            "Wavelength", "Batch", "Compute", "ApplicationAutoScaling",
        },
        "diagrams.aws.database": {
            "Aurora", "AuroraInstance", "Database", "DatabaseMigrationService",
            "DatabaseMigrationServiceDatabaseMigrationWorkflow", "DocumentDB",
            "DynamoDB", "DynamodbDax", "DynamodbGlobalSecondaryIndex", "DynamodbItem",
            "DynamodbItems", "DynamodbTable", "ElastiCache", "ElasticacheForMemcached",
            "ElasticacheForRedis", "Keyspaces", "Neptune", "QuantumLedgerDatabase",
            "RDS", "RDSInstance", "RDSOnVmware", "Redshift", "RedshiftDenseComputeNode",
            "RedshiftDenseStorageNode", "Timestream", "DocumentdbMongodbCompatibility",
            "DB", "Dynamodb", "Elasticache", "QLDB", "Memorydb",
        },
        "diagrams.aws.network": {
            "APIGateway", "APIGatewayEndpoint", "AppMesh", "ClientVpn", "CloudFront",
            "CloudFrontDownloadDistribution", "CloudFrontEdgeLocation",
            "CloudFrontStreamingDistribution", "CloudMap", "DirectConnect",
            "ElasticLoadBalancing", "ELB", "ALB", "NLB", "CLB",
            "GlobalAccelerator", "InternetGateway", "NATGateway", "NetworkingAndContentDelivery",
            "PrivateLink", "Privatelink", "Route53", "Route53HostedZone",
            "SiteToSiteVpn", "TransitGateway", "VPC", "VPCCustomerGateway",
            "VPCElasticNetworkAdapter", "VPCElasticNetworkInterface", "VPCFlowLogs",
            "VPCPeering", "VPCRouter", "VPCTrafficMirroring", "VpnConnection",
            "VpnGateway", "Endpoint", "Nacl", "NetworkAcl", "RouteTable",
            "VpcPeering", "TransitGatewayAttachment", "ElbApplicationLoadBalancer",
            "ElbClassicLoadBalancer", "ElbNetworkLoadBalancer", "AppMeshVirtualGateway",
            "CloudfrontEdgeLocation", "ElbGatewayLoadBalancer", "Cloudfront",
        },
        "diagrams.aws.storage": {
            "Backup", "CloudendureDisasterRecovery", "EBS", "EBSSnapshot",
            "EBSVolume", "EFS", "ElasticBlockStore", "ElasticFileSystem",
            "FsxForLustre", "FsxForWindowsFileServer", "Fsx", "S3",
            "S3Bucket", "S3Glacier", "S3GlacierArchive", "S3GlacierVault",
            "S3Object", "SimpleStorageService", "SnowFamily", "Snowball",
            "SnowballEdge", "Snowcone", "Snowmobile", "StorageGateway",
            "StorageGatewayCachedVolume", "StorageGatewayStoredVolume",
            "StorageGatewayVirtualTapeLibrary", "Storage", "S3Standard",
            "ElasticBlockStoreEBS", "SimpleStorageServiceS3",
        },
        "diagrams.aws.analytics": {
            "Analytics", "Athena", "CloudSearch", "CloudsearchSearchDocuments",
            "DataExchange", "DataPipeline", "ElasticsearchService", "EMR",
            "EMRCluster", "EMREngineMaprM3", "EMREngineMaprM5", "EMREngineMaprM7",
            "EMRHdfsCluster", "Glue", "GlueCrawlers", "GlueDataCatalog",
            "Kinesis", "KinesisDataAnalytics", "KinesisDataFirehose",
            "KinesisDataStreams", "KinesisVideoStreams", "LakeFormation",
            "ManagedStreamingForApacheKafka", "Quicksight", "Redshift",
            "ES", "OpenSearch", "OpenSearchService",
        },
        "diagrams.aws.integration": {
            "APIGateway", "AppSync", "ConsoleMobileApplication", "EventBridge",
            "EventBridgeCustomEventBusResource", "EventBridgeDefaultEventBusResource",
            "EventBridgeSaasPartnerEventBusResource", "MQ", "SimpleNotificationService",
            "SimpleQueueService", "SNS", "SQS", "StepFunctions", "Eventbridge",
        },
        "diagrams.aws.security": {
            "ACM", "Artifact", "CertificateAuthority", "CloudDirectory", "CloudHsm",
            "Cognito", "Detective", "DirectoryService", "FirewallManager", "Guardduty",
            "IAM", "IAMAccessAnalyzer", "IAMPermissions", "IAMRole", "IdentityAndAccessManagement",
            "Inspector", "KeyManagementService", "KMS", "Macie", "ManagedMicrosoftAd",
            "ResourceAccessManager", "RAM", "SecretsManager", "SecurityHub",
            "SecurityIdentityAndCompliance", "Shield", "ShieldAdvanced", "SingleSignOn",
            "SSO", "WAF", "WAFFilteringRule", "Detective", "InspectorAgent",
        },
        "diagrams.aws.management": {
            "AutoScaling", "Cloudformation", "CloudformationChangeSet",
            "CloudformationStack", "CloudformationTemplate", "Cloudtrail",
            "Cloudwatch", "CloudwatchAlarm", "CloudwatchEventEventBased",
            "CloudwatchEventTimeBased", "CloudwatchRule", "Codeguru", "CommandLineInterface",
            "Config", "ControlTower", "LicenseManager", "ManagedServices",
            "ManagementConsole", "OpsworksApps", "OpsworksDeployments",
            "OpsworksInstances", "OpsworksLayers", "OpsworksMonitoring",
            "OpsworksPermissions", "OpsworksResources", "OpsworksStack",
            "Organizations", "OrganizationsAccount", "OrganizationsOrganizationalUnit",
            "PersonalHealthDashboard", "ServiceCatalog", "SystemsManager",
            "SystemsManagerAutomation", "SystemsManagerDocuments",
            "SystemsManagerInventory", "SystemsManagerMaintenanceWindows",
            "SystemsManagerOpscenter", "SystemsManagerParameterStore",
            "SystemsManagerPatchManager", "SystemsManagerRunCommand",
            "SystemsManagerStateManager", "TrustedAdvisor", "TrustedAdvisorChecklist",
            "TrustedAdvisorChecklistCost", "TrustedAdvisorChecklistFaultTolerant",
            "TrustedAdvisorChecklistPerformance", "TrustedAdvisorChecklistSecurity",
            "WellArchitectedTool", "SSM", "CloudWatch", "ParameterStore",
        },
        "diagrams.aws.ml": {
            "ApacheMxnetOnAWS", "AugmentedAi", "Comprehend", "DeepLearningAmis",
            "DeepLearningContainers", "Deepcomposer", "Deeplens", "Deepracer",
            "ElasticInference", "Forecast", "FraudDetector", "Kendra", "Lex",
            "MachineLearning", "Personalize", "Polly", "Rekognition",
            "SagemakerGroundTruth", "SagemakerModel", "SagemakerNotebook",
            "SagemakerTrainingJob", "Sagemaker", "Textract", "Transcribe",
            "Translate", "Bedrock",
        },
        "diagrams.aws.general": {
            "Client", "Disk", "Forums", "General", "GenericDatabase",
            "GenericFirewall", "GenericOfficeBuilding", "GenericSamlToken",
            "GenericSDK", "InternetAlt1", "InternetAlt2", "InternetGateway",
            "Marketplace", "MobileClient", "Multimedia", "OfficeBuilding",
            "SamlToken", "SDK", "SslPadlock", "TapeStorage", "Toolkit",
            "TraditionalServer", "User", "Users",
        },

        # ==========================================================================
        # GCP
        # ==========================================================================
        "diagrams.gcp.compute": {
            "AppEngine", "ComputeEngine", "ContainerOptimizedOS", "Functions",
            "GKEOnPrem", "GPU", "KubernetesEngine", "Run", "GCE", "GKE",
            "CloudFunctions", "CloudRun",
        },
        "diagrams.gcp.database": {
            "Bigtable", "Datastore", "Firestore", "Memorystore", "Spanner", "SQL",
            "BigTable", "CloudSQL", "CloudSpanner",
        },
        "diagrams.gcp.network": {
            "Armor", "CDN", "DedicatedInterconnect", "DNS", "ExternalIpAddresses",
            "FirewallRules", "LoadBalancing", "NAT", "Network", "PartnerInterconnect",
            "PremiumNetworkTier", "Router", "Routes", "StandardNetworkTier",
            "TrafficDirector", "VirtualPrivateCloud", "VPN", "CloudDNS", "CloudCDN",
            "CloudArmor", "CloudNAT", "CloudRouter", "CloudVPN", "VPC",
        },
        "diagrams.gcp.storage": {
            "Filestore", "GCS", "PersistentDisk", "Storage", "CloudStorage",
        },
        "diagrams.gcp.analytics": {
            "BigQuery", "Composer", "DataCatalog", "DataFusion", "Dataflow",
            "Datalab", "Dataprep", "Dataproc", "Genomics", "Pubsub", "PubSub",
        },
        "diagrams.gcp.ml": {
            "AdvancedSolutionsLab", "AIHub", "AIPlatformDataLabelingService",
            "AIPlatform", "AutoML", "AutomlNaturalLanguage", "AutomlTables",
            "AutomlTranslation", "AutomlVideoIntelligence", "AutomlVision",
            "Automl", "DialogFlowEnterpriseEdition", "DialogFlow", "InferenceAPI",
            "JobsAPI", "NaturalLanguageAPI", "RecommendationsAI", "SpeechToText",
            "TextToSpeech", "TPU", "TranslationAPI", "VideoIntelligenceAPI",
            "VisionAPI", "VertexAI",
        },
        "diagrams.gcp.security": {
            "Iam", "IAP", "KeyManagementService", "ResourceManager",
            "SecurityCommandCenter", "SecurityScanner", "KMS", "IAM",
        },
        "diagrams.gcp.devtools": {
            "Build", "CodeForIntellij", "Code", "ContainerRegistry", "GradleAppEnginePlugin",
            "IdePlugins", "MavenAppEnginePlugin", "Scheduler", "SDK", "SourceRepositories",
            "Tasks", "TestLab", "ToolsForEclipse", "ToolsForPowershell", "ToolsForVisualStudio",
            "CloudBuild", "CloudCode", "CloudScheduler", "CloudTasks", "ArtifactRegistry",
        },

        # ==========================================================================
        # Azure
        # ==========================================================================
        "diagrams.azure.compute": {
            "AppServices", "AutomanagedVM", "AvailabilitySets", "BatchAccounts",
            "CitrixVirtualDesktopsEssentials", "CloudServices", "CloudServicesClassic",
            "CloudsimpleVirtualMachines", "ContainerInstances", "ContainerRegistries",
            "DiskEncryptionSets", "Disks", "FunctionApps", "ImageDefinitions",
            "ImageVersions", "KubernetesServices", "MeshApplications", "OsImages",
            "SAPHANAOnAzure", "ServiceFabricClusters", "SharedImageGalleries",
            "Snapshots", "SpringCloud", "VMClassic", "VMImages", "VMLinux",
            "VMScaleSet", "VMWindows", "VM", "Workspaces", "AKS", "ACR", "ACR",
        },
        "diagrams.azure.database": {
            "BlobStorage", "CacheForRedis", "CosmosDb", "DatabaseForMariadbServers",
            "DatabaseForMysqlServers", "DatabaseForPostgresqlServers", "DataFactory",
            "DataLake", "ElasticDatabasePools", "ElasticJobAgents", "InstancePools",
            "ManagedDatabases", "SQLDatabases", "SQLDatawarehouse", "SQLManagedInstances",
            "SQLServerStretchDatabases", "SQLServers", "SQLVM", "VirtualClusters",
            "VirtualDatacenter", "SQL", "SQLDatabase", "MySQL", "PostgreSQL",
        },
        "diagrams.azure.network": {
            "ApplicationGateway", "ApplicationSecurityGroups", "CDNProfiles",
            "Connections", "DDOSProtectionPlans", "DNSPrivateZones", "DNSZones",
            "ExpressrouteCircuits", "Firewall", "FrontDoors", "LoadBalancers",
            "LocalNetworkGateways", "NATGateways", "NetworkInterfaces",
            "NetworkSecurityGroupsClassic", "NetworkWatcher", "OnPremisesDataGateways",
            "PrivateEndpoints", "PrivateLinkHub", "PrivateLinkServices",
            "PublicIpAddresses", "PublicIpPrefixes", "ReservedIpAddressesClassic",
            "RouteFilters", "RouteTables", "ServiceEndpointPolicies",
            "Subnets", "TrafficManagerProfiles", "VirtualNetworkClassic",
            "VirtualNetworkGateways", "VirtualNetworks", "VirtualWans",
            "VPN", "VPNGateway", "LoadBalancer", "DNS", "TrafficManager", "VNet",
        },
        "diagrams.azure.storage": {
            "ArchiveStorage", "Azurefxtedgefiler", "BlobStorage", "DataBox",
            "DataBoxEdge", "DataLakeStorage", "GeneralStorage", "NetappFiles",
            "QueuesStorage", "StorageAccountsClassic", "StorageAccounts",
            "StorageExplorer", "StorageSyncServices", "StorsimpleDataManagers",
            "StorsimpleDeviceManagers", "TableStorage", "Storage", "FileStorage",
        },
        "diagrams.azure.analytics": {
            "AnalysisServices", "DataExplorerClusters", "DataFactories",
            "DataLakeAnalytics", "DataLakeStoreGen1", "Databricks",
            "EventHubClusters", "EventHubs", "Hdinsightclusters", "LogAnalyticsWorkspaces",
            "StreamAnalyticsJobs", "SynapseAnalytics", "HDInsight", "EventHub",
            "Synapse",
        },
        "diagrams.azure.integration": {
            "APIForFhir", "APIManagement", "AppConfiguration", "DataCatalog",
            "EventGridDomains", "EventGridSubscriptions", "EventGridTopics",
            "IntegrationAccounts", "IntegrationServiceEnvironments", "LogicApps",
            "PartnerTopic", "SendgridAccounts", "ServiceBus", "ServiceBusRelays",
            "SignalR", "SoftwareAsAService", "StorsimpleDeviceManagers",
            "SystemTopic", "ServiceCatalogManagedApplicationDefinitions",
            "LogicApp", "APIM",
        },
        "diagrams.azure.security": {
            "ApplicationSecurityGroups", "ConditionalAccess", "Defender",
            "ExtendedSecurityUpdates", "KeyVaults", "SecurityCenter", "Sentinel",
            "KeyVault",
        },
        "diagrams.azure.ml": {
            "BatchAI", "BotServices", "CognitiveServices", "GenomicsAccounts",
            "MachineLearningServiceWorkspaces", "MachineLearningStudioClassicWebServices",
            "MachineLearningStudioWebServicePlans", "MachineLearningStudioWorkspaces",
            "MachineLearning", "AzureML", "OpenAI",
        },
        "diagrams.azure.general": {
            "Allresources", "Azurehome", "Developertools", "Helpsupport",
            "Information", "Managementgroups", "Marketplace", "Quickstartcenter",
            "Recent", "Reservations", "Resource", "Resourcegroups", "Servicehealth",
            "Shareddashboard", "Subscriptions", "Support", "Supportrequests", "Tag",
            "Tags", "Templates", "Twousericon", "Userhealthicon", "Usericon",
            "Userprivacy", "Userresource", "Whatsnew",
        },
        "diagrams.azure.devops": {
            "ApplicationInsights", "Artifacts", "Boards", "Devops", "DevtestLabs",
            "LabServices", "Pipelines", "Repos", "TestPlans",
        },

        # ==========================================================================
        # Kubernetes
        # ==========================================================================
        "diagrams.k8s.compute": {
            "Cronjob", "Deploy", "Deployment", "DS", "DaemonSet", "Job", "Pod",
            "RS", "ReplicaSet", "STS", "StatefulSet",
        },
        "diagrams.k8s.network": {
            "Endpoint", "Ep", "Ing", "Ingress", "Netpol", "NetworkPolicy", "SVC", "Service",
        },
        "diagrams.k8s.storage": {
            "PV", "PersistentVolume", "PVC", "PersistentVolumeClaim", "SC", "StorageClass", "Vol", "Volume",
        },
        "diagrams.k8s.clusterconfig": {
            "ClusterRole", "ClusterRoleBinding", "HPA", "HorizontalPodAutoscaler",
            "LimitRange", "Limits", "Ns", "Namespace", "Quota", "ResourceQuota",
            "Role", "RoleBinding", "Secret", "SA", "ServiceAccount",
        },
        "diagrams.k8s.controlplane": {
            "API", "APIServer", "CCM", "ControllerManager", "KProxy", "Kubelet",
            "Sched", "Scheduler", "CM", "ConfigMap",
        },
        "diagrams.k8s.group": {
            "NS", "Namespace",
        },
        "diagrams.k8s.infra": {
            "ETCD", "Master", "Node",
        },
        "diagrams.k8s.others": {
            "CRD", "PSP",
        },
        "diagrams.k8s.podconfig": {
            "CM", "ConfigMap", "Secret",
        },
        "diagrams.k8s.rbac": {
            "CRole", "CRB", "ClusterRole", "ClusterRoleBinding", "Group",
            "RB", "RoleBinding", "Role", "SA", "ServiceAccount", "User",
        },

        # ==========================================================================
        # On-Premises
        # ==========================================================================
        "diagrams.onprem.compute": {
            "Nomad", "Server",
        },
        "diagrams.onprem.database": {
            "Cassandra", "Clickhouse", "CockroachDB", "Couchbase", "CouchDB",
            "Dgraph", "Druid", "HBase", "InfluxDB", "Janusgraph", "MariaDB",
            "Mongodb", "MongoDB", "Mssql", "MSSQL", "Mysql", "MySQL", "Neo4J",
            "Oracle", "PostgreSQL", "Postgresql", "Scylla", "TimescaleDB",
            "Questdb", "ClickHouse",
        },
        "diagrams.onprem.inmemory": {
            "Aerospike", "Hazelcast", "Memcached", "Redis",
        },
        "diagrams.onprem.queue": {
            "ActiveMQ", "Activemq", "Celery", "Kafka", "Nats", "RabbitMQ",
            "Rabbitmq", "Zeromq", "ZeroMQ",
        },
        "diagrams.onprem.network": {
            "Apache", "Bind9", "Caddy", "Consul", "Envoy", "Etcd", "ETCD",
            "Glassfish", "Gunicorn", "HAProxy", "Haproxy", "Internet", "Istio",
            "Jetty", "Kong", "Linkerd", "Nginx", "Ocelot", "OpenServiceMesh",
            "Pfsense", "Pomerium", "Powerdns", "Tomcat", "Traefik", "Vyos",
            "Wildfly", "Zookeeper",
        },
        "diagrams.onprem.container": {
            "Containerd", "Crio", "Docker", "Firecracker", "Gvisor", "K3S",
            "LXC", "Lxc", "RKT", "Rkt",
        },
        "diagrams.onprem.ci": {
            "Circleci", "CircleCI", "Concourseci", "ConcourseCI", "Droneci",
            "DroneCI", "GithubActions", "Gitlabci", "GitlabCI", "Jenkins",
            "Teamcity", "TeamCity", "Travisci", "TravisCI", "Zuul", "Tekton",
            "Spinnaker", "ArgoCD", "FluxCD",
        },
        "diagrams.onprem.gitops": {
            "Argocd", "ArgoCD", "Flagger", "Flux",
        },
        "diagrams.onprem.iac": {
            "Ansible", "Atlantis", "Awx", "AWX", "Puppet", "Terraform",
        },
        "diagrams.onprem.monitoring": {
            "Cortex", "Datadog", "Dynatrace", "Grafana", "Humio", "Jaeger",
            "Kibana", "Loki", "Newrelic", "NewRelic", "Prometheus", "PrometheusOperator",
            "Sentry", "Splunk", "Thanos", "Zabbix", "Zipkin",
        },
        "diagrams.onprem.logging": {
            "Fluentbit", "FluentBit", "Graylog", "Loki", "RSyslog", "SyslogNg",
        },
        "diagrams.onprem.aggregator": {
            "Fluentd", "Vector",
        },
        "diagrams.onprem.mlops": {
            "Mlflow", "MLflow", "Polyaxon",
        },
        "diagrams.onprem.analytics": {
            "Beam", "Databricks", "Dbt", "DBT", "Flink", "Hadoop", "Hive",
            "Metabase", "Norikra", "Presto", "Singer", "Spark", "Storm",
            "Superset", "Tableau", "Trino",
        },
        "diagrams.onprem.workflow": {
            "Airflow", "Azkaban", "Conductor", "Digdag", "Kubeflow", "Nifi",
            "NiFi",
        },
        "diagrams.onprem.client": {
            "Client", "User", "Users",
        },
        "diagrams.onprem.security": {
            "Bitwarden", "Trivy", "Vault",
        },
        "diagrams.onprem.vcs": {
            "Git", "Gitea", "Github", "GitHub", "Gitlab", "GitLab", "Svn", "SVN",
        },
        "diagrams.onprem.tracing": {
            "Jaeger", "Tempo",
        },
        "diagrams.onprem.certificates": {
            "CertManager", "LetsEncrypt",
        },
        "diagrams.onprem.storage": {
            "CephOsd", "Ceph", "Glusterfs", "GlusterFS", "HDFS", "Portworx",
        },
        "diagrams.onprem.proxmox": {
            "ProxmoxVE", "Pve",
        },
        "diagrams.onprem.auth": {
            "Boundary", "BuzzfeedSso", "Oauth2Proxy",
        },
        "diagrams.onprem.dns": {
            "Coredns", "CoreDNS", "Powerdns", "PowerDNS",
        },
        "diagrams.onprem.registry": {
            "Harbor", "Jfrog",
        },
        "diagrams.onprem.search": {
            "Solr",
        },

        # ==========================================================================
        # Programming Languages
        # ==========================================================================
        "diagrams.programming.language": {
            "Bash", "C", "Cpp", "Csharp", "Dart", "Elixir", "Erlang", "Go",
            "Java", "Javascript", "Kotlin", "Latex", "Matlab", "Nodejs", "NodeJS",
            "Php", "PHP", "Python", "R", "Ruby", "Rust", "Scala", "Swift",
            "Typescript", "TypeScript",
        },
        "diagrams.programming.framework": {
            "Angular", "Backbone", "Django", "Ember", "Fastapi", "FastAPI",
            "Flask", "Flutter", "Graphql", "GraphQL", "Laravel", "Micronaut",
            "Rails", "React", "Spring", "Starlette", "Svelte", "Vue",
        },
        "diagrams.programming.flowchart": {
            "Action", "Collate", "Database", "Decision", "Delay", "Display",
            "Document", "InputOutput", "Inspection", "InternalStorage",
            "LoopLimit", "ManualInput", "ManualLoop", "Merge", "MultipleDocuments",
            "OffPageConnectorLeft", "OffPageConnectorRight", "Or",
            "PredefinedProcess", "Preparation", "Sort", "StartEnd",
            "StoredData", "SummingJunction",
        },

        # ==========================================================================
        # Generic
        # ==========================================================================
        "diagrams.generic.compute": {
            "Rack",
        },
        "diagrams.generic.database": {
            "SQL",
        },
        "diagrams.generic.device": {
            "Mobile", "Tablet",
        },
        "diagrams.generic.network": {
            "Firewall", "Router", "Subnet", "Switch", "VPN",
        },
        "diagrams.generic.os": {
            "Android", "Centos", "CentOS", "Debian", "IOS", "LinuxGeneral",
            "Raspbian", "RedHat", "Suse", "SUSE", "Ubuntu", "Windows",
        },
        "diagrams.generic.place": {
            "Datacenter",
        },
        "diagrams.generic.storage": {
            "Storage",
        },
        "diagrams.generic.virtualization": {
            "Virtualbox", "VirtualBox", "Vmware", "VMware", "XEN",
        },
        "diagrams.generic.blank": {
            "Blank",
        },

        # ==========================================================================
        # SaaS
        # ==========================================================================
        "diagrams.saas.alerting": {
            "Opsgenie", "Pagerduty", "PagerDuty", "Pushover", "Xmatters",
        },
        "diagrams.saas.analytics": {
            "Datadog", "Snowflake", "Stitch",
        },
        "diagrams.saas.cdn": {
            "Akamai", "Cloudflare",
        },
        "diagrams.saas.chat": {
            "Discord", "Line", "Mattermost", "Messenger", "RocketChat", "Slack",
            "Teams", "Telegram",
        },
        "diagrams.saas.communication": {
            "Twilio",
        },
        "diagrams.saas.filesharing": {
            "Nextcloud",
        },
        "diagrams.saas.identity": {
            "Auth0", "Okta",
        },
        "diagrams.saas.logging": {
            "Datadog", "Newrelic", "Papertrail",
        },
        "diagrams.saas.media": {
            "Cloudinary",
        },
        "diagrams.saas.recommendation": {
            "Recombee",
        },
        "diagrams.saas.social": {
            "Facebook", "Twitter",
        },

        # ==========================================================================
        # Elastic Stack
        # ==========================================================================
        "diagrams.elastic.elasticsearch": {
            "Elasticsearch", "Kibana", "Logstash", "Beats", "ElasticSearch",
        },
        "diagrams.elastic.beats": {
            "APM", "Auditbeat", "Filebeat", "Functionbeat", "Heartbeat",
            "Metricbeat", "Packetbeat",
        },
        "diagrams.elastic.agent": {
            "Agent", "Endpoint", "Fleet", "Integrations",
        },
        "diagrams.elastic.enterprisesearch": {
            "AppSearch", "EnterpriseSearch", "SiteSearch", "WorkplaceSearch",
        },
        "diagrams.elastic.observability": {
            "APM", "Logs", "Metrics", "Observability", "Uptime",
        },
        "diagrams.elastic.saas": {
            "Cloud", "Elastic",
        },
        "diagrams.elastic.security": {
            "Endpoint", "Security", "SIEM", "Xdr",
        },

        # ==========================================================================
        # DigitalOcean
        # ==========================================================================
        "diagrams.digitalocean.compute": {
            "Containers", "Docker", "Droplet", "K8SCluster", "K8SNode",
            "K8SNodePool",
        },
        "diagrams.digitalocean.database": {
            "DbaasPrimary", "DbaasReadOnly", "DbaasStandby",
        },
        "diagrams.digitalocean.network": {
            "Certificate", "Domain", "DomainRegistration", "Firewall",
            "FloatingIp", "InternetGateway", "LoadBalancer", "ManagedVpn",
            "Vpc",
        },
        "diagrams.digitalocean.storage": {
            "Folder", "Space", "Volume",
        },

        # ==========================================================================
        # OpenStack
        # ==========================================================================
        "diagrams.openstack.compute": {
            "Nova", "Placement", "Qinling", "Zun",
        },
        "diagrams.openstack.storage": {
            "Cinder", "Manila", "Swift",
        },
        "diagrams.openstack.network": {
            "Designate", "Neutron", "Octavia",
        },
        "diagrams.openstack.sharedservices": {
            "Barbican", "Glance", "Karbor", "Keystone", "Searchlight",
        },
        "diagrams.openstack.orchestration": {
            "Blazar", "Heat", "Mistral", "Senlin", "Zaqar",
        },
        "diagrams.openstack.applicationlifecycle": {
            "Freezer", "Masakari", "Murano", "Solum",
        },
        "diagrams.openstack.deployment": {
            "Ansible", "Charms", "Helm", "Kolla", "Tripleo",
        },
        "diagrams.openstack.containerservices": {
            "Kuryr", "Zun",
        },
        "diagrams.openstack.monitoring": {
            "Monasca", "Telemetry",
        },
        "diagrams.openstack.user": {
            "Openstackclient",
        },
        "diagrams.openstack.workloadprovisioning": {
            "Magnum", "Sahara", "Trove",
        },
        "diagrams.openstack.frontend": {
            "Horizon",
        },
        "diagrams.openstack.billing": {
            "Cloudkitty",
        },
        "diagrams.openstack.optimization": {
            "Congress", "Rally", "Vitrage", "Watcher",
        },
        "diagrams.openstack.multiregion": {
            "Tricircle",
        },

        # ==========================================================================
        # C4 Model
        # ==========================================================================
        "diagrams.c4": {
            "C4Node", "Container", "Database", "Person", "Relationship",
            "System", "SystemBoundary",
        },

        # ==========================================================================
        # Custom
        # ==========================================================================
        "diagrams.custom": {
            "Custom",
        },

        # ==========================================================================
        # IBM
        # ==========================================================================
        "diagrams.ibm.compute": {
            "BareMetalServer", "CitrixVirtualAppsAndDesktops", "ImageTemplate",
            "Instance", "Key", "PowerInstance",
        },
        "diagrams.ibm.network": {
            "Bridge", "DirectLink", "Enterprise", "Firewall", "FloatingIp",
            "Gateway", "InternetServices", "LoadBalancer", "PublicGateway",
            "Router", "Subnet", "TransitGateway", "Vpc", "VpnConnection",
            "VpnGateway", "VpnPolicy",
        },
        "diagrams.ibm.storage": {
            "BlockStorage", "ObjectStorage",
        },
        "diagrams.ibm.security": {
            "ApiGateway", "Appid", "BeyondTrustPrivilegedAccessManagement",
            "CloudHsmV2", "DataShield", "KeyProtect", "Secrets",
        },
        "diagrams.ibm.data": {
            "Cdl", "CloudantForIbmCloud", "ConversationsForCloud",
            "DataStax", "Db2", "Db2OnCloud", "Db2Warehouse",
            "DistributedLedger", "Dv", "EtcdV3", "InformixOnCloud",
            "MongoDbV3", "NetworkTimeSeries", "PostgresqlV2",
        },
        "diagrams.ibm.devops": {
            "ContinuousDelivery", "ContainerRegistry",
        },
        "diagrams.ibm.analytics": {
            "Analytics", "AnalyticsEngine", "DataPak", "DataReplication",
            "Infosphere", "Streaming", "StreamingAnalytics",
        },
        "diagrams.ibm.applications": {
            "ActionableInsight", "Annotate", "ApiDocs", "AppConnectivity",
            "ApplicationForIbmCloud", "BlankApp", "DataConnector", "Devops",
            "EnterpriseApplications", "EnterpriseMessagingForIbmCloud",
            "EventScheduler", "HostedApplicationForIbmCloud", "Launch",
            "OpenSourceAppsForIbmCloud", "Parse", "PersonalityInsights",
            "ProductRecommendations", "SaasIntegrationWithHybridCloud",
            "Salesforce", "ServiceDiscovery", "Streaming",
            "StreamingAnalytics", "Tone", "Transformation",
            "VisualRecognition", "Watson",
        },
        "diagrams.ibm.infrastructure": {
            "Cdn", "Channels", "CloudMessaging", "Databases", "Diagnostics",
            "EdgeServices", "Enterprise", "FileStorage", "HybridConnectivity",
            "HybridNetworking", "InfrastructureAsCode", "InfraServices",
            "InternetServices", "MicroservicesApplication", "MobileDevices",
            "Monitoring", "MonitoringLogging", "Objectstorage", "Peer",
            "PowerVs", "PublicCloudClasic", "ResourceManagement", "Resources",
            "Runtime", "SaasConnectivity", "SaasIntegration", "SatelliteL2L3",
            "Serverless", "Services", "Sysdig", "VirtualMachines",
        },
        "diagrams.ibm.blockchain": {
            "Blockchain", "BlockchainDeveloper", "CertificateAuthority",
            "ClientApplication", "Communication", "Consensus", "Event",
            "EventListener", "ExistingEnterpriseSystems", "HyperledgerFabric",
            "Ibm", "KeyManagement", "Ledger", "Membership", "MessageBus",
            "Node", "Services", "SmartContract", "StateDatabase",
            "TransactionMgr", "Wallet",
        },
        "diagrams.ibm.general": {
            "CloudServices", "CloudFoundry", "EnterpriseStack", "HybridCloud",
            "Marketplace", "ObjectStorageAccessor", "OfflineCapabilities",
            "Openwhisk", "PeerCloud", "ProviderCloudPortal", "User",
        },
        "diagrams.ibm.management": {
            "AlertNotification", "ApiManagement", "CloudManagement",
            "ClusterManagement", "ContentMgt", "DataServices", "DeviceManagement",
            "InformationGovernance", "ItServiceManagement", "Launch", "Monitoring",
            "ProcessManagement", "ProviderCloudPortalService",
            "PushNotificationService", "ServiceManagement", "TextToSpeech",
        },
        "diagrams.ibm.user": {
            "Browser", "Device", "IntegratedDigitalExperiences", "Physicalentity",
            "Sensor", "User",
        },

        # ==========================================================================
        # Firebase
        # ==========================================================================
        "diagrams.firebase.base": {
            "Firebase",
        },
        "diagrams.firebase.develop": {
            "Authentication", "Firestore", "Functions", "Hosting", "MLKit",
            "RealtimeDatabase", "Storage",
        },
        "diagrams.firebase.extentions": {
            "Extensions",
        },
        "diagrams.firebase.grow": {
            "ABTesting", "AppIndexing", "DynamicLinks", "InAppMessaging",
            "Invites", "Messaging", "Predictions", "RemoteConfig",
        },
        "diagrams.firebase.quality": {
            "AppDistribution", "CrashReporting", "Crashlytics",
            "PerformanceMonitoring", "TestLab",
        },

        # ==========================================================================
        # Alibaba Cloud
        # ==========================================================================
        "diagrams.alibabacloud.compute": {
            "AutoScaling", "BatchCompute", "ContainerRegistry", "ContainerService",
            "ElasticComputeService", "ElasticContainerInstance", "ElasticHighPerformanceComputing",
            "ElasticSearch", "FunctionCompute", "OperationOrchestrationService",
            "ResourceOrchestrationService", "ServerLoadBalancer", "ServerlessAppEngine",
            "SimpleApplicationServer", "WebAppService",
        },
        "diagrams.alibabacloud.database": {
            "ApsaradbCassandra", "ApsaradbHbase", "ApsaradbMemcache", "ApsaradbMongodb",
            "ApsaradbOceanbase", "ApsaradbPolardb", "ApsaradbPostgresql", "ApsaradbPpas",
            "ApsaradbRedis", "ApsaradbSqlserver", "DataManagementService",
            "DataTransmissionService", "DatabaseBackupService", "DisributeRelationalDatabaseService",
            "GraphDatabaseService", "HybriddbForMysql", "RelationalDatabaseService",
        },
        "diagrams.alibabacloud.network": {
            "Cdn", "CloudEnterpriseNetwork", "ElasticIpAddress", "ExpressConnect",
            "NatGateway", "ServerLoadBalancer", "SmartAccessGateway",
            "VirtualPrivateCloud", "VpnGateway",
        },
        "diagrams.alibabacloud.storage": {
            "CloudStorageGateway", "FileStorageHdfs", "FileStorageNas", "HybridBackupRecovery",
            "HybridCloudDisasterRecovery", "Imm", "ObjectStorageService", "ObjectTableStore",
        },
        "diagrams.alibabacloud.analytics": {
            "AnalyticDb", "ClickHouse", "DataLakeAnalytics", "ElaticMapReduce",
            "OpenSearch", "PublicDataManagement",
        },
        "diagrams.alibabacloud.security": {
            "AntiBotService", "AntiDdosBasic", "AntiDdosPro", "AntifraudService",
            "BastionHost", "CloudFirewall", "CloudSecurityScanner", "ContentModeration",
            "CrowdsourcedSecurityTesting", "DataEncryptionService", "DbAudit",
            "GameShield", "IdVerification", "ManagedSecurityService", "SecurityCenter",
            "ServerGuard", "SslCertificates", "WebApplicationFirewall",
        },
        "diagrams.alibabacloud.application": {
            "ApiGateway", "BeeBot", "BlockchainAsAService", "CloudCallCenter",
            "CodePipeline", "DirectMail", "LogService", "MessageNotificationService",
            "NodeJsPerformancePlatform", "OpenSearch", "PerformanceTestingService",
            "RdCloud", "SmartConversationAnalysis", "Yida",
        },
        "diagrams.alibabacloud.communication": {
            "DirectMail", "MobilePush",
        },
        "diagrams.alibabacloud.iot": {
            "IotInternetDeviceId", "IotLinkWan", "IotMobileConnectionPackage", "IotPlatform",
        },
        "diagrams.alibabacloud.web": {
            "Dns", "Domain",
        },

        # ==========================================================================
        # Outscale
        # ==========================================================================
        "diagrams.outscale.compute": {
            "Compute", "DirectConnect",
        },
        "diagrams.outscale.network": {
            "ClientGateway", "InternetService", "LoadBalancer", "NatService",
            "Net", "NIC", "SiteToSiteVpng", "VirtualPrivateCloud",
        },
        "diagrams.outscale.security": {
            "Firewall", "IdentityAndAccessManagement",
        },
        "diagrams.outscale.storage": {
            "SimpleStorageService", "Storage",
        },
    }

    # Common mistakes made by LLMs and their corrections
    COMMON_FIXES: Dict[Tuple[str, str], Tuple[str, str]] = {
        # Fluentd is in aggregator, not logging
        ("diagrams.onprem.logging", "Fluentd"): ("diagrams.onprem.aggregator", "Fluentd"),
        ("diagrams.onprem.logging", "Vector"): ("diagrams.onprem.aggregator", "Vector"),
        # Elasticsearch imports
        ("diagrams.elastic.elasticsearch", "ElasticSearch"): ("diagrams.elastic.elasticsearch", "Elasticsearch"),
        ("diagrams.onprem.monitoring", "Elasticsearch"): ("diagrams.elastic.elasticsearch", "Elasticsearch"),
        ("diagrams.onprem.logging", "Elasticsearch"): ("diagrams.elastic.elasticsearch", "Elasticsearch"),
        ("diagrams.onprem.logging", "Logstash"): ("diagrams.elastic.elasticsearch", "Logstash"),
        ("diagrams.onprem.logging", "Kibana"): ("diagrams.elastic.elasticsearch", "Kibana"),
        ("diagrams.onprem.logging", "Beats"): ("diagrams.elastic.elasticsearch", "Beats"),
        # Common database fixes
        ("diagrams.onprem.database", "Postgres"): ("diagrams.onprem.database", "PostgreSQL"),
        ("diagrams.onprem.database", "Mongo"): ("diagrams.onprem.database", "MongoDB"),
        ("diagrams.onprem.database", "MySQL"): ("diagrams.onprem.database", "Mysql"),
        # AWS fixes
        ("diagrams.aws.storage", "S3Bucket"): ("diagrams.aws.storage", "S3"),
        ("diagrams.aws.compute", "ECS"): ("diagrams.aws.compute", "ElasticContainerService"),
        ("diagrams.aws.compute", "EKS"): ("diagrams.aws.compute", "ElasticKubernetesService"),
        # GCP fixes
        ("diagrams.gcp.compute", "GCE"): ("diagrams.gcp.compute", "ComputeEngine"),
        ("diagrams.gcp.compute", "GKE"): ("diagrams.gcp.compute", "KubernetesEngine"),
        # Azure fixes
        ("diagrams.azure.compute", "AKS"): ("diagrams.azure.compute", "KubernetesServices"),
        # K8s fixes
        ("diagrams.k8s.compute", "Deployment"): ("diagrams.k8s.compute", "Deploy"),
        ("diagrams.k8s.network", "Service"): ("diagrams.k8s.network", "SVC"),
        # CI/CD fixes
        ("diagrams.onprem.ci", "GitHub"): ("diagrams.onprem.vcs", "Github"),
        ("diagrams.onprem.ci", "GitLab"): ("diagrams.onprem.vcs", "Gitlab"),
    }

    # Alternative names mapping (name -> (module, canonical_name))
    ALTERNATIVE_NAMES: Dict[str, Tuple[str, str]] = {
        # Common aliases
        "ECS": ("diagrams.aws.compute", "ElasticContainerService"),
        "EKS": ("diagrams.aws.compute", "ElasticKubernetesService"),
        "RDS": ("diagrams.aws.database", "RDS"),
        "DynamoDB": ("diagrams.aws.database", "Dynamodb"),
        "S3": ("diagrams.aws.storage", "S3"),
        "SQS": ("diagrams.aws.integration", "SQS"),
        "SNS": ("diagrams.aws.integration", "SNS"),
        "Lambda": ("diagrams.aws.compute", "Lambda"),
        "APIGateway": ("diagrams.aws.network", "APIGateway"),
        "CloudFront": ("diagrams.aws.network", "CloudFront"),
        "Route53": ("diagrams.aws.network", "Route53"),
        "VPC": ("diagrams.aws.network", "VPC"),
        "ALB": ("diagrams.aws.network", "ALB"),
        "NLB": ("diagrams.aws.network", "NLB"),
        "ElastiCache": ("diagrams.aws.database", "ElastiCache"),
        "Redshift": ("diagrams.aws.database", "Redshift"),
        "Athena": ("diagrams.aws.analytics", "Athena"),
        "Glue": ("diagrams.aws.analytics", "Glue"),
        "Kinesis": ("diagrams.aws.analytics", "Kinesis"),
        "EMR": ("diagrams.aws.analytics", "EMR"),
        "SageMaker": ("diagrams.aws.ml", "Sagemaker"),
        "Cognito": ("diagrams.aws.security", "Cognito"),
        "IAM": ("diagrams.aws.security", "IAM"),
        "KMS": ("diagrams.aws.security", "KMS"),
        "CloudWatch": ("diagrams.aws.management", "Cloudwatch"),
        "CloudFormation": ("diagrams.aws.management", "Cloudformation"),
        "SecretsManager": ("diagrams.aws.security", "SecretsManager"),
        # GCP
        "GCE": ("diagrams.gcp.compute", "ComputeEngine"),
        "GKE": ("diagrams.gcp.compute", "KubernetesEngine"),
        "CloudRun": ("diagrams.gcp.compute", "Run"),
        "CloudFunctions": ("diagrams.gcp.compute", "Functions"),
        "BigQuery": ("diagrams.gcp.analytics", "BigQuery"),
        "PubSub": ("diagrams.gcp.analytics", "Pubsub"),
        "CloudSQL": ("diagrams.gcp.database", "SQL"),
        "CloudSpanner": ("diagrams.gcp.database", "Spanner"),
        "CloudStorage": ("diagrams.gcp.storage", "GCS"),
        "Firestore": ("diagrams.gcp.database", "Firestore"),
        "VertexAI": ("diagrams.gcp.ml", "VertexAI"),
        # Azure
        "AKS": ("diagrams.azure.compute", "KubernetesServices"),
        "ACR": ("diagrams.azure.compute", "ContainerRegistries"),
        "AzureSQL": ("diagrams.azure.database", "SQLDatabases"),
        "CosmosDB": ("diagrams.azure.database", "CosmosDb"),
        "AzureML": ("diagrams.azure.ml", "MachineLearningServiceWorkspaces"),
        "KeyVault": ("diagrams.azure.security", "KeyVaults"),
        "EventHub": ("diagrams.azure.analytics", "EventHubs"),
        "ServiceBus": ("diagrams.azure.integration", "ServiceBus"),
        "LogicApps": ("diagrams.azure.integration", "LogicApps"),
        "APIM": ("diagrams.azure.integration", "APIManagement"),
        # Kubernetes
        "Pod": ("diagrams.k8s.compute", "Pod"),
        "Deployment": ("diagrams.k8s.compute", "Deploy"),
        "Service": ("diagrams.k8s.network", "SVC"),
        "Ingress": ("diagrams.k8s.network", "Ing"),
        "ConfigMap": ("diagrams.k8s.podconfig", "CM"),
        "Secret": ("diagrams.k8s.podconfig", "Secret"),
        "PersistentVolume": ("diagrams.k8s.storage", "PV"),
        "PersistentVolumeClaim": ("diagrams.k8s.storage", "PVC"),
        "Namespace": ("diagrams.k8s.clusterconfig", "Ns"),
        "HPA": ("diagrams.k8s.clusterconfig", "HPA"),
        # On-prem
        "Kafka": ("diagrams.onprem.queue", "Kafka"),
        "RabbitMQ": ("diagrams.onprem.queue", "RabbitMQ"),
        "Redis": ("diagrams.onprem.inmemory", "Redis"),
        "Memcached": ("diagrams.onprem.inmemory", "Memcached"),
        "PostgreSQL": ("diagrams.onprem.database", "PostgreSQL"),
        "MySQL": ("diagrams.onprem.database", "Mysql"),
        "MongoDB": ("diagrams.onprem.database", "MongoDB"),
        "Cassandra": ("diagrams.onprem.database", "Cassandra"),
        "Elasticsearch": ("diagrams.elastic.elasticsearch", "Elasticsearch"),
        "Kibana": ("diagrams.elastic.elasticsearch", "Kibana"),
        "Logstash": ("diagrams.elastic.elasticsearch", "Logstash"),
        "Fluentd": ("diagrams.onprem.aggregator", "Fluentd"),
        "Fluentbit": ("diagrams.onprem.logging", "Fluentbit"),
        "Prometheus": ("diagrams.onprem.monitoring", "Prometheus"),
        "Grafana": ("diagrams.onprem.monitoring", "Grafana"),
        "Jaeger": ("diagrams.onprem.monitoring", "Jaeger"),
        "Nginx": ("diagrams.onprem.network", "Nginx"),
        "HAProxy": ("diagrams.onprem.network", "HAProxy"),
        "Traefik": ("diagrams.onprem.network", "Traefik"),
        "Envoy": ("diagrams.onprem.network", "Envoy"),
        "Istio": ("diagrams.onprem.network", "Istio"),
        "Consul": ("diagrams.onprem.network", "Consul"),
        "Docker": ("diagrams.onprem.container", "Docker"),
        "Jenkins": ("diagrams.onprem.ci", "Jenkins"),
        "ArgoCD": ("diagrams.onprem.gitops", "Argocd"),
        "Terraform": ("diagrams.onprem.iac", "Terraform"),
        "Ansible": ("diagrams.onprem.iac", "Ansible"),
        "Vault": ("diagrams.onprem.security", "Vault"),
        "Airflow": ("diagrams.onprem.workflow", "Airflow"),
        "Spark": ("diagrams.onprem.analytics", "Spark"),
        "Flink": ("diagrams.onprem.analytics", "Flink"),
    }

    @classmethod
    def validate_import(cls, module: str, name: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate an import and return corrected module/name if needed.

        Returns:
            Tuple of (is_valid, corrected_module, corrected_name)
            If valid, corrected values are None.
        """
        # Check if it's a known fix
        fix_key = (module, name)
        if fix_key in cls.COMMON_FIXES:
            corrected = cls.COMMON_FIXES[fix_key]
            return False, corrected[0], corrected[1]

        # Check if module exists and name is valid
        if module in cls.VALID_IMPORTS:
            if name in cls.VALID_IMPORTS[module]:
                return True, None, None

            # Try case-insensitive match
            for valid_name in cls.VALID_IMPORTS[module]:
                if valid_name.lower() == name.lower():
                    return False, module, valid_name

        # Try to find the name in alternative names
        if name in cls.ALTERNATIVE_NAMES:
            corrected = cls.ALTERNATIVE_NAMES[name]
            return False, corrected[0], corrected[1]

        # Try to find the name in any module
        for mod, names in cls.VALID_IMPORTS.items():
            if name in names:
                return False, mod, name
            # Case-insensitive search
            for valid_name in names:
                if valid_name.lower() == name.lower():
                    return False, mod, valid_name

        return False, None, None

    @classmethod
    def fix_imports(cls, code: str) -> Tuple[str, List[str], List[str]]:
        """
        Fix all imports in the given Python code.

        Returns:
            Tuple of (fixed_code, errors, warnings)
        """
        errors = []
        warnings = []
        fixed_lines = []

        # Pattern to match import statements
        import_pattern = re.compile(
            r'^(\s*from\s+)([\w.]+)(\s+import\s+)(.+)$',
            re.MULTILINE
        )

        lines = code.split('\n')

        for line in lines:
            match = import_pattern.match(line)
            if match:
                prefix = match.group(1)  # 'from '
                module = match.group(2)  # 'diagrams.xxx.yyy'
                import_kw = match.group(3)  # ' import '
                imports = match.group(4)  # 'Foo, Bar, Baz'

                # Parse individual imports
                import_names = [n.strip() for n in imports.split(',')]
                fixed_imports = []
                module_fixes = {}  # Track if we need to split into multiple import lines

                for name in import_names:
                    # Handle 'as' aliases
                    alias = None
                    if ' as ' in name:
                        name, alias = name.split(' as ')
                        name = name.strip()
                        alias = alias.strip()

                    is_valid, corrected_module, corrected_name = cls.validate_import(module, name)

                    if is_valid:
                        if alias:
                            fixed_imports.append(f"{name} as {alias}")
                        else:
                            fixed_imports.append(name)
                    elif corrected_module and corrected_name:
                        if corrected_module == module:
                            # Same module, just fix the name
                            if alias:
                                fixed_imports.append(f"{corrected_name} as {alias}")
                            else:
                                fixed_imports.append(corrected_name)
                            warnings.append(f"Fixed: {name} -> {corrected_name}")
                        else:
                            # Different module, need separate import
                            if corrected_module not in module_fixes:
                                module_fixes[corrected_module] = []
                            if alias:
                                module_fixes[corrected_module].append(f"{corrected_name} as {alias}")
                            else:
                                module_fixes[corrected_module].append(corrected_name)
                            warnings.append(f"Fixed: {module}.{name} -> {corrected_module}.{corrected_name}")
                    else:
                        errors.append(f"Unknown import: {module}.{name}")
                        # Keep original to not break code completely
                        if alias:
                            fixed_imports.append(f"{name} as {alias}")
                        else:
                            fixed_imports.append(name)

                # Build fixed line(s)
                if fixed_imports:
                    fixed_lines.append(f"{prefix}{module}{import_kw}{', '.join(fixed_imports)}")

                # Add additional import lines for module fixes
                for fixed_module, fixed_names in module_fixes.items():
                    fixed_lines.append(f"from {fixed_module} import {', '.join(fixed_names)}")
            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines), errors, warnings

    @classmethod
    def get_suggested_imports(cls, description: str) -> List[Tuple[str, str]]:
        """
        Suggest likely imports based on a description.

        Returns list of (module, name) tuples.
        """
        suggestions = []
        description_lower = description.lower()

        # Keywords to imports mapping
        keyword_imports = {
            'aws': [
                ("diagrams.aws.compute", "EC2"),
                ("diagrams.aws.storage", "S3"),
                ("diagrams.aws.database", "RDS"),
                ("diagrams.aws.network", "VPC"),
            ],
            'lambda': [("diagrams.aws.compute", "Lambda")],
            's3': [("diagrams.aws.storage", "S3")],
            'ec2': [("diagrams.aws.compute", "EC2")],
            'rds': [("diagrams.aws.database", "RDS")],
            'dynamodb': [("diagrams.aws.database", "Dynamodb")],
            'sqs': [("diagrams.aws.integration", "SQS")],
            'sns': [("diagrams.aws.integration", "SNS")],

            'gcp': [
                ("diagrams.gcp.compute", "ComputeEngine"),
                ("diagrams.gcp.storage", "GCS"),
                ("diagrams.gcp.database", "SQL"),
            ],
            'bigquery': [("diagrams.gcp.analytics", "BigQuery")],
            'gke': [("diagrams.gcp.compute", "KubernetesEngine")],

            'azure': [
                ("diagrams.azure.compute", "VM"),
                ("diagrams.azure.storage", "BlobStorage"),
                ("diagrams.azure.database", "SQLDatabases"),
            ],
            'aks': [("diagrams.azure.compute", "KubernetesServices")],

            'kubernetes': [
                ("diagrams.k8s.compute", "Pod"),
                ("diagrams.k8s.compute", "Deploy"),
                ("diagrams.k8s.network", "SVC"),
            ],
            'k8s': [
                ("diagrams.k8s.compute", "Pod"),
                ("diagrams.k8s.compute", "Deploy"),
            ],

            'kafka': [("diagrams.onprem.queue", "Kafka")],
            'rabbitmq': [("diagrams.onprem.queue", "RabbitMQ")],
            'redis': [("diagrams.onprem.inmemory", "Redis")],
            'postgres': [("diagrams.onprem.database", "PostgreSQL")],
            'mysql': [("diagrams.onprem.database", "Mysql")],
            'mongodb': [("diagrams.onprem.database", "MongoDB")],
            'elasticsearch': [("diagrams.elastic.elasticsearch", "Elasticsearch")],
            'nginx': [("diagrams.onprem.network", "Nginx")],
            'docker': [("diagrams.onprem.container", "Docker")],
            'prometheus': [("diagrams.onprem.monitoring", "Prometheus")],
            'grafana': [("diagrams.onprem.monitoring", "Grafana")],
        }

        for keyword, imports in keyword_imports.items():
            if keyword in description_lower:
                suggestions.extend(imports)

        return list(set(suggestions))
