# MLOps Components on Google Cloud, AWS, and Azure

*Report compiled April 26, 2026. Pricing values are list rates from official provider documentation and reflect
US-region defaults; actual figures vary by region, instance choice, commitment level, and configuration.*

> **How to read the citations:** every specific factual claim is followed by `[N]` where `N` is the source number in
> the [Sources](#sources) list at the end. Multiple sources are separated by commas — e.g. `[1, 4]`. Statements without a
> citation are editorial synthesis or definitions that do not attach to a single source.

## A Note on the Approach

There is a near-perfect symmetry across the three vendors at the *category* level — each one has a flagship platform (
Vertex AI, SageMaker, Azure ML), each one offers the same canonical list of components (a pipeline orchestrator, a model
registry, a feature store, a model monitor, etc.), and each one bills broadly the same way (you pay for the underlying
compute and storage, not for the platform abstraction itself) [1, 19, 30]. The actual differences are about three
things:

1. **What's truly first-party** vs. what assumes you'll bring open-source tooling. Azure ML and Vertex AI ship a managed
   feature store; SageMaker's Feature Store is one of the older offerings on the market [11, 20, 26].
2. **How tightly the components are wired together.** SageMaker integrates Pipelines → Model Registry → Model Monitor as
   a tight loop; Vertex AI integrates Pipelines (KFP/TFX) → Model Registry → Model Monitoring with deep BigQuery hooks;
   Azure ML integrates Pipelines → Registries (workspace and cross-workspace) → Model Monitoring with Azure DevOps and
   Event Grid as the glue [1, 8, 19].
3. **Pricing model for the orchestrator and the serving endpoint.** This is where the surprises live. Vertex AI
   charges $0.03 per pipeline run on top of compute [37, 38]; SageMaker Pipelines doesn't charge for orchestration
   itself but every step is a billed compute job [21]; Azure ML charges no platform fee at all, only the underlying
   compute [27, 28]. Endpoints in all three keep billing while idle unless explicitly undeployed — this is the most
   common source of unexpected MLOps cost [37, 39].

The metrics chosen for comparison reflect what actually moves the needle in production MLOps adoption: managed-service
maturity, pipeline orchestration model, registry and lineage, feature-store availability, drift detection,
deployment-target options, CI/CD integration, generative-AI support, lock-in characteristics, and pricing model.

---

## 1. Executive Summary

| Dimension                      | Google Cloud (Vertex AI)                                              | AWS (SageMaker AI)                                               | Azure (Azure Machine Learning)                                      |
|--------------------------------|-----------------------------------------------------------------------|------------------------------------------------------------------|---------------------------------------------------------------------|
| Flagship platform              | Vertex AI [1, 4]                                                      | Amazon SageMaker AI [13, 14, 19]                                 | Azure Machine Learning [22, 24, 25]                                 |
| Pipeline SDK                   | Kubeflow Pipelines (KFP) and TFX [3, 6]                               | SageMaker Pipelines (proprietary Python SDK) [13, 19]            | Azure ML SDK v2, CLI v2, components [25, 28]                        |
| Native feature store (managed) | Vertex AI Feature Store [4, 5, 38]                                    | SageMaker Feature Store [20, 30, 31]                             | Managed feature store (workspace type) [26]                         |
| Model registry                 | Vertex AI Model Registry [4, 7]                                       | SageMaker Model Registry / Model Packages [14, 15, 16, 17]       | Azure ML model registry + cross-workspace registries [11, 22, 23]   |
| First-party model monitoring   | Vertex AI Model Monitoring (skew/drift) [4, 7]                        | SageMaker Model Monitor [14, 19]                                 | Azure ML monitoring (data drift, model performance) [11, 22]        |
| CI/CD path                     | Cloud Build + Cloud Deploy + Artifact Registry [6, 8]                 | SageMaker Projects + CodePipeline / GitHub Actions [13, 14, 16]  | Azure Pipelines / GitHub Actions + MLOps v2 accelerator [9, 10, 25] |
| Headline pricing wrinkle       | Pipelines $0.03/run + compute; endpoints don't scale to zero [37, 38] | No charge for Pipelines orchestration; pay per step compute [21] | No platform charge; pay only for underlying compute [27, 28]        |
| Strongest GenAI integration    | Model Garden (200+ models incl. Gemini, Anthropic) [33]               | Bedrock + JumpStart in registry [17]                             | Model catalog with OpenAI, Hugging Face, Meta, Cohere [24]          |

---

## 2. Side-by-side Component Map

The same nine components map across all three platforms. The names differ; the responsibilities don't.

| MLOps capability                    | Vertex AI (GCP)                                                                                   | SageMaker AI (AWS)                                                                                             | Azure Machine Learning                                                               |
|-------------------------------------|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| **Notebook IDE / workbench**        | Vertex AI Workbench [10]                                                                          | SageMaker Studio (recently "Unified Studio") [13]                                                              | Azure ML Studio + compute instances [22]                                             |
| **Data prep / feature engineering** | Dataflow + BigQuery + custom KFP components [6]                                                   | SageMaker Data Wrangler (300+ transforms) [18, 19]                                                             | Azure ML data assets + designer + Synapse Spark integration [22]                     |
| **Pipeline orchestration**          | Vertex AI Pipelines (KFP / TFX SDK, DAG of containers) [2, 3, 6]                                  | SageMaker Pipelines (DAG, native UI + SDK) [16, 17, 19]                                                        | Azure ML Pipelines (v2 components, registry-shareable) [25, 28]                      |
| **Feature store**                   | Vertex AI Feature Store (online + offline; new gen pricing follows BigQuery for offline) [4, 38]  | SageMaker Feature Store (online + offline; Standard tier on DynamoDB; InMemory tier for sub-10ms) [20, 30, 31] | Managed feature store (Spark materialization; ADLS Gen2 offline + Redis online) [26] |
| **Model registry**                  | Vertex AI Model Registry (free; pay only on deploy) [4, 7]                                        | SageMaker Model Registry / Model Packages (groups, versions, approval workflow) [15, 16, 17]                   | Azure ML registry (per-workspace and cross-workspace) [11, 23]                       |
| **Experiment tracking / metadata**  | Vertex AI Experiments + Vertex ML Metadata ($10/GiB-month for metadata storage) [37]              | SageMaker Experiments + integration with MLflow on SageMaker [21]                                              | Azure ML jobs + MLflow tracking server (native) [22, 24]                             |
| **Model monitoring**                | Vertex AI Model Monitoring ($3.50 per GB analyzed for both training and prediction data) [37, 38] | SageMaker Model Monitor (data quality, model quality, bias drift, feature attribution; with Clarify) [13, 14]  | Azure ML monitoring (data drift + model performance; Event Grid alerts) [11, 22]     |
| **Real-time serving**               | Online Prediction endpoints (do not scale to zero) [38, 39]                                       | SageMaker Endpoints (real-time, async, serverless inference) [19, 21]                                          | Managed online endpoints (autoscale via Azure Monitor) [27, 28]                      |
| **Batch inference**                 | Batch Prediction jobs [4]                                                                         | SageMaker Batch Transform [19]                                                                                 | Batch endpoints / pipeline jobs [25]                                                 |
| **CI/CD glue**                      | Cloud Build + Cloud Deploy; Artifact Registry for component versioning [6, 8]                     | SageMaker Projects (templates) + CodePipeline / GitHub Actions [13, 14, 16]                                    | Azure Pipelines / GitHub Actions + MLOps v2 solution accelerator [9, 10, 25]         |

---

## 3. Pipeline Orchestration

### Authoring model

- **Vertex AI Pipelines** uses Kubeflow Pipelines (KFP) and TFX SDKs; pipelines are DAGs of containerized tasks compiled
  to YAML and run on Google's managed pipeline service [2, 3, 6]. Google's own guidance recommends KFP over TFX unless
  you're already TFX-deep [3].
- **SageMaker Pipelines** uses a proprietary Python SDK; pipelines are DAGs you can author with code, JSON, or a visual
  UI in Studio [16, 17, 19]. The cost of this tighter integration is portability — migrating a SageMaker Pipeline to
  Airflow or Argo Workflows generally requires a complete rewrite [13].
- **Azure ML Pipelines (v2)** uses YAML-defined components and the `az ml` CLI / Python SDK v2; components can be
  registered to a per-workspace or cross-workspace registry and reused across teams [25, 28].

### Pricing for orchestration itself

| Platform            | Orchestration fee                                 | Step compute                                                                       |
|---------------------|---------------------------------------------------|------------------------------------------------------------------------------------|
| Vertex AI Pipelines | **$0.03 per pipeline run**, plus compute [37, 38] | Billed per node-hour of underlying VMs [37]                                        |
| SageMaker Pipelines | **No fee** for orchestration itself [21]          | Each step (training/processing/transform) is billed as a normal SageMaker job [21] |
| Azure ML Pipelines  | **No platform fee** [27, 28]                      | Compute targets billed at underlying VM rates; AML adds nothing on top [27]        |

### Cross-environment promotion

- **Vertex AI**: cross-project promotion via Artifact Registry (compiled pipeline YAML) and shared Model
  Registry [6, 8].
- **SageMaker**: cross-account promotion via Model Registry shared through AWS RAM, plus SageMaker Projects
  templates [14, 17].
- **Azure ML**: cross-workspace registries are an explicit first-class feature — register a component once, run it in
  dev, test, and prod workspaces with the same identifier [11, 23].

Azure ML's cross-workspace registry is the cleanest of the three for the dev → test → prod promotion flow because it's
the only one designed for that pattern from the start [11, 23].

---

## 4. Model Registry

All three platforms provide model versioning, metadata tracking, approval workflows, and the ability to deploy from
registry [4, 15, 22].

| Capability                      | Vertex AI Model Registry                         | SageMaker Model Registry                                                                                       | Azure ML Registry                                                    |
|---------------------------------|--------------------------------------------------|----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| Cost to store a model           | $0 — no cost for keeping models in registry [38] | $0 for the registry itself; charges accrue when deployed [21]                                                  | $0 for the registry; storage in associated Blob Storage account [27] |
| Versioning                      | Yes [4, 7]                                       | Yes (Model Packages within Model Package Groups) [15, 17]                                                      | Yes [11, 22]                                                         |
| Approval workflow               | Yes (via metadata) [4]                           | Yes — `Approved` / `PendingManualApproval` / `Rejected` states; can be automated by metric thresholds [15, 16] | Yes (via tags + Event Grid) [22]                                     |
| Lineage to training run         | Vertex ML Metadata [6]                           | Auto-linked to SageMaker Pipeline execution [16, 17]                                                           | Auto-linked to Azure ML job [22]                                     |
| Cross-account / cross-workspace | Via shared Cloud Storage [6]                     | Via AWS RAM [17]                                                                                               | Native cross-workspace registries [11, 23]                           |
| Foundation-model integration    | Model Garden (Gemini, Claude, third-party) [33]  | SageMaker JumpStart can register foundation models in the same registry [17]                                   | Model catalog (OpenAI, Hugging Face, Meta, Cohere, Mistral) [24]     |

---

## 5. Feature Store

| Aspect                       | Vertex AI Feature Store                                               | SageMaker Feature Store                                                                                                                          | Azure ML Managed Feature Store                         |
|------------------------------|-----------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| Online store backend         | Bigtable-class store; offline operations follow BigQuery pricing [38] | DynamoDB-backed Standard tier; in-memory tier for sub-10ms reads (max 50 GiB) [29, 31]                                                           | Redis (online); ADLS Gen2 (offline) [26]               |
| Online + offline consistency | Yes [4]                                                               | Yes — synced automatically [20]                                                                                                                  | Yes — same feature pipeline serves both [26]           |
| Pricing model (online)       | Storage per GB plus reads/writes [38]                                 | Per-KB write request units, per-4KB read request units, plus per-GB-month storage; in-memory tier billed per GB-hour with 5 GiB minimum [30, 31] | Underlying Redis / Spark compute, no platform fee [27] |
| Hidden cost trap             | Offline operations are billed at BigQuery rates separately [38]       | Athena queries against the offline store are billed separately from Feature Store [32]                                                           | Spark materialization compute time                     |
| Point-in-time correctness    | Yes [4]                                                               | Yes (point-in-time queries on offline store) [20]                                                                                                | Yes (declarative training data generation) [26]        |
| Discoverability UI           | Feature catalog in Vertex AI console [4]                              | Feature catalog in SageMaker Studio [20]                                                                                                         | Feature set browser in AML studio [26]                 |

A practical note: SageMaker Feature Store can incur significant *secondary* costs because high-throughput reads pass
through DynamoDB and high-volume queries through Athena/Glue [13, 32]. Vertex's offline feature store similarly inherits
BigQuery pricing for offline ops [38]. Azure's managed feature store is the youngest of the three but is structurally
simpler because both backends are well-known commodity services [26].

---

## 6. Model Monitoring

All three detect data drift, prediction drift, and feature attribution drift. The differences are mostly in the billing
model and the alerting integrations.

| Capability           | Vertex AI Model Monitoring                                                     | SageMaker Model Monitor                                                                               | Azure ML Monitoring                                                                         |
|----------------------|--------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Drift types          | Skew (training vs serving) and drift (over time) [4, 6]                        | Data quality, model quality, bias drift, feature attribution drift (paired with Clarify) [13, 14, 19] | Data drift, prediction drift, model performance [11, 22]                                    |
| Pricing              | $3.50 per GB of data analyzed (training and prediction logs combined) [37, 38] | Per-instance-hour for the monitoring schedule; one ml.m5.xlarge job typically costs <$1/day [21]      | Underlying compute only (no per-GB fee) [27, 28]                                            |
| Alerting             | Cloud Monitoring alerts [6]                                                    | CloudWatch alarms + EventBridge [14]                                                                  | Azure Monitor alerts + Event Grid (for lifecycle events including drift detection) [11, 22] |
| Triggers retraining? | Yes, via Cloud Functions / Pub/Sub from alerts [6]                             | Yes, via EventBridge → SageMaker Pipelines [14]                                                       | Yes, via Event Grid → Azure Pipelines or AML jobs [11]                                      |

Vertex AI's per-GB pricing is the easiest to estimate up-front but punishes high-cardinality logging; SageMaker's
per-instance-hour pricing rewards short, scheduled monitoring runs but can become expensive if you keep monitoring jobs
always-on; Azure's compute-only model is the simplest but pushes the complexity onto you to size the compute
right [21, 27, 37].

---

## 7. Serving / Deployment Targets

| Target type                 | Vertex AI                                                                         | SageMaker                                                                                                 | Azure ML                                                                               |
|-----------------------------|-----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| Real-time HTTPS endpoint    | Online Prediction endpoint (per-node-hour billing; **no scale-to-zero**) [38, 39] | Real-time endpoint (per-instance-hour billing; **no scale-to-zero** for traditional endpoints) [13, 21]   | Managed online endpoint (autoscale via Azure Monitor; managed VM-backed) [27, 28]      |
| Serverless inference        | Vertex AI auto-scaling endpoints (still requires ≥1 node) [38]                    | SageMaker Serverless Inference (memory-MB per second + per-GB data in/out — actually scales to zero) [29] | Not a directly equivalent first-party offering; ACI virtual nodes can approximate [40] |
| Batch                       | Batch Prediction job [4]                                                          | Batch Transform [19]                                                                                      | Batch endpoints [25]                                                                   |
| Multi-model on one endpoint | Yes (co-hosting models) [38]                                                      | Yes (Multi-Model Endpoints) [19]                                                                          | Yes (deployments under one endpoint with traffic splitting) [22]                       |
| GPU serving                 | Per-GPU node-hour (e.g., NVIDIA L4 endpoint roughly $800/month at 24/7 [36])      | Per-GPU instance-hour [21]                                                                                | Per-GPU VM-hour, ACI or AKS-backed [27, 28]                                            |

The single most expensive MLOps mistake on all three platforms is forgetting to undeploy idle endpoints — this is
documented as a primary source of "billing shock" specifically on SageMaker and Vertex AI, where dedicated endpoints
continue to charge regardless of traffic [13, 36, 39, 40]. Azure ML has the same property; the cost-management docs
explicitly recommend autoscaling and setting minimum nodes to 0 on training clusters (though not endpoints) [27, 28].

---

## 8. CI/CD and DevOps Integration

| Element                               | Vertex AI                                                            | SageMaker                                                                                    | Azure ML                                                                                    |
|---------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Native CI/CD service used             | Cloud Build, Cloud Deploy [6, 8]                                     | CodePipeline + CodeBuild + CodeCommit, or third-party (GitHub Actions, Jenkins) [12, 13, 14] | Azure Pipelines (Azure DevOps) or GitHub Actions [9, 10]                                    |
| First-party reference architecture    | "MLOps with Vertex AI" GitHub samples and TFX-on-GCP guide [3, 6, 8] | "MLOps with SageMaker, GitHub, GitHub Actions" reference architecture [14]                   | MLOps v2 solution accelerator [9, 10, 25]                                                   |
| Component reuse across environments   | Pipeline definitions in Artifact Registry [6]                        | SageMaker Projects templates published via AWS Service Catalog [16, 17]                      | Cross-workspace registries — same component runs in dev/test/prod with one command [11, 23] |
| Source of truth for model approval    | Model Registry approval state [4]                                    | Model Registry approval state, often gated by CodePipeline [16]                              | Tag + Event Grid event triggers downstream pipeline [22]                                    |
| Multi-account / multi-project pattern | Multi-project setup with admin / dev / test / prod separation [8]    | Multi-account with central Model Registry account [12, 14]                                   | Multi-workspace, single subscription, MLOps v2 reference [9]                                |

Azure ML and SageMaker both have *opinionated*, vendor-published reference architectures (MLOps v2 [9] and SageMaker
Projects [14, 16]); Vertex AI's reference is more sample-code than turnkey scaffolding [6, 8]. For organizations that
want a shrink-wrapped CI/CD-for-ML setup, Azure's MLOps v2 accelerator is the most batteries-included starting
point [9, 10].

---

## 9. Generative AI / Foundation Model Integration

This has rapidly become a first-class MLOps concern in 2025–2026. All three integrate foundation-model deployment into
the same registry/endpoint primitives used for traditional ML, but the catalogs differ substantially.

| Aspect                                     | Vertex AI                                                                            | SageMaker                                                                                     | Azure ML                                                                   |
|--------------------------------------------|--------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| First-party foundation models              | Gemini family (2.5 Pro, 2.5 Flash, 2.0 Flash, Imagen, Veo) [33, 34]                  | Amazon Nova family + AWS partner models via Bedrock                                           | OpenAI models via Azure OpenAI Service (GPT family, embedding models) [24] |
| Third-party catalog                        | Model Garden — 200+ models including Anthropic Claude, Meta Llama, Hugging Face [33] | SageMaker JumpStart — registers foundation models inside the customer's private registry [17] | Model catalog — OpenAI, Hugging Face, Meta, Cohere, Mistral [24]           |
| Token-based pricing alongside infra        | Yes, for Gemini and partner models [33, 34]                                          | Yes (Bedrock-backed)                                                                          | Yes (Azure OpenAI)                                                         |
| Fine-tuning workflow inside MLOps platform | Yes, supervised fine-tuning + adapter (LoRA) + RLHF in Vertex AI [33]                | Yes, JumpStart fine-tuning into Model Registry [17]                                           | Yes, fine-tune via Azure ML jobs; result lands in registry [22]            |

For traditional MLOps (your own custom-trained models), the three are roughly equivalent. For GenAI-native MLOps, *
*Vertex AI** has the broadest first-party + partner model selection, **Azure ML** has the deepest OpenAI integration,
and **SageMaker** depends heavily on Bedrock as the foundation-model layer underneath [17, 24, 33].

---

## 10. Pricing Comparison Summary

The platforms have converged on a "you pay for the resources, not the platform" model, with a few notable
exceptions [27, 28, 30, 37].

| Cost line item                                               | Vertex AI                                                               | SageMaker                                                                 | Azure ML                                      |
|--------------------------------------------------------------|-------------------------------------------------------------------------|---------------------------------------------------------------------------|-----------------------------------------------|
| Platform fee (just to use it)                                | None                                                                    | None                                                                      | None [27, 28]                                 |
| Pipeline orchestration fee                                   | **$0.03 per run** [37]                                                  | None                                                                      | None [27]                                     |
| Model Registry storage                                       | Free [38]                                                               | Free                                                                      | Free (Blob Storage charges apply) [27]        |
| Feature Store online (per ~10M reads/month, simple workload) | Reads/writes + per-GB-month storage [38]                                | $0.0285 per million reads (4KB units) + storage [30]                      | Underlying Redis cost only [26, 27]           |
| Model Monitoring                                             | $3.50/GB analyzed [37, 38]                                              | Per-instance-hour (~$0.92/hr for ml.m5.xlarge) [21]                       | Underlying compute only [27]                  |
| Real-time endpoint (1× CPU node, 24/7)                       | Roughly $130–$250/month for n1-standard-4 class [35]                    | Comparable for ml.m5.xlarge (~$165/month at $0.23/hr × 730) [21]          | Comparable for Standard_DS3_v2 class [27, 28] |
| Real-time endpoint (1× NVIDIA L4 GPU, 24/7)                  | ~$800/month [36]                                                        | Comparable on g6/g5 instances [21]                                        | Comparable on NCas_T4_v3 / NC L4 v5 [27]      |
| Endpoint scales to zero by default?                          | **No** [38, 39]                                                         | **No** for traditional endpoints; Yes for Serverless Inference [29]       | **No** for managed online endpoints [27, 28]  |
| Free credit / trial                                          | $300 GCP credits + Vertex free tier (5 GB online prediction/month) [35] | 2-month SageMaker free tier (50 hr training, 125 hr inference, etc.) [34] | Underlying free Azure tier [27]               |

---

## 11. Lock-in and Portability

Worth being explicit about because it affects which platform you'd choose for a long-lived ML platform investment.

- **Vertex AI** uses open-source Kubeflow Pipelines and TFX SDKs natively, so pipeline definitions are theoretically
  portable to any KFP-compliant runtime (including on-prem or other clouds running Kubeflow) [3, 6]. This is the most
  portable of the three at the orchestration layer.
- **SageMaker** uses a proprietary pipeline SDK; migrating SageMaker Pipelines to Airflow or Argo typically requires a
  complete rewrite [13]. Models trained inside SageMaker are often wrapped in SageMaker-specific container formats,
  making cross-cloud moves harder [13]. The lock-in is the most pronounced of the three.
- **Azure ML** uses a YAML-defined component model that is Azure-specific but has good MLflow interoperability — MLflow
  experiments, models, and registry calls work natively against Azure ML [22, 24]. This makes Azure ML somewhat portable
  for MLflow-centric organizations.

---

## 12. Decision Heuristic

Treat this as a tie-breaker, not a rule:

- **You're already deep in BigQuery, want TPUs, or are building heavily on Gemini / Anthropic models** → Vertex
  AI [27, 33].
- **You're already in AWS, your data is in S3, and your platform team prefers tightly integrated managed services** →
  SageMaker AI [13, 19].
- **You're a Microsoft enterprise shop, use Azure DevOps or GitHub Actions for CI/CD, and value cross-workspace
  registries for dev → test → prod promotion** → Azure ML [9, 11, 23].
- **You want maximum portability of your ML pipelines themselves** → Vertex AI (KFP-native) [3, 6].
- **You want the cheapest entry point for "register a model and serve it" with no platform fee** → Azure ML, which adds
  nothing to underlying compute charges [27, 28].
- **You want the most opinionated, batteries-included CI/CD-for-ML reference architecture** → Azure ML's MLOps v2
  accelerator [9, 10].

---

## Sources

1. MLOps on Vertex AI — Google Cloud
   Documentation — https://docs.cloud.google.com/vertex-ai/docs/start/introduction-mlops
2. Introduction to Vertex AI Pipelines — Google Cloud
   Documentation — https://docs.cloud.google.com/vertex-ai/docs/pipelines/introduction
3. How to get started with Vertex AI & MLOps —
   Recordly — https://www.recordlydata.com/blog/how-to-get-started-with-vertex-ai-mlops
4. Vertex AI Platform overview — Google Cloud — https://cloud.google.com/vertex-ai
5. Building a Simple End-to-End MLOps Pipeline with Vertex AI — Medium / Ken
   Maeda — https://medium.com/@whiteking64/building-a-simple-end-to-end-mlops-pipeline-with-google-vertex-ai-2317ac77cb82
6. mlops-with-vertex-ai (end-to-end TFX example) — GitHub /
   GoogleCloudPlatform — https://github.com/GoogleCloudPlatform/mlops-with-vertex-ai
7. Architecture for MLOps using TFX, Vertex AI Pipelines, and Cloud Build — Google Cloud Architecture
   Center — https://docs.cloud.google.com/architecture/architecture-for-mlops-using-tfx-kubeflow-pipelines-and-cloud-build
8. Production-Ready MLOps on GCP Part 1: Architecture Overview — Medium / Saoussen
   Chaabnia — https://medium.com/google-cloud/production-ready-mlops-on-gcp-part-1-architecture-overview-8b2294c41e1d
9. Set up MLOps with Azure DevOps — Microsoft
   Learn — https://learn.microsoft.com/en-us/azure/machine-learning/how-to-setup-mlops-azureml
10. MLOps machine learning model management v1 — Microsoft
    Learn — https://learn.microsoft.com/lb-lu/azure/machine-learning/concept-model-management-and-deployment?view=azureml-api-1
11. MLOps: Enabling the Enterprise With Azure AI —
    Lingaro — https://lingarogroup.com/blog/mlops-enabling-the-enterprise-with-azure-ai
12. Promote pipelines using SageMaker Model Registry, Terraform, GitHub, Jenkins —
    AWS — https://aws.amazon.com/blogs/machine-learning/promote-pipelines-in-a-multi-environment-setup-using-amazon-sagemaker-model-registry-hashicorp-terraform-github-and-jenkins-ci-cd/
13. Amazon SageMaker Review 2026: Features, Pricing, Pros & Cons —
    TrueFoundry — https://www.truefoundry.com/blog/amazon-sagemaker-review-features-pricing-pros-and-cons-better-alternative
14. Build an end-to-end MLOps pipeline using SageMaker Pipelines, GitHub, and GitHub Actions —
    AWS — https://aws.amazon.com/blogs/machine-learning/build-an-end-to-end-mlops-pipeline-using-amazon-sagemaker-pipelines-github-and-github-actions/
15. How to Use SageMaker Model Registry —
    OneUptime — https://oneuptime.com/blog/post/2026-02-12-sagemaker-model-registry/view
16. Automating ML Workflows with SageMaker Pipelines and Model Registry — Medium / W
    Shamim — https://medium.com/@Shamimw/automating-ml-workflows-with-amazon-sagemaker-pipelines-and-model-registry-24ad89805d88
17. Amazon SageMaker Model Registry Cheat Sheet — Tutorials
    Dojo — https://tutorialsdojo.com/amazon-sagemaker-model-registry-cheat-sheet/
18. Building End-To-End MLOps on AWS — Caylent — https://caylent.com/blog/building-end-to-end-mlops-on-aws
19. New – Amazon SageMaker Pipelines (announcement) —
    AWS — https://aws.amazon.com/blogs/aws/amazon-sagemaker-pipelines-brings-devops-to-machine-learning-projects/
20. Amazon SageMaker Feature Store — AWS — https://aws.amazon.com/sagemaker/ai/feature-store/
21. amazon-sagemaker-mlops-with-featurestore-and-datawrangler (cost notes) — GitHub /
    aws-samples — https://github.com/aws-samples/amazon-sagemaker-mlops-with-featurestore-and-datawrangler
22. MLOps machine learning model management — Microsoft
    Learn — https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-management-and-deployment?view=azureml-api-2
23. Share models, components, and environments across workspaces with registries — Microsoft
    Learn — https://learn.microsoft.com/en-us/azure/machine-learning/how-to-share-models-pipelines-across-workspaces-with-registries?view=azureml-api-2
24. Azure Machine Learning product page — Microsoft Azure — https://azure.microsoft.com/en-us/products/machine-learning
25. Set up MLOps with Azure DevOps (v2 components) — Microsoft
    Learn — https://learn.microsoft.com/en-us/azure/machine-learning/how-to-setup-mlops-azureml
26. What is managed feature store? — Microsoft
    Learn — https://learn.microsoft.com/en-us/azure/machine-learning/concept-what-is-managed-feature-store
27. Plan to manage costs — Azure Machine Learning — Microsoft
    Learn — https://learn.microsoft.com/en-us/azure/machine-learning/concept-plan-manage-cost
28. Manage and optimize costs — Azure Machine Learning — Microsoft
    Learn — https://docs.azure.cn/en-us/machine-learning/how-to-manage-optimize-cost
29. Online store — Amazon SageMaker AI — AWS
    Docs — https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-storage-configurations-online-store.html
30. Amazon SageMaker AI Pricing — AWS — https://aws.amazon.com/sagemaker/ai/pricing/
31. Create, store, and share features with Feature Store — AWS
    Docs — https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html
32. SageMaker Feature Store pricing (Athena query interactions) — AWS re:
    Post — https://repost.aws/questions/QUyHSe39RJQmybHZSZDufrtw/sagemaker-feature-store-pricing
33. Vertex AI Pricing Review — Lindy — https://www.lindy.ai/blog/vertex-ai-pricing
34. Amazon SageMaker Pricing 2026 — TrustRadius — https://www.trustradius.com/products/amazon-sagemaker/pricing
35. What Is GCP Vertex AI? — Leanware — https://www.leanware.co/insights/what-is-gcp-vertex-ai
36. AI Cost Tracking on GCP (Vertex AI endpoint cost example) — Medium / Matias
    Coca — https://medium.com/@cocamatias/ai-cost-tracking-on-gcp-a-practical-guide-to-vertex-ai-gemini-api-and-model-spend-63b3c87d8ee4
37. Vertex AI: Pricing for Top 16 Vertex Services in 2026 —
    Finout — https://www.finout.io/blog/top-16-vertex-services-in-2026
38. Vertex AI pricing (Google Cloud) — https://cloud.google.com/vertex-ai/pricing
39. Vertex AI Pricing Complete 2026 Guide — nOps — https://www.nops.io/blog/vertex-ai-pricing/
40. Managing Your Azure Machine Learning Costs — Matt on ML.NET — https://accessibleai.dev/post/azure_ml_pricing_tips/