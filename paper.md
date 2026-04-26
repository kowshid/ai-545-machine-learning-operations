# Managed Kubernetes Comparison: Amazon EKS vs Google GKE vs Azure AKS

*Report compiled April 26, 2026. Pricing and SLA values are list rates from official provider pages and reflect
US-region defaults; actual figures vary by region, commitment, and configuration.*

> **How to read the citations:** every specific factual claim is followed by `[N]` where `N` is the source number in
> the [Sources](#sources) list at the end. Multiple sources are separated by commas — e.g. `[1, 8]`. Statements without a
> citation are editorial synthesis or definitions that don't attach to a single source.

## A Note on the Requested Metrics

The metrics requested fall into three categories, and being upfront about this is important for the report to be useful:

1. **Platform-defined values** — pricing, SLAs, encryption standards, max cluster size, accelerator availability. These
   have authoritative answers and are tabulated below.
2. **Workload-dependent values** — request latency (p50/p95/p99), throughput, network bandwidth, IOPS, max throughput
   under load, model deployment latency, geographic latency variance. These depend on instance type, region, network
   topology, application code, and load profile. There is no single "EKS latency" number; what exists are the *ceilings
   and configuration knobs* the platform exposes.
3. **Organizational (DORA) metrics** — Deployment Frequency, Change Failure Rate, Rollback Rate, MTTR, MTBF, Apdex
   Score, Storage Backup Success Rate. These describe **how a team operates** on a platform, not the platform itself.
   The platforms are described in terms of what they offer to support these practices, not invented values.

---

## 1. Executive Summary

All three are CNCF-conformant managed Kubernetes services, and on equivalent hardware their headline compute prices are
now within a few percent of each other [10, 13]. The meaningful differences sit at the edges:

- **AKS** is structurally cheapest for many small clusters because the control plane is free on the Free tier [9, 10];
  the trade-off is a 99.5% SLO (not a financially backed SLA) unless you upgrade to the Standard tier [7].
- **GKE** is regarded as the closest to upstream Kubernetes and ships new versions fastest [12], with the strongest
  native AI/ML story (TPUs, Autopilot, A3 H100/H200 availability) [21, 22].
- **EKS** has the deepest enterprise AWS integration and the largest third-party tool ecosystem [11], and — with
  Karpenter — the most mature node autoscaler [24, 25, 44]. Its control plane fee applies to every cluster, which
  compounds in fleets [10].

## 2. Quick Comparison Table

| Dimension                                  | Amazon EKS                                                                                                  | Google GKE                                                                                            | Azure AKS                                                                                                             |
|--------------------------------------------|-------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| Control plane fee                          | \$0.10/hr per cluster (standard support); $0.60/hr after 14 months in extended support [1]                  | \$0.10/hr per cluster after a \$74.40/month free credit (covers ~one zonal/Autopilot cluster) [3, 16] | Free (Free tier); \$0.10/hr (Standard tier with Uptime SLA) [7, 9]                                                    |
| Highest financially-backed SLA             | 99.95% (Standard control plane); higher tiers via Provisioned Control Plane [1, 2]                          | 99.95% (regional Standard, Autopilot); 99.5% (zonal Standard) [4]                                     | 99.95% (Standard tier + Availability Zones); 99.9% (Standard tier without AZs); 99.5% SLO only on Free tier [5, 6, 7] |
| Native autoscaler                          | Karpenter (production-grade) + Cluster Autoscaler [24, 25]                                                  | Cluster Autoscaler + Node Auto-Provisioning + ComputeClasses (Karpenter not native) [27]              | Karpenter for AKS (GA early 2026) + Cluster Autoscaler [27]                                                           |
| K8s version cadence after upstream release | ~4–8 weeks                                                                                                  | ~2 weeks (fastest)                                                                                    | ~3–6 weeks                                                                                                            | (all three values: [12]) |
| Max nodes per cluster                      | Up to 13,500 with Provisioned Control Plane scaling tiers [14]                                              | 15,000 (Standard); 65,000 in v1.31+ for AI workloads [41]                                             | 5,000 (with quota increase) [42]                                                                                      |
| Native managed K8s-resource backup         | None — AWS Backup does not cover EKS Kubernetes resources; Velero is the de facto third-party tool [37, 38] | Velero widely used; backups commonly via CSI snapshots [46, 47]                                       | Velero widely used; backups commonly via CSI snapshots [45, 46]                                                       |

---

## 3. Performance Metrics (Latency, Throughput, Bandwidth, IOPS)

The honest answer for this section is that **none of these are properties of EKS, GKE, or AKS as services** — they are
properties of the EC2/Compute Engine/Azure VM you choose, the disk you attach, the region you run in, and the
application you deploy. Published third-party benchmarks vary by an order of magnitude depending on those choices. The
relevant question is what each platform exposes as a ceiling.

### Latency (p50 / p95 / p99)

No vendor publishes managed-Kubernetes request-latency percentiles, because the Kubernetes service does not own the
request path of your application. Industry comparisons that do exist measure *control-plane API latency* and
*pod-scheduling latency*. One often-cited observation is that GKE's network policy implementation shows roughly 15–20%
lower latency than AWS VPC CNI in like-for-like microbenchmarks [11], but this is configuration-sensitive and not a
guarantee. For application latency, the dominant variables are instance generation, placement strategy, service mesh
choice, and load-balancer type — none of which are differentiated by the managed-Kubernetes layer itself.

### Throughput (requests/sec) and Max Throughput Under Load

Same caveat as latency. What can be compared is **cluster-level scaling ceilings**:

- **EKS** with Provisioned Control Plane offers explicit scaling tiers up to 13,500 nodes per cluster, with
  control-plane capacity priced and provisioned in advance for predictable burst handling [1, 14].
- **GKE** Standard supports up to 15,000 nodes per cluster, and up to 65,000 nodes in version 1.31+ for large-scale AI
  workloads [41].
- **AKS** is documented at up to 5,000 nodes per cluster with a quota increase, lower than the other two [42].

### Network Bandwidth

Driven entirely by VM choice. For top-end AI nodes in April 2026, all three vendors have converged on roughly 3,200 Gbps
fabric (AWS EFA on P5en, Google's RDMA on A3 Ultra [22], Azure InfiniBand on ND-H200 v5). For general-purpose pods,
bandwidth is whatever the chosen VM SKU provides.

### IOPS (storage performance)

This is a property of the **block storage service**, not the Kubernetes service. The CSI driver simply exposes the
underlying disk. Headline numbers:

| Storage tier                    | Max IOPS per volume                                                    | Max throughput per volume | Source       |
|---------------------------------|------------------------------------------------------------------------|---------------------------|--------------|
| AWS gp3 (general-purpose SSD)   | 16,000                                                                 | 1,000 MiB/s               | [28, 29, 30] |
| AWS io2 Block Express (premium) | 256,000+ (sub-ms latency)                                              | 4,000 MiB/s               | [28, 29]     |
| Azure Premium SSD v2            | 80,000                                                                 | 1,200 MiB/s               | [26]         |
| Google Persistent Disk SSD      | 100,000 read / 80,000 write                                            | varies by size            | [31]         |
| Google Hyperdisk                | IOPS/throughput provisioned independently of capacity (gp3-like model) | —                         | [26]         |

GCP's persistent disks have historically advertised the highest per-volume read IOPS ceilings; AWS io2 Block Express is
the lowest-latency option (sub-millisecond) [29, 48]. For most workloads the differences are immaterial —
over-provisioning is what drives cost on all three platforms [26].

---

## 4. Availability and Reliability

### Uptime / Availability SLA (control plane)

| Configuration                      | EKS                                                                | GKE                                                                              | AKS                                                |
|------------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------|----------------------------------------------------|
| Highest available SLA              | 99.95% (Standard); higher with Provisioned Control Plane tiers [2] | 99.95% (regional Standard or Autopilot, multi-zone Autopilot pods get 99.9%) [4] | 99.95% (Standard tier + Availability Zones) [5, 6] |
| Lowest tier                        | n/a — single SLA [2]                                               | 99.5% (zonal Standard) [4]                                                       | 99.5% SLO (Free tier — not financially backed) [7] |
| Financially backed at lowest tier? | Yes [2]                                                            | Yes [4]                                                                          | **No** — Free tier is SLO only [7]                 |

Microsoft documents the AKS Standard-tier SLA as 99.95% with Availability Zones and 99.9% without AZs [5, 6] — both
numbers come directly from the SLA text.

### MTTR (Mean Time To Recovery) and MTBF (Mean Time Between Failures)

These are operational metrics owned by the team running the workload, not values published by the cloud providers. The
platform features that *influence* them:

- **EKS**: AWS Auto-Recovery on EC2, managed node group rolling updates, Karpenter consolidation/replacement (~45–60 sec
  to bring a replacement node online in production tests [44]), AWS Health API integration.
- **GKE**: Auto-repair and auto-upgrade are on by default; node auto-provisioning replaces failed nodes; the regional
  control plane survives single-zone outage [11].
- **AKS**: Built-in node auto-repair runs by default — this is one of AKS's quieter advantages over EKS [14].

### Error Rate

A property of the application, not the platform. The platforms expose the building blocks (probes, PDBs, multi-AZ
scheduling) equally.

---

## 5. Auto-Scaling and Provisioning

### Auto-Scaling Response Time and Resource Provisioning Time

| Platform | Autoscaler                                  | Typical scale-up time                                                                                                                                   | Source |
|----------|---------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|--------|
| EKS      | Karpenter (recommended)                     | ~45–60 seconds to bring a node online in production tests; bypasses ASGs and calls EC2 APIs directly                                                    | [44]   |
| EKS      | Cluster Autoscaler (legacy)                 | 3–4 minutes typical (10-second scan cycle + ASG provisioning)                                                                                           | [44]   |
| GKE      | Cluster Autoscaler + Node Auto-Provisioning | Generally faster cluster *creation* than EKS (3–5 minutes vs 10–15) per third-party reports; node scale-up similar to or slightly faster than CA on EKS | [11]   |
| AKS      | Karpenter for AKS (GA early 2026)           | Comparable to EKS Karpenter (~45–60 seconds), now the recommended autoscaler                                                                            | [27]   |
| AKS      | Cluster Autoscaler                          | Similar to other CA implementations                                                                                                                     | [44]   |

For raw cluster startup time, GKE has historically been the fastest [11]. For ongoing node-level scaling under load,
Karpenter on EKS and AKS now wins by a wide margin over Cluster Autoscaler [44].

---

## 6. Cost

### Cost per Compute Unit (CPU-hour, GB-hour) — Serverless modes

For an apples-to-apples serverless comparison, an Oracle benchmark using a 16-vCPU/64-GiB pod configuration found [13]:

| Service                                           | vCPU rate        | Memory rate (GiB) |
|---------------------------------------------------|------------------|-------------------|
| AWS Fargate (with EKS)                            | $0.04048/vCPU-hr | $0.004445/GiB-hr  |
| AKS Virtual Nodes (via Azure Container Instances) | $0.0405/vCPU-hr  | $0.00445/GiB-hr   |
| GKE Autopilot                                     | $0.0445/vCPU-hr  | $0.0049225/GiB-hr |

For an identical 20-pod, 24/7 workload, monthly costs were within ~10% of each other across the three:
roughly $13,945 (EKS+Fargate), $13,954 (AKS Virtual Nodes), and $15,282 (GKE Autopilot) [13]. For Standard mode (you
manage nodes), on-demand compute pricing is essentially identical across all three — by design [34].

### Cost per Cluster (control plane) — the cleanest differentiator

- **EKS**: $0.10/hr × 730 hr ≈ **$73/cluster/month** for every cluster, no exceptions [1, 9]; rises to $0.60/hr in
  extended support [1].
- **GKE**: $0.10/hr per cluster, but a $74.40/month free credit covers approximately one zonal or Autopilot cluster per
  billing account [3, 16].
- **AKS**: **$0** on the Free tier (no SLA); **$73/cluster/month** ($0.10/hr) on the Standard tier if you want the
  financially-backed Uptime SLA [7, 9].

For a 10-team org with 4 clusters each (40 total), EKS and GKE
charge $2,920/month in control-plane fees while AKS Free can charge $0 — a structural advantage for fleet-heavy
environments [10].

### Storage Cost per GB (block storage, US East baseline)

- AWS gp3: $0.08/GiB-month + $0.005/provisioned IOPS-month above 3,000 free [26, 30]
- AWS io2: $0.125/GiB-month + $0.065/provisioned IOPS-month [26, 29]
- Azure Premium SSD v2: independently provisioned IOPS/throughput, comparable to gp3 model [26]
- GCP Persistent Disk Standard: $0.17/GiB-month [8]
- GCP Persistent Disk SSD: $0.32/GiB-month [8]
- GCP Hyperdisk: priced like gp3 (independent capacity/IOPS/throughput dimensions) [26]

GCP's standard PD pricing is the highest per-GiB; Hyperdisk closes the gap for performance-sensitive workloads [26].

### Data Egress / Data Transfer Cost

| Provider     | Internet egress (first tier, US) | Cross-region (within US) | Free monthly egress |
|--------------|----------------------------------|--------------------------|---------------------|
| AWS          | $0.09/GB (first 10 TB) [33, 34]  | $0.02/GB [34]            | 100 GB [40]         |
| Azure        | $0.087/GB (first 5 TB) [33]      | $0.02/GB [34]            | 100 GB [40]         |
| Google Cloud | $0.12/GB (first 1 TB) [33, 34]   | $0.01/GB [34]            | 100 GB [40]         |

Notes that matter for Kubernetes workloads:

- **AWS** charges $0.01/GB for cross-AZ traffic, which adds up for service meshes spread across AZs [10].
- **Azure** offers free inter-AZ traffic — a real advantage for highly-zoned AKS deployments [10].
- **Google** has the highest per-GB egress to the internet, but the cheapest cross-region transfer ($0.01/GB) and free
  same-region egress from Cloud Storage to Compute Engine — a meaningful win for read-heavy ML training pipelines [34].

### Idle Resource Cost

A property of how clusters are used, not of the platforms. Published comparisons show ~67% reduction in cluster cost by
running 8 hours × weekdays vs 24/7, applied equally across all three providers [10]. The structural difference is that
AKS Free incurs no control-plane charge while idle, whereas EKS and GKE keep charging $0.10/hr regardless [10].

### Total Cost of Ownership

For equivalent production workloads (3-node cluster, 24/7), reported real-world TCO is within ~1% across the three:
roughly $1,382/month on EKS, $1,397/month on GKE, $1,382/month on AKS [10]. **TCO differences therefore come from
non-compute factors**: control-plane fleet fees (favors AKS [10]), egress patterns (depends on architecture),
commitment-discount flexibility (GCP CUDs are most flexible [34]), and operational toil. For most organizations,
existing cloud commitments and team familiarity dominate the TCO calculation.

### Cost per Request / Cost per Transaction

A function of your application's compute and bandwidth per request. The platforms do not differ at the per-request
layer — they differ at the per-vCPU-hour and per-GB-egress layer documented above [33, 34].

---

## 7. DevOps Process Metrics (Deployment Frequency, Change Failure Rate, Rollback Rate)

These are the four DORA metrics — **organizational metrics, not platform metrics**. Whether your team deploys 1× or 100×
per day, and what your change failure and rollback rates are, depends on your CI/CD pipeline, test coverage, and
deployment strategy. The platforms differ in the *ergonomics* they offer:

- **EKS** has the deepest third-party CI/CD ecosystem (Argo CD, Flux, Spinnaker, GitHub Actions); over 400 community
  Helm charts target AWS specifically [11].
- **GKE** has the tightest first-party CI/CD integration through Cloud Build, Cloud Deploy, and Artifact Registry [11].
- **AKS** has the strongest Microsoft-stack DevOps integration through Azure DevOps and GitHub Actions [11].

All three support standard rollback patterns (blue/green, canary, progressive delivery) equally well through Argo
Rollouts, Flagger, or native Deployment strategies.

---

## 8. Storage: Backup, Restore, Replication, Consistency

### Storage Backup Success Rate, Restore Time, Replication Lag

These are operational metrics. The platforms offer:

| Capability                        | EKS                                                                      | GKE                                                    | AKS                                                 |
|-----------------------------------|--------------------------------------------------------------------------|--------------------------------------------------------|-----------------------------------------------------|
| First-party managed K8s backup    | None — AWS Backup covers EBS volumes only, not Kubernetes resources [37] | Snapshot-based via CSI; Velero widely used [46]        | Snapshot-based via CSI; Velero widely used [45, 46] |
| Volume snapshots                  | EBS snapshots via CSI snapshotter [36, 38]                               | Persistent Disk snapshots via CSI [46]                 | Managed Disk snapshots via CSI [45]                 |
| Cross-region replication (block)  | EBS snapshot copy [36]                                                   | PD snapshot copy [47]                                  | Managed disk copy [45]                              |
| Consistency model (block storage) | Strongly consistent at volume level (single-attach default)              | Strongly consistent at volume level                    | Strongly consistent at volume level                 |
| Multi-attach support (block)      | Yes (io1/io2) [28]                                                       | Yes (Hyperdisk; Regional PD for sync replication) [26] | Yes (Shared Disks) [31]                             |

In practice, with Velero, a published GKE-to-GKE backup of a 100 GB PV (51 GB used) took 7–8 seconds for backup and
under 1 minute for restore [47]; backup performance is generally bounded by snapshot service throughput rather than the
Kubernetes layer.

### Storage Replication Lag

For synchronous-replicated options (Regional PD, Azure ZRS disks, AWS Multi-AZ EFS): effectively zero
application-visible lag (synchronous writes). For async cross-region replication: lag is service-defined and typically
measured in seconds-to-minutes depending on the snapshot interval.

### Storage Consistency Model

All three default to strong consistency for block volumes (single-attach pattern) and read-after-write consistency for
their object stores (S3, GCS, Azure Blob). This has been the case across all three since 2020.

---

## 9. Geographic Latency Variance

Driven by region count and pricing-by-region, not by the Kubernetes service.

A practical pricing wrinkle: serverless Kubernetes (Fargate, AKS Virtual Nodes, GKE Autopilot) pricing varies by region
by 50–100% in extreme cases. AWS Fargate is reported as 72% more expensive in São Paulo vs N. Virginia, AKS Virtual
Nodes 100% more in Brazil South vs East US, and GKE Autopilot 59% more in São Paulo vs Iowa [13]. This is true across
all three, not unique to one.

---

## 10. SDK and CLI Maturity

| Platform | Primary CLI                                        | SDK ecosystem                                                                                                                                                   |
|----------|----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| EKS      | `aws` CLI + `eksctl` (de facto installer)          | The widest set of mature SDKs (Boto3 for Python, AWS SDKs for Java/Go/Node.js/etc.); largest ecosystem of community Helm charts targeting AWS specifically [11] |
| GKE      | `gcloud` CLI (sub-module) + `kubectl` plugins [15] | Google Cloud client libraries; deep Cloud Build integration [11]                                                                                                |
| AKS      | `az` CLI                                           | Azure SDKs in major languages; strong Visual Studio / GitHub Actions integration [11]                                                                           |

All three are mature and production-viable [11].

---

## 11. AI/ML and Accelerator Availability

This category has become a major differentiator since 2024 and is worth a careful look.

### Top-tier accelerator support (as of April 2026)

| Accelerator            | EKS                                                                                             | GKE                                                                            | AKS                |
|------------------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|--------------------|
| NVIDIA H200            | EC2 P5 family                                                                                   | A3 Ultra (capacity-constrained, requires reservation/Spot/Flex-start) [21, 22] | ND-H200 v5         |
| NVIDIA H100            | EC2 P5/P5e                                                                                      | A3 High / A3 Mega / A3 Edge [21, 22]                                           | ND-H100 v5         |
| NVIDIA A100 (40/80 GB) | EC2 P4d/P4de                                                                                    | A2 Standard / A2 Ultra [21, 22]                                                | ND-A100 v4         |
| NVIDIA L4              | EC2 G6                                                                                          | G2 (broad availability) [21, 22]                                               | NC L4 v5           |
| Custom accelerators    | AWS Trainium / Inferentia (exposed via Karpenter custom resources `aws.amazon.com/neuron`) [25] | Google TPU v5p, v5e, Trillium [21]                                             | None — NVIDIA-only |

Practical notes:

- **GKE** has the strongest AI story overall: TPUs are exclusive to Google [21], GKE Autopilot supports GPUs
  natively [32], and Multi-Instance GPU (MIG) is supported on A100/H100 [23]. GPU drivers are pre-installed on
  Container-Optimized OS nodes [35].
- **EKS** matches GKE on NVIDIA hardware; AWS's own Trainium/Inferentia silicon is exposed to Kubernetes via Karpenter
  custom resources [25]; GPU drivers are typically installed via the NVIDIA GPU Operator [35].
- **AKS** has full NVIDIA support including H200; it has no equivalent of TPUs or Trainium.

### Model Deployment Latency

Not a vendor-published metric. What can be compared is *how fast a GPU pod becomes ready*: this is dominated by
container-image pull time (mitigated equally by all three via image streaming/pre-pulling), driver-init time (similar
across platforms), and GPU node availability — where capacity constraints on H100/H200 are the real bottleneck on all
three providers in 2026 [22, 33].

---

## 12. Encryption Standards

All three support equivalent baseline encryption. The differences are in the certifications and the responsibility
model.

| Aspect                                | EKS                                                                                                                             | GKE                                                                                                                               | AKS                                                                                                                         |
|---------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Encryption at rest (etcd, default)    | EBS volume encryption with AWS-managed or customer-managed KMS keys [19, 39]                                                    | Default at-rest encryption with FIPS 140-2 validated BoringCrypto [17]                                                            | Azure Storage Service Encryption with platform or customer-managed keys [20]                                                |
| Envelope encryption for K8s secrets   | KMS provider plugin (AWS KMS) — recommended; uses HSM-backed CMKs [18, 19, 43]                                                  | Application-layer secrets encryption using Cloud KMS [17]                                                                         | Azure Key Vault provider for Secrets Store CSI Driver [20]                                                                  |
| FIPS 140-2 validated modules          | Yes for AWS KMS (HSM-backed) [43]; FIPS endpoints available; node images **not** FIPS by default — customer responsibility [20] | Yes by default at the storage layer (BoringCrypto); FedRAMP High authorized [17]; node-level FIPS is customer responsibility [20] | Yes for Key Vault/Managed HSM; FIPS endpoints available; node images **not** FIPS by default — customer responsibility [20] |
| Encryption in transit (intra-cluster) | Customer responsibility (typically a service mesh) [20]                                                                         | Automatic encryption inside Google's VPC [17]; in-cluster mTLS still customer's job for FIPS [20]                                 | Customer responsibility (Istio, Linkerd) [20]                                                                               |
| TLS standards                         | TLS 1.2 / 1.3 across all managed endpoints                                                                                      | TLS 1.2 / 1.3 across all managed endpoints                                                                                        | TLS 1.2 / 1.3 across all managed endpoints                                                                                  |
| Key algorithms                        | AES-256-GCM at rest; standard NIST suites for KMS [19]                                                                          | AES-256-GCM at rest; standard NIST suites; BoringCrypto module [17]                                                               | AES-256 at rest; standard NIST suites                                                                                       |

The honest summary: **for FIPS compliance, all three put intra-cluster encryption-in-transit on the customer**, and a
service mesh is the standard answer regardless of platform [20].

---

## 13. Apdex Score

Apdex is a calculated score (Satisfied + Tolerating/2, divided by total) based on a target latency that *the user
defines*. There is no Apdex score "for EKS" — it is a property of your monitored application. CloudWatch (AWS), Cloud
Monitoring (GCP), and Azure Monitor each support computing Apdex from custom metrics; third-party APMs (Datadog, New
Relic, Dynatrace) compute it the same way regardless of which Kubernetes you run on.

---

## 14. Decision Heuristic

For a quick sanity check on which platform fits a given workload:

- **You run many small clusters and don't need a control-plane SLA on all of them** → AKS Free has the strongest
  structural cost advantage [10].
- **You're investing in AI/ML, especially TPU-based or fastest-version-of-Kubernetes** → GKE is the cleanest
  fit [12, 21].
- **You're already deep in AWS for data services, IAM, networking, or you want Karpenter without third-party support** →
  EKS, despite the per-cluster fee [11, 24].
- **You're a Microsoft enterprise shop with Active Directory, .NET workloads, or Windows containers** → AKS, which has
  the best Windows container support [11].
- **You need the highest single-cluster scale ceiling** → GKE (15,000 nodes; 65,000 in v1.31+) [41] or EKS Provisioned
  Control Plane (13,500 nodes) [14] > AKS (5,000) [42].

---

## Sources

1. Amazon EKS Pricing — AWS — https://aws.amazon.com/eks/pricing/
2. Amazon EKS Service Level Agreement — AWS — https://aws.amazon.com/eks/sla/
3. Google Kubernetes Engine pricing — Google Cloud — https://cloud.google.com/kubernetes-engine/pricing
4. Google Kubernetes Engine Service Level Agreement (SLA) — Google
   Cloud — https://cloud.google.com/kubernetes-engine/sla
5. SLA for Azure Kubernetes Service (AKS) — Microsoft — https://www.azure.cn/en-us/support/sla/kubernetes-service/
6. Azure Kubernetes Service (AKS) with Uptime SLA — Microsoft Docs — https://docs.azure.cn/en-us/aks/uptime-sla
7. AKS Introduces Uptime SLA — Microsoft Community
   Hub — https://techcommunity.microsoft.com/t5/apps-on-azure-blog/aks-introduces-uptime-sla/ba-p/1350832
8. Kubernetes Pricing 2026: EKS vs AKS vs GKE Comparison Guide —
   Sedai — https://sedai.io/blog/kubernetes-cost-eks-vs-aks-vs-gke
9. EKS vs GKE vs AKS: Managed Kubernetes Comparison 2026 —
   Reintech — https://reintech.io/blog/eks-vs-gke-vs-aks-managed-kubernetes-comparison-2026
10. EKS vs GKE vs AKS: A FinOps Cost Comparison in 2026 — DEV
    Community — https://dev.to/muskan_8abedcc7e12/eks-vs-gke-vs-aks-a-finops-cost-comparison-in-2026-2m12
11. Amazon EKS vs Azure AKS vs Google GKE Platform Comparison 2026 —
    Index.dev — https://www.index.dev/skill-vs-skill/devops-amazon-eks-vs-azure-aks-vs-google-gke
12. EKS vs GKE vs AKS (2026): Pricing, Performance & Feature Comparison —
    Atmosly — https://atmosly.com/blog/eks-vs-gke-vs-aks-which-managed-kubernetes-is-best-2025
13. Serverless Kubernetes costs for EKS, AKS, GKE, and OKE —
    Oracle — https://blogs.oracle.com/cloud-infrastructure/serverless-kubernetes-costs-eks-aks-gke-oke
14. EKS vs. AKS: Choosing the Right Managed Kubernetes —
    Qovery — https://www.qovery.com/blog/managed-kubernetes-comparison-eks-vs-aks
15. Managed Kubernetes Comparison: EKS vs. GKE —
    Qovery — https://www.qovery.com/blog/managed-kubernetes-comparison-eks-vs-gke
16. GKE Pricing Explained — CloudChipr — https://cloudchipr.com/blog/gke-pricing
17. About FIPS-validated encryption in GKE — Google
    Cloud — https://docs.cloud.google.com/kubernetes-engine/docs/concepts/gke-fips-compliance
18. Data Encryption and Secrets Management — Amazon EKS Best
    Practices — https://aws.github.io/aws-eks-best-practices/security/docs/data/
19. Encryption best practices for Amazon EKS — AWS Prescriptive
    Guidance — https://docs.aws.amazon.com/prescriptive-guidance/latest/encryption-best-practices/eks.html
20. Container Encryption: FIPS Validated for EKS/GKE/AKS —
    Coalfire — https://coalfire.com/the-coalfire-blog/container-encryption-fips-validated-and-compliant-encryption-for-cloud-native-workloads
21. Accelerator (GPU and TPU) locations — Google
    Cloud — https://docs.cloud.google.com/compute/docs/regions-zones/accelerator-zones
22. GPU machine types — Google Cloud Compute Engine — https://docs.cloud.google.com/compute/docs/gpus
23. Multi Instance GPUs (MIG) in GKE — Medium / Jayesh
    Mahajan — https://medium.com/@jayeshmahajan/multi-instance-gpus-mig-in-gke-43674ce977a0
24. Introducing Karpenter —
    AWS — https://aws.amazon.com/blogs/aws/introducing-karpenter-an-open-source-high-performance-kubernetes-cluster-autoscaler/
25. Scale cluster compute with Karpenter and Cluster Autoscaler — AWS
    Docs — https://docs.aws.amazon.com/eks/latest/userguide/autoscaling.html
26. How to optimize block storage costs across AWS, Azure and GCP —
    Holori — https://holori.com/how-to-optimize-block-storage-costs-across-aws-azure-and-gcp/
27. Is There a Karpenter Equivalent on GKE? —
    daily.dev — https://app.daily.dev/posts/is-there-a-karpenter-equivalent-on-gke--9xeya0tyh
28. Explore all EBS volume types — Lucidity — https://www.lucidity.cloud/blog/ebs-volume-types
29. GP3 vs io1/io2: EBS Volumes (2026 Guide) — CloudFix — https://cloudfix.com/blog/aws-gp3-vs-io1-io2/
30. GP2 vs GP3: Pricing, Performance & Migration (2026) — CloudZero — https://www.cloudzero.com/blog/aws-gp2-vs-gp3/
31. AWS Storage vs Azure Storage vs GCP Storage —
    Lucidity — https://www.lucidity.cloud/blog/aws-storage-vs-azure-storage-vs-gcp-storage
32. Deploy GPU workloads in Autopilot — Google
    Cloud — https://docs.cloud.google.com/kubernetes-engine/docs/how-to/autopilot-gpus
33. Cloud Egress Costs (2026) — SpendArk — https://spendark.com/blog/cloud-egress-costs-guide/
34. AWS vs Azure vs GCP Pricing in 2026 —
    Zop.dev — https://zop.dev/resources/blogs/aws-vs-azure-vs-gcp-pricing-in-2026-compute-storage-and-network-compared
35. Enabling LLMs: GPUs on Kubernetes — TrueFoundry — https://www.truefoundry.com/blog/using-gpus-with-kubernetes
36. Backup and restore your Amazon EKS cluster resources using Velero —
    AWS — https://aws.amazon.com/blogs/containers/backup-and-restore-your-amazon-eks-cluster-resources-using-velero/
37. Native back-up support for EKS clusters — AWS re:
    Post — https://repost.aws/questions/QUZN5p9JdYTNeetOcxubvi6Q/native-back-up-support-for-eks-clusters
38. EKS Back Up: How to Back Up and Restore EKS with Velero —
    NetApp — https://www.netapp.com/learn/cbs-aws-blg-eks-back-up-how-to-back-up-and-restore-eks-with-velero/
39. Encryption At Rest — Amazon EKS Blueprints
    Patterns — https://aws-samples.github.io/cdk-eks-blueprints-patterns/patterns/security/encryption-at-rest/
40. Cloud Egress Costs Explained — EgressCost.com — https://www.egresscost.com/
41. Plan for large GKE clusters — Google Cloud
    Documentation — https://docs.cloud.google.com/kubernetes-engine/docs/concepts/planning-large-clusters
42. Performance and scaling best practices for large workloads in AKS — Microsoft
    Learn — https://learn.microsoft.com/en-us/azure/aks/best-practices-performance-scale-large
43. AWS EKS Secret Encryption with AWS KMS —
    Devoriales — https://devoriales.com/aws-eks-secret-encryption-securing-your-eks-secrets-at-rest-with-aws-kms
44. Karpenter vs. Cluster Autoscaler (production-test latency numbers) —
    Chkk — https://www.chkk.io/blog/karpenter-vs-cluster-autoscaler
45. Velero: Kubernetes Backup and Restore for Azure AKS — Medium / Mehmet
    Kanus — https://medium.com/hedgus/velero-kubernetes-backup-and-restore-solution-0cbd56f449be
46. How to Set Up Velero for Kubernetes Backup —
    OneUptime — https://oneuptime.com/blog/post/2026-01-25-velero-kubernetes-backup/view
47. Backup and restore Kubernetes cluster resources using Velero (contains GKE 100GB PVC backup timing) — CloudBees
    Docs — https://docs.cloudbees.com/docs/cloudbees-ci/latest/backup-restore/velero-dr
48. Understanding EBS Latency: Storage Type Performance in AWS —
    Percona — https://www.percona.com/blog/performance-of-various-ebs-storage-types-in-aws/