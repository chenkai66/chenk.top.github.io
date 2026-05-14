---
title: "Terraform for AI Agents (4): Compute — ECS, ACK, or Function Compute?"
date: 2026-03-18 09:00:00
tags:
  - Terraform
  - Alibaba Cloud
  - ECS
  - ACK
  - Function Compute
  - AI Agents
categories: Terraform
lang: en
mathjax: false
series: terraform-agents
series_title: "Terraform for AI Agents on Alibaba Cloud"
series_order: 4
description: "The three places an agent's main loop can live on Aliyun: a long-running ECS instance with pm2, a Kubernetes pod on ACK, or a Function Compute invocation. The cost-crossover model I use to pick between them, and a real cloud-init bootstrap that goes from bare Ubuntu to running agent in 90 seconds."
disableNunjucks: true
translationKey: "terraform-agents-4"
---

The single most important architectural decision in an agent system is *where the agent loop process runs*. There are three good options on Aliyun, plus a fourth that almost everyone forgets. Picking the wrong one isn't catastrophic — you can migrate later — but it costs weeks of unnecessary work and several thousand RMB a month in idle compute.

This article covers all four options with working Terraform, cost crossovers, and operational gotchas I often encounter.

---

## The four patterns

![Three primary places to run an agent: ECS, ACK, FC](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/04-compute-for-agent-runtime/fig1_three_compute_patterns.png)

![Auto-scaling compute cluster dynamically adjusting to workload](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/04-compute-for-agent-runtime/wanxiang_compute_scale.png)


Each has a sweet spot:

- **ECS** is a Linux VM. Long-lived, stateful, easy to SSH into when you're debugging. The right answer for prototypes, single-tenant agents, and anywhere you want to keep one machine "warm" with cached models or local state.
- **ACK** (Container Service for Kubernetes) is the prod answer at scale. Multiple agent kinds, autoscaling, rolling deploys, GPU scheduling. Worth the operational weight only when you have at least three or four agent services and an SRE who's comfortable in K8s.
- **Function Compute (FC)** is per-invocation, scale-to-zero. Cold start 200-800ms, hard cap of 24 hours per invocation. Right for webhook-triggered agents, scheduled crawlers, and anything that runs in bursts and idles otherwise.
- **Elastic Container Instance (ECI)** is the one people forget — a container without a node underneath. Cold start ~5s, billed per second, no node pool to manage. The sweet spot is bursty batch jobs that run 2-30 minutes per shot, several times an hour.

The first three cover most cases. ECI fills the gap between "too long for FC" and "too bursty for ECS".

## The cost crossover

Here's the rough monthly cost picture as a function of sustained QPS:

![Compute cost crossover — rough model](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/04-compute-for-agent-runtime/fig2_compute_cost_curve.png)

Under ~1 QPS sustained, FC dominates — you pay almost nothing during idle. From ~1 to ~30 QPS sustained, a single ECS box wins. Above that, ACK's higher fixed cost gets amortised over enough load to be cheaper than packing more onto ECS. ECI sits orthogonally: cheaper than ECS below 50% utilisation, more expensive above.

The model is rough — your actual numbers depend on the instance family, network, and the agent's chattiness — but the *shape* is reliable. Here's the decision rule I use:

> Bursty + low average → Function Compute
>
> Steady + low-to-mid → ECS with pm2
>
> Multi-agent + sustained mid-to-high → ACK
>
> Bursty batch, 2-30 min per run → ECI

With the framing in place, let's walk through each pattern and the Terraform that ships them.

## Pattern 1: ECS with pm2

For 80% of agent projects, this is what you want. One or two ECS instances behind an ALB, each running `pm2` as the supervisor for the Python or Node agent process.

```hcl
data "alicloud_images" "ubuntu" {
  owners      = "system"
  name_regex  = "^ubuntu_22_04_x64.*"
  most_recent = true
}

data "alicloud_instance_types" "agent" {
  cpu_core_count       = 4
  memory_size          = 16
  availability_zone    = "cn-shanghai-l"
  instance_type_family = "ecs.c7"
}

resource "alicloud_instance" "agent" {
  count = var.agent_count

  instance_name        = "agent-${terraform.workspace}-${count.index + 1}"
  image_id             = data.alicloud_images.ubuntu.images[0].id
  instance_type        = data.alicloud_instance_types.agent.instance_types[0].id
  availability_zone    = "cn-shanghai-l"

  vswitch_id      = module.vpc.private_vswitch_ids[count.index % 3]
  security_groups = [module.vpc.agent_runtime_sg_id]

  system_disk_category   = "cloud_essd"
  system_disk_size       = 80
  system_disk_encrypted  = true
  system_disk_kms_key_id = module.vpc.kms_keys["memory"]

  user_data = base64encode(templatefile("${path.module}/cloud-init.sh", {
    repo_url     = var.agent_repo_url
    branch       = var.agent_branch
    gateway_url  = "http://${alicloud_alb_listener.gateway.id}.alb.aliyuncs.com"
    sls_project  = alicloud_log_project.agents.name
    sls_logstore = alicloud_log_store.agent_runs.name
    role_name    = alicloud_ram_role.agent.name
    region       = "cn-shanghai"
  }))

  tags = {
    Role = "agent-runtime"
    App  = "research-agent"
  }

  lifecycle {
    create_before_destroy = true
    ignore_changes        = [user_data]   # don't replace on every cloud-init bump
  }
}
```

Three things worth highlighting:

1. **`data` blocks pick the image and instance type** instead of hardcoding them. `ubuntu_22_04_x64` resolves to the latest patched image; `data.alicloud_instance_types.agent` finds an `ecs.c7` with 4 vCPUs and 16 GiB. When Aliyun deprecates an image SKU, your next plan picks the new one automatically. Hardcoding `ecs.c7.xlarge` works until that exact SKU is out of stock in your zone, at which point Terraform fails — letting the data source pick gives you graceful fallback.
2. **`system_disk_kms_key_id` ties the disk to the `memory` CMK** from article 3. Encryption-at-rest costs nothing extra and removes a whole compliance headache.
3. **`lifecycle { create_before_destroy = true }`** means a planned replace creates the new instance, attaches it to the ALB, drains the old, then destroys it — zero-downtime rotation. The trade-off is you briefly need 2× capacity, which is fine for two-instance fleets and starts to matter at 50.

### The cloud-init that actually survives a real bootstrap

Most blog posts show a happy-path cloud-init: `apt-get install`, `git clone`, `pm2 start`, done. That works for a demo. Production bootstraps need to handle six things the happy path glosses over: DNS races, slow public mirrors, RAM credential timing, secret leakage via boot-disk snapshots, root-vs-service-account, and a bootstrap-done marker the ALB can gate on.

Here's the version I actually run:

```bash
#!/bin/bash
set -euxo pipefail

# 1. Wait for network — DNS isn't always ready when cloud-init starts
for i in {1..30}; do
  curl -sf https://mirrors.aliyun.com/ -o /dev/null && break
  sleep 2
done

# 2. Use Aliyun apt mirror — public Ubuntu mirrors are painfully slow from cn-shanghai
sed -i 's|http://.*\.ubuntu\.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list

apt-get update -y
apt-get install -y python3.11 python3.11-venv git curl ca-certificates jq nodejs npm
npm install -g pm2 uv

# 3. Fetch instance-role credentials from metadata BEFORE any tool that needs them
TOKEN_URL="http://100.100.100.200/latest/meta-data/ram/security-credentials/${role_name}"
TOKEN_JSON=$(curl -s --max-time 5 "$TOKEN_URL")
export ALICLOUD_ACCESS_KEY_ID=$(echo $TOKEN_JSON | jq -r .AccessKeyId)
export ALICLOUD_ACCESS_KEY_SECRET=$(echo $TOKEN_JSON | jq -r .AccessKeySecret)
export ALICLOUD_SECURITY_TOKEN=$(echo $TOKEN_JSON | jq -r .SecurityToken)

# 4. Pull secrets from KMS into tmpfs — they never hit disk, never appear in snapshots
mkdir -p /run/agent
mount -t tmpfs -o size=64M tmpfs /run/agent
python3 - <<'PY'
import os
from alibabacloud_kms20160120.client import Client as KmsClient
from alibabacloud_tea_openapi.models import Config
client = KmsClient(Config(
    region_id='${region}',
    access_key_id=os.environ['ALICLOUD_ACCESS_KEY_ID'],
    access_key_secret=os.environ['ALICLOUD_ACCESS_KEY_SECRET'],
    security_token=os.environ['ALICLOUD_SECURITY_TOKEN'],
))
for name in ['rds-password', 'litellm-master-key']:
    resp = client.get_secret_value(...)
    open(f'/run/agent/{name}', 'w').write(resp.body.secret_data)
PY
chmod 600 /run/agent/*

# 5. Clone the agent runtime, install deps with uv (10x faster than pip)
mkdir -p /opt/agent && cd /opt/agent
git clone --depth 1 -b ${branch} ${repo_url} src
cd src && uv venv .venv && uv pip sync requirements.txt

cat > /opt/agent/src/.env <<EOF
LLM_GATEWAY_URL=${gateway_url}
SLS_PROJECT=${sls_project}
SLS_LOGSTORE=${sls_logstore}
EOF

# 6. Run as a non-root service account, not root
useradd -r -s /bin/false agent || true
chown -R agent:agent /opt/agent /run/agent

# 7. Bootstrap-done marker for the ALB health check to gate on
sudo -u agent pm2 start ecosystem.config.js
sudo -u agent pm2 save
echo "$(date -Iseconds)" > /run/agent/bootstrap-done
```

The flow:

![Cloud-init bootstrap flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/04-compute-for-agent-runtime/fig3_cloud_init_flow.png)

About 90 seconds from `apply` to `pm2 status` showing the agent as `online`. The first `apt-get install` is the slow step (~60s). Once you have a stable image, **bake it with Packer** so future ECS instances skip apt entirely and boot in 25 seconds.

> **Real-world tip:** `user_data` is logged to `/var/log/cloud-init-output.log` on the instance. When an agent doesn't come up, that's where you look first. The `set -euxo pipefail` at the top makes failures loud and traceable — without it I lost two hours debugging a silent `pip install` failure that turned out to be a missing `gcc`.

## Pattern 2: ACK for production fleets

Once you have three or more agent kinds running side by side, the per-VM operational cost dominates and ECS stops scaling. ACK gives you one cluster, one scheduler, one upgrade path, and a single place to wire autoscaling and observability.

![Container orchestration managing distributed agent workloads](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/terraform-agents/04-compute-for-agent-runtime/wanxiang_container_orchestration.png)


The minimal Terraform to get a managed K8s cluster on Aliyun:

```hcl
resource "alicloud_cs_managed_kubernetes" "agents" {
  name_prefix                  = "agents-${terraform.workspace}"
  version                      = "1.30.1-aliyun.1"
  cluster_spec                 = "ack.pro.small"
  vpc_id                       = module.vpc.vpc_id
  worker_vswitch_ids           = module.vpc.private_vswitch_ids
  pod_vswitch_ids              = module.vpc.private_vswitch_ids
  service_cidr                 = "172.21.0.0/20"
  proxy_mode                   = "ipvs"
  load_balancer_spec           = "slb.s2.small"
  enable_ssh                   = false
  delete_protection            = true
  control_plane_log_components = ["apiserver", "audit"]

  addons { name = "managed-arms-prometheus" }
  addons { name = "logtail-ds" }
}

resource "alicloud_cs_kubernetes_node_pool" "agents" {
  cluster_id            = alicloud_cs_managed_kubernetes.agents.id
  node_pool_name        = "agent-workers"
  vswitch_ids           = module.vpc.private_vswitch_ids
  instance_types        = ["ecs.c7.xlarge"]
  desired_size          = 3
  system_disk_category  = "cloud_essd"
  system_disk_size      = 80
  install_cloud_monitor = true
  scaling_config {
    enable   = true
    min_size = 2
    max_size = 10
  }
}
```

A few notes:

- **`ack.pro.small`** is the managed control plane SKU. Aliyun runs the masters; you only pay for the worker ECS — about ¥350/month for the control plane on top of node cost. Don't pick the unmanaged SKU unless you have a strong reason.
- **`pod_vswitch_ids`** is for Terway, the Aliyun-native CNI. Each pod gets a real VPC IP — no overlay network, security groups apply directly. This is the right default; Flannel makes networking debugging miserable.
- **`delete_protection = true`** does what it says — `terraform destroy` won't kill the cluster. Set this on every prod cluster.
- The `addons` block enables ARMS Prometheus (article 7) and the SLS log collector. Provisioning these via Terraform means new clusters come pre-instrumented.

The actual agent pods come from a Kubernetes Deployment manifest — usually applied by a separate `kubectl` step or via the `kubernetes` Terraform provider. I keep the cluster in this `terraform` project and the workloads in a separate Helm chart, because they have different release cadences. The cluster changes once a quarter; the agent image changes ten times a day.

## Pattern 3: Function Compute for event-driven agents

Some agents only run when triggered — a webhook fires, a cron tick happens, an OSS object lands. For those, FC is unbeatable: zero idle cost, automatic scale-out, and the cloud handles the runtime entirely.

```hcl
resource "alicloud_fc_service" "agent" {
  name        = "agent-${terraform.workspace}"
  description = "Event-triggered agent functions"

  log_config {
    project  = alicloud_log_project.agents.name
    logstore = alicloud_log_store.agent_runs.name
  }

  vpc_config {
    vswitch_ids       = module.vpc.private_vswitch_ids
    security_group_id = module.vpc.agent_runtime_sg_id
  }

  role            = alicloud_ram_role.fc_agent.arn
  internet_access = false
}

resource "alicloud_fc_function" "scheduled_research" {
  service     = alicloud_fc_service.agent.name
  name        = "scheduled-research"
  description = "Daily research agent run"
  filename    = "${path.module}/dist/scheduled-research.zip"
  handler     = "index.handler"
  runtime     = "python3.11"
  memory_size = 1024
  timeout     = 600
  ca_port     = 8080

  environment_variables = {
    LLM_GATEWAY_URL = "http://${alicloud_alb_listener.gateway.id}.alb.aliyuncs.com"
  }
}

resource "alicloud_fc_trigger" "daily" {
  service  = alicloud_fc_service.agent.name
  function = alicloud_fc_function.scheduled_research.name
  name     = "daily-9am"
  type     = "timer"
  config = jsonencode({
    cronExpression = "0 0 9 * * *"
    enable         = true
    payload        = "{}"
  })
}
```

What this gives you: a Python 3.11 function, 1 GiB RAM, 10-minute timeout, attached to the same VPC and security group as the rest of your stack, triggered every day at 9am. Zero servers to maintain. Cost: roughly ¥0.10 per invocation at this size, plus ¥0.0001 per GB-second of execution. A daily cron that runs for 5 minutes costs about ¥3/month. The same agent on an idle ECS would be ~¥250/month.

Three caveats I keep tripping on:

1. **Cold start.** First invocation after idle takes 200-800ms more than subsequent ones. For a webhook with sub-second SLA, this matters; for a cron task, it doesn't. Provisioned concurrency exists but defeats the point of FC — once you're paying for warm instances 24/7, you may as well be on ECS.
2. **VPC attachment** adds another 200-400ms to cold starts because FC has to attach an ENI to your VPC. Worth it if the function needs to reach RDS/OpenSearch over private network; skip the `vpc_config` block if it only calls public APIs.
3. **24-hour max runtime.** For long agent loops, FC is a bad fit. Either chunk the loop into shorter steps (with state in OSS or Redis between steps) or move to ECS/ECI.

That third caveat is exactly the gap ECI fills.

## Pattern 4: ECI for bursty batch agents

ECS, ACK, and FC cover most cases, but there's a fourth pattern that comes into its own for *bursty* batch agent work: **Elastic Container Instance**. ECI is "a container without a Kubernetes node underneath" — Aliyun runs the container directly on a fleet of bare-metal hosts they manage, you only pay for the seconds it runs.

The use case is sharp. You have an agent that runs for 2-30 minutes per invocation, several times an hour, often in parallel. Function Compute caps out at 24h but the cold start is rough for batch; ECS is wasteful (you're paying for an idle box between bursts); ACK adds K8s overhead you don't want. ECI hits the middle: cold start ~5s, no idle cost, no node pool to manage.

```hcl
resource "alicloud_eci_container_group" "research_batch" {
  container_group_name = "research-batch-${formatdate("YYYYMMDDhhmm", timestamp())}"
  security_group_id    = module.vpc.agent_runtime_sg_id
  vswitch_id           = module.vpc.private_vswitch_ids[0]
  cpu                  = 4
  memory               = 16
  restart_policy       = "OnFailure"
  ram_role_name        = alicloud_ram_role.agent_eci.name

  containers {
    name              = "research"
    image             = "registry-vpc.cn-shanghai.aliyuncs.com/agents/research:${var.image_tag}"
    image_pull_policy = "IfNotPresent"
    cpu               = 4
    memory            = 16
    commands          = ["python", "-m", "research_agent"]
    args              = ["--session-id", var.session_id]

    environment_vars {
      key   = "LLM_GATEWAY_URL"
      value = module.gateway.url
    }
    volume_mounts {
      name       = "scratch"
      mount_path = "/scratch"
    }
  }

  volumes {
    name = "scratch"
    type = "EmptyDirVolume"
  }

  image_registry_credential {
    server    = "registry-vpc.cn-shanghai.aliyuncs.com"
    user_name = "agent-puller"
    password  = data.alicloud_kms_secret.acr_pull.secret_data
  }

  lifecycle {
    create_before_destroy = false                       # ECI is single-shot
    ignore_changes        = [container_group_name]      # timestamp-suffixed name drifts
  }
}
```

Production gotchas I've hit:

- **The image registry must use the `registry-vpc` endpoint**, not the public one. Public goes through your NAT and costs egress; VPC endpoint is free and faster. I once burned ¥200 in a week on egress before I noticed.
- **`restart_policy = "OnFailure"` not `"Always"`.** ECI is single-shot by design; `"Always"` makes it loop on success too, which is the opposite of what batch jobs want.
- **`EmptyDirVolume` is per-container-group ephemeral**, exactly like a Kubernetes emptyDir. Don't store anything here you can't reproduce — write outputs to OSS or RDS.
- **`ram_role_name`** lets the container fetch credentials from instance metadata, same pattern as ECS. No AK/SK in env vars.

The `alicloud_eci_container_group` resource has an annoying quirk: changes to `containers[*]` don't always trigger a clean replace because the API is partly merge-based. For production batch jobs I provision the container group once via Terraform with `ignore_changes` on the runtime spec, then trigger new runs by calling the ECI API directly from the orchestrator, using the role and SG that Terraform provisioned. **Treat the Terraform side as the *template*, not the *executor*.**

Cost rough math: a 4 vCPU / 16 GB ECI is ~¥0.96/hour, billed by the second after a 1-minute minimum. An 8-minute research run costs ~¥0.13. 100 runs/day = ¥13/day = ¥390/month. The same workload on a permanently-running ECS would be ¥250-400/month even when idle. **Below ~50% utilisation, ECI wins; above, ECS or ACK wins.**

## A real example: hybrid

Most production agent stacks I've shipped end up hybrid:

- ECS for the always-on conversational agent that holds session state in memory
- ACK for the worker fleet that processes background jobs across three or four agent kinds
- FC for webhook receivers and daily cron tasks
- ECI for batch jobs that take 5-20 minutes and run dozens of times a day

Terraform makes this trivial — four modules in the same project, sharing the VPC and security groups from article 3. The skill is knowing which pattern fits which workload, not learning all the resource syntaxes.

## Right-sizing the instance

A common question: which `ecs.*` family for an agent runtime? My defaults:

| Workload                        | Family               | Why |
|---------------------------------|----------------------|-----|
| Conversational agent, no GPU    | `ecs.c7`             | CPU-bound on tokenisation + I/O on LLM calls |
| Memory-heavy (large context)    | `ecs.r7`             | More RAM per vCPU |
| Batch / scheduled with bursts   | `ecs.c7a` (AMD)      | ~15% cheaper, slightly slower per core |
| GPU inference of small models   | `ecs.gn7i`           | T4-class, cheapest GPU on Aliyun |
| Pretraining / large fine-tune   | Use PAI-DLC, not ECS | Don't reinvent the orchestration |

Avoid the burstable `ecs.t6` family for agent runtime — CPU credits run out under sustained load and your latency goes off a cliff. They're fine for the bastion that runs `terraform apply` and not much else.

## When `alicloud_pai_*` doesn't exist (and the fallback)

If you're shipping an agent that does its own training or hosts a custom LLM, the obvious first instinct is to spin up PAI-EAS via Terraform. Reality check: as of `alicloud` provider 1.230, PAI resource coverage is thin. You have `alicloud_pai_workspace` and a few related resources, but full EAS service deployment is not first-class — there's no `alicloud_pai_eas_service` that lets you declare a model serving endpoint declaratively in HCL.

The practical fallbacks I use, in order of preference:

**Fallback 1: provision the workspace, run EAS via API.** Terraform owns the workspace and IAM; an imperative tool (`eascmd` or the EAS Python SDK) owns the runtime spec.

```hcl
resource "alicloud_pai_workspace" "agents" {
  workspace_name = "agents-${terraform.workspace}"
  description    = "PAI workspace for agent inference"
  env_types      = ["prod"]
  display_name   = "Agents"
}

output "pai_workspace_id" {
  value = alicloud_pai_workspace.agents.id
}
```

A `Makefile` target then calls `eascmd create eas-service.json` where the JSON contains image, processor, instance type, and autoscaling config. This is honest about the split: declarative for the stable substrate, imperative for the moving runtime.

**Fallback 2: ROS for the EAS bits.** ROS (Aliyun's native IaC) often supports resources before the Terraform provider does. You can call ROS from Terraform via `alicloud_ros_stack`:

```hcl
resource "alicloud_ros_stack" "eas_serving" {
  stack_name    = "eas-serving-${terraform.workspace}"
  template_body = file("${path.module}/eas-service.ros.json")

  parameters {
    parameter_key   = "WorkspaceId"
    parameter_value = alicloud_pai_workspace.agents.id
  }
  parameters {
    parameter_key   = "ImageUrl"
    parameter_value = "registry-vpc.cn-shanghai.aliyuncs.com/agents/qwen-server:${var.model_version}"
  }
  timeout_in_minutes = 30
}
```

The `eas-service.ros.json` is a ROS template — Aliyun-flavoured CloudFormation — and exposes EAS properties the alicloud Terraform provider doesn't yet. Terraform owns the stack, ROS owns the deep-cloud bits, and you get a unified `terraform plan` across both.

**Fallback 3: write your own provider.** Don't. I tried it once — two weeks of Go work, then the official provider shipped the resource I needed a month later and I had a fork to maintain forever. The Aliyun team ships new alicloud provider resources monthly; you'd rather their work catch up than carry a fork.

The general principle for any "Terraform doesn't have it yet" situation: **Terraform owns the stable substrate, lighter tools own the moving parts.** RAM roles, VPC, KMS, OSS — Terraform. Brand-new resources, beta APIs — `null_resource` + `local-exec`, ROS embedded, or a separate orchestrator. As the provider catches up, migrate the `null_resource` blocks into proper resources with `terraform import`.

> **Real-world tip:** `alicloud_ros_stack` is genuinely underrated. When the alicloud Terraform provider lags, ROS often has the API a quarter ahead. Embedding ROS templates inside Terraform gives you the best of both.

## What's next

Article 5 fills in the storage layer — vector store, relational, object store, backups — that everything we just provisioned needs to talk to. ECS instances, ACK pods, FC functions, and ECI containers are all useless until they have somewhere to put memory.

Then article 6 builds the LLM gateway in front of all the compute, article 7 wires observability and cost alarms, and article 8 stitches everything into one `terraform apply`.
