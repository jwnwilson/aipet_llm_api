# Self-Hosted GitHub Actions Runner Setup

The `deploy` job in `.github/workflows/deploy.yml` uses `runs-on: [self-hosted, k3s]`.
You need one registered runner on the machine running k3s (the Raspberry Pi or dev box).

## Prerequisites

- The machine running k3s has internet access
- `kubectl` is installed and `~/.kube/config` points to the local cluster
- GitHub Personal Access Token with `repo` scope (or use the `gh` CLI)

## Install the runner

Run these commands **on the k3s host** (not your dev machine).

Check https://github.com/actions/runner/releases for the latest ARM64 version before running.

```bash
# 1. Download the runner package
mkdir -p ~/actions-runner && cd ~/actions-runner
curl -o actions-runner-linux-arm64-2.321.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.321.0/actions-runner-linux-arm64-2.321.0.tar.gz
tar xzf ./actions-runner-linux-arm64-2.321.0.tar.gz

# 2. Configure — get the registration token from:
#    https://github.com/<owner>/<repo>/settings/actions/runners/new
./config.sh \
  --url https://github.com/<owner>/aipet-llm \
  --token <RUNNER_TOKEN> \
  --name rpi5-k3s \
  --labels self-hosted,k3s,linux,arm64 \
  --unattended

# 3. Install as a systemd service so it survives reboots
sudo ./svc.sh install
sudo ./svc.sh start
```

## Verify registration

After `./config.sh` completes, the runner appears at:
`https://github.com/<owner>/aipet-llm/settings/actions/runners`

Status should be **Idle**.

## AWS credentials on the runner

The deploy job calls `aws ecr get-login-password` to refresh the ECR pull secret in k8s.
The runner machine needs AWS credentials. Preferred: attach an IAM instance profile to the
machine. Alternative: set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in the runner's
environment file (`~/actions-runner/.env`).

The required IAM policy is the same `ecr-push` policy created by Terraform:

```bash
terraform -chdir=infra/terraform output ecr_push_policy_arn
```

## Removing the runner

```bash
cd ~/actions-runner
sudo ./svc.sh stop
sudo ./svc.sh uninstall
./config.sh remove --token <REMOVE_TOKEN>
```
